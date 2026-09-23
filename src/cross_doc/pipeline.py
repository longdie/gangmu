from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.common.config import GangmuConfig, get_config
from src.common.files import (
    ProjectLogger,
    TokenUsageTracker,
    clean_whitespace,
    read_json,
    read_text,
    utc_now,
)
from src.common.paths import PROMPT_DIR
from src.cross_doc.clients import CanonicalStatementEmbedder, RerankerClient
from src.cross_doc.pacing import CrossDocumentRequestPacer
from src.cross_doc.resolution import (
    FrozenCandidateResolver,
    candidate_record as frozen_candidate_record,
    evidence_payload as frozen_evidence_payload,
)
from src.cross_doc.schema import (
    CONTENT_ALGORITHM_VERSION,
    DATA_FILES,
    LEGACY_DATA_FILES,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    validate_canonical_graph,
)
from src.evidence_graph.derivation import (
    EMBEDDING_DIMENSIONS,
    clean_model_text,
    normalized_statement,
)
from src.evidence_graph.schema import (
    DATA_FILES as EVIDENCE_DATA_FILES,
    validate_evidence_graph,
)
from src.single_doc.llm import LLMClient


MANIFEST_FILE = "manifest.json"
CHECKPOINT_DIR_NAME = ".checkpoint"
CHECKPOINT_POINTER_FILE = "LATEST"
CHECKPOINT_VERSION = 5
LEGACY_CHECKPOINT_VERSION = 1
PRE_DOCUMENT_CHECKPOINT_VERSION = 2
PRE_ONE_TO_ONE_CHECKPOINT_VERSION = 3
PRE_CONSTRUCTION_RECORDS_CHECKPOINT_VERSION = 4
ALIGNMENT_ALGORITHM_VERSION = 2
CANONICAL_CONTENT_MAX_ATTEMPTS = 3
CONSTRUCTION_RECORDS_DIR_NAME = "construction_records"
CHECKPOINT_RECORDS_DIR_NAME = "construction_records"
CONSTRUCTION_RECORDS_SCHEMA_VERSION = "1.0"
LEGACY_CANONICAL_STATEMENT_PROMPT_SHA256 = (
    "de3ae659e40dcb267e7e7200df61793dfff932ccf08e2707ac3b186d49e2fb07"
)
RELATIONS = {
    "equivalent",
    "possible_equivalent",
    "not_equivalent",
}
TYPE_ORDER = {"dependency": 0, "proof": 1}
PROMPT_PATHS = {
    "alignment": PROMPT_DIR / "cross_doc_alignment.md",
    "strict_cluster_pair": PROMPT_DIR / "cross_doc_strict_cluster_pair.md",
    "statement": PROMPT_DIR / "canonical_statement.md",
    "statement_validation": (PROMPT_DIR / "canonical_statement_validation.md"),
}


class CanonicalGraphPipeline:
    """Build or incrementally update one Canonical Graph."""

    def __init__(
        self,
        *,
        canonical_graph_dir: str | Path,
        evidence_graph_dir: str | Path,
        config: GangmuConfig | None = None,
        logger: ProjectLogger | None = None,
        usage_tracker: TokenUsageTracker | None = None,
        llm_client: Any | None = None,
        reranker_client: Any | None = None,
        statement_embedder: Any | None = None,
        request_pacer: CrossDocumentRequestPacer | None = None,
        checkpoint_interval: int | None = None,
        prompt_paths: dict[str, str | Path] | None = None,
        checkpoint_model_transitions: (
            dict[str, dict[str, dict[str, str]]] | None
        ) = None,
    ) -> None:
        self.graph_dir = Path(canonical_graph_dir).expanduser().resolve()
        self.evidence_graph_dir = Path(evidence_graph_dir).expanduser().resolve()
        self.graph_id = self.graph_dir.name
        self.evidence_graph_id = self.evidence_graph_dir.name
        if not self.graph_id or not self.evidence_graph_id:
            raise ValueError("Graph directories must have non-empty names.")
        if self.graph_dir == self.evidence_graph_dir:
            raise ValueError(
                "Canonical Graph and Evidence Graph directories must differ."
            )

        self.config = config or get_config()
        self.logger = logger or ProjectLogger(
            self.config.log_dir("cross_document"),
            run_name=self.graph_id,
        )
        self.usage_tracker = usage_tracker or TokenUsageTracker()
        configured_interval = self.config.get_int(
            "cross_document.checkpoint_every_stage_results",
            minimum=1,
        )
        self.checkpoint_interval = (
            configured_interval if checkpoint_interval is None else checkpoint_interval
        )
        if (
            isinstance(self.checkpoint_interval, bool)
            or not isinstance(self.checkpoint_interval, int)
            or self.checkpoint_interval < 1
        ):
            raise ValueError("checkpoint_interval must be a positive integer.")

        configured_prompt_paths = {
            key: Path(path).expanduser().resolve() for key, path in PROMPT_PATHS.items()
        }
        for key, path in (prompt_paths or {}).items():
            if key not in configured_prompt_paths:
                raise ValueError(f"Unknown cross-document prompt key: {key}")
            configured_prompt_paths[key] = Path(path).expanduser().resolve()
        self.prompt_paths = configured_prompt_paths
        self.prompts = {key: read_text(path) for key, path in self.prompt_paths.items()}
        # These hashes describe the prompts that actually produced the stored
        # alignment decisions. They can intentionally differ from the prompt
        # files currently loaded for a postprocessing-only checkpoint resume.
        self.alignment_prompt_sha256_used = self._prompt_sha256("alignment")
        self.strict_cluster_pair_prompt_sha256_used = self._prompt_sha256(
            "strict_cluster_pair"
        )
        self.checkpoint_model_transitions = _normalize_checkpoint_model_transitions(
            checkpoint_model_transitions or {}
        )

        self.alignment_model = self.config.model("cross_document_alignment")
        self.strict_model = self.config.model("cross_document_strict_verification")
        self.statement_model = self.config.model("canonical_statement")
        self.embedding_model = self.config.model("embedding")
        self.reranker_model = self.config.model("reranker")
        cross_document_llm_identities = {
            (model.name, model.provider)
            for model in (
                self.alignment_model,
                self.strict_model,
                self.statement_model,
            )
        }
        if len(cross_document_llm_identities) != 1:
            raise ValueError(
                "Cross-document alignment, strict verification, and Canonical "
                "content generation must use one shared model and provider."
            )
        provider_failure_limit = self.config.get_int(
            "cross_document.provider_failures_before_abort",
            minimum=1,
        )
        llm_call_attempts = self.config.get_int(
            "cross_document.llm_call_attempts",
            minimum=1,
        )
        llm_retry_delay_seconds = self.config.get_float(
            "cross_document.llm_retry_delay_seconds",
            minimum=0,
        )
        self.llm_request_pacer = (
            request_pacer
            or CrossDocumentRequestPacer.from_config(
                self.config,
                announce=self._announce_pacing,
            )
        )
        self.llm_client = llm_client or LLMClient(
            config=self.config,
            logger=self.logger,
            usage_tracker=self.usage_tracker,
            consecutive_failures_before_abort=provider_failure_limit,
            call_attempts=llm_call_attempts,
            retry_delay_seconds=llm_retry_delay_seconds,
            request_pacer=self.llm_request_pacer,
        )
        self.reranker_client = reranker_client or RerankerClient(
            config=self.config,
            logger=self.logger,
            usage_tracker=self.usage_tracker,
        )
        self.statement_embedder = statement_embedder or CanonicalStatementEmbedder(
            config=self.config,
            logger=self.logger,
            usage_tracker=self.usage_tracker,
        )

        self.evidence_retrieval_top_k = self.config.get_int(
            "cross_document.evidence_retrieval_top_k",
            minimum=1,
        )
        self.evidence_similarity_threshold = self.config.get_float(
            "cross_document.evidence_similarity_threshold",
            minimum=-1,
            maximum=1,
        )
        self.reranker_top_k = self.config.get_int(
            "cross_document.reranker_top_k",
            minimum=1,
        )
        self.reranker_parallel_workers = self.config.get_int(
            "cross_document.reranker_parallel_workers",
            minimum=1,
            maximum=5,
        )
        self.llm_parallel_workers = self.config.get_int(
            "cross_document.llm_parallel_workers",
            minimum=1,
            maximum=5,
        )
        self.content_parallel_workers = self.config.get_int(
            "cross_document.content_parallel_workers",
            minimum=1,
            maximum=5,
        )
        self.low_cohesion_similarity = self.config.get_float(
            "cross_document.low_cohesion_similarity",
            minimum=-1,
            maximum=1,
        )
        self.representative_min_similarity = self.config.get_float(
            "cross_document.representative_min_centroid_similarity",
            minimum=-1,
            maximum=1,
        )

        self.checkpoint_dir = self.graph_dir / CHECKPOINT_DIR_NAME
        self.checkpoint_records_dir = (
            self.checkpoint_dir / CHECKPOINT_RECORDS_DIR_NAME
        )
        safe_lock_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.graph_id)
        self.lock_path = self.logger.log_dir / f"{safe_lock_name}.lock"

        # Populated after validated input loading.
        self.evidence_manifest: dict[str, Any] = {}
        self.evidence_sources: list[dict[str, Any]] = []
        self.evidence_nodes: list[dict[str, Any]] = []
        self.evidence_edges: list[dict[str, Any]] = []
        self.evidence_embeddings = np.empty(
            (0, EMBEDDING_DIMENSIONS),
            dtype=np.float32,
        )
        self.source_by_id: dict[str, dict[str, Any]] = {}
        self._evidence_fingerprint_cache: dict[str, Any] | None = None

        # Mutable construction state.
        self.nodes: list[dict[str, Any]] = []
        self.mapping = np.empty((0,), dtype=np.int64)
        self.embedding_sums = np.empty(
            (0, EMBEDDING_DIMENSIONS),
            dtype=np.float32,
        )
        self.centroid_embeddings = np.empty(
            (0, EMBEDDING_DIMENSIONS),
            dtype=np.float32,
        )
        self.statement_embeddings = np.empty(
            (0, EMBEDDING_DIMENSIONS),
            dtype=np.float32,
        )
        self.title_embeddings = np.empty(
            (0, EMBEDDING_DIMENSIONS),
            dtype=np.float32,
        )
        self.center_evidence_embeddings = np.empty(
            (0, EMBEDDING_DIMENSIONS),
            dtype=np.float32,
        )
        self.possible_pairs: set[tuple[int, int]] = set()
        self.dirty_content_node_ids: set[int] = set()
        self.fallback_statement_node_ids: set[int] = set()
        self.fallback_title_node_ids: set[int] = set()
        self.content_validation_failures: dict[int, dict[str, Any]] = {}
        self.processed_evidence_node_count = 0
        self.base_revision = 0
        self.target_revision = 1
        self.base_existed = False
        self.created_at = ""
        self.base_usage: dict[str, Any] = {}
        self.phase = "alignment"
        self.alignment_model_history: list[dict[str, Any]] = []
        self.strict_verification_model_history: list[dict[str, Any]] = []
        self._checkpoint_model_migrated = False
        self._checkpoint_model_migration_messages: list[str] = []
        self._checkpoint_schema_migrated = False
        self._checkpoint_schema_migration_messages: list[str] = []
        # A document is planned against one frozen Canonical snapshot. Paid
        # stage results remain here until every node in the document has a
        # proposal and the proposals have been committed serially.
        self.active_document_source_id = ""
        self.active_document_node_ids: list[int] = []
        self.active_document_results: dict[str, dict[str, Any]] = {}
        self.active_document_snapshot_canonical_node_count = 0
        self._active_document_stage_changes = 0
        self._document_commit_in_progress = False
        self.construction_recording_started_at = utc_now()
        self.construction_recording_start_evidence_node_id = 1
        self.construction_records_complete = True
        self.construction_prompt_snapshots = dict(self.prompts)
        self._candidate_resolver_cache: (
            tuple[tuple[int, int, int], FrozenCandidateResolver] | None
        ) = None

    def _announce_pacing(self, message: str) -> None:
        print(message)
        self.logger.info(
            "Cross-document LLM pacing event",
            pacing_message=message,
        )

    def run(self) -> dict[str, Any]:
        with self._exclusive_lock():
            self._load_evidence_graph()
            existing = self._load_existing_canonical()
            checkpoint = self._load_checkpoint(existing=existing)
            if checkpoint is not None:
                self._restore_checkpoint(checkpoint)
                self.logger.info(
                    "Resuming Canonical Graph from checkpoint",
                    graph_id=self.graph_id,
                    target_revision=self.target_revision,
                    phase=self.phase,
                    processed_evidence_nodes=(self.processed_evidence_node_count),
                    dirty_content=len(self.dirty_content_node_ids),
                )
                if self._checkpoint_model_migrated:
                    for message in self._checkpoint_model_migration_messages:
                        print(
                            f"[canonical-graph] checkpoint model migration: {message}"
                        )
                        self.logger.warning(
                            "Canonical Graph checkpoint model migrated",
                            migration=message,
                            processed_evidence_nodes=(
                                self.processed_evidence_node_count
                            ),
                        )
                    # Publish the new signature and auditable model histories
                    # before making the first paid request with the new model.
                    self._save_checkpoint()
                if self._checkpoint_schema_migrated:
                    for message in self._checkpoint_schema_migration_messages:
                        print(
                            f"[canonical-graph] checkpoint schema migration: {message}"
                        )
                        self.logger.warning(
                            "Canonical Graph checkpoint schema migrated",
                            migration=message,
                            processed_evidence_nodes=(
                                self.processed_evidence_node_count
                            ),
                        )
                    self._save_checkpoint()
                reopened_title_ids = self._reopen_invalid_finalization_titles()
                if reopened_title_ids:
                    print(
                        "[canonical-graph] finalization title repair: "
                        f"reopened={reopened_title_ids}"
                    )
                    self.logger.warning(
                        "Reopened invalid Canonical titles before finalization",
                        canonical_node_ids=reopened_title_ids,
                    )
                    # Persist the repaired phase before issuing any paid call.
                    self._save_checkpoint()
            else:
                if (
                    existing["existed"]
                    and existing.get("schema_version") == SCHEMA_VERSION
                    and int(
                        existing["manifest"].get(
                            "evidence_graph_revision",
                            0,
                        )
                        or 0
                    )
                    == int(self.evidence_manifest["revision"])
                    and int(
                        existing["manifest"]
                        .get("construction", {})
                        .get("processed_evidence_node_count", 0)
                        or 0
                    )
                    == len(self.evidence_nodes)
                    and int(
                        existing["manifest"].get("counts", {}).get("evidence_edges", 0)
                        or 0
                    )
                    == len(self.evidence_edges)
                    and int(
                        existing["manifest"]
                        .get("construction", {})
                        .get("content_algorithm_version", 0)
                        or 0
                    )
                    == CONTENT_ALGORITHM_VERSION
                ):
                    return self._no_op_summary(existing["manifest"])
                self._initialize_state(existing)
                # Preserve all imported state before the first paid call.
                self._save_checkpoint()

            try:
                if self.phase == "alignment":
                    self._run_alignment()
                    self.phase = "content"
                    self._save_checkpoint()
                if self.phase == "content":
                    self._run_content_postprocessing()
                    self.phase = "finalization"
                    self._save_checkpoint()
                result = self._finalize_and_commit()
            except BaseException as exc:
                # A forced checkpoint also records paid attempts since the most
                # recent periodic snapshot. Do not hide the original failure.
                if self._document_commit_in_progress:
                    # The complete proposal set was saved immediately before
                    # the local commit began. A half-mutated in-memory graph is
                    # not a valid recovery point; leave the pre-commit snapshot
                    # untouched so restart can replay the commit without APIs.
                    self.logger.warning(
                        "Document commit was interrupted; preserving the "
                        "pre-commit checkpoint",
                        error=str(exc),
                        source_id=self.active_document_source_id,
                    )
                else:
                    try:
                        self._save_checkpoint()
                    except Exception as checkpoint_error:
                        self.logger.exception(
                            "Failed to save emergency Canonical Graph checkpoint",
                            checkpoint_error,
                            original_error=str(exc),
                        )
                raise
            self._delete_checkpoint()

        self.logger.info("Canonical Graph update completed", **result)
        return result

    def _load_evidence_graph(self) -> None:
        if not self.evidence_graph_dir.is_dir():
            raise FileNotFoundError(
                f"Evidence Graph directory does not exist: {self.evidence_graph_dir}"
            )
        required = {
            EVIDENCE_DATA_FILES["sources"],
            EVIDENCE_DATA_FILES["nodes"],
            EVIDENCE_DATA_FILES["edges"],
            EVIDENCE_DATA_FILES["embeddings"],
            EVIDENCE_DATA_FILES["usage"],
            MANIFEST_FILE,
        }
        missing = sorted(
            filename
            for filename in required
            if not (self.evidence_graph_dir / filename).is_file()
        )
        if missing:
            raise ValueError(
                "Evidence Graph is incomplete; missing: " + ", ".join(missing)
            )

        manifest = read_json(self.evidence_graph_dir / MANIFEST_FILE)
        sources = read_json(self.evidence_graph_dir / EVIDENCE_DATA_FILES["sources"])
        nodes = read_json(self.evidence_graph_dir / EVIDENCE_DATA_FILES["nodes"])
        edges = read_json(self.evidence_graph_dir / EVIDENCE_DATA_FILES["edges"])
        embeddings = np.load(
            self.evidence_graph_dir / EVIDENCE_DATA_FILES["embeddings"],
            allow_pickle=False,
            mmap_mode="r",
        )
        usage = read_json(self.evidence_graph_dir / EVIDENCE_DATA_FILES["usage"])
        errors = validate_evidence_graph(
            manifest=manifest,
            sources=sources,
            nodes=nodes,
            edges=edges,
            embeddings=embeddings,
            usage=usage,
            expected_graph_id=self.evidence_graph_id,
        )
        if errors:
            raise ValueError("Evidence Graph is invalid: " + "; ".join(errors))
        if any(not normalized_statement(node) for node in nodes):
            raise ValueError("Every Evidence Node must have a normalized statement.")

        self.evidence_manifest = manifest
        self.evidence_sources = sources
        self.evidence_nodes = nodes
        self.evidence_edges = edges
        self.evidence_embeddings = embeddings
        self.source_by_id = {
            clean_whitespace(str(source["id"])): source for source in sources
        }
        closed_sources: set[str] = set()
        previous_source = ""
        for node in nodes:
            source_id = clean_whitespace(str(node["source_id"]))
            if source_id == previous_source:
                continue
            if source_id in closed_sources:
                raise ValueError(
                    "Evidence Nodes for one source must form one contiguous "
                    f"range; source {source_id!r} appears more than once."
                )
            if previous_source:
                closed_sources.add(previous_source)
            previous_source = source_id

    def _load_existing_canonical(self) -> dict[str, Any]:
        empty = {
            "manifest": {},
            "nodes": [],
            "edges": [],
            "possible_equivalence_edges": [],
            "mapping": np.empty((0,), dtype=np.int64),
            "embedding_sums": np.empty(
                (0, EMBEDDING_DIMENSIONS),
                dtype=np.float32,
            ),
            "centroid_embeddings": np.empty(
                (0, EMBEDDING_DIMENSIONS),
                dtype=np.float32,
            ),
            "statement_embeddings": np.empty(
                (0, EMBEDDING_DIMENSIONS),
                dtype=np.float32,
            ),
            "title_embeddings": np.empty(
                (0, EMBEDDING_DIMENSIONS),
                dtype=np.float32,
            ),
            "center_evidence_embeddings": np.empty(
                (0, EMBEDDING_DIMENSIONS),
                dtype=np.float32,
            ),
            "diagnostics": {},
            "usage": {},
            "schema_version": SCHEMA_VERSION,
            "existed": False,
        }
        if not self.graph_dir.exists():
            return empty
        if not self.graph_dir.is_dir():
            raise NotADirectoryError(
                f"Canonical Graph path is not a directory: {self.graph_dir}"
            )

        entries = [
            path
            for path in self.graph_dir.iterdir()
            if path.name != CHECKPOINT_DIR_NAME
        ]
        if not entries:
            return empty
        present = {path.name for path in entries if path.is_file()}
        if MANIFEST_FILE not in present:
            raise ValueError(
                "Canonical Graph directory is non-empty but has no manifest: "
                f"{self.graph_dir}"
            )
        manifest = read_json(self.graph_dir / MANIFEST_FILE)
        if not isinstance(manifest, dict):
            raise ValueError("Canonical Graph manifest must be an object.")
        schema_version = manifest.get("schema_version")
        if schema_version == SCHEMA_VERSION:
            graph_files = DATA_FILES
        elif schema_version == LEGACY_SCHEMA_VERSION:
            graph_files = LEGACY_DATA_FILES
        else:
            raise ValueError(
                f"Unsupported Canonical Graph schema version: {schema_version!r}"
            )
        required = {*graph_files.values(), MANIFEST_FILE}
        if present & required != required:
            missing = sorted(required - present)
            raise ValueError(
                "Canonical Graph is incomplete; missing: " + ", ".join(missing)
            )
        records_path = self.graph_dir / CONSTRUCTION_RECORDS_DIR_NAME
        if records_path.exists() and not records_path.is_dir():
            raise ValueError(
                "Canonical Graph construction_records entry must be a directory."
            )
        allowed_entries = {*required, CONSTRUCTION_RECORDS_DIR_NAME}
        unexpected = sorted(
            path.name for path in entries if path.name not in allowed_entries
        )
        if unexpected:
            raise ValueError(
                "Canonical Graph directory contains unexpected entries: "
                + ", ".join(unexpected)
            )

        loaded = {
            "manifest": manifest,
            "nodes": read_json(self.graph_dir / graph_files["nodes"]),
            "edges": read_json(self.graph_dir / graph_files["edges"]),
            "possible_equivalence_edges": read_json(
                self.graph_dir / graph_files["possible_equivalence_edges"]
            ),
            "mapping": np.load(
                self.graph_dir / graph_files["evidence_node_to_canonical"],
                allow_pickle=False,
            ),
            "embedding_sums": np.load(
                self.graph_dir / graph_files["embedding_sums"],
                allow_pickle=False,
            ),
            "centroid_embeddings": np.load(
                self.graph_dir / graph_files["centroid_embeddings"],
                allow_pickle=False,
            ),
            "statement_embeddings": np.load(
                self.graph_dir / graph_files["statement_embeddings"],
                allow_pickle=False,
            ),
            "diagnostics": read_json(self.graph_dir / graph_files["diagnostics"]),
            "usage": read_json(self.graph_dir / graph_files["usage"]),
            "schema_version": schema_version,
            "existed": True,
        }
        if schema_version == SCHEMA_VERSION:
            loaded["title_embeddings"] = np.load(
                self.graph_dir / DATA_FILES["title_embeddings"],
                allow_pickle=False,
            )
            loaded["center_evidence_embeddings"] = np.load(
                self.graph_dir / DATA_FILES["center_evidence_embeddings"],
                allow_pickle=False,
            )
        else:
            node_count = len(loaded["nodes"])
            loaded["title_embeddings"] = np.zeros(
                (node_count, EMBEDDING_DIMENSIONS),
                dtype=np.float32,
            )
            loaded["center_evidence_embeddings"] = self._center_embedding_matrix(
                loaded["nodes"]
            )
        manifest = loaded["manifest"]
        if manifest.get("evidence_graph_id") != self.evidence_graph_id:
            raise ValueError(
                "The Canonical Graph is bound to a different Evidence Graph."
            )
        previous_node_count = int(
            manifest.get("counts", {}).get("evidence_nodes", 0) or 0
        )
        previous_edge_count = int(
            manifest.get("counts", {}).get("evidence_edges", 0) or 0
        )
        if previous_node_count > len(self.evidence_nodes):
            raise ValueError(
                "Evidence Graph has fewer nodes than the existing Canonical "
                "Graph processed."
            )
        if previous_edge_count > len(self.evidence_edges):
            raise ValueError(
                "Evidence Graph has fewer edges than the existing Canonical "
                "Graph processed."
            )
        self._validate_evidence_prefix(
            manifest=manifest,
            node_count=previous_node_count,
            edge_count=previous_edge_count,
        )
        previous_evidence_manifest = deepcopy(self.evidence_manifest)
        previous_evidence_manifest["revision"] = manifest.get("evidence_graph_revision")
        previous_evidence_manifest["counts"] = {
            **previous_evidence_manifest.get("counts", {}),
            "nodes": previous_node_count,
            "edges": previous_edge_count,
        }
        errors = validate_canonical_graph(
            manifest=manifest,
            nodes=loaded["nodes"],
            edges=loaded["edges"],
            possible_equivalence_edges=loaded["possible_equivalence_edges"],
            evidence_node_to_canonical=loaded["mapping"],
            embedding_sums=loaded["embedding_sums"],
            centroid_embeddings=loaded["centroid_embeddings"],
            statement_embeddings=loaded["statement_embeddings"],
            title_embeddings=(
                loaded["title_embeddings"] if schema_version == SCHEMA_VERSION else None
            ),
            center_evidence_embeddings=(
                loaded["center_evidence_embeddings"]
                if schema_version == SCHEMA_VERSION
                else None
            ),
            diagnostics=loaded["diagnostics"],
            usage=loaded["usage"],
            evidence_manifest=previous_evidence_manifest,
            evidence_nodes=self.evidence_nodes[:previous_node_count],
            evidence_edges=self.evidence_edges[:previous_edge_count],
            evidence_embeddings=self.evidence_embeddings[:previous_node_count],
            expected_graph_id=self.graph_id,
        )
        if errors:
            raise ValueError(
                "Existing Canonical Graph is invalid: " + "; ".join(errors)
            )
        return loaded

    def _initialize_state(self, existing: dict[str, Any]) -> None:
        self.alignment_prompt_sha256_used = self._prompt_sha256("alignment")
        self.strict_cluster_pair_prompt_sha256_used = self._prompt_sha256(
            "strict_cluster_pair"
        )
        self.base_existed = bool(existing["existed"])
        manifest = existing["manifest"]
        self.base_revision = int(manifest.get("revision", 0) or 0)
        self.target_revision = self.base_revision + 1
        self.created_at = (
            clean_whitespace(str(manifest.get("created_at", "") or "")) or utc_now()
        )
        self.base_usage = deepcopy(existing["usage"])
        self.nodes = deepcopy(existing["nodes"])
        for node in self.nodes:
            node["in_edges"] = []
            node["out_edges"] = []
            node["possible_equivalence_edge_ids"] = []
        old_mapping = np.asarray(existing["mapping"], dtype=np.int64)
        self.mapping = np.zeros(len(self.evidence_nodes), dtype=np.int64)
        self.mapping[: old_mapping.shape[0]] = old_mapping
        self.embedding_sums = np.asarray(
            existing["embedding_sums"],
            dtype=np.float32,
        ).copy()
        self.centroid_embeddings = np.asarray(
            existing["centroid_embeddings"],
            dtype=np.float32,
        ).copy()
        self.statement_embeddings = np.asarray(
            existing["statement_embeddings"],
            dtype=np.float32,
        ).copy()
        self.title_embeddings = np.asarray(
            existing["title_embeddings"],
            dtype=np.float32,
        ).copy()
        self.center_evidence_embeddings = np.asarray(
            existing["center_evidence_embeddings"],
            dtype=np.float32,
        ).copy()
        self.possible_pairs = {
            (int(edge["node_a"]), int(edge["node_b"]))
            for edge in existing["possible_equivalence_edges"]
        }
        self.fallback_statement_node_ids = {
            int(node_id)
            for node_id in existing.get("diagnostics", {}).get(
                "fallback_statement_canonical_node_ids",
                [],
            )
        }
        self.fallback_title_node_ids = {
            int(node_id)
            for node_id in existing.get("diagnostics", {}).get(
                "fallback_title_canonical_node_ids",
                [],
            )
        }
        self.content_validation_failures = (
            _content_validation_failures_from_diagnostics(
                existing.get("diagnostics", {}),
                canonical_node_count=len(self.nodes),
            )
        )
        construction = (
            manifest.get("construction", {})
            if isinstance(manifest.get("construction"), dict)
            else {}
        )
        if existing.get("schema_version") == LEGACY_SCHEMA_VERSION:
            # Schema 2.0 has no final title or center-Evidence matrices. Mark
            # every imported node for one postprocessing-only schema upgrade;
            # already completed Evidence-to-Canonical memberships are reused.
            self.dirty_content_node_ids = set(range(1, len(self.nodes) + 1))
        elif (
            int(construction.get("content_algorithm_version", 0) or 0)
            != CONTENT_ALGORITHM_VERSION
        ):
            # A content-algorithm revision changes the generated title,
            # statement, validation contract, or retry policy. Reprocess every
            # row so one formal revision never claims a mixed content version.
            self.dirty_content_node_ids = set(range(1, len(self.nodes) + 1))
            self.fallback_statement_node_ids.clear()
            self.fallback_title_node_ids.clear()
            self.content_validation_failures.clear()
        else:
            self.dirty_content_node_ids = set()
        for node_id in self.dirty_content_node_ids:
            self.content_validation_failures.pop(node_id, None)
        self.processed_evidence_node_count = old_mapping.shape[0]
        self.phase = "alignment"
        self.alignment_model_history = self._history_for_new_update(
            stored_history=construction.get("alignment_model_history"),
            stored_identity={
                "name": clean_whitespace(
                    str(construction.get("alignment_model", "") or "")
                ),
                "provider": clean_whitespace(
                    str(construction.get("alignment_provider", "") or "")
                ),
            },
            current_identity=self._model_identity(self.alignment_model),
            processed_count=self.processed_evidence_node_count,
            total_count=len(self.evidence_nodes),
        )
        self.strict_verification_model_history = self._history_for_new_update(
            stored_history=construction.get("strict_verification_model_history"),
            stored_identity={
                "name": clean_whitespace(
                    str(
                        construction.get(
                            "strict_verification_model",
                            "",
                        )
                        or ""
                    )
                ),
                "provider": clean_whitespace(
                    str(
                        construction.get(
                            "strict_verification_provider",
                            "",
                        )
                        or ""
                    )
                ),
            },
            current_identity=self._model_identity(self.strict_model),
            processed_count=self.processed_evidence_node_count,
            total_count=len(self.evidence_nodes),
        )
        self.active_document_source_id = ""
        self.active_document_node_ids = []
        self.active_document_results = {}
        self.active_document_snapshot_canonical_node_count = 0
        self._active_document_stage_changes = 0
        self._document_commit_in_progress = False
        self.construction_recording_started_at = utc_now()
        self.construction_recording_start_evidence_node_id = (
            self.processed_evidence_node_count + 1
        )
        self.construction_records_complete = True
        self.construction_prompt_snapshots = dict(self.prompts)

    def _no_op_summary(self, manifest: dict[str, Any]) -> dict[str, Any]:
        counts = dict(manifest["counts"])
        summary = {
            "graph_id": self.graph_id,
            "graph_dir": str(self.graph_dir),
            "evidence_graph_id": self.evidence_graph_id,
            "evidence_graph_revision": self.evidence_manifest["revision"],
            "revision": manifest["revision"],
            "created": False,
            "updated": False,
            "counts": counts,
            "status": "already_up_to_date",
        }
        self.logger.info("Canonical Graph is already up to date", **summary)
        return summary

    def _run_alignment(self) -> None:
        total = len(self.evidence_nodes)
        if self.processed_evidence_node_count >= total:
            return
        print(
            "[canonical-graph] alignment: "
            f"pending={total - self.processed_evidence_node_count}, "
            f"reranker_workers={self.reranker_parallel_workers}, "
            f"llm_workers={self.llm_parallel_workers}, "
            f"checkpoint_every_stages={self.checkpoint_interval}"
        )
        while self.processed_evidence_node_count < total:
            self._activate_next_document()
            proposals = self._plan_active_document()
            # Preserve the complete paid proposal set before the first graph
            # mutation. A failed local commit can then be replayed for free.
            self._save_checkpoint()
            self._document_commit_in_progress = True
            self._commit_active_document(proposals)
            self._document_commit_in_progress = False
            self._save_checkpoint()

    def _activate_next_document(self) -> None:
        if self.active_document_node_ids:
            return
        start_index = self.processed_evidence_node_count
        if start_index >= len(self.evidence_nodes):
            return
        source_id = clean_whitespace(str(self.evidence_nodes[start_index]["source_id"]))
        node_ids: list[int] = []
        for index in range(start_index, len(self.evidence_nodes)):
            node_source_id = clean_whitespace(
                str(self.evidence_nodes[index]["source_id"])
            )
            if node_source_id != source_id:
                break
            node_ids.append(index + 1)
        if not node_ids:
            raise RuntimeError("The next Evidence document has no nodes.")
        self.active_document_source_id = source_id
        self.active_document_node_ids = node_ids
        self.active_document_results = {}
        self.active_document_snapshot_canonical_node_count = len(self.nodes)
        self._active_document_stage_changes = 0
        legacy_partial_document = (
            start_index > 0
            and clean_whitespace(str(self.evidence_nodes[start_index - 1]["source_id"]))
            == source_id
        )
        print(
            "[canonical-graph] document snapshot: "
            f"source={source_id}, nodes={node_ids[0]}-{node_ids[-1]}, "
            f"canonical_snapshot={len(self.nodes)}, "
            f"legacy_partial={legacy_partial_document}"
        )
        self.logger.info(
            "Canonical alignment document snapshot activated",
            source_id=source_id,
            evidence_node_id_start=node_ids[0],
            evidence_node_id_end=node_ids[-1],
            canonical_snapshot_node_count=len(self.nodes),
            legacy_partial_document=legacy_partial_document,
            processed_evidence_node_count=(self.processed_evidence_node_count),
        )
        # Publish the document boundary before its first paid request.
        self._save_checkpoint()

    def _plan_active_document(self) -> dict[int, dict[str, Any]]:
        if not self.active_document_node_ids:
            raise RuntimeError("No active Evidence document is available.")
        snapshot_processed_count = self.processed_evidence_node_count
        restored_proposals: dict[int, dict[str, Any]] = {}
        restored_candidates: dict[
            int, tuple[list[dict[str, Any]], dict[str, Any]]
        ] = {}
        pending_node_ids: deque[int] = deque()
        for evidence_node_id in self.active_document_node_ids:
            result = self.active_document_results.get(str(evidence_node_id))
            if result is None:
                pending_node_ids.append(evidence_node_id)
            elif result["stage"] == "proposed":
                restored_proposals[evidence_node_id] = deepcopy(result["proposal"])
            elif result["stage"] == "candidates_ready":
                restored_candidates[evidence_node_id] = (
                    deepcopy(result["candidates"]),
                    deepcopy(
                        result.get(
                            "decision_trace",
                            self._legacy_candidate_trace(result["candidates"]),
                        )
                    ),
                )
            else:  # The checkpoint validator should already reject this.
                raise ValueError("Unknown active-document result stage.")

        proposals = dict(restored_proposals)
        llm_ready: deque[
            tuple[int, list[dict[str, Any]], dict[str, Any]]
        ] = deque(
            (node_id, candidates, trace)
            for node_id, (candidates, trace) in sorted(
                restored_candidates.items()
            )
        )
        reranker_futures: dict[
            Future[Any], tuple[int, dict[str, Any]]
        ] = {}
        llm_futures: dict[Future[Any], int] = {}
        reranker_executor = ThreadPoolExecutor(
            max_workers=self.reranker_parallel_workers,
            thread_name_prefix="canonical-reranker",
        )
        llm_executor = ThreadPoolExecutor(
            max_workers=self.llm_parallel_workers,
            thread_name_prefix="canonical-llm",
        )
        fatal_error: BaseException | None = None
        try:
            while pending_node_ids or llm_ready or reranker_futures or llm_futures:
                while llm_ready and len(llm_futures) < self.llm_parallel_workers:
                    evidence_node_id, reranked, decision_trace = llm_ready.popleft()
                    future = llm_executor.submit(
                        self._plan_evidence_alignment,
                        evidence_node_id=evidence_node_id,
                        candidates=reranked,
                        decision_trace=decision_trace,
                    )
                    llm_futures[future] = evidence_node_id

                while (
                    pending_node_ids
                    and len(reranker_futures) < self.reranker_parallel_workers
                ):
                    evidence_node_id = pending_node_ids.popleft()
                    candidates, decision_trace = self._generate_candidates(
                        evidence_node_id,
                        processed_limit=snapshot_processed_count,
                    )
                    if not candidates:
                        proposal = self._no_candidate_proposal(
                            evidence_node_id,
                            decision_trace=decision_trace,
                        )
                        proposals[evidence_node_id] = proposal
                        self._record_active_document_proposal(proposal)
                        continue
                    if len(candidates) <= self.reranker_top_k:
                        decision_trace = self._candidate_trace_after_reranker(
                            decision_trace,
                            candidates,
                            applied=False,
                        )
                        self._record_active_document_candidates(
                            evidence_node_id,
                            candidates,
                            decision_trace,
                        )
                        llm_ready.append(
                            (evidence_node_id, candidates, decision_trace)
                        )
                        print(
                            "[canonical-graph] candidates ready "
                            f"{evidence_node_id}/{len(self.evidence_nodes)}: "
                            f"candidates={len(candidates)}, reranker=skipped"
                        )
                        # Let the next loop iteration immediately submit this
                        # node to the LLM pool instead of draining the whole
                        # document's local candidate generation first.
                        break
                    future = reranker_executor.submit(
                        self._rerank_candidates,
                        evidence_node=(self.evidence_nodes[evidence_node_id - 1]),
                        candidates=candidates,
                    )
                    reranker_futures[future] = (
                        evidence_node_id,
                        decision_trace,
                    )

                all_futures = set(reranker_futures) | set(llm_futures)
                if not all_futures:
                    continue
                done, _ = wait(all_futures, return_when=FIRST_COMPLETED)
                # Handle all results that became available together. This
                # keeps every completed paid stage recoverable if one sibling
                # future failed at the same time.
                errors: list[BaseException] = []
                for future in done:
                    if future in reranker_futures:
                        evidence_node_id, decision_trace = (
                            reranker_futures.pop(future)
                        )
                        try:
                            reranked = future.result()
                        except BaseException as exc:
                            errors.append(exc)
                            continue
                        decision_trace = self._candidate_trace_after_reranker(
                            decision_trace,
                            reranked,
                            applied=True,
                        )
                        self._record_active_document_candidates(
                            evidence_node_id,
                            reranked,
                            decision_trace,
                        )
                        llm_ready.append(
                            (evidence_node_id, reranked, decision_trace)
                        )
                        print(
                            "[canonical-graph] reranked "
                            f"{evidence_node_id}/{len(self.evidence_nodes)}: "
                            f"candidates={len(reranked)}, reranker=applied"
                        )
                    else:
                        evidence_node_id = llm_futures.pop(future)
                        try:
                            proposal = future.result()
                        except BaseException as exc:
                            errors.append(exc)
                            continue
                        proposals[evidence_node_id] = proposal
                        self._record_active_document_proposal(proposal)
                        print(
                            "[canonical-graph] proposed "
                            f"{evidence_node_id}/{len(self.evidence_nodes)}: "
                            f"action={proposal['action']}"
                        )
                if errors:
                    fatal_error = errors[0]
                    break
        except BaseException as exc:
            fatal_error = exc
        finally:
            if fatal_error is not None:
                for future in (*reranker_futures, *llm_futures):
                    future.cancel()
            reranker_executor.shutdown(
                wait=True,
                cancel_futures=fatal_error is not None,
            )
            llm_executor.shutdown(
                wait=True,
                cancel_futures=fatal_error is not None,
            )
            if fatal_error is not None:
                # Calls already in flight may have completed successfully
                # while the sibling failure was propagating. Persist their
                # paid outputs even though no new downstream work is started.
                for future, value in list(reranker_futures.items()):
                    evidence_node_id, decision_trace = value
                    if future.cancelled() or not future.done():
                        continue
                    try:
                        reranked = future.result()
                    except BaseException:
                        continue
                    decision_trace = self._candidate_trace_after_reranker(
                        decision_trace,
                        reranked,
                        applied=True,
                    )
                    self._record_active_document_candidates(
                        evidence_node_id,
                        reranked,
                        decision_trace,
                    )
                for future, evidence_node_id in list(llm_futures.items()):
                    if future.cancelled() or not future.done():
                        continue
                    try:
                        proposal = future.result()
                    except BaseException:
                        continue
                    proposals[evidence_node_id] = proposal
                    self._record_active_document_proposal(proposal)
        if fatal_error is not None:
            # The outer run guard writes an emergency snapshot that includes
            # all stage results recorded above.
            raise fatal_error
        missing = sorted(set(self.active_document_node_ids) - set(proposals))
        if missing:
            raise RuntimeError(
                f"Active document planning ended with missing proposals: {missing[:10]}"
            )
        return proposals

    def _record_active_document_candidates(
        self,
        evidence_node_id: int,
        candidates: list[dict[str, Any]],
        decision_trace: dict[str, Any],
    ) -> None:
        self.active_document_results[str(evidence_node_id)] = {
            "stage": "candidates_ready",
            "candidates": deepcopy(candidates),
            "decision_trace": deepcopy(decision_trace),
        }
        self._active_document_stage_completed()

    def _record_active_document_proposal(
        self,
        proposal: dict[str, Any],
    ) -> None:
        evidence_node_id = int(proposal["evidence_node_id"])
        self.active_document_results[str(evidence_node_id)] = {
            "stage": "proposed",
            "proposal": deepcopy(proposal),
        }
        self._active_document_stage_completed()

    def _active_document_stage_completed(self) -> None:
        self._active_document_stage_changes += 1
        if self._active_document_stage_changes >= self.checkpoint_interval:
            self._save_checkpoint()
            self._active_document_stage_changes = 0

    @staticmethod
    def _no_candidate_proposal(
        evidence_node_id: int,
        *,
        decision_trace: dict[str, Any],
    ) -> dict[str, Any]:
        trace = deepcopy(decision_trace)
        trace.setdefault("alignment_judgments", [])
        trace.setdefault("strict_cluster_comparisons", [])
        return {
            "evidence_node_id": evidence_node_id,
            "action": "create",
            "action_label": "created_no_candidates",
            "target_anchor": None,
            "possible_target_anchors": [],
            "confirmed_anchors": [],
            "merge_groups": [],
            "possible_anchor_pairs": [],
            "candidate_count": 0,
            "judged_count": 0,
            "decision_trace": trace,
        }

    def _generate_candidates(
        self,
        evidence_node_id: int,
        *,
        processed_limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if processed_limit is None:
            processed_limit = self.processed_evidence_node_count
        trace: dict[str, Any] = {
            "snapshot_processed_evidence_node_count": int(processed_limit),
            "snapshot_canonical_node_count": int(
                self.active_document_snapshot_canonical_node_count
            ),
            "retrieval": {
                "same_type_only": True,
                "evidence_top_k": self.evidence_retrieval_top_k,
                "minimum_similarity": self.evidence_similarity_threshold,
                "embedding_hits": [],
                "canonical_candidates_before_reranker": [],
            },
            "reranker": {
                "applied": False,
                "top_k": self.reranker_top_k,
                "canonical_candidates_after_reranker": [],
            },
        }
        if processed_limit == 0 or not self.nodes:
            return [], trace
        prior_ids = np.fromiter(
            range(1, processed_limit + 1),
            dtype=np.int64,
        )
        ordered, retrieval_trace = self._candidate_resolver().generate_candidates(
            query_node=self.evidence_nodes[evidence_node_id - 1],
            query_embedding=self.evidence_embeddings[evidence_node_id - 1],
            candidate_evidence_ids=prior_ids,
        )
        trace["retrieval"] = retrieval_trace
        return ordered, trace

    def _candidate_resolver(self) -> FrozenCandidateResolver:
        cache_key = (
            self.processed_evidence_node_count,
            len(self.nodes),
            id(self.mapping),
        )
        cached = self._candidate_resolver_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        resolver = FrozenCandidateResolver(
            canonical_nodes=self.nodes,
            evidence_nodes=self.evidence_nodes,
            evidence_embeddings=self.evidence_embeddings,
            evidence_node_to_canonical=self.mapping,
            evidence_top_k=self.evidence_retrieval_top_k,
            minimum_similarity=self.evidence_similarity_threshold,
            reranker_top_k=self.reranker_top_k,
            require_complete_mapping=False,
        )
        self._candidate_resolver_cache = (cache_key, resolver)
        return resolver

    @staticmethod
    def _candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
        return frozen_candidate_record(candidate)

    def _candidate_trace_after_reranker(
        self,
        trace: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        applied: bool,
    ) -> dict[str, Any]:
        result = deepcopy(trace)
        result["reranker"] = {
            "applied": applied,
            "top_k": self.reranker_top_k,
            "canonical_candidates_after_reranker": [
                self._candidate_record(candidate) for candidate in candidates
            ],
        }
        return result

    def _legacy_candidate_trace(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a minimal trace when resuming a pre-recording checkpoint."""
        trace = {
            "snapshot_processed_evidence_node_count": (
                self.processed_evidence_node_count
            ),
            "snapshot_canonical_node_count": len(self.nodes),
            "retrieval": {
                "same_type_only": True,
                "evidence_top_k": self.evidence_retrieval_top_k,
                "minimum_similarity": self.evidence_similarity_threshold,
                "embedding_hits": [],
                "canonical_candidates_before_reranker": [],
                "legacy_trace_incomplete": True,
            },
        }
        return self._candidate_trace_after_reranker(
            trace,
            candidates,
            applied=any("reranker_score" in item for item in candidates),
        )

    def _rerank_candidates(
        self,
        *,
        evidence_node: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(candidates) <= self.reranker_top_k:
            raise ValueError(
                "Reranker must be skipped when all Canonical candidates can "
                "proceed directly to strict-equivalence judgment."
            )
        reranked, _trace = self._candidate_resolver().rerank_candidates(
            query_node=evidence_node,
            candidates=candidates,
            reranker_client=self.reranker_client,
            usage_task="cross_document_reranker",
            retain_all_scores=False,
        )
        return reranked

    def _format_reranker_query(
        self,
        evidence_node: dict[str, Any],
    ) -> str:
        return self._candidate_resolver().format_reranker_query(evidence_node)

    def _format_reranker_document(
        self,
        candidate: dict[str, Any],
    ) -> str:
        return self._candidate_resolver().format_reranker_document(candidate)

    def _candidate_representatives(
        self,
        candidate: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self._candidate_resolver().candidate_representatives(candidate)

    def _judge_alignment_candidates(
        self,
        *,
        evidence_node: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[int, str]:
        relations: dict[int, str] = {}
        for candidate in candidates:
            canonical_node_id = int(candidate["canonical_node_id"])
            payload = {
                "current_evidence_node": self._evidence_payload(
                    int(evidence_node["id"]),
                ),
                "candidate_canonical_node": self._canonical_payload(
                    canonical_node_id,
                    query_match_evidence_node_id=candidate.get(
                        "query_match_evidence_node_id"
                    ),
                ),
            }
            result = self.llm_client.call_json(
                model_task="cross_document_alignment",
                prompt=self.prompts["alignment"],
                input_text=json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                usage_task="cross_document_alignment",
                validator=_validate_single_relation,
                preserve_latex_escapes=True,
                repair_instructions=(
                    read_text(PROMPT_DIR / "ternary_relation_output.md")
                ),
            )
            if result is None:
                raise RuntimeError(
                    "Alignment model did not return a valid one-to-one result."
                )
            relations[canonical_node_id] = str(result["relation"])
        return relations

    def _plan_evidence_alignment(
        self,
        *,
        evidence_node_id: int,
        candidates: list[dict[str, Any]],
        decision_trace: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce an immutable alignment proposal against the snapshot."""
        evidence_node = self.evidence_nodes[evidence_node_id - 1]
        relations = self._judge_alignment_candidates(
            evidence_node=evidence_node,
            candidates=candidates,
        )
        decision_trace = deepcopy(decision_trace)
        decision_trace["alignment_judgments"] = []
        for candidate in candidates:
            canonical_node_id = int(candidate["canonical_node_id"])
            representatives = self._candidate_representatives(candidate)
            judgment = self._candidate_record(candidate)
            judgment.update(
                {
                    "canonical_anchor_evidence_node_id": (
                        self._canonical_anchor(canonical_node_id)
                    ),
                    "representative_evidence_nodes": [
                        {
                            "role": str(representative["role"]),
                            "evidence_node_id": int(
                                representative["evidence_node_id"]
                            ),
                        }
                        for representative in representatives
                    ],
                    "relation": relations[canonical_node_id],
                }
            )
            decision_trace["alignment_judgments"].append(judgment)
        decision_trace["strict_cluster_comparisons"] = []
        equivalent_ids = [
            canonical_node_id
            for canonical_node_id, relation in relations.items()
            if relation == "equivalent"
        ]
        possible_ids = [
            canonical_node_id
            for canonical_node_id, relation in relations.items()
            if relation == "possible_equivalent"
        ]
        possible_anchors = [
            self._canonical_anchor(canonical_node_id)
            for canonical_node_id in possible_ids
        ]
        if not equivalent_ids:
            return self._alignment_proposal(
                evidence_node_id=evidence_node_id,
                action="create",
                action_label=(
                    "created_possible_equivalence"
                    if possible_anchors
                    else "created_no_match"
                ),
                possible_target_anchors=possible_anchors,
                candidate_count=len(candidates),
                judged_count=len(relations),
                decision_trace=decision_trace,
            )

        original_anchors = {
            canonical_node_id: self._canonical_anchor(canonical_node_id)
            for canonical_node_id in equivalent_ids
        }
        confirmed_ids = sorted(equivalent_ids)
        confirmed_anchors = [
            original_anchors[canonical_node_id] for canonical_node_id in confirmed_ids
        ]
        if len(confirmed_ids) == 1:
            return self._alignment_proposal(
                evidence_node_id=evidence_node_id,
                action="join",
                action_label="joined_single_equivalent",
                target_anchor=confirmed_anchors[0],
                possible_target_anchors=possible_anchors,
                confirmed_anchors=confirmed_anchors,
                candidate_count=len(candidates),
                judged_count=len(relations),
                decision_trace=decision_trace,
            )

        pair_relations: dict[tuple[int, int], str] = {}
        for index, node_a in enumerate(confirmed_ids):
            for node_b in confirmed_ids[index + 1 :]:
                relation, comparison = self._strict_cluster_pair(
                    node_a=node_a,
                    node_b=node_b,
                )
                pair_relations[(node_a, node_b)] = relation
                decision_trace["strict_cluster_comparisons"].append(
                    comparison
                )
        equivalent_graph = {
            canonical_node_id: set() for canonical_node_id in confirmed_ids
        }
        possible_anchor_pairs: list[list[int]] = []
        for (node_a, node_b), relation in pair_relations.items():
            if relation == "equivalent":
                equivalent_graph[node_a].add(node_b)
                equivalent_graph[node_b].add(node_a)
            elif relation == "possible_equivalent":
                possible_anchor_pairs.append(
                    [original_anchors[node_a], original_anchors[node_b]]
                )

        merge_groups: list[list[int]] = []
        for component in _connected_components(equivalent_graph):
            if len(component) < 2:
                continue
            if all(
                node_b in equivalent_graph[node_a]
                for index, node_a in enumerate(sorted(component))
                for node_b in sorted(component)[index + 1 :]
            ):
                merge_groups.append(
                    sorted(original_anchors[node_id] for node_id in component)
                )

        all_confirmed_form_one_clique = len(merge_groups) == 1 and set(
            merge_groups[0]
        ) == set(confirmed_anchors)
        if all_confirmed_form_one_clique:
            return self._alignment_proposal(
                evidence_node_id=evidence_node_id,
                action="join",
                action_label="merged_clusters_and_joined",
                target_anchor=min(confirmed_anchors),
                possible_target_anchors=possible_anchors,
                confirmed_anchors=confirmed_anchors,
                merge_groups=merge_groups,
                possible_anchor_pairs=possible_anchor_pairs,
                candidate_count=len(candidates),
                judged_count=len(relations),
                decision_trace=decision_trace,
            )

        # The current node cannot safely choose between mutually inconsistent
        # strict positives. Preserve safe disjoint cluster merges, create a new
        # Canonical node, and expose every ambiguous target for review.
        return self._alignment_proposal(
            evidence_node_id=evidence_node_id,
            action="create",
            action_label="created_after_ambiguous_strict_verification",
            possible_target_anchors=[
                *possible_anchors,
                *confirmed_anchors,
            ],
            confirmed_anchors=confirmed_anchors,
            merge_groups=merge_groups,
            possible_anchor_pairs=possible_anchor_pairs,
            candidate_count=len(candidates),
            judged_count=len(relations),
            decision_trace=decision_trace,
        )

    @staticmethod
    def _alignment_proposal(
        *,
        evidence_node_id: int,
        action: str,
        action_label: str,
        target_anchor: int | None = None,
        possible_target_anchors: list[int] | None = None,
        confirmed_anchors: list[int] | None = None,
        merge_groups: list[list[int]] | None = None,
        possible_anchor_pairs: list[list[int]] | None = None,
        candidate_count: int,
        judged_count: int,
        decision_trace: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "evidence_node_id": evidence_node_id,
            "action": action,
            "action_label": action_label,
            "target_anchor": target_anchor,
            "possible_target_anchors": sorted(set(possible_target_anchors or [])),
            "confirmed_anchors": sorted(set(confirmed_anchors or [])),
            "merge_groups": sorted(
                {
                    tuple(sorted(set(group)))
                    for group in (merge_groups or [])
                    if len(set(group)) >= 2
                }
            ),
            "possible_anchor_pairs": sorted(
                {
                    tuple(sorted(set(pair)))
                    for pair in (possible_anchor_pairs or [])
                    if len(set(pair)) == 2
                }
            ),
            "candidate_count": candidate_count,
            "judged_count": judged_count,
            "decision_trace": deepcopy(decision_trace),
        }

    def _commit_active_document(
        self,
        proposals: dict[int, dict[str, Any]],
    ) -> None:
        expected_ids = self.active_document_node_ids
        if sorted(proposals) != expected_ids:
            raise ValueError(
                "Active document proposals do not cover its nodes exactly."
            )

        requested_groups = sorted(
            {
                tuple(int(anchor) for anchor in group)
                for proposal in proposals.values()
                for group in proposal["merge_groups"]
            }
        )
        blocked_groups: set[tuple[int, ...]] = set()
        for index, group_a in enumerate(requested_groups):
            set_a = set(group_a)
            for group_b in requested_groups[index + 1 :]:
                set_b = set(group_b)
                if set_a & set_b and set_a != set_b:
                    blocked_groups.add(group_a)
                    blocked_groups.add(group_b)
        if blocked_groups:
            self.logger.warning(
                "Overlapping document merge proposals were conservatively deferred",
                source_id=self.active_document_source_id,
                blocked_merge_groups=[list(group) for group in sorted(blocked_groups)],
            )

        applied_merge_groups: list[list[int]] = []
        for anchor_group in requested_groups:
            if anchor_group in blocked_groups:
                continue
            canonical_ids = self._canonical_ids_from_anchors(list(anchor_group))
            if len(canonical_ids) >= 2:
                self._merge_canonical_groups([set(canonical_ids)])
                applied_merge_groups.append(list(anchor_group))

        possible_anchor_pairs = {
            tuple(int(anchor) for anchor in pair)
            for proposal in proposals.values()
            for pair in proposal["possible_anchor_pairs"]
        }
        for anchor_a, anchor_b in sorted(possible_anchor_pairs):
            canonical_ids = self._canonical_ids_from_anchors([anchor_a, anchor_b])
            if len(canonical_ids) == 2:
                self._add_possible_pair(*canonical_ids)

        committed_outcomes: list[dict[str, Any]] = []
        for evidence_node_id in expected_ids:
            proposal = proposals[evidence_node_id]
            proposal_groups = {
                tuple(int(anchor) for anchor in group)
                for group in proposal["merge_groups"]
            }
            merge_conflict = bool(proposal_groups & blocked_groups)
            action = str(proposal["action"])
            possible_anchors = list(proposal["possible_target_anchors"])
            if merge_conflict:
                action = "create"
                possible_anchors.extend(proposal["confirmed_anchors"])

            if action == "join":
                target_anchor = int(proposal["target_anchor"])
                target_ids = self._canonical_ids_from_anchors([target_anchor])
                if len(target_ids) != 1:
                    raise RuntimeError("A proposal target anchor no longer resolves.")
                canonical_node_id = target_ids[0]
                self._add_evidence_member(
                    canonical_node_id,
                    evidence_node_id,
                )
                action_label = str(proposal["action_label"])
            elif action == "create":
                canonical_node_id = self._create_canonical_node(evidence_node_id)
                action_label = (
                    "created_after_overlapping_merge_conflict"
                    if merge_conflict
                    else str(proposal["action_label"])
                )
            else:
                raise ValueError(f"Unknown proposal action: {action!r}")

            possible_targets = self._canonical_ids_from_anchors(possible_anchors)
            self._add_possible_links(canonical_node_id, possible_targets)
            outcome = {
                "action": action_label,
                "canonical_node_id": canonical_node_id,
                "candidate_count": int(proposal["candidate_count"]),
                "judged_count": int(proposal["judged_count"]),
            }
            committed_outcomes.append(
                {
                    "evidence_node_id": evidence_node_id,
                    **outcome,
                }
            )
            print(
                "[canonical-graph] aligned "
                f"{evidence_node_id}/{len(self.evidence_nodes)}: "
                f"action={outcome['action']}, "
                f"canonical_node_id={outcome['canonical_node_id']}, "
                f"candidates={outcome['candidate_count']}, "
                f"judged={outcome['judged_count']}"
            )
            self.logger.info(
                "Evidence Node alignment committed",
                evidence_node_id=evidence_node_id,
                source_id=self.active_document_source_id,
                **outcome,
            )

        self.processed_evidence_node_count = expected_ids[-1]
        for outcome in committed_outcomes:
            evidence_node_id = int(outcome["evidence_node_id"])
            proposal = proposals[evidence_node_id]
            self._write_checkpoint_record(
                "alignment",
                f"{evidence_node_id:08d}",
                self._alignment_construction_record(
                    proposal=proposal,
                    outcome=outcome,
                ),
            )
        self._write_checkpoint_record(
            "documents",
            f"{expected_ids[0]:08d}_{expected_ids[-1]:08d}",
            self._document_construction_record(
                outcomes=committed_outcomes,
                requested_groups=requested_groups,
                blocked_groups=blocked_groups,
                applied_groups=applied_merge_groups,
            ),
        )
        print(
            "[canonical-graph] document committed: "
            f"source={self.active_document_source_id}, "
            f"nodes={expected_ids[0]}-{expected_ids[-1]}"
        )
        self.active_document_source_id = ""
        self.active_document_node_ids = []
        self.active_document_results = {}
        self.active_document_snapshot_canonical_node_count = 0
        self._active_document_stage_changes = 0

    def _alignment_construction_record(
        self,
        *,
        proposal: dict[str, Any],
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        evidence_node_id = int(proposal["evidence_node_id"])
        evidence_node = self.evidence_nodes[evidence_node_id - 1]
        proposal_record = {
            key: deepcopy(proposal[key])
            for key in (
                "action",
                "action_label",
                "target_anchor",
                "possible_target_anchors",
                "confirmed_anchors",
                "merge_groups",
                "possible_anchor_pairs",
                "candidate_count",
                "judged_count",
            )
        }
        return {
            "record_schema_version": CONSTRUCTION_RECORDS_SCHEMA_VERSION,
            "canonical_graph_id": self.graph_id,
            "target_revision": self.target_revision,
            "evidence_graph_id": self.evidence_graph_id,
            "evidence_graph_revision": self.evidence_manifest["revision"],
            "evidence_node_id": evidence_node_id,
            "source_id": str(evidence_node["source_id"]),
            "type": str(evidence_node["type"]),
            "decision_trace": deepcopy(proposal.get("decision_trace", {})),
            "proposal": proposal_record,
            "outcome": {
                "action": str(outcome["action"]),
                "canonical_node_id_at_document_commit": int(
                    outcome["canonical_node_id"]
                ),
                "final_canonical_node_id": int(
                    self.mapping[evidence_node_id - 1]
                ),
            },
            "recorded_at": utc_now(),
        }

    def _document_construction_record(
        self,
        *,
        outcomes: list[dict[str, Any]],
        requested_groups: list[tuple[int, ...]],
        blocked_groups: set[tuple[int, ...]],
        applied_groups: list[list[int]],
    ) -> dict[str, Any]:
        action_counts: dict[str, int] = {}
        for outcome in outcomes:
            action = str(outcome["action"])
            action_counts[action] = action_counts.get(action, 0) + 1
        node_ids = [int(item["evidence_node_id"]) for item in outcomes]
        return {
            "record_schema_version": CONSTRUCTION_RECORDS_SCHEMA_VERSION,
            "canonical_graph_id": self.graph_id,
            "target_revision": self.target_revision,
            "evidence_graph_id": self.evidence_graph_id,
            "evidence_graph_revision": self.evidence_manifest["revision"],
            "source_id": self.active_document_source_id,
            "evidence_node_id_start": min(node_ids),
            "evidence_node_id_end": max(node_ids),
            "evidence_node_count": len(node_ids),
            "snapshot_canonical_node_count": (
                self.active_document_snapshot_canonical_node_count
            ),
            "canonical_node_count_after_commit": len(self.nodes),
            "action_counts": dict(sorted(action_counts.items())),
            "requested_merge_anchor_groups": [
                list(group) for group in requested_groups
            ],
            "blocked_merge_anchor_groups": [
                list(group) for group in sorted(blocked_groups)
            ],
            "applied_merge_anchor_groups": deepcopy(applied_groups),
            "possible_equivalence_pair_count_after_commit": len(
                self.possible_pairs
            ),
            "committed_at": utc_now(),
        }

    def _write_checkpoint_record(
        self,
        category: str,
        key: str,
        payload: dict[str, Any],
    ) -> None:
        if category not in {"alignment", "documents", "content"}:
            raise ValueError(f"Unknown construction-record category: {category}")
        if not re.fullmatch(r"[0-9_]+", key):
            raise ValueError("Construction-record key is invalid.")
        directory = self.checkpoint_records_dir / category
        directory.mkdir(parents=True, exist_ok=True)
        final_path = directory / f"{key}.json"
        temporary_path = directory / f".{key}.{uuid.uuid4().hex}.tmp"
        try:
            self._write_json_fsynced(temporary_path, payload)
            os.replace(temporary_path, final_path)
            self._fsync_directory(directory)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _strict_cluster_pair(
        self,
        *,
        node_a: int,
        node_b: int,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "canonical_node_a": self._canonical_payload(node_a),
            "canonical_node_b": self._canonical_payload(node_b),
        }
        result = self.llm_client.call_json(
            model_task="cross_document_strict_verification",
            prompt=self.prompts["strict_cluster_pair"],
            input_text=json.dumps(payload, ensure_ascii=False, indent=2),
            usage_task="cross_document_strict_cluster_pair",
            validator=_validate_single_relation,
            preserve_latex_escapes=True,
        )
        if result is None:
            raise RuntimeError(
                "Strict Canonical-cluster pair verification returned no valid result."
            )
        relation = str(result["relation"])
        return relation, {
            "canonical_node_a": node_a,
            "canonical_anchor_a": self._canonical_anchor(node_a),
            "representative_evidence_node_ids_a": [
                int(item["evidence_node_id"])
                for item in payload["canonical_node_a"][
                    "representative_evidence_nodes"
                ]
            ],
            "canonical_node_b": node_b,
            "canonical_anchor_b": self._canonical_anchor(node_b),
            "representative_evidence_node_ids_b": [
                int(item["evidence_node_id"])
                for item in payload["canonical_node_b"][
                    "representative_evidence_nodes"
                ]
            ],
            "relation": relation,
        }

    def _evidence_payload(
        self,
        evidence_node_id: int,
    ) -> dict[str, Any]:
        return frozen_evidence_payload(
            self.evidence_nodes[evidence_node_id - 1]
        )

    def _canonical_content_evidence_payload(
        self,
        evidence_node_id: int,
    ) -> dict[str, Any]:
        node = self.evidence_nodes[evidence_node_id - 1]
        return {
            "evidence_node_id": evidence_node_id,
            "type": node["type"],
            "title": node["title"],
            "normalized_statement": normalized_statement(node),
        }

    def _canonical_payload(
        self,
        canonical_node_id: int,
        *,
        query_match_evidence_node_id: int | None = None,
    ) -> dict[str, Any]:
        return self._candidate_resolver().canonical_payload(
            canonical_node_id,
            query_match_evidence_node_id=query_match_evidence_node_id,
        )

    def _canonical_anchor(self, canonical_node_id: int) -> int:
        return int(self.nodes[canonical_node_id - 1]["center_evidence_node_id"])

    def _canonical_ids_from_anchors(
        self,
        anchors: list[int],
    ) -> list[int]:
        return sorted(
            {
                int(self.mapping[anchor - 1])
                for anchor in anchors
                if 1 <= anchor <= self.mapping.shape[0]
                and int(self.mapping[anchor - 1]) > 0
            }
        )

    def _add_possible_links(
        self,
        canonical_node_id: int,
        target_ids: list[int],
    ) -> None:
        for target_id in target_ids:
            self._add_possible_pair(canonical_node_id, target_id)

    def _add_possible_pair(self, node_a: int, node_b: int) -> None:
        if node_a == node_b:
            return
        if not 1 <= node_a <= len(self.nodes) or not 1 <= node_b <= len(self.nodes):
            raise ValueError("Possible-equivalence endpoint is out of range.")
        self.possible_pairs.add((min(node_a, node_b), max(node_a, node_b)))

    def _create_canonical_node(self, evidence_node_id: int) -> int:
        if int(self.mapping[evidence_node_id - 1]) != 0:
            raise ValueError(f"Evidence Node {evidence_node_id} is already assigned.")
        evidence_node = self.evidence_nodes[evidence_node_id - 1]
        canonical_node_id = len(self.nodes) + 1
        vector = np.asarray(
            self.evidence_embeddings[evidence_node_id - 1],
            dtype=np.float32,
        )
        node = {
            "id": canonical_node_id,
            "type": evidence_node["type"],
            "title": clean_model_text(str(evidence_node["title"])),
            "statement": normalized_statement(evidence_node),
            "field": evidence_node["field"],
            "subfield": evidence_node["subfield"],
            "evidence_node_ids": [evidence_node_id],
            "center_evidence_node_id": evidence_node_id,
            "diverse_evidence_node_ids": [],
            "in_edges": [],
            "out_edges": [],
            "possible_equivalence_edge_ids": [],
        }
        self.nodes.append(node)
        self.mapping[evidence_node_id - 1] = canonical_node_id
        self.embedding_sums = _append_matrix_row(
            self.embedding_sums,
            vector,
        )
        self.centroid_embeddings = _append_matrix_row(
            self.centroid_embeddings,
            vector,
        )
        # The working statement equals this Evidence normalized statement, so
        # its existing vector is already the correct statement vector.
        self.statement_embeddings = _append_matrix_row(
            self.statement_embeddings,
            vector,
        )
        self.title_embeddings = _append_matrix_row(
            self.title_embeddings,
            np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32),
        )
        self.center_evidence_embeddings = _append_matrix_row(
            self.center_evidence_embeddings,
            vector,
        )
        self.dirty_content_node_ids.add(canonical_node_id)
        return canonical_node_id

    def _add_evidence_member(
        self,
        canonical_node_id: int,
        evidence_node_id: int,
    ) -> None:
        if int(self.mapping[evidence_node_id - 1]) != 0:
            raise ValueError(f"Evidence Node {evidence_node_id} is already assigned.")
        node = self.nodes[canonical_node_id - 1]
        evidence_node = self.evidence_nodes[evidence_node_id - 1]
        if evidence_node["type"] != node["type"]:
            raise ValueError(
                "Cannot assign an Evidence Node to a Canonical Node of a "
                "different type."
            )
        node["evidence_node_ids"] = sorted(
            [*node["evidence_node_ids"], evidence_node_id]
        )
        self.mapping[evidence_node_id - 1] = canonical_node_id
        self.embedding_sums[canonical_node_id - 1] += np.asarray(
            self.evidence_embeddings[evidence_node_id - 1],
            dtype=np.float32,
        )
        self.centroid_embeddings[canonical_node_id - 1] = _l2_normalize(
            self.embedding_sums[canonical_node_id - 1]
        )
        self._update_representatives(canonical_node_id)
        self.dirty_content_node_ids.add(canonical_node_id)
        self.fallback_statement_node_ids.discard(canonical_node_id)
        self.fallback_title_node_ids.discard(canonical_node_id)
        self.content_validation_failures.pop(canonical_node_id, None)

    def _update_representatives(self, canonical_node_id: int) -> None:
        node = self.nodes[canonical_node_id - 1]
        member_ids = np.asarray(
            node["evidence_node_ids"],
            dtype=np.int64,
        )
        centroid = self.centroid_embeddings[canonical_node_id - 1]
        member_vectors = self.evidence_embeddings[member_ids - 1]
        centroid_scores = np.asarray(
            member_vectors @ centroid,
            dtype=np.float32,
        )
        center_local_index = int(np.argmax(centroid_scores))
        center_id = int(member_ids[center_local_index])

        candidate_ids = [
            int(member_ids[index])
            for index in range(len(member_ids))
            if int(member_ids[index]) != center_id
        ]
        centroid_score_by_id = {
            int(member_ids[index]): float(score)
            for index, score in enumerate(centroid_scores)
        }
        selected = [center_id]
        selected_sources = {str(self.evidence_nodes[center_id - 1]["source_id"])}
        diverse_ids: list[int] = []
        while candidate_ids and len(diverse_ids) < 2:
            selected_vectors = self.evidence_embeddings[
                np.asarray(selected, dtype=np.int64) - 1
            ]

            def diversity_key(
                candidate_id: int,
            ) -> tuple[int, int, float, float, int]:
                candidate_vector = self.evidence_embeddings[candidate_id - 1]
                max_similarity = float(np.max(selected_vectors @ candidate_vector))
                source_id = str(self.evidence_nodes[candidate_id - 1]["source_id"])
                centroid_similarity = centroid_score_by_id[candidate_id]
                return (
                    int(source_id not in selected_sources),
                    int(centroid_similarity >= self.representative_min_similarity),
                    1.0 - max_similarity,
                    centroid_similarity,
                    -candidate_id,
                )

            chosen = max(candidate_ids, key=diversity_key)
            candidate_ids.remove(chosen)
            diverse_ids.append(chosen)
            selected.append(chosen)
            selected_sources.add(str(self.evidence_nodes[chosen - 1]["source_id"]))

        node["center_evidence_node_id"] = center_id
        node["diverse_evidence_node_ids"] = diverse_ids
        center_node = self.evidence_nodes[center_id - 1]
        node["field"] = center_node["field"]
        node["subfield"] = center_node["subfield"]
        self.center_evidence_embeddings[canonical_node_id - 1] = np.asarray(
            self.evidence_embeddings[center_id - 1],
            dtype=np.float32,
        )

    def _center_embedding_matrix(
        self,
        nodes: list[dict[str, Any]],
    ) -> np.ndarray:
        if not nodes:
            return np.empty(
                (0, EMBEDDING_DIMENSIONS),
                dtype=np.float32,
            )
        center_ids = np.asarray(
            [int(node["center_evidence_node_id"]) for node in nodes],
            dtype=np.int64,
        )
        if np.any(center_ids < 1) or np.any(center_ids > len(self.evidence_nodes)):
            raise ValueError("Canonical center Evidence Node ID is out of range.")
        return np.asarray(
            self.evidence_embeddings[center_ids - 1],
            dtype=np.float32,
        ).copy()

    def _merge_canonical_groups(
        self,
        groups: list[set[int]],
    ) -> dict[int, int]:
        if not groups:
            return {node_id: node_id for node_id in range(1, len(self.nodes) + 1)}
        parent = list(range(len(self.nodes) + 1))

        def find(node_id: int) -> int:
            while parent[node_id] != node_id:
                parent[node_id] = parent[parent[node_id]]
                node_id = parent[node_id]
            return node_id

        def union(node_a: int, node_b: int) -> None:
            root_a = find(node_a)
            root_b = find(node_b)
            if root_a == root_b:
                return
            smaller, larger = sorted((root_a, root_b))
            parent[larger] = smaller

        for group in groups:
            valid_group = sorted(set(group))
            if len(valid_group) < 2:
                continue
            if any(not 1 <= node_id <= len(self.nodes) for node_id in valid_group):
                raise ValueError("Canonical merge group contains invalid ID.")
            base_type = self.nodes[valid_group[0] - 1]["type"]
            if any(
                self.nodes[node_id - 1]["type"] != base_type for node_id in valid_group
            ):
                raise ValueError("Canonical merge group contains different node types.")
            for node_id in valid_group[1:]:
                union(valid_group[0], node_id)

        components: dict[int, list[int]] = {}
        for node_id in range(1, len(self.nodes) + 1):
            components.setdefault(find(node_id), []).append(node_id)
        ordered_components = sorted(
            components.values(),
            key=lambda component: min(component),
        )
        old_to_new: dict[int, int] = {}
        for new_id, component in enumerate(ordered_components, start=1):
            for old_id in component:
                old_to_new[old_id] = new_id

        old_nodes = self.nodes
        old_sums = self.embedding_sums
        old_statement_embeddings = self.statement_embeddings
        old_title_embeddings = self.title_embeddings
        old_dirty = set(self.dirty_content_node_ids)
        old_fallback = set(self.fallback_statement_node_ids)
        old_title_fallback = set(self.fallback_title_node_ids)
        old_validation_failures = deepcopy(self.content_validation_failures)
        new_nodes: list[dict[str, Any]] = []
        new_sums: list[np.ndarray] = []
        new_centroids: list[np.ndarray] = []
        new_statement_embeddings: list[np.ndarray] = []
        new_title_embeddings: list[np.ndarray] = []
        new_center_embeddings: list[np.ndarray] = []
        new_dirty: set[int] = set()
        new_fallback: set[int] = set()
        new_title_fallback: set[int] = set()
        new_validation_failures: dict[int, dict[str, Any]] = {}

        for new_id, component in enumerate(ordered_components, start=1):
            base_old_id = min(component)
            base_node = deepcopy(old_nodes[base_old_id - 1])
            members = sorted(
                {
                    evidence_node_id
                    for old_id in component
                    for evidence_node_id in old_nodes[old_id - 1]["evidence_node_ids"]
                }
            )
            vector_sum = np.sum(
                old_sums[np.asarray(component, dtype=np.int64) - 1],
                axis=0,
                dtype=np.float32,
            ).astype(np.float32, copy=False)
            base_node.update(
                {
                    "id": new_id,
                    "evidence_node_ids": members,
                    "in_edges": [],
                    "out_edges": [],
                    "possible_equivalence_edge_ids": [],
                }
            )
            new_nodes.append(base_node)
            new_sums.append(vector_sum)
            new_centroids.append(_l2_normalize(vector_sum))
            if len(component) == 1:
                new_statement_embeddings.append(
                    old_statement_embeddings[base_old_id - 1].copy()
                )
                new_title_embeddings.append(
                    old_title_embeddings[base_old_id - 1].copy()
                )
                old_center_id = int(
                    old_nodes[base_old_id - 1]["center_evidence_node_id"]
                )
                new_center_embeddings.append(
                    np.asarray(
                        self.evidence_embeddings[old_center_id - 1],
                        dtype=np.float32,
                    ).copy()
                )
                if base_old_id in old_dirty:
                    new_dirty.add(new_id)
                if base_old_id in old_fallback:
                    new_fallback.add(new_id)
                if base_old_id in old_title_fallback:
                    new_title_fallback.add(new_id)
                if base_old_id in old_validation_failures:
                    failure = deepcopy(old_validation_failures[base_old_id])
                    failure["canonical_node_id"] = new_id
                    new_validation_failures[new_id] = failure
            else:
                new_statement_embeddings.append(
                    np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
                )
                new_title_embeddings.append(
                    np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
                )
                new_center_embeddings.append(
                    np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
                )
                new_dirty.add(new_id)

        lookup = np.zeros(len(old_nodes) + 1, dtype=np.int64)
        for old_id, new_id in old_to_new.items():
            lookup[old_id] = new_id
        nonzero = self.mapping > 0
        self.mapping[nonzero] = lookup[self.mapping[nonzero]]
        self.nodes = new_nodes
        self.embedding_sums = _stack_rows(new_sums)
        self.centroid_embeddings = _stack_rows(new_centroids)
        self.statement_embeddings = _stack_rows(new_statement_embeddings)
        self.title_embeddings = _stack_rows(new_title_embeddings)
        self.center_evidence_embeddings = _stack_rows(new_center_embeddings)
        self.dirty_content_node_ids = new_dirty
        self.fallback_statement_node_ids = new_fallback
        self.fallback_title_node_ids = new_title_fallback
        self.content_validation_failures = new_validation_failures
        self.possible_pairs = {
            (
                min(old_to_new[node_a], old_to_new[node_b]),
                max(old_to_new[node_a], old_to_new[node_b]),
            )
            for node_a, node_b in self.possible_pairs
            if old_to_new[node_a] != old_to_new[node_b]
        }
        for canonical_node_id in range(1, len(self.nodes) + 1):
            self._update_representatives(canonical_node_id)
        return old_to_new

    def _run_content_postprocessing(self) -> None:
        pending = sorted(self.dirty_content_node_ids)
        if not pending:
            return
        print(
            "[canonical-graph] title/statement postprocessing: "
            f"pending={len(pending)}, "
            f"workers={self.content_parallel_workers}"
        )
        ready: dict[int, dict[str, Any]] = {}
        scheduled: set[int] = set()
        futures: dict[Future[Any], int] = {}
        completed_since_checkpoint = 0
        failed: dict[int, BaseException] = {}
        executor = ThreadPoolExecutor(
            max_workers=self.content_parallel_workers,
            thread_name_prefix="canonical-content",
        )
        fatal_error: BaseException | None = None
        try:
            while self.dirty_content_node_ids or futures or ready:
                while len(futures) + len(ready) < self.content_parallel_workers:
                    available = sorted(self.dirty_content_node_ids - scheduled)
                    if not available:
                        break
                    canonical_node_id = available[0]
                    job = self._content_job_snapshot(canonical_node_id)
                    future = executor.submit(
                        self._prepare_content_result,
                        job,
                    )
                    futures[future] = canonical_node_id
                    scheduled.add(canonical_node_id)

                if futures:
                    done, _ = wait(
                        set(futures),
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        canonical_node_id = futures.pop(future)
                        try:
                            ready[canonical_node_id] = future.result()
                        except BaseException as exc:
                            failed[canonical_node_id] = exc

                made_progress = False
                while self.dirty_content_node_ids:
                    canonical_node_id = min(self.dirty_content_node_ids)
                    if canonical_node_id in failed:
                        fatal_error = failed[canonical_node_id]
                        break
                    result = ready.pop(canonical_node_id, None)
                    if result is None:
                        break
                    scheduled.discard(canonical_node_id)
                    self._commit_content_result(result)
                    completed_since_checkpoint += 1
                    made_progress = True
                    if completed_since_checkpoint >= self.checkpoint_interval:
                        self._save_checkpoint()
                        completed_since_checkpoint = 0
                if fatal_error is not None or failed:
                    fatal_error = fatal_error or next(iter(failed.values()))
                    break
                if (
                    not futures
                    and not ready
                    and self.dirty_content_node_ids
                    and not made_progress
                ):
                    raise RuntimeError(
                        "Content postprocessing scheduler made no progress."
                    )
        except BaseException as exc:
            fatal_error = exc
        finally:
            if fatal_error is not None:
                for future in futures:
                    future.cancel()
            executor.shutdown(
                wait=True,
                cancel_futures=fatal_error is not None,
            )
            if fatal_error is not None:
                # Retain every completed result that precedes the first failed
                # Canonical ID. Later successful siblings are intentionally
                # retried so commits remain deterministic by ID.
                for future, canonical_node_id in list(futures.items()):
                    if future.cancelled() or not future.done():
                        continue
                    try:
                        ready[canonical_node_id] = future.result()
                    except BaseException as exc:
                        failed[canonical_node_id] = exc
                while self.dirty_content_node_ids:
                    canonical_node_id = min(self.dirty_content_node_ids)
                    if canonical_node_id in failed:
                        break
                    result = ready.pop(canonical_node_id, None)
                    if result is None:
                        break
                    scheduled.discard(canonical_node_id)
                    self._commit_content_result(result)
                    completed_since_checkpoint += 1
        if fatal_error is not None:
            # The outer run guard persists all serially committed results in
            # an emergency checkpoint before returning control to the caller.
            raise fatal_error

    def _content_job_snapshot(
        self,
        canonical_node_id: int,
    ) -> dict[str, Any]:
        node = self.nodes[canonical_node_id - 1]
        member_ids = [
            int(evidence_node_id) for evidence_node_id in node["evidence_node_ids"]
        ]
        center_id = int(node["center_evidence_node_id"])
        center_node = self.evidence_nodes[center_id - 1]
        return {
            "canonical_node_id": canonical_node_id,
            "type": str(node["type"]),
            "member_ids": member_ids,
            "center_id": center_id,
            "center_title": clean_model_text(str(center_node["title"])),
            "center_statement": normalized_statement(center_node),
            "center_embedding": np.asarray(
                self.evidence_embeddings[center_id - 1],
                dtype=np.float32,
            ).copy(),
            "previous_title": clean_model_text(str(node.get("title", ""))),
            "previous_statement": clean_model_text(str(node.get("statement", ""))),
            "previous_title_embedding": self.title_embeddings[
                canonical_node_id - 1
            ].copy(),
            "previous_statement_embedding": self.statement_embeddings[
                canonical_node_id - 1
            ].copy(),
            "evidence_nodes": [
                self._canonical_content_evidence_payload(evidence_node_id)
                for evidence_node_id in member_ids
            ],
        }

    def _prepare_content_result(
        self,
        job: dict[str, Any],
    ) -> dict[str, Any]:
        canonical_node_id = int(job["canonical_node_id"])
        member_ids = list(job["member_ids"])
        center_id = int(job["center_id"])
        center_title = str(job["center_title"])
        center_statement = str(job["center_statement"])
        previous_title = str(job["previous_title"])
        previous_statement = str(job["previous_statement"])
        used_statement_fallback = False
        used_title_fallback = False
        content_attempts = 0
        attempt_records: list[dict[str, Any]] = []
        validation_passed = True
        validation_failure: dict[str, Any] | None = None
        requires_generation = len(member_ids) > 1 or not _valid_canonical_title(
            center_title
        )
        if not requires_generation:
            title = center_title
            statement = center_statement
            statement_embedding = job["center_embedding"].copy()
        else:
            evidence_nodes = deepcopy(job["evidence_nodes"])
            candidate_title = ""
            candidate_statement = ""
            validation_result: dict[str, Any] | None = None
            for attempt in range(1, CANONICAL_CONTENT_MAX_ATTEMPTS + 1):
                generation_payload: dict[str, Any] = {
                    "canonical_node_id": canonical_node_id,
                    "type": job["type"],
                    "previous_candidate": "",
                    "validation_feedback": "",
                    "evidence_nodes": evidence_nodes,
                }
                if validation_result is not None:
                    generation_payload["previous_candidate"] = {
                        "title": candidate_title,
                        "statement": candidate_statement,
                    }
                    generation_payload["validation_feedback"] = str(
                        validation_result["feedback"]
                    )
                input_previous_candidate = deepcopy(
                    generation_payload["previous_candidate"]
                )
                input_validation_feedback = str(
                    generation_payload["validation_feedback"]
                )
                result = self.llm_client.call_json(
                    model_task="canonical_statement",
                    prompt=self.prompts["statement"],
                    input_text=json.dumps(
                        generation_payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    usage_task="canonical_content_generation",
                    validator=_validate_content_result,
                    preserve_latex_escapes=True,
                )
                if result is None:
                    raise RuntimeError(
                        "Canonical content generation returned no valid result."
                    )
                candidate_title = clean_model_text(str(result["title"]))
                candidate_statement = clean_model_text(str(result["statement"]))
                validation_payload = {
                    "canonical_node_id": canonical_node_id,
                    "type": job["type"],
                    "candidate": {
                        "title": candidate_title,
                        "statement": candidate_statement,
                    },
                    "evidence_nodes": evidence_nodes,
                }
                validation_result = self.llm_client.call_json(
                    model_task="canonical_statement",
                    prompt=self.prompts["statement_validation"],
                    input_text=json.dumps(
                        validation_payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    usage_task="canonical_statement_validation",
                    validator=_validate_content_validation,
                    preserve_latex_escapes=True,
                )
                if validation_result is None:
                    raise RuntimeError(
                        "Canonical content validation returned no valid result."
                    )
                attempt_records.append(
                    {
                        "attempt": attempt,
                        "previous_candidate": input_previous_candidate,
                        "validation_feedback": input_validation_feedback,
                        "generated_candidate": {
                            "title": candidate_title,
                            "statement": candidate_statement,
                        },
                        "validation": deepcopy(validation_result),
                    }
                )
                content_attempts = attempt
                if validation_result["valid"] is True:
                    break
                self.logger.warning(
                    "Canonical content validation failed; feedback will be "
                    "used by the next generation attempt",
                    canonical_node_id=canonical_node_id,
                    attempt=attempt,
                    max_attempts=CANONICAL_CONTENT_MAX_ATTEMPTS,
                    validation_feedback=validation_result,
                )
            title = candidate_title
            statement = candidate_statement
            validation_passed = bool(
                validation_result and validation_result.get("valid") is True
            )
            if not validation_passed:
                validation_failure = {
                    "canonical_node_id": canonical_node_id,
                    "attempts": content_attempts,
                    "final_feedback": str(validation_result["feedback"]),
                }
                self.logger.warning(
                    "Canonical content remained invalid after the maximum "
                    "feedback rounds; accepting the final candidate and "
                    "recording diagnostics",
                    canonical_node_id=canonical_node_id,
                    attempts=content_attempts,
                    validation_feedback=validation_result,
                )

            if not _valid_canonical_title(title):
                title = _fallback_canonical_title(
                    node_type=str(job["type"]),
                    center_title=center_title,
                    center_statement=center_statement,
                )
                used_title_fallback = True
                self.logger.warning(
                    "Canonical title remained generic after model repair; "
                    "using semantic fallback title",
                    canonical_node_id=canonical_node_id,
                    fallback_title=title,
                )

            if statement == center_statement:
                statement_embedding = job["center_embedding"].copy()
            elif statement == previous_statement and _is_normalized_vector(
                job["previous_statement_embedding"]
            ):
                statement_embedding = job["previous_statement_embedding"].copy()
            else:
                statement_embedding = self.statement_embedder.embed(
                    statement,
                    usage_task="canonical_statement_embedding",
                )

        if title == previous_title and _is_normalized_vector(
            job["previous_title_embedding"]
        ):
            title_embedding = job["previous_title_embedding"].copy()
        else:
            title_embedding = self.statement_embedder.embed(
                title,
                usage_task="canonical_title_embedding",
            )
        return {
            "action": "finalize",
            "canonical_node_id": canonical_node_id,
            "member_ids": member_ids,
            "title": title,
            "statement": statement,
            "title_embedding": title_embedding,
            "statement_embedding": statement_embedding,
            "used_statement_fallback": used_statement_fallback,
            "used_title_fallback": used_title_fallback,
            "content_attempts": content_attempts,
            "validation_passed": validation_passed,
            "validation_failure": validation_failure,
            "construction_record": {
                "record_schema_version": (
                    CONSTRUCTION_RECORDS_SCHEMA_VERSION
                ),
                "canonical_graph_id": self.graph_id,
                "target_revision": self.target_revision,
                "evidence_graph_id": self.evidence_graph_id,
                "evidence_graph_revision": self.evidence_manifest["revision"],
                "canonical_node_id": canonical_node_id,
                "evidence_node_ids": member_ids,
                "mode": (
                    "generated_and_validated"
                    if requires_generation
                    else "reused_single_evidence_content"
                ),
                "attempts": attempt_records,
                "final": {
                    "title": title,
                    "statement": statement,
                    "validation_passed": validation_passed,
                    "used_statement_fallback": used_statement_fallback,
                    "used_title_fallback": used_title_fallback,
                },
            },
        }

    def _commit_content_result(self, result: dict[str, Any]) -> None:
        canonical_node_id = int(result["canonical_node_id"])
        if canonical_node_id not in self.dirty_content_node_ids:
            raise RuntimeError(
                "Prepared content result targets an already completed "
                f"Canonical Node: {canonical_node_id}"
            )
        node = self.nodes[canonical_node_id - 1]
        if list(node["evidence_node_ids"]) != list(result["member_ids"]):
            raise RuntimeError(
                "Canonical membership changed while content was being "
                f"prepared for node {canonical_node_id}."
            )
        title_embedding = np.asarray(
            result["title_embedding"],
            dtype=np.float32,
        )
        statement_embedding = np.asarray(
            result["statement_embedding"],
            dtype=np.float32,
        )
        if not _is_normalized_vector(title_embedding):
            raise ValueError(
                f"Canonical title embedding is invalid for node {canonical_node_id}."
            )
        if not _is_normalized_vector(statement_embedding):
            raise ValueError(
                "Canonical statement embedding is invalid for node "
                f"{canonical_node_id}."
            )
        node["title"] = str(result["title"])
        node["statement"] = str(result["statement"])
        self.title_embeddings[canonical_node_id - 1] = title_embedding
        self.statement_embeddings[canonical_node_id - 1] = statement_embedding
        self.dirty_content_node_ids.remove(canonical_node_id)
        used_statement_fallback = bool(result["used_statement_fallback"])
        used_title_fallback = bool(result["used_title_fallback"])
        if used_statement_fallback:
            self.fallback_statement_node_ids.add(canonical_node_id)
        else:
            self.fallback_statement_node_ids.discard(canonical_node_id)
        if used_title_fallback:
            self.fallback_title_node_ids.add(canonical_node_id)
        else:
            self.fallback_title_node_ids.discard(canonical_node_id)
        validation_failure = result.get("validation_failure")
        if validation_failure is None:
            self.content_validation_failures.pop(canonical_node_id, None)
        else:
            self.content_validation_failures[canonical_node_id] = deepcopy(
                validation_failure
            )
        construction_record = deepcopy(result["construction_record"])
        construction_record["recorded_at"] = utc_now()
        self._write_checkpoint_record(
            "content",
            f"{canonical_node_id:08d}",
            construction_record,
        )
        print(
            "[canonical-graph] finalized title/statement "
            f"canonical_node_id={canonical_node_id}, "
            f"remaining={len(self.dirty_content_node_ids)}, "
            f"attempts={int(result['content_attempts'])}, "
            f"validation_passed={bool(result['validation_passed'])}, "
            f"statement_fallback={used_statement_fallback}, "
            f"title_fallback={used_title_fallback}"
        )

    def _finalize_and_commit(self) -> dict[str, Any]:
        if self.dirty_content_node_ids:
            raise RuntimeError(
                "Cannot finalize with unresolved Canonical titles/statements."
            )
        if self.processed_evidence_node_count != len(self.evidence_nodes):
            raise RuntimeError(
                "Cannot finalize before every Evidence Node is processed."
            )
        if np.any(self.mapping == 0):
            raise RuntimeError("Cannot finalize with unmapped Evidence Nodes.")

        edges, projected_self_loops = self._project_canonical_edges()
        possible_edges = self._materialize_possible_edges()
        low_cohesion_ids = self._low_cohesion_node_ids()
        now = utc_now()
        diagnostics = {
            "revision": self.target_revision,
            "projected_self_loop_evidence_edge_ids": projected_self_loops,
            "fallback_statement_canonical_node_ids": sorted(
                self.fallback_statement_node_ids
            ),
            "fallback_title_canonical_node_ids": sorted(self.fallback_title_node_ids),
            "low_cohesion_canonical_node_ids": low_cohesion_ids,
            "canonical_content_validation": {
                "max_attempts": CANONICAL_CONTENT_MAX_ATTEMPTS,
                "accepted_after_failed_validation": [
                    deepcopy(self.content_validation_failures[node_id])
                    for node_id in sorted(self.content_validation_failures)
                ],
            },
            "validation": {
                "passed": True,
                "checked_at": now,
            },
        }
        usage = self._merged_usage()
        manifest = {
            "graph_type": "canonical",
            "schema_version": SCHEMA_VERSION,
            "graph_id": self.graph_id,
            "name": self.graph_id,
            "revision": self.target_revision,
            "evidence_graph_id": self.evidence_graph_id,
            "evidence_graph_revision": self.evidence_manifest["revision"],
            "created_at": self.created_at,
            "updated_at": now,
            "counts": {
                "evidence_nodes": len(self.evidence_nodes),
                "evidence_edges": len(self.evidence_edges),
                "canonical_nodes": len(self.nodes),
                "canonical_edges": len(edges),
                "possible_equivalence_edges": len(possible_edges),
            },
            "files": dict(DATA_FILES),
            "embedding": {
                "model": self.embedding_model.name,
                "provider": self.embedding_model.provider,
                "dimensions": EMBEDDING_DIMENSIONS,
                "dtype": "float32",
                "row_mapping": "canonical_node_id_minus_1",
                "centroid_l2_normalized": True,
                "statement_l2_normalized": True,
                "title_l2_normalized": True,
                "center_evidence_l2_normalized": True,
            },
            "construction": {
                "operation": "construction",
                "processed_evidence_node_count": len(self.evidence_nodes),
                "alignment_unit": "source_document_snapshot",
                "document_commit_order": "evidence_node_id",
                "reranker_parallel_workers": (self.reranker_parallel_workers),
                "llm_parallel_workers": self.llm_parallel_workers,
                "content_parallel_workers": self.content_parallel_workers,
                "alignment_algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
                "content_algorithm_version": CONTENT_ALGORITHM_VERSION,
                "evidence_retrieval_top_k": self.evidence_retrieval_top_k,
                "evidence_similarity_threshold": (
                    self.evidence_similarity_threshold
                ),
                "reranker_top_k": self.reranker_top_k,
                "alignment_model": self.alignment_model.name,
                "alignment_provider": self.alignment_model.provider,
                "alignment_model_history": self._closed_model_history(
                    self.alignment_model_history,
                    end_node_id=len(self.evidence_nodes),
                ),
                "strict_verification_model": self.strict_model.name,
                "strict_verification_provider": self.strict_model.provider,
                "strict_verification_model_history": (
                    self._closed_model_history(
                        self.strict_verification_model_history,
                        end_node_id=len(self.evidence_nodes),
                    )
                ),
                "reranker_model": self.reranker_model.name,
                "reranker_provider": self.reranker_model.provider,
                "content_model": self.statement_model.name,
                "content_provider": self.statement_model.provider,
                "alignment_prompt_sha256": self.alignment_prompt_sha256_used,
                "strict_cluster_pair_prompt_sha256": (
                    self.strict_cluster_pair_prompt_sha256_used
                ),
                "content_prompt_sha256": self._prompt_sha256("statement"),
                "statement_validation_prompt_sha256": (
                    self._prompt_sha256("statement_validation")
                ),
            },
            "evidence_prefix": self._evidence_prefix_metadata(
                node_count=len(self.evidence_nodes),
                edge_count=len(self.evidence_edges),
            ),
            "construction_records": {
                "schema_version": CONSTRUCTION_RECORDS_SCHEMA_VERSION,
                "directory": CONSTRUCTION_RECORDS_DIR_NAME,
                "revision_directory": (
                    f"{CONSTRUCTION_RECORDS_DIR_NAME}/"
                    f"revision_{self.target_revision:04d}"
                ),
            },
        }
        errors = validate_canonical_graph(
            manifest=manifest,
            nodes=self.nodes,
            edges=edges,
            possible_equivalence_edges=possible_edges,
            evidence_node_to_canonical=self.mapping,
            embedding_sums=self.embedding_sums,
            centroid_embeddings=self.centroid_embeddings,
            statement_embeddings=self.statement_embeddings,
            title_embeddings=self.title_embeddings,
            center_evidence_embeddings=(self.center_evidence_embeddings),
            diagnostics=diagnostics,
            usage=usage,
            evidence_manifest=self.evidence_manifest,
            evidence_nodes=self.evidence_nodes,
            evidence_edges=self.evidence_edges,
            evidence_embeddings=self.evidence_embeddings,
            expected_graph_id=self.graph_id,
        )
        if errors:
            raise ValueError("Canonical Graph validation failed: " + "; ".join(errors))

        self._save_transactionally(
            manifest=manifest,
            edges=edges,
            possible_edges=possible_edges,
            diagnostics=diagnostics,
            usage=usage,
            construction_records=self._materialize_construction_records(
                manifest=manifest,
            ),
        )
        return {
            "graph_id": self.graph_id,
            "graph_dir": str(self.graph_dir),
            "evidence_graph_id": self.evidence_graph_id,
            "evidence_graph_revision": self.evidence_manifest["revision"],
            "revision": self.target_revision,
            "created": not self.base_existed,
            "updated": True,
            "counts": dict(manifest["counts"]),
            "fallback_statement_count": len(self.fallback_statement_node_ids),
            "fallback_title_count": len(self.fallback_title_node_ids),
            "content_validation_failure_count": len(
                self.content_validation_failures
            ),
            "low_cohesion_node_count": len(low_cohesion_ids),
            "projected_self_loop_count": len(projected_self_loops),
        }

    def _project_canonical_edges(
        self,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        grouped: dict[tuple[int, int], list[int]] = {}
        self_loops: list[int] = []
        for edge in self.evidence_edges:
            evidence_edge_id = int(edge["id"])
            source = int(self.mapping[int(edge["source"]) - 1])
            target = int(self.mapping[int(edge["target"]) - 1])
            if source == target:
                self_loops.append(evidence_edge_id)
                continue
            grouped.setdefault((source, target), []).append(evidence_edge_id)

        for node in self.nodes:
            node["in_edges"] = []
            node["out_edges"] = []
        edges: list[dict[str, Any]] = []
        for source, target in sorted(grouped):
            evidence_edge_ids = sorted(grouped[(source, target)])
            types = sorted(
                {
                    str(self.evidence_edges[evidence_edge_id - 1]["type"])
                    for evidence_edge_id in evidence_edge_ids
                },
                key=lambda value: TYPE_ORDER[value],
            )
            edge_id = len(edges) + 1
            edges.append(
                {
                    "id": edge_id,
                    "types": types,
                    "source": source,
                    "target": target,
                    "evidence_edge_ids": evidence_edge_ids,
                }
            )
            self.nodes[source - 1]["out_edges"].append(edge_id)
            self.nodes[target - 1]["in_edges"].append(edge_id)
        return edges, sorted(self_loops)

    def _materialize_possible_edges(self) -> list[dict[str, Any]]:
        for node in self.nodes:
            node["possible_equivalence_edge_ids"] = []
        possible_edges: list[dict[str, Any]] = []
        for node_a, node_b in sorted(self.possible_pairs):
            if node_a == node_b:
                continue
            edge_id = len(possible_edges) + 1
            possible_edges.append(
                {
                    "id": edge_id,
                    "type": "possible_equivalence",
                    "node_a": node_a,
                    "node_b": node_b,
                    "status": "needs_review",
                }
            )
            self.nodes[node_a - 1]["possible_equivalence_edge_ids"].append(edge_id)
            self.nodes[node_b - 1]["possible_equivalence_edge_ids"].append(edge_id)
        return possible_edges

    def _low_cohesion_node_ids(self) -> list[int]:
        low_cohesion: list[int] = []
        for node in self.nodes:
            canonical_node_id = int(node["id"])
            member_ids = np.asarray(
                node["evidence_node_ids"],
                dtype=np.int64,
            )
            scores = (
                self.evidence_embeddings[member_ids - 1]
                @ self.centroid_embeddings[canonical_node_id - 1]
            )
            if float(np.min(scores)) < self.low_cohesion_similarity:
                low_cohesion.append(canonical_node_id)
        return low_cohesion

    def _load_checkpoint(
        self,
        *,
        existing: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.checkpoint_dir.exists():
            return None
        if not self.checkpoint_dir.is_dir():
            raise NotADirectoryError(
                f"Canonical Graph checkpoint is not a directory: {self.checkpoint_dir}"
            )
        pointer_path = self.checkpoint_dir / CHECKPOINT_POINTER_FILE
        if not pointer_path.is_file():
            # A process may have stopped before atomically publishing its first
            # complete snapshot. Such unreferenced temporary data is unusable.
            self._delete_checkpoint()
            return None
        snapshot_name = pointer_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"snapshot-[0-9a-f]{32}", snapshot_name):
            raise ValueError("Canonical Graph checkpoint pointer is invalid.")
        snapshot_dir = (self.checkpoint_dir / snapshot_name).resolve()
        if snapshot_dir.parent != self.checkpoint_dir.resolve():
            raise ValueError("Unsafe Canonical Graph checkpoint snapshot path.")
        if not snapshot_dir.is_dir():
            raise ValueError(
                f"Canonical Graph checkpoint snapshot is missing: {snapshot_dir}"
            )

        present = {path.name for path in snapshot_dir.iterdir() if path.is_file()}
        if "state.json" not in present:
            raise ValueError("Canonical Graph checkpoint state is missing.")
        state = read_json(snapshot_dir / "state.json")
        if not isinstance(state, dict):
            raise ValueError("Canonical Graph checkpoint state must be an object.")
        if (
            state.get("revision_operation", "construction") != "construction"
            or state.get("revision_audit_records", {}) != {}
        ):
            raise ValueError(
                "Only construction checkpoints are supported. This checkpoint "
                "contains an unsupported review operation or revision audit "
                "records; use a new output directory for construction."
            )
        checkpoint_version = state.get("checkpoint_version")
        if checkpoint_version not in {
            LEGACY_CHECKPOINT_VERSION,
            PRE_DOCUMENT_CHECKPOINT_VERSION,
            PRE_ONE_TO_ONE_CHECKPOINT_VERSION,
            PRE_CONSTRUCTION_RECORDS_CHECKPOINT_VERSION,
            CHECKPOINT_VERSION,
        }:
            raise ValueError(
                "Unsupported Canonical Graph checkpoint version: "
                f"{checkpoint_version!r}"
            )
        required = {
            "state.json",
            DATA_FILES["nodes"],
            DATA_FILES["possible_equivalence_edges"],
            DATA_FILES["evidence_node_to_canonical"],
            DATA_FILES["embedding_sums"],
            DATA_FILES["centroid_embeddings"],
            DATA_FILES["statement_embeddings"],
            DATA_FILES["usage"],
        }
        if checkpoint_version >= PRE_DOCUMENT_CHECKPOINT_VERSION:
            required.update(
                {
                    DATA_FILES["title_embeddings"],
                    DATA_FILES["center_evidence_embeddings"],
                }
            )
        if present != required:
            raise ValueError(
                "Canonical Graph checkpoint files are invalid: "
                f"missing={sorted(required - present)}, "
                f"unexpected={sorted(present - required)}"
            )
        if checkpoint_version == LEGACY_CHECKPOINT_VERSION:
            if state.get("phase") == "statements":
                state["phase"] = "content"
            legacy_dirty = state.pop("dirty_statement_node_ids", [])
            state["dirty_content_node_ids"] = legacy_dirty
            state.setdefault("fallback_title_node_ids", [])
            self._checkpoint_schema_migrated = True
            self._checkpoint_schema_migration_messages.append(
                "checkpoint v1 -> v2; Canonical title and center-Evidence "
                "matrices will be materialized without rerunning alignment"
            )
        if checkpoint_version < CHECKPOINT_VERSION:
            state["checkpoint_version"] = CHECKPOINT_VERSION
            state.setdefault("active_document_source_id", "")
            state.setdefault("active_document_node_ids", [])
            state.setdefault("active_document_results", {})
            state.setdefault("active_document_snapshot_canonical_node_count", 0)
            state.setdefault("construction_recording_started_at", utc_now())
            state.setdefault(
                "construction_recording_start_evidence_node_id",
                int(state.get("processed_evidence_node_count", 0)) + 1,
            )
            state.setdefault("construction_records_complete", False)
            state.setdefault("construction_prompt_snapshots", dict(self.prompts))
            self._checkpoint_schema_migrated = True
            self._checkpoint_schema_migration_messages.append(
                "checkpoint v1/v2/v3/v4 -> v5; structured construction "
                "records start from the resumable Evidence Node suffix"
            )
        if state.get("graph_id") != self.graph_id:
            raise ValueError("Canonical Graph checkpoint graph_id mismatch.")
        if state.get("evidence_graph_id") != self.evidence_graph_id:
            raise ValueError("Canonical Graph checkpoint Evidence Graph mismatch.")

        base_revision = int(existing["manifest"].get("revision", 0) or 0)
        target_revision = int(state.get("target_revision", 0) or 0)
        if target_revision <= base_revision:
            self._delete_checkpoint()
            return None
        if int(state.get("base_revision", -1)) != base_revision:
            raise ValueError(
                "Canonical Graph formal revision changed after this checkpoint "
                "was created."
            )
        if bool(state.get("base_existed")) != bool(existing["existed"]):
            raise ValueError("Canonical Graph checkpoint base state does not match.")
        if state.get("evidence_fingerprint") != self._evidence_fingerprint():
            raise ValueError(
                "Evidence Graph changed after this Canonical checkpoint was "
                "created. Restore the matching Evidence Graph revision or "
                "explicitly discard the checkpoint."
            )
        stored_signature = state.get("construction_signature")
        current_signature = self._construction_signature()
        if stored_signature != current_signature:
            migration_messages = self._apply_checkpoint_content_prompt_transition(
                state=state,
                stored_signature=stored_signature,
                current_signature=current_signature,
            )
            if state.get("construction_signature") != current_signature:
                migration_messages.extend(
                    self._apply_checkpoint_model_transitions(
                        state=state,
                        stored_signature=state.get("construction_signature"),
                        current_signature=current_signature,
                    )
                )
            if state.get("construction_signature") != current_signature:
                raise ValueError(
                    "Cross-document models, prompts, or retrieval settings "
                    "changed after this checkpoint was created."
                )
            if migration_messages:
                self._checkpoint_schema_migrated = True
                self._checkpoint_schema_migration_messages.extend(migration_messages)

        checkpoint_nodes = read_json(snapshot_dir / DATA_FILES["nodes"])
        if (
            state.get("active_document_node_ids")
            and not state.get("active_document_snapshot_canonical_node_count")
        ):
            state["active_document_snapshot_canonical_node_count"] = len(
                checkpoint_nodes
            )
        state.setdefault("content_validation_failures", [])
        signature_prompts = state["construction_signature"]["prompts"]
        state.setdefault(
            "alignment_prompt_sha256_used",
            signature_prompts["alignment"]["sha256"],
        )
        state.setdefault(
            "strict_cluster_pair_prompt_sha256_used",
            signature_prompts["strict_cluster_pair"]["sha256"],
        )
        if state.pop("reprocess_all_content_after_prompt_transition", False):
            state["dirty_content_node_ids"] = list(
                range(1, len(checkpoint_nodes) + 1)
            )
            state["fallback_statement_node_ids"] = []
            state["fallback_title_node_ids"] = []
            state["content_validation_failures"] = []
            if state.get("phase") != "alignment":
                state["phase"] = "content"
        checkpoint = {
            "state": state,
            "nodes": checkpoint_nodes,
            "possible_equivalence_edges": read_json(
                snapshot_dir / DATA_FILES["possible_equivalence_edges"]
            ),
            "mapping": np.load(
                snapshot_dir / DATA_FILES["evidence_node_to_canonical"],
                allow_pickle=False,
            ),
            "embedding_sums": np.load(
                snapshot_dir / DATA_FILES["embedding_sums"],
                allow_pickle=False,
            ),
            "centroid_embeddings": np.load(
                snapshot_dir / DATA_FILES["centroid_embeddings"],
                allow_pickle=False,
            ),
            "statement_embeddings": np.load(
                snapshot_dir / DATA_FILES["statement_embeddings"],
                allow_pickle=False,
            ),
            "usage": read_json(snapshot_dir / DATA_FILES["usage"]),
        }
        if checkpoint_version >= PRE_DOCUMENT_CHECKPOINT_VERSION:
            checkpoint["title_embeddings"] = np.load(
                snapshot_dir / DATA_FILES["title_embeddings"],
                allow_pickle=False,
            )
            checkpoint["center_evidence_embeddings"] = np.load(
                snapshot_dir / DATA_FILES["center_evidence_embeddings"],
                allow_pickle=False,
            )
        else:
            checkpoint["title_embeddings"] = np.zeros(
                (len(checkpoint_nodes), EMBEDDING_DIMENSIONS),
                dtype=np.float32,
            )
            checkpoint["center_evidence_embeddings"] = self._center_embedding_matrix(
                checkpoint_nodes
            )
            # Every legacy checkpoint node needs a final title vector. During
            # first-revision alignment these nodes are already dirty; the union
            # also safely upgrades incremental checkpoints with clean old rows.
            state["dirty_content_node_ids"] = list(range(1, len(checkpoint_nodes) + 1))
        self._validate_checkpoint(checkpoint)
        self._prune_checkpoint_records(state=state, nodes=checkpoint_nodes)
        return checkpoint

    def _restore_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint["state"]
        self.nodes = checkpoint["nodes"]
        self.mapping = checkpoint["mapping"]
        self.embedding_sums = checkpoint["embedding_sums"]
        self.centroid_embeddings = checkpoint["centroid_embeddings"]
        self.statement_embeddings = checkpoint["statement_embeddings"]
        self.title_embeddings = checkpoint["title_embeddings"]
        self.center_evidence_embeddings = checkpoint["center_evidence_embeddings"]
        self.possible_pairs = {
            (int(edge["node_a"]), int(edge["node_b"]))
            for edge in checkpoint["possible_equivalence_edges"]
        }
        self.base_usage = checkpoint["usage"]
        self.dirty_content_node_ids = {
            int(node_id) for node_id in state["dirty_content_node_ids"]
        }
        self.fallback_statement_node_ids = {
            int(node_id) for node_id in state["fallback_statement_node_ids"]
        }
        self.fallback_title_node_ids = {
            int(node_id) for node_id in state["fallback_title_node_ids"]
        }
        self.content_validation_failures = {
            int(item["canonical_node_id"]): deepcopy(item)
            for item in state.get("content_validation_failures", [])
        }
        self.processed_evidence_node_count = int(state["processed_evidence_node_count"])
        self.base_revision = int(state["base_revision"])
        self.target_revision = int(state["target_revision"])
        self.base_existed = bool(state["base_existed"])
        self.created_at = str(state["created_at"])
        self.phase = str(state["phase"])
        self.alignment_prompt_sha256_used = str(
            state["alignment_prompt_sha256_used"]
        )
        self.strict_cluster_pair_prompt_sha256_used = str(
            state["strict_cluster_pair_prompt_sha256_used"]
        )
        signature = state["construction_signature"]
        signature_models = signature["models"]
        self.alignment_model_history = self._checkpoint_model_history(
            state=state,
            key="alignment_model_history",
            fallback_identity=signature_models["alignment"],
        )
        self.strict_verification_model_history = self._checkpoint_model_history(
            state=state,
            key="strict_verification_model_history",
            fallback_identity=signature_models["strict_verification"],
        )
        self.active_document_source_id = str(state.get("active_document_source_id", ""))
        self.active_document_node_ids = [
            int(node_id) for node_id in state.get("active_document_node_ids", [])
        ]
        self.active_document_results = deepcopy(
            state.get("active_document_results", {})
        )
        self.active_document_snapshot_canonical_node_count = int(
            state.get("active_document_snapshot_canonical_node_count", len(self.nodes))
        )
        self.construction_recording_started_at = str(
            state.get("construction_recording_started_at", utc_now())
        )
        self.construction_recording_start_evidence_node_id = int(
            state.get(
                "construction_recording_start_evidence_node_id",
                self.processed_evidence_node_count + 1,
            )
        )
        self.construction_records_complete = bool(
            state.get("construction_records_complete", False)
        )
        prompt_snapshots = state.get("construction_prompt_snapshots", {})
        self.construction_prompt_snapshots = (
            {
                str(key): str(value)
                for key, value in prompt_snapshots.items()
            }
            if isinstance(prompt_snapshots, dict)
            else dict(self.prompts)
        )
        self.llm_request_pacer.restore_state(state.get("llm_pacing_state"))
        self._active_document_stage_changes = 0
        self._document_commit_in_progress = False

    def _reopen_invalid_finalization_titles(self) -> list[int]:
        """Return invalid completed titles to postprocessing on resume."""
        if self.phase != "finalization":
            return []
        invalid_title_ids = [
            int(node["id"])
            for node in self.nodes
            if not _valid_canonical_title(str(node.get("title", "")))
        ]
        if not invalid_title_ids:
            return []
        self.dirty_content_node_ids.update(invalid_title_ids)
        for node_id in invalid_title_ids:
            self.content_validation_failures.pop(node_id, None)
        self.phase = "content"
        return invalid_title_ids

    def _validate_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint["state"]
        nodes = checkpoint["nodes"]
        possible_edges = checkpoint["possible_equivalence_edges"]
        mapping = checkpoint["mapping"]
        embedding_sums = checkpoint["embedding_sums"]
        centroids = checkpoint["centroid_embeddings"]
        statement_embeddings = checkpoint["statement_embeddings"]
        title_embeddings = checkpoint["title_embeddings"]
        center_embeddings = checkpoint["center_evidence_embeddings"]
        usage = checkpoint["usage"]
        if not isinstance(nodes, list):
            raise ValueError("Checkpoint nodes must be a list.")
        if not isinstance(possible_edges, list):
            raise ValueError("Checkpoint possible-equivalence edges must be a list.")
        if not isinstance(usage, dict):
            raise ValueError("Checkpoint usage must be an object.")
        pacing_state = state.get("llm_pacing_state")
        if pacing_state is not None:
            self.llm_request_pacer.validated_state(pacing_state)
        for key in (
            "alignment_prompt_sha256_used",
            "strict_cluster_pair_prompt_sha256_used",
        ):
            value = state.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"Checkpoint {key} is invalid.")
        processed = int(state.get("processed_evidence_node_count", -1))
        if not 0 <= processed <= len(self.evidence_nodes):
            raise ValueError("Checkpoint processed Evidence Node count is invalid.")
        if state.get("phase") not in {
            "alignment",
            "content",
            "finalization",
        }:
            raise ValueError("Checkpoint phase is invalid.")
        recording_started_at = state.get("construction_recording_started_at")
        recording_start_id = state.get(
            "construction_recording_start_evidence_node_id"
        )
        prompt_snapshots = state.get("construction_prompt_snapshots")
        if not isinstance(recording_started_at, str) or not recording_started_at:
            raise ValueError("Checkpoint construction-record start time is invalid.")
        if (
            isinstance(recording_start_id, bool)
            or not isinstance(recording_start_id, int)
            or not 1 <= recording_start_id <= len(self.evidence_nodes) + 1
        ):
            raise ValueError(
                "Checkpoint construction-record Evidence start ID is invalid."
            )
        if not isinstance(state.get("construction_records_complete"), bool):
            raise ValueError(
                "Checkpoint construction-record completeness flag is invalid."
            )
        if (
            not isinstance(prompt_snapshots, dict)
            or set(prompt_snapshots) != set(PROMPT_PATHS)
            or any(not isinstance(value, str) for value in prompt_snapshots.values())
        ):
            raise ValueError("Checkpoint construction prompt snapshots are invalid.")
        self._validate_active_document_checkpoint_state(
            state=state,
            processed=processed,
            canonical_node_count=len(nodes),
        )
        for history_key in (
            "alignment_model_history",
            "strict_verification_model_history",
        ):
            if history_key in state:
                self._validated_model_history(
                    state[history_key],
                    processed_count=processed,
                    allow_open_end=True,
                )
        if (
            not isinstance(mapping, np.ndarray)
            or mapping.dtype != np.int64
            or mapping.shape != (len(self.evidence_nodes),)
            or np.any(mapping[:processed] <= 0)
            or np.any(mapping[processed:] != 0)
            or (processed and int(np.max(mapping[:processed])) > len(nodes))
        ):
            raise ValueError("Checkpoint Evidence-to-Canonical mapping is invalid.")

        memberships = np.zeros(processed, dtype=np.int64)
        for expected_id, node in enumerate(nodes, start=1):
            if not isinstance(node, dict) or node.get("id") != expected_id:
                raise ValueError(
                    f"Checkpoint Canonical Node ID invalid at row {expected_id}."
                )
            member_ids = node.get("evidence_node_ids")
            if (
                not isinstance(member_ids, list)
                or not member_ids
                or member_ids != sorted(set(member_ids))
                or any(not 1 <= node_id <= processed for node_id in member_ids)
            ):
                raise ValueError(
                    f"Checkpoint membership invalid at Canonical Node {expected_id}."
                )
            for evidence_node_id in member_ids:
                if memberships[evidence_node_id - 1] != 0:
                    raise ValueError(
                        "Checkpoint Evidence Node has duplicate memberships."
                    )
                memberships[evidence_node_id - 1] = expected_id
                if self.evidence_nodes[evidence_node_id - 1]["type"] != node.get(
                    "type"
                ):
                    raise ValueError("Checkpoint Canonical Node contains mixed types.")
            center = node.get("center_evidence_node_id")
            diverse = node.get("diverse_evidence_node_ids")
            if center not in member_ids:
                raise ValueError("Checkpoint center Evidence Node is not a member.")
            if (
                not isinstance(diverse, list)
                or len(diverse) > 2
                or len(set(diverse)) != len(diverse)
                or center in diverse
                or any(item not in member_ids for item in diverse)
            ):
                raise ValueError("Checkpoint diverse Evidence Node list is invalid.")
        if not np.array_equal(memberships, mapping[:processed]):
            raise ValueError("Checkpoint mappings disagree with Canonical memberships.")

        for name, matrix in (
            ("embedding sums", embedding_sums),
            ("centroids", centroids),
            ("statement embeddings", statement_embeddings),
            ("title embeddings", title_embeddings),
            ("center Evidence embeddings", center_embeddings),
        ):
            if (
                not isinstance(matrix, np.ndarray)
                or matrix.dtype != np.float32
                or matrix.shape != (len(nodes), EMBEDDING_DIMENSIONS)
                or not np.all(np.isfinite(matrix))
            ):
                raise ValueError(f"Checkpoint {name} matrix is invalid.")
        if len(nodes):
            sum_norms = np.linalg.norm(embedding_sums, axis=1)
            centroid_norms = np.linalg.norm(centroids, axis=1)
            if np.any(sum_norms <= 0) or not np.allclose(
                centroid_norms,
                1.0,
                rtol=1e-4,
                atol=1e-4,
            ):
                raise ValueError("Checkpoint sum or centroid vectors are invalid.")
            expected_centroids = embedding_sums / sum_norms[:, None]
            if not np.allclose(
                expected_centroids,
                centroids,
                rtol=1e-4,
                atol=1e-4,
            ):
                raise ValueError("Checkpoint centroids disagree with embedding sums.")
            center_norms = np.linalg.norm(center_embeddings, axis=1)
            if not np.allclose(
                center_norms,
                1.0,
                rtol=1e-4,
                atol=1e-4,
            ):
                raise ValueError("Checkpoint center Evidence vectors are invalid.")
            expected_center_embeddings = self._center_embedding_matrix(nodes)
            if not np.allclose(
                center_embeddings,
                expected_center_embeddings,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError(
                    "Checkpoint center Evidence vectors disagree with nodes."
                )

        dirty = state.get("dirty_content_node_ids")
        fallback = state.get("fallback_statement_node_ids")
        title_fallback = state.get("fallback_title_node_ids")
        for name, values in (
            ("dirty", dirty),
            ("statement fallback", fallback),
            ("title fallback", title_fallback),
        ):
            if (
                not isinstance(values, list)
                or values != sorted(set(values))
                or any(not 1 <= node_id <= len(nodes) for node_id in values)
            ):
                raise ValueError(f"Checkpoint {name} statement node IDs are invalid.")
        validation_failures = state.get("content_validation_failures")
        validation_failure_map = _validated_content_validation_failures(
            validation_failures,
            canonical_node_count=len(nodes),
        )
        if set(validation_failure_map) & set(dirty):
            raise ValueError(
                "Checkpoint content-validation diagnostics may only reference "
                "completed Canonical Nodes."
            )
        clean_rows = [
            node_id - 1
            for node_id in range(1, len(nodes) + 1)
            if node_id not in set(dirty)
        ]
        if clean_rows and not np.allclose(
            np.linalg.norm(statement_embeddings[clean_rows], axis=1),
            1.0,
            rtol=1e-4,
            atol=1e-4,
        ):
            raise ValueError(
                "Checkpoint contains an invalid completed statement vector."
            )
        if clean_rows and not np.allclose(
            np.linalg.norm(title_embeddings[clean_rows], axis=1),
            1.0,
            rtol=1e-4,
            atol=1e-4,
        ):
            raise ValueError("Checkpoint contains an invalid completed title vector.")

        seen_pairs: set[tuple[int, int]] = set()
        for edge in possible_edges:
            if not isinstance(edge, dict):
                raise ValueError("Checkpoint possible-equivalence edge is invalid.")
            node_a = edge.get("node_a")
            node_b = edge.get("node_b")
            if (
                not isinstance(node_a, int)
                or not isinstance(node_b, int)
                or not 1 <= node_a < node_b <= len(nodes)
                or (node_a, node_b) in seen_pairs
            ):
                raise ValueError(
                    "Checkpoint possible-equivalence endpoints are invalid."
                )
            seen_pairs.add((node_a, node_b))

    def _validate_active_document_checkpoint_state(
        self,
        *,
        state: dict[str, Any],
        processed: int,
        canonical_node_count: int,
    ) -> None:
        source_id = state.get("active_document_source_id", "")
        node_ids = state.get("active_document_node_ids", [])
        results = state.get("active_document_results", {})
        snapshot_canonical_count = state.get(
            "active_document_snapshot_canonical_node_count",
            0,
        )
        if not isinstance(source_id, str):
            raise ValueError("Checkpoint active document source is invalid.")
        if not isinstance(node_ids, list) or any(
            isinstance(node_id, bool) or not isinstance(node_id, int)
            for node_id in node_ids
        ):
            raise ValueError("Checkpoint active document node IDs are invalid.")
        if not isinstance(results, dict):
            raise ValueError("Checkpoint active document results are invalid.")
        if (
            isinstance(snapshot_canonical_count, bool)
            or not isinstance(snapshot_canonical_count, int)
            or snapshot_canonical_count < 0
            or snapshot_canonical_count > canonical_node_count
        ):
            raise ValueError(
                "Checkpoint active document Canonical snapshot count is invalid."
            )
        if not node_ids:
            if source_id or results or snapshot_canonical_count:
                raise ValueError(
                    "Checkpoint has active document data without node IDs."
                )
            return
        if snapshot_canonical_count != canonical_node_count:
            raise ValueError(
                "Checkpoint active document snapshot must match the frozen "
                "Canonical state."
            )
        if state.get("phase") != "alignment":
            raise ValueError("Only the alignment phase may contain an active document.")
        if node_ids != list(range(processed + 1, processed + 1 + len(node_ids))):
            raise ValueError(
                "Checkpoint active document must be the next contiguous "
                "Evidence Node range."
            )
        if node_ids[-1] > len(self.evidence_nodes):
            raise ValueError("Checkpoint active document exceeds Evidence data.")
        if not source_id or any(
            clean_whitespace(str(self.evidence_nodes[node_id - 1]["source_id"]))
            != source_id
            for node_id in node_ids
        ):
            raise ValueError(
                "Checkpoint active document nodes do not share its source."
            )
        valid_keys = {str(node_id) for node_id in node_ids}
        if any(key not in valid_keys for key in results):
            raise ValueError("Checkpoint active document contains an unrelated result.")
        for key, result in results.items():
            if not isinstance(result, dict):
                raise ValueError("Checkpoint document stage result is invalid.")
            stage = result.get("stage")
            if stage == "candidates_ready":
                candidates = result.get("candidates")
                if not isinstance(candidates, list) or not candidates:
                    raise ValueError("Checkpoint candidate stage result is invalid.")
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        raise ValueError("Checkpoint Canonical candidate is invalid.")
                    canonical_node_id = candidate.get("canonical_node_id")
                    if (
                        isinstance(canonical_node_id, bool)
                        or not isinstance(canonical_node_id, int)
                        or not 1 <= canonical_node_id <= canonical_node_count
                    ):
                        raise ValueError("Checkpoint Canonical candidate ID is invalid.")
            elif stage == "proposed":
                proposal = result.get("proposal")
                if (
                    not isinstance(proposal, dict)
                    or proposal.get("evidence_node_id") != int(key)
                    or proposal.get("action") not in {"create", "join"}
                ):
                    raise ValueError("Checkpoint alignment proposal is invalid.")
                target_anchor = proposal.get("target_anchor")
                if proposal["action"] == "join" and (
                    isinstance(target_anchor, bool)
                    or not isinstance(target_anchor, int)
                    or not 1 <= target_anchor <= processed
                ):
                    raise ValueError("Checkpoint alignment target anchor is invalid.")
                for list_key in (
                    "possible_target_anchors",
                    "confirmed_anchors",
                    "merge_groups",
                    "possible_anchor_pairs",
                ):
                    if not isinstance(proposal.get(list_key), list):
                        raise ValueError("Checkpoint alignment proposal is incomplete.")
                if any(
                    not isinstance(group, list) or len(set(group)) < 2
                    for group in proposal["merge_groups"]
                ) or any(
                    not isinstance(pair, list) or len(set(pair)) != 2
                    for pair in proposal["possible_anchor_pairs"]
                ):
                    raise ValueError(
                        "Checkpoint alignment proposal groups are invalid."
                    )
                flat_anchors: list[Any] = [
                    *proposal["possible_target_anchors"],
                    *proposal["confirmed_anchors"],
                    *[
                        anchor
                        for group in proposal["merge_groups"]
                        if isinstance(group, list)
                        for anchor in group
                    ],
                    *[
                        anchor
                        for pair in proposal["possible_anchor_pairs"]
                        if isinstance(pair, list)
                        for anchor in pair
                    ],
                ]
                if any(
                    isinstance(anchor, bool)
                    or not isinstance(anchor, int)
                    or not 1 <= anchor <= processed
                    for anchor in flat_anchors
                ):
                    raise ValueError("Checkpoint alignment proposal anchor is invalid.")
            else:
                raise ValueError(
                    "Checkpoint document stage must be candidates_ready or proposed."
                )

    def _prune_checkpoint_records(
        self,
        *,
        state: dict[str, Any],
        nodes: list[dict[str, Any]],
    ) -> None:
        """Discard record files newer than the authoritative checkpoint."""
        if not self.checkpoint_records_dir.is_dir():
            return
        processed = int(state["processed_evidence_node_count"])
        dirty_content_ids = {
            int(node_id) for node_id in state.get("dirty_content_node_ids", [])
        }
        rules = {
            "alignment": lambda stem: int(stem) <= processed,
            "documents": lambda stem: int(stem.split("_")[1]) <= processed,
            "content": lambda stem: (
                int(stem) <= len(nodes) and int(stem) not in dirty_content_ids
            ),
        }
        for category, keep in rules.items():
            directory = self.checkpoint_records_dir / category
            if not directory.is_dir():
                continue
            pattern = (
                r"[0-9]{8}_[0-9]{8}"
                if category == "documents"
                else r"[0-9]{8}"
            )
            for path in directory.iterdir():
                if not path.is_file() or path.suffix != ".json":
                    continue
                if not re.fullmatch(pattern, path.stem) or not keep(path.stem):
                    path.unlink()
            self._fsync_directory(directory)

    def _save_checkpoint(self) -> None:
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_records_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "graph_id": self.graph_id,
            "evidence_graph_id": self.evidence_graph_id,
            "base_revision": self.base_revision,
            "target_revision": self.target_revision,
            "base_existed": self.base_existed,
            "evidence_fingerprint": self._evidence_fingerprint(),
            "construction_signature": self._construction_signature(),
            "alignment_prompt_sha256_used": (
                self.alignment_prompt_sha256_used
            ),
            "strict_cluster_pair_prompt_sha256_used": (
                self.strict_cluster_pair_prompt_sha256_used
            ),
            "processed_evidence_node_count": (self.processed_evidence_node_count),
            "dirty_content_node_ids": sorted(self.dirty_content_node_ids),
            "fallback_statement_node_ids": sorted(self.fallback_statement_node_ids),
            "fallback_title_node_ids": sorted(self.fallback_title_node_ids),
            "content_validation_failures": [
                deepcopy(self.content_validation_failures[node_id])
                for node_id in sorted(self.content_validation_failures)
            ],
            "alignment_model_history": deepcopy(self.alignment_model_history),
            "strict_verification_model_history": deepcopy(
                self.strict_verification_model_history
            ),
            "active_document_source_id": (self.active_document_source_id),
            "active_document_node_ids": list(self.active_document_node_ids),
            "active_document_results": deepcopy(self.active_document_results),
            "active_document_snapshot_canonical_node_count": (
                self.active_document_snapshot_canonical_node_count
            ),
            "construction_recording_started_at": (
                self.construction_recording_started_at
            ),
            "construction_recording_start_evidence_node_id": (
                self.construction_recording_start_evidence_node_id
            ),
            "construction_records_complete": self.construction_records_complete,
            "construction_prompt_snapshots": deepcopy(
                self.construction_prompt_snapshots
            ),
            "llm_pacing_state": self.llm_request_pacer.snapshot_state(),
            "phase": self.phase,
            "created_at": self.created_at,
            "updated_at": utc_now(),
        }
        possible_edges = [
            {"node_a": node_a, "node_b": node_b}
            for node_a, node_b in sorted(self.possible_pairs)
        ]
        usage = self._merged_usage()
        transaction_id = uuid.uuid4().hex
        temporary_dir = self.checkpoint_dir / (f".snapshot-{transaction_id}.tmp")
        snapshot_name = f"snapshot-{transaction_id}"
        snapshot_dir = self.checkpoint_dir / snapshot_name
        pointer_temp = self.checkpoint_dir / (
            f".{CHECKPOINT_POINTER_FILE}.{transaction_id}.tmp"
        )
        temporary_dir.mkdir()
        try:
            json_payloads = {
                "state.json": state,
                DATA_FILES["nodes"]: self.nodes,
                DATA_FILES["possible_equivalence_edges"]: possible_edges,
                DATA_FILES["usage"]: usage,
            }
            for filename, payload in json_payloads.items():
                self._write_json_fsynced(
                    temporary_dir / filename,
                    payload,
                )
            array_payloads = {
                DATA_FILES["evidence_node_to_canonical"]: self.mapping,
                DATA_FILES["embedding_sums"]: self.embedding_sums,
                DATA_FILES["centroid_embeddings"]: (self.centroid_embeddings),
                DATA_FILES["statement_embeddings"]: (self.statement_embeddings),
                DATA_FILES["title_embeddings"]: self.title_embeddings,
                DATA_FILES["center_evidence_embeddings"]: (
                    self.center_evidence_embeddings
                ),
            }
            for filename, matrix in array_payloads.items():
                self._write_npy_fsynced(
                    temporary_dir / filename,
                    matrix,
                )
            self._fsync_directory(temporary_dir)
            os.replace(temporary_dir, snapshot_dir)
            self._fsync_directory(self.checkpoint_dir)
            with pointer_temp.open("x", encoding="utf-8") as file:
                file.write(snapshot_name + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(
                pointer_temp,
                self.checkpoint_dir / CHECKPOINT_POINTER_FILE,
            )
            self._fsync_directory(self.checkpoint_dir)
            for path in self.checkpoint_dir.iterdir():
                if (
                    path == snapshot_dir
                    or path.name == CHECKPOINT_POINTER_FILE
                    or path == self.checkpoint_records_dir
                ):
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            self._fsync_directory(self.checkpoint_dir)
        except Exception:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            if pointer_temp.exists():
                pointer_temp.unlink()
            raise

    def _delete_checkpoint(self) -> None:
        if not self.checkpoint_dir.exists():
            return
        if not self.checkpoint_dir.is_dir():
            raise NotADirectoryError(
                f"Canonical Graph checkpoint is not a directory: {self.checkpoint_dir}"
            )
        shutil.rmtree(self.checkpoint_dir)
        if self.graph_dir.is_dir():
            self._fsync_directory(self.graph_dir)

    def _load_checkpoint_record_category(
        self,
        category: str,
    ) -> list[dict[str, Any]]:
        directory = self.checkpoint_records_dir / category
        if not directory.is_dir():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            record = read_json(path)
            if not isinstance(record, dict):
                raise ValueError(
                    f"Construction record must be an object: {path}"
                )
            records.append(record)
        return records

    def _materialize_construction_records(
        self,
        *,
        manifest: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        alignment_records = self._load_checkpoint_record_category("alignment")
        for record in alignment_records:
            evidence_node_id = int(record["evidence_node_id"])
            record.setdefault("outcome", {})["final_canonical_node_id"] = int(
                self.mapping[evidence_node_id - 1]
            )

        strict_records: list[dict[str, Any]] = []
        for alignment_record in alignment_records:
            trigger_id = int(alignment_record["evidence_node_id"])
            comparisons = alignment_record.get("decision_trace", {}).get(
                "strict_cluster_comparisons",
                [],
            )
            for index, comparison in enumerate(comparisons, start=1):
                item = deepcopy(comparison)
                item.update(
                    {
                        "record_schema_version": (
                            CONSTRUCTION_RECORDS_SCHEMA_VERSION
                        ),
                        "canonical_graph_id": self.graph_id,
                        "target_revision": self.target_revision,
                        "trigger_evidence_node_id": trigger_id,
                        "comparison_index": index,
                        "final_canonical_node_a": int(
                            self.mapping[int(item["canonical_anchor_a"]) - 1]
                        ),
                        "final_canonical_node_b": int(
                            self.mapping[int(item["canonical_anchor_b"]) - 1]
                        ),
                    }
                )
                strict_records.append(item)

        document_records = self._load_checkpoint_record_category("documents")
        content_records = self._load_checkpoint_record_category("content")
        expected_alignment_count = max(
            0,
            len(self.evidence_nodes)
            - self.construction_recording_start_evidence_node_id
            + 1,
        )
        records_complete = bool(
            self.construction_records_complete
            and len(alignment_records) == expected_alignment_count
        )
        prompt_hashes = {
            key: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for key, value in self.construction_prompt_snapshots.items()
        }
        run_manifest = {
            "record_schema_version": CONSTRUCTION_RECORDS_SCHEMA_VERSION,
            "operation": "construction",
            "canonical_graph_id": self.graph_id,
            "canonical_graph_revision": self.target_revision,
            "base_canonical_graph_revision": self.base_revision,
            "evidence_graph_id": self.evidence_graph_id,
            "evidence_graph_revision": self.evidence_manifest["revision"],
            "started_at": self.construction_recording_started_at,
            "completed_at": manifest["updated_at"],
            "models": {
                "alignment": self._model_identity(self.alignment_model),
                "strict_verification": self._model_identity(self.strict_model),
                "canonical_content": self._model_identity(self.statement_model),
                "reranker": self._model_identity(self.reranker_model),
                "embedding": self._model_identity(self.embedding_model),
            },
            "model_history": {
                "alignment": deepcopy(
                    manifest["construction"]["alignment_model_history"]
                ),
                "strict_verification": deepcopy(
                    manifest["construction"][
                        "strict_verification_model_history"
                    ]
                ),
            },
            "prompts": {
                key: {
                    "file": self.prompt_paths[key].name,
                    "sha256": prompt_hashes[key],
                }
                for key in sorted(self.construction_prompt_snapshots)
            },
            "retrieval": {
                "same_type_only": True,
                "evidence_top_k": self.evidence_retrieval_top_k,
                "minimum_similarity": self.evidence_similarity_threshold,
                "reranker_top_k": self.reranker_top_k,
            },
            "concurrency": {
                "reranker_workers": self.reranker_parallel_workers,
                "llm_workers": self.llm_parallel_workers,
                "content_workers": self.content_parallel_workers,
            },
            "algorithms": {
                "alignment_version": ALIGNMENT_ALGORITHM_VERSION,
                "content_version": CONTENT_ALGORITHM_VERSION,
            },
            "trace_coverage": {
                "evidence_node_id_start": (
                    self.construction_recording_start_evidence_node_id
                ),
                "evidence_node_id_end": len(self.evidence_nodes),
                "expected_alignment_records": expected_alignment_count,
                "saved_alignment_records": len(alignment_records),
                "complete_for_revision": records_complete,
                "note": (
                    "Complete for this revision."
                    if records_complete
                    else "The run resumed from a checkpoint created before "
                    "structured construction recording was available."
                ),
            },
            "counts": {
                "alignment_decisions": len(alignment_records),
                "strict_cluster_comparisons": len(strict_records),
                "document_commits": len(document_records),
                "content_postprocessing": len(content_records),
            },
            "id_semantics": {
                "candidate_canonical_node_ids": (
                    "IDs in the frozen document snapshot"
                ),
                "canonical_anchor": (
                    "stable center Evidence Node ID used across compact "
                    "Canonical renumbering"
                ),
                "final_canonical_node_id": (
                    "Canonical Node ID in the submitted revision"
                ),
            },
        }

        def jsonl(records: list[dict[str, Any]]) -> str:
            return "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            )

        readme = "# Construction records\n\n"
        readme += (
            "These files are immutable experiment and audit records for this "
            "Canonical Graph revision. Canonical candidate IDs refer to the "
            "frozen document snapshot; use anchors or final IDs when comparing "
            "against the submitted graph. API keys and full Evidence statements "
            "are intentionally not duplicated here.\n"
        )
        records = {
            "json": {"run_manifest.json": run_manifest},
            "text": {
                "README.md": readme,
                "alignment_decisions.jsonl": jsonl(alignment_records),
                "strict_cluster_comparisons.jsonl": jsonl(strict_records),
                "document_commits.jsonl": jsonl(document_records),
                "content_postprocessing.jsonl": jsonl(content_records),
                **{
                    f"prompts/{self.prompt_paths[key].name}": value
                    for key, value in self.construction_prompt_snapshots.items()
                },
            },
        }
        return records

    def _save_transactionally(
        self,
        *,
        manifest: dict[str, Any],
        edges: list[dict[str, Any]],
        possible_edges: list[dict[str, Any]],
        diagnostics: dict[str, Any],
        usage: dict[str, Any],
        construction_records: dict[str, dict[str, Any]],
    ) -> None:
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        json_payloads = {
            DATA_FILES["nodes"]: self.nodes,
            DATA_FILES["edges"]: edges,
            DATA_FILES["possible_equivalence_edges"]: possible_edges,
            DATA_FILES["diagnostics"]: diagnostics,
            DATA_FILES["usage"]: usage,
            MANIFEST_FILE: manifest,
        }
        array_payloads = {
            DATA_FILES["evidence_node_to_canonical"]: self.mapping,
            DATA_FILES["embedding_sums"]: self.embedding_sums,
            DATA_FILES["centroid_embeddings"]: self.centroid_embeddings,
            DATA_FILES["statement_embeddings"]: self.statement_embeddings,
            DATA_FILES["title_embeddings"]: self.title_embeddings,
            DATA_FILES["center_evidence_embeddings"]: (self.center_evidence_embeddings),
        }
        # Manifest is the commit marker and must be replaced last.
        filenames = [
            DATA_FILES["nodes"],
            DATA_FILES["edges"],
            DATA_FILES["possible_equivalence_edges"],
            DATA_FILES["evidence_node_to_canonical"],
            DATA_FILES["embedding_sums"],
            DATA_FILES["centroid_embeddings"],
            DATA_FILES["statement_embeddings"],
            DATA_FILES["title_embeddings"],
            DATA_FILES["center_evidence_embeddings"],
            DATA_FILES["diagnostics"],
            DATA_FILES["usage"],
            MANIFEST_FILE,
        ]
        transaction_id = uuid.uuid4().hex
        temporary_paths: dict[str, Path] = {}
        backup_paths: dict[str, Path] = {}
        committed: list[str] = []
        records_parent = self.graph_dir / CONSTRUCTION_RECORDS_DIR_NAME
        records_parent.mkdir(parents=True, exist_ok=True)
        records_name = f"revision_{self.target_revision:04d}"
        records_final_path = records_parent / records_name
        records_temporary_path = records_parent / (
            f".{records_name}.{transaction_id}.tmp"
        )
        records_backup_path = records_parent / (
            f".{records_name}.{transaction_id}.bak"
        )
        records_committed = False
        try:
            for filename, payload in json_payloads.items():
                path = self.graph_dir / (f".{filename}.{transaction_id}.tmp")
                self._write_json_fsynced(path, payload)
                temporary_paths[filename] = path
            for filename, matrix in array_payloads.items():
                path = self.graph_dir / (f".{filename}.{transaction_id}.tmp")
                self._write_npy_fsynced(path, matrix)
                temporary_paths[filename] = path

            records_temporary_path.mkdir()
            for relative_name, payload in construction_records["json"].items():
                self._write_json_fsynced(
                    records_temporary_path / relative_name,
                    payload,
                )
            for relative_name, payload in construction_records["text"].items():
                self._write_text_fsynced(
                    records_temporary_path / relative_name,
                    str(payload),
                )
            for directory in sorted(
                {
                    path.parent
                    for path in records_temporary_path.rglob("*")
                    if path.is_file()
                },
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                self._fsync_directory(directory)
            self._fsync_directory(records_temporary_path)

            for filename in filenames:
                final_path = self.graph_dir / filename
                if final_path.exists():
                    backup_path = self.graph_dir / (f".{filename}.{transaction_id}.bak")
                    try:
                        os.link(final_path, backup_path)
                    except OSError:
                        shutil.copy2(final_path, backup_path)
                    backup_paths[filename] = backup_path
            if records_final_path.exists():
                os.replace(records_final_path, records_backup_path)
            os.replace(records_temporary_path, records_final_path)
            records_committed = True
            self._fsync_directory(records_parent)
            for filename in filenames:
                os.replace(
                    temporary_paths[filename],
                    self.graph_dir / filename,
                )
                committed.append(filename)
            self._fsync_directory(self.graph_dir)
        except Exception:
            for filename in reversed(committed):
                final_path = self.graph_dir / filename
                backup_path = backup_paths.get(filename)
                if backup_path and backup_path.exists():
                    os.replace(backup_path, final_path)
                elif final_path.exists():
                    final_path.unlink()
            if records_committed and records_final_path.exists():
                shutil.rmtree(records_final_path)
            if records_backup_path.exists():
                os.replace(records_backup_path, records_final_path)
            self._fsync_directory(records_parent)
            self._fsync_directory(self.graph_dir)
            raise
        finally:
            for path in temporary_paths.values():
                if path.exists():
                    path.unlink()
            for path in backup_paths.values():
                if path.exists():
                    path.unlink()
            if records_temporary_path.exists():
                shutil.rmtree(records_temporary_path)
            if records_backup_path.exists():
                shutil.rmtree(records_backup_path)

    def _merged_usage(self) -> dict[str, Any]:
        current = self.usage_tracker.summary()
        previous = self.base_usage
        previous_by_model = (
            previous.get("by_model", {})
            if isinstance(previous.get("by_model"), dict)
            else {}
        )
        current_by_model = (
            current.get("by_model", {})
            if isinstance(current.get("by_model"), dict)
            else {}
        )
        by_model = deepcopy(previous_by_model)
        numeric_fields = (
            "calls",
            "successful_calls",
            "failed_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )
        for key, current_item in current_by_model.items():
            if not isinstance(current_item, dict):
                continue
            item = by_model.setdefault(
                key,
                {
                    "provider": current_item.get("provider", ""),
                    "model": current_item.get("model", ""),
                    **{field: 0 for field in numeric_fields},
                },
            )
            for field in numeric_fields:
                item[field] = int(item.get(field, 0) or 0) + int(
                    current_item.get(field, 0) or 0
                )
        total = {
            field: sum(
                int(item.get(field, 0) or 0)
                for item in by_model.values()
                if isinstance(item, dict)
            )
            for field in numeric_fields
        }
        previous_calls = (
            previous.get("calls", []) if isinstance(previous.get("calls"), list) else []
        )
        current_calls = (
            current.get("calls", []) if isinstance(current.get("calls"), list) else []
        )
        return {
            "started_at": clean_whitespace(str(previous.get("started_at", "") or ""))
            or clean_whitespace(str(current.get("started_at", "") or ""))
            or utc_now(),
            "updated_at": utc_now(),
            "graph_id": self.graph_id,
            "revision": self.target_revision,
            "total": total,
            "by_model": by_model,
            "calls": [*previous_calls, *current_calls],
        }

    @staticmethod
    def _model_identity(model: Any) -> dict[str, str]:
        return {
            "name": clean_whitespace(str(model.name)),
            "provider": clean_whitespace(str(model.provider)),
        }

    def _apply_checkpoint_content_prompt_transition(
        self,
        *,
        state: dict[str, Any],
        stored_signature: Any,
        current_signature: dict[str, Any],
    ) -> list[str]:
        """Safely migrate a checkpoint after prompt-contract changes.

        Content-only changes can preserve alignment at any phase. Alignment or
        strict-verification prompt changes are accepted only after the complete
        Evidence prefix has been committed and no active document remains. In
        that case the old prompt hashes are retained as alignment provenance.
        """
        if not isinstance(stored_signature, dict):
            return []
        if state.get("phase") not in {"alignment", "content", "finalization"}:
            return []
        migrated_signature = deepcopy(stored_signature)
        stored_prompts = migrated_signature.get("prompts")
        current_prompts = current_signature.get("prompts")
        if not isinstance(stored_prompts, dict) or not isinstance(
            current_prompts,
            dict,
        ):
            return []
        content_prompt_keys = {
            "statement",
            "statement_validation",
            "strict_evidence_pair",
        }
        alignment_prompt_keys = {"alignment", "strict_cluster_pair"}
        changed_prompt_keys = {
            key
            for key in set(stored_prompts) | set(current_prompts)
            if stored_prompts.get(key) != current_prompts.get(key)
        }
        if changed_prompt_keys - content_prompt_keys - alignment_prompt_keys:
            return []

        stored_models = migrated_signature.get("models")
        current_models = current_signature.get("models")
        stored_settings = migrated_signature.get("settings")
        current_settings = current_signature.get("settings")
        if (
            stored_models != current_models
            or not isinstance(stored_settings, dict)
            or not isinstance(current_settings, dict)
        ):
            return []
        stored_non_content_settings = {
            key: value
            for key, value in stored_settings.items()
            if key != "content_algorithm_version"
        }
        current_non_content_settings = {
            key: value
            for key, value in current_settings.items()
            if key != "content_algorithm_version"
        }
        if stored_non_content_settings != current_non_content_settings:
            return []

        alignment_prompts_changed = bool(
            changed_prompt_keys & alignment_prompt_keys
        )
        if alignment_prompts_changed:
            alignment_is_fully_committed = (
                state.get("phase") in {"content", "finalization"}
                and int(state.get("processed_evidence_node_count", -1))
                == len(self.evidence_nodes)
                and not state.get("active_document_source_id", "")
                and not state.get("active_document_node_ids", [])
                and not state.get("active_document_results", {})
            )
            if not alignment_is_fully_committed:
                return []
            try:
                stored_alignment_hash = str(
                    stored_prompts["alignment"]["sha256"]
                )
                stored_strict_hash = str(
                    stored_prompts["strict_cluster_pair"]["sha256"]
                )
            except (KeyError, TypeError):
                return []
            if not re.fullmatch(r"[0-9a-f]{64}", stored_alignment_hash) or not (
                re.fullmatch(r"[0-9a-f]{64}", stored_strict_hash)
            ):
                return []
            state.setdefault(
                "alignment_prompt_sha256_used",
                stored_alignment_hash,
            )
            state.setdefault(
                "strict_cluster_pair_prompt_sha256_used",
                stored_strict_hash,
            )

        content_changed = bool(changed_prompt_keys & content_prompt_keys) or (
            stored_settings.get("content_algorithm_version")
            != current_settings.get("content_algorithm_version")
        )

        migrated_signature["prompts"] = deepcopy(current_prompts)
        migrated_signature["settings"] = deepcopy(current_settings)
        if migrated_signature != current_signature:
            return []
        state["construction_signature"] = deepcopy(current_signature)
        messages: list[str] = []
        if alignment_prompts_changed:
            messages.append(
                "alignment and strict-verification prompt files changed after "
                "all Evidence Nodes were committed; paid alignment was "
                "preserved and its original prompt hashes were retained as "
                "provenance"
            )
        if content_changed:
            state["reprocess_all_content_after_prompt_transition"] = True
            snapshots = state.setdefault(
                "construction_prompt_snapshots",
                dict(self.prompts),
            )
            if isinstance(snapshots, dict):
                for key in changed_prompt_keys & content_prompt_keys:
                    if key in self.prompts:
                        snapshots[key] = self.prompts[key]
            messages.append(
                "Canonical title/statement postprocessing upgraded to the "
                "current content-generation contract; all Canonical content "
                "will be regenerated while paid alignment state is preserved"
            )
        return messages

    def _apply_checkpoint_model_transitions(
        self,
        *,
        state: dict[str, Any],
        stored_signature: Any,
        current_signature: dict[str, Any],
    ) -> list[str]:
        if not self.checkpoint_model_transitions:
            return []
        if not isinstance(stored_signature, dict):
            return []
        if state.get("phase") != "alignment":
            return []
        processed = int(state.get("processed_evidence_node_count", -1))
        if not 0 <= processed < len(self.evidence_nodes):
            return []

        migrated_signature = deepcopy(stored_signature)
        stored_models = migrated_signature.get("models")
        current_models = current_signature.get("models")
        if not isinstance(stored_models, dict) or not isinstance(
            current_models,
            dict,
        ):
            return []

        # Validate every requested transition before mutating checkpoint
        # state. This keeps a misspelled or stale migration declaration from
        # partially rewriting an otherwise recoverable snapshot.
        for task, transition in self.checkpoint_model_transitions.items():
            if stored_models.get(task) != transition["from"]:
                return []
            if current_models.get(task) != transition["to"]:
                return []

        messages: list[str] = []
        active_results = state.get("active_document_results", {})
        proposal_ids: list[int] = []
        if isinstance(active_results, dict) and any(
            task in self.checkpoint_model_transitions
            for task in ("alignment", "strict_verification")
        ):
            proposal_ids = sorted(
                int(evidence_node_id)
                for evidence_node_id, result in active_results.items()
                if isinstance(result, dict) and result.get("stage") == "proposed"
            )
            if proposal_ids:
                # A completed proposal embeds decisions from both alignment
                # and (when multiple strict positives occur) strict-cluster
                # verification. Preserve provider-independent candidate lists,
                # but rerun old-provider proposals so one active document never
                # mixes two LLM identities.
                state["active_document_results"] = {
                    evidence_node_id: deepcopy(result)
                    for evidence_node_id, result in active_results.items()
                    if not (
                        isinstance(result, dict)
                        and result.get("stage") == "proposed"
                    )
                }
                messages.append(
                    "active document: discarded "
                    f"{len(proposal_ids)} old-provider alignment proposals "
                    "and preserved all candidate-only stage results; rerun "
                    f"Evidence Nodes {proposal_ids[0]}-{proposal_ids[-1]}"
                )

        histories = {
            "alignment": "alignment_model_history",
            "strict_verification": ("strict_verification_model_history"),
        }
        for task, transition in self.checkpoint_model_transitions.items():
            old_identity = transition["from"]
            new_identity = transition["to"]
            history_key = histories.get(task)
            if history_key is not None:
                history = self._checkpoint_model_history(
                    state=state,
                    key=history_key,
                    fallback_identity=old_identity,
                )
                history = self._transition_model_history(
                    history=history,
                    old_identity=old_identity,
                    new_identity=new_identity,
                    processed_count=processed,
                )
                state[history_key] = history
            stored_models[task] = deepcopy(new_identity)
            if history_key is None:
                messages.append(
                    f"{task}: {old_identity['name']} via "
                    f"{old_identity['provider']} -> {new_identity['name']} via "
                    f"{new_identity['provider']}; new provider will generate "
                    "Canonical content after alignment"
                )
            else:
                messages.append(
                    f"{task}: {old_identity['name']} via "
                    f"{old_identity['provider']} -> {new_identity['name']} via "
                    f"{new_identity['provider']}; new model starts at Evidence "
                    f"Node {processed + 1}"
                )

        if migrated_signature != current_signature:
            return []
        state["construction_signature"] = deepcopy(current_signature)
        return messages

    def _checkpoint_model_history(
        self,
        *,
        state: dict[str, Any],
        key: str,
        fallback_identity: dict[str, Any],
    ) -> list[dict[str, Any]]:
        processed = int(state.get("processed_evidence_node_count", 0) or 0)
        raw_history = state.get(key)
        if raw_history is None:
            if processed == 0:
                return []
            return [
                {
                    "model": clean_whitespace(
                        str(fallback_identity.get("name", "") or "")
                    ),
                    "provider": clean_whitespace(
                        str(fallback_identity.get("provider", "") or "")
                    ),
                    "evidence_node_id_start": 1,
                    "evidence_node_id_end": None,
                }
            ]
        return self._validated_model_history(
            raw_history,
            processed_count=processed,
            allow_open_end=True,
        )

    def _history_for_new_update(
        self,
        *,
        stored_history: Any,
        stored_identity: dict[str, str],
        current_identity: dict[str, str],
        processed_count: int,
        total_count: int,
    ) -> list[dict[str, Any]]:
        if processed_count == 0:
            return [
                {
                    "model": current_identity["name"],
                    "provider": current_identity["provider"],
                    "evidence_node_id_start": 1,
                    "evidence_node_id_end": None,
                }
            ]

        if stored_history is None:
            if not stored_identity["name"] or not stored_identity["provider"]:
                raise ValueError(
                    "Existing Canonical Graph does not identify its construction model."
                )
            history = [
                {
                    "model": stored_identity["name"],
                    "provider": stored_identity["provider"],
                    "evidence_node_id_start": 1,
                    "evidence_node_id_end": processed_count,
                }
            ]
        else:
            # Older postprocessing-only revisions could materialize an empty
            # model-history tail such as start=450, end=449. It represents no
            # calls and can be removed without losing provenance.
            if isinstance(stored_history, list):
                stored_history = [
                    item
                    for item in stored_history
                    if not (
                        isinstance(item, dict)
                        and item.get("evidence_node_id_start") == processed_count + 1
                        and item.get("evidence_node_id_end") == processed_count
                    )
                ]
            history = self._validated_model_history(
                stored_history,
                processed_count=processed_count,
                allow_open_end=False,
            )

        if processed_count >= total_count:
            return history

        last = history[-1]
        if (
            last["model"] == current_identity["name"]
            and last["provider"] == current_identity["provider"]
        ):
            last["evidence_node_id_end"] = None
        else:
            history.append(
                {
                    "model": current_identity["name"],
                    "provider": current_identity["provider"],
                    "evidence_node_id_start": processed_count + 1,
                    "evidence_node_id_end": None,
                }
            )
        return history

    def _transition_model_history(
        self,
        *,
        history: list[dict[str, Any]],
        old_identity: dict[str, str],
        new_identity: dict[str, str],
        processed_count: int,
    ) -> list[dict[str, Any]]:
        if processed_count == 0:
            return [
                {
                    "model": new_identity["name"],
                    "provider": new_identity["provider"],
                    "evidence_node_id_start": 1,
                    "evidence_node_id_end": None,
                }
            ]
        if not history:
            raise ValueError("Checkpoint model history is empty.")
        last = history[-1]
        if (
            last["model"] != old_identity["name"]
            or last["provider"] != old_identity["provider"]
        ):
            raise ValueError(
                "Checkpoint model history does not match its stored model."
            )
        last_end = last["evidence_node_id_end"]
        if last_end is None:
            last["evidence_node_id_end"] = processed_count
        elif last_end != processed_count:
            raise ValueError(
                "Checkpoint model history does not end at the processed Evidence Node."
            )
        history.append(
            {
                "model": new_identity["name"],
                "provider": new_identity["provider"],
                "evidence_node_id_start": processed_count + 1,
                "evidence_node_id_end": None,
            }
        )
        return history

    @staticmethod
    def _validated_model_history(
        raw_history: Any,
        *,
        processed_count: int,
        allow_open_end: bool,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_history, list) or not raw_history:
            raise ValueError("Canonical model history must be a non-empty list.")
        history = deepcopy(raw_history)
        expected_start = 1
        for index, item in enumerate(history):
            if not isinstance(item, dict) or set(item) != {
                "model",
                "provider",
                "evidence_node_id_start",
                "evidence_node_id_end",
            }:
                raise ValueError("Canonical model history entry is invalid.")
            model = clean_whitespace(str(item["model"] or ""))
            provider = clean_whitespace(str(item["provider"] or ""))
            start = item["evidence_node_id_start"]
            end = item["evidence_node_id_end"]
            if (
                not model
                or not provider
                or isinstance(start, bool)
                or not isinstance(start, int)
                or start != expected_start
            ):
                raise ValueError("Canonical model history entry is invalid.")
            if end is None:
                if not allow_open_end or index != len(history) - 1:
                    raise ValueError(
                        "Only the active checkpoint model history entry may "
                        "have an open end."
                    )
                expected_start = processed_count + 1
            elif (
                isinstance(end, bool)
                or not isinstance(end, int)
                or end < start
                or end > processed_count
            ):
                raise ValueError("Canonical model history range is invalid.")
            else:
                expected_start = end + 1
            item["model"] = model
            item["provider"] = provider
        if expected_start != processed_count + 1:
            raise ValueError(
                "Canonical model history does not cover the processed "
                "Evidence Node prefix."
            )
        return history

    @staticmethod
    def _closed_model_history(
        history: list[dict[str, Any]],
        *,
        end_node_id: int,
    ) -> list[dict[str, Any]]:
        result = deepcopy(history)
        if not result:
            return result
        if result[-1]["evidence_node_id_end"] is None:
            result[-1]["evidence_node_id_end"] = end_node_id
        return result

    def _construction_signature(self) -> dict[str, Any]:
        return {
            "models": {
                "alignment": {
                    "name": self.alignment_model.name,
                    "provider": self.alignment_model.provider,
                },
                "strict_verification": {
                    "name": self.strict_model.name,
                    "provider": self.strict_model.provider,
                },
                "canonical_statement": {
                    "name": self.statement_model.name,
                    "provider": self.statement_model.provider,
                },
                "embedding": {
                    "name": self.embedding_model.name,
                    "provider": self.embedding_model.provider,
                    "dimensions": self.embedding_model.dimensions,
                },
                "reranker": {
                    "name": self.reranker_model.name,
                    "provider": self.reranker_model.provider,
                },
            },
            "prompts": {
                key: {
                    "file": path.name,
                    "sha256": self._prompt_sha256(key),
                }
                for key, path in self.prompt_paths.items()
            },
            "settings": {
                "alignment_algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
                "content_algorithm_version": CONTENT_ALGORITHM_VERSION,
                "evidence_retrieval_top_k": (self.evidence_retrieval_top_k),
                "evidence_similarity_threshold": (
                    self.evidence_similarity_threshold
                ),
                "reranker_top_k": self.reranker_top_k,
                "low_cohesion_similarity": self.low_cohesion_similarity,
                "representative_min_centroid_similarity": (
                    self.representative_min_similarity
                ),
            },
        }

    def _evidence_fingerprint(self) -> dict[str, Any]:
        if self._evidence_fingerprint_cache is not None:
            return deepcopy(self._evidence_fingerprint_cache)
        files = (
            MANIFEST_FILE,
            EVIDENCE_DATA_FILES["nodes"],
            EVIDENCE_DATA_FILES["edges"],
            EVIDENCE_DATA_FILES["embeddings"],
        )
        fingerprint = {
            "graph_id": self.evidence_graph_id,
            "revision": self.evidence_manifest["revision"],
            "files": {
                filename: _sha256_file(self.evidence_graph_dir / filename)
                for filename in files
            },
        }
        self._evidence_fingerprint_cache = fingerprint
        return deepcopy(fingerprint)

    def _evidence_prefix_metadata(
        self,
        *,
        node_count: int,
        edge_count: int,
    ) -> dict[str, Any]:
        node_payload = self.evidence_nodes[:node_count]
        edge_payload = self.evidence_edges[:edge_count]
        node_digest = _sha256_json(node_payload)
        edge_digest = _sha256_json(edge_payload)
        embedding_digest = hashlib.sha256()
        if node_count:
            contiguous = np.ascontiguousarray(
                self.evidence_embeddings[:node_count],
                dtype=np.float32,
            )
            embedding_digest.update(contiguous.tobytes(order="C"))
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "nodes_sha256": node_digest,
            "edges_sha256": edge_digest,
            "embeddings_sha256": embedding_digest.hexdigest(),
        }

    def _validate_evidence_prefix(
        self,
        *,
        manifest: dict[str, Any],
        node_count: int,
        edge_count: int,
    ) -> None:
        expected = manifest.get("evidence_prefix")
        if not isinstance(expected, dict):
            raise ValueError(
                "Existing Canonical Graph does not contain Evidence prefix "
                "fingerprints required for safe incremental update."
            )
        actual = self._evidence_prefix_metadata(
            node_count=node_count,
            edge_count=edge_count,
        )
        if expected != actual:
            raise ValueError(
                "The processed Evidence Graph prefix was modified. Canonical "
                "incremental update requires an append-only Evidence Graph."
            )

    def _prompt_sha256(self, key: str) -> str:
        return hashlib.sha256(self.prompts[key].encode("utf-8")).hexdigest()

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_json_fsynced(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _write_text_fsynced(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _write_npy_fsynced(path: Path, matrix: np.ndarray) -> None:
        with path.open("xb") as file:
            np.save(file, matrix, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return (array / norm).astype(np.float32, copy=False)


def _append_matrix_row(
    matrix: np.ndarray,
    row: np.ndarray,
) -> np.ndarray:
    row_array = np.asarray(row, dtype=np.float32).reshape(
        1,
        EMBEDDING_DIMENSIONS,
    )
    if matrix.shape[0] == 0:
        return row_array.copy()
    return np.vstack((matrix, row_array)).astype(np.float32, copy=False)


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty(
            (0, EMBEDDING_DIMENSIONS),
            dtype=np.float32,
        )
    return np.vstack(rows).astype(np.float32, copy=False)


def _normalize_checkpoint_model_transitions(
    transitions: dict[str, dict[str, dict[str, str]]],
) -> dict[str, dict[str, dict[str, str]]]:
    allowed_tasks = {
        "alignment",
        "strict_verification",
        "canonical_statement",
    }
    if not isinstance(transitions, dict):
        raise ValueError("checkpoint_model_transitions must be a dictionary.")
    normalized: dict[str, dict[str, dict[str, str]]] = {}
    for task, transition in transitions.items():
        if task not in allowed_tasks:
            raise ValueError(f"Unsupported checkpoint model transition task: {task}")
        if not isinstance(transition, dict) or set(transition) != {
            "from",
            "to",
        }:
            raise ValueError(
                f"Checkpoint model transition {task!r} must contain "
                "exactly 'from' and 'to'."
            )
        normalized_transition: dict[str, dict[str, str]] = {}
        for endpoint in ("from", "to"):
            identity = transition[endpoint]
            if not isinstance(identity, dict) or set(identity) != {
                "name",
                "provider",
            }:
                raise ValueError(
                    f"Checkpoint model transition {task!r}.{endpoint} "
                    "must contain exactly 'name' and 'provider'."
                )
            name = clean_whitespace(str(identity["name"] or ""))
            provider = clean_whitespace(str(identity["provider"] or ""))
            if not name or not provider:
                raise ValueError(
                    f"Checkpoint model transition {task!r}.{endpoint} cannot be empty."
                )
            normalized_transition[endpoint] = {
                "name": name,
                "provider": provider,
            }
        if normalized_transition["from"] == normalized_transition["to"]:
            raise ValueError(
                f"Checkpoint model transition {task!r} does not change model."
            )
        normalized[task] = normalized_transition
    return normalized


def _connected_components(
    graph: dict[int, set[int]],
) -> list[set[int]]:
    unseen = set(graph)
    components: list[set[int]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            stack.extend(graph[node] - component)
        components.append(component)
    return components


def _validate_single_relation(
    result: dict[str, Any],
) -> tuple[bool, str]:
    if (
        not isinstance(result, dict)
        or set(result) != {"relation"}
        or result.get("relation") not in RELATIONS
    ):
        return (
            False,
            "Strict verification output must contain exactly one valid relation.",
        )
    return True, ""


def _validate_content_result(
    result: dict[str, Any],
) -> tuple[bool, str]:
    if (
        not isinstance(result, dict)
        or set(result) != {"title", "statement"}
        or not isinstance(result.get("title"), str)
        or not clean_model_text(result["title"])
        or len(clean_model_text(result["title"])) > 240
        or not isinstance(result.get("statement"), str)
        or not clean_model_text(result["statement"])
    ):
        return (
            False,
            "Canonical content output must contain exactly one non-empty title "
            "of at most 240 characters and one non-empty statement.",
        )
    return True, ""


def _validate_content_validation(
    result: dict[str, Any],
) -> tuple[bool, str]:
    if (
        not isinstance(result, dict)
        or set(result) != {"valid", "feedback"}
        or not isinstance(result.get("valid"), bool)
        or not isinstance(result.get("feedback"), str)
    ):
        return (
            False,
            "Canonical content validation must contain exactly one boolean "
            "valid field and one string feedback field.",
        )
    feedback = clean_whitespace(result["feedback"])
    if result["valid"] is True and feedback:
        return False, "Valid Canonical content must have empty feedback."
    if result["valid"] is False and not feedback:
        return False, "Invalid Canonical content must include concrete feedback."
    return True, ""


def _validated_content_validation_failures(
    value: Any,
    *,
    canonical_node_count: int,
) -> dict[int, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Content-validation diagnostics must be a list.")
    failures: dict[int, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "canonical_node_id",
            "attempts",
            "final_feedback",
        }:
            raise ValueError("Content-validation diagnostic entry is invalid.")
        canonical_node_id = item.get("canonical_node_id")
        attempts = item.get("attempts")
        feedback = item.get("final_feedback")
        if (
            isinstance(canonical_node_id, bool)
            or not isinstance(canonical_node_id, int)
            or not 1 <= canonical_node_id <= canonical_node_count
            or canonical_node_id in failures
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts != CANONICAL_CONTENT_MAX_ATTEMPTS
            or not isinstance(feedback, str)
            or not clean_whitespace(feedback)
        ):
            raise ValueError("Content-validation diagnostic entry is invalid.")
        failures[canonical_node_id] = deepcopy(item)
    return failures


def _content_validation_failures_from_diagnostics(
    diagnostics: Any,
    *,
    canonical_node_count: int,
) -> dict[int, dict[str, Any]]:
    if not isinstance(diagnostics, dict):
        return {}
    section = diagnostics.get("canonical_content_validation")
    if section is None:
        return {}
    if (
        not isinstance(section, dict)
        or section.get("max_attempts") != CANONICAL_CONTENT_MAX_ATTEMPTS
    ):
        raise ValueError("Canonical content-validation diagnostics are invalid.")
    return _validated_content_validation_failures(
        section.get("accepted_after_failed_validation"),
        canonical_node_count=canonical_node_count,
    )


def _valid_canonical_title(value: str) -> bool:
    title = clean_model_text(str(value or ""))
    if not title or len(title) > 240:
        return False
    normalized = clean_whitespace(title).strip(" .:;,-_").lower()
    generic = re.fullmatch(
        r"(?:main\s+)?(?:theorem|lemma|proposition|corollary|definition|"
        r"claim|property|result)(?:\s+(?:\d+(?:\.\d+)*|[ivxlcdm]+))?",
        normalized,
    )
    return generic is None


def _fallback_canonical_title(
    *,
    node_type: str,
    center_title: str,
    center_statement: str,
) -> str:
    """Build a meaningful last-resort title when model repair is exhausted."""
    title = clean_model_text(center_title)
    if _valid_canonical_title(title):
        return title
    normalized = clean_whitespace(title).strip(" .:;,-_").lower()
    if node_type == "definition" and normalized:
        if normalized.startswith(("a ", "an ", "the ")):
            term = normalized
        else:
            article = "an" if normalized[0] in "aeiou" else "a"
            term = f"{article} {normalized}"
        fallback = f"Definition of {term}"
    else:
        statement = clean_whitespace(clean_model_text(center_statement))
        statement_excerpt = statement[:180].rstrip(" .:;,-_")
        label = clean_whitespace(node_type).replace("_", " ").title()
        fallback = f"{label}: {statement_excerpt}"
    if not _valid_canonical_title(fallback):
        raise ValueError("Unable to construct a meaningful Canonical title.")
    return fallback


def _is_normalized_vector(value: Any) -> bool:
    if not isinstance(value, np.ndarray) or value.shape != (EMBEDDING_DIMENSIONS,):
        return False
    if value.dtype != np.float32 or not np.all(np.isfinite(value)):
        return False
    return bool(np.isclose(np.linalg.norm(value), 1.0, rtol=1e-4, atol=1e-4))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
