from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import uuid
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
    utc_now,
)
from src.evidence_graph.derivation import (
    EMBEDDING_DIMENSIONS,
    EvidenceEmbeddingBuilder,
    EvidenceNodeNormalizer,
)
from src.evidence_graph.schema import (
    DATA_FILES,
    SCHEMA_VERSION,
    validate_evidence_graph,
)
from src.single_doc.schema import validate_single_document_graph


MANIFEST_FILE = "manifest.json"
CHECKPOINT_DIR_NAME = ".checkpoint"
CHECKPOINT_POINTER_FILE = "LATEST"
CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_INTERVAL = 100
ALL_GRAPH_FILES = (
    DATA_FILES["sources"],
    DATA_FILES["nodes"],
    DATA_FILES["edges"],
    DATA_FILES["embeddings"],
    DATA_FILES["usage"],
    MANIFEST_FILE,
)


class EvidenceGraphPipeline:
    """Create or incrementally update one JSON-backed Evidence Graph."""

    def __init__(
        self,
        *,
        evidence_graph_dir: str | Path,
        single_document_graph_paths: list[str | Path],
        logger: ProjectLogger | None = None,
        config: GangmuConfig | None = None,
        usage_tracker: TokenUsageTracker | None = None,
        normalizer: Any | None = None,
        embedding_builder: Any | None = None,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    ) -> None:
        self.graph_dir = Path(evidence_graph_dir).expanduser().resolve()
        self.graph_id = self.graph_dir.name
        if not self.graph_id:
            raise ValueError("Evidence Graph directory must have a name.")

        self.input_paths = [
            Path(path).expanduser().resolve() for path in single_document_graph_paths
        ]
        if len(self.input_paths) != len(set(self.input_paths)):
            raise ValueError("The single-document graph path list contains duplicates.")
        if (
            isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, int)
            or checkpoint_interval < 1
        ):
            raise ValueError("checkpoint_interval must be a positive integer.")
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = self.graph_dir / CHECKPOINT_DIR_NAME

        self.config = config or get_config()
        self.logger = logger or ProjectLogger(
            self.config.log_dir("evidence_graph"),
            run_name=self.graph_id,
        )
        self.usage_tracker = usage_tracker or TokenUsageTracker()
        self.normalizer = normalizer or EvidenceNodeNormalizer(
            config=self.config,
            logger=self.logger,
            usage_tracker=self.usage_tracker,
        )
        self.embedding_builder = embedding_builder or EvidenceEmbeddingBuilder(
            config=self.config,
            logger=self.logger,
            usage_tracker=self.usage_tracker,
        )
        safe_lock_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.graph_id)
        self.lock_path = self.logger.log_dir / f"{safe_lock_name}.lock"

    def run(self) -> dict[str, Any]:
        input_graphs = [
            self._load_single_document_graph(path) for path in self.input_paths
        ]
        self._validate_batch_source_ids(input_graphs)
        input_fingerprints = self._input_fingerprints()

        with self._exclusive_lock():
            (
                manifest,
                sources,
                nodes,
                edges,
                embeddings,
                previous_usage,
                base_existed,
            ) = self._load_existing()
            if not base_existed and not input_graphs:
                raise ValueError(
                    "At least one single-document graph is required when "
                    "creating an Evidence Graph."
                )
            base_revision = int(manifest.get("revision", 0) or 0)
            checkpoint = self._load_checkpoint(
                input_fingerprints=input_fingerprints,
                base_revision=base_revision,
                base_existed=base_existed,
            )
            if checkpoint is None:
                existing_source_ids = {
                    clean_whitespace(str(source.get("id", "")))
                    for source in sources
                    if isinstance(source, dict)
                }
                duplicate_source_ids = sorted(
                    {
                        clean_whitespace(str(graph["source"]["id"]))
                        for graph in input_graphs
                    }
                    & existing_source_ids
                )
                if duplicate_source_ids:
                    raise ValueError(
                        "Sources already exist in the Evidence Graph: "
                        + ", ".join(duplicate_source_ids)
                    )
                import_results: list[dict[str, Any]] = []
                for graph in input_graphs:
                    import_results.append(
                        self._merge_single_document(
                            graph=graph,
                            sources=sources,
                            nodes=nodes,
                            edges=edges,
                        )
                    )
                revision = base_revision + 1
                checkpoint_base_usage = previous_usage
                normalized_node_ids: set[int] = set()
                embedded_node_ids: set[int] = set()
                embedding_pending_node_ids = set(
                    range(int(embeddings.shape[0]) + 1, len(nodes) + 1)
                )
                checkpoint_state: dict[str, Any] = {
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "graph_id": self.graph_id,
                    "base_revision": base_revision,
                    "target_revision": revision,
                    "base_existed": base_existed,
                    "input_fingerprints": input_fingerprints,
                    "imported_documents": import_results,
                    "normalization_signature": self._normalization_signature(),
                    "embedding_signature": self._embedding_signature(),
                    "normalized_node_ids": [],
                    "embedded_node_ids": [],
                    "embedding_pending_node_ids": sorted(embedding_pending_node_ids),
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                }
            else:
                checkpoint_state = checkpoint["state"]
                sources = checkpoint["sources"]
                nodes = checkpoint["nodes"]
                edges = checkpoint["edges"]
                embeddings = checkpoint["embeddings"]
                checkpoint_base_usage = checkpoint["usage"]
                import_results = list(checkpoint_state.get("imported_documents", []))
                revision = int(checkpoint_state["target_revision"])
                normalized_node_ids = {
                    int(node_id)
                    for node_id in checkpoint_state.get(
                        "normalized_node_ids",
                        [],
                    )
                }
                embedded_node_ids = {
                    int(node_id)
                    for node_id in checkpoint_state.get(
                        "embedded_node_ids",
                        [],
                    )
                }
                embedding_pending_node_ids = {
                    int(node_id)
                    for node_id in checkpoint_state.get(
                        "embedding_pending_node_ids",
                        [],
                    )
                }
                self.logger.info(
                    "Resuming Evidence Graph from checkpoint",
                    graph_id=self.graph_id,
                    revision=revision,
                    normalized_nodes=len(normalized_node_ids),
                    embedded_nodes=len(embedded_node_ids),
                    embedding_pending_nodes=len(embedding_pending_node_ids),
                )

            def checkpoint_usage() -> dict[str, Any]:
                return self._merge_usage(
                    previous=checkpoint_base_usage,
                    current=self.usage_tracker.summary(),
                    revision=revision,
                )

            def save_checkpoint(*, phase: str) -> None:
                checkpoint_state["normalized_node_ids"] = sorted(normalized_node_ids)
                checkpoint_state["embedded_node_ids"] = sorted(embedded_node_ids)
                checkpoint_state["embedding_pending_node_ids"] = sorted(
                    embedding_pending_node_ids
                )
                checkpoint_state["phase"] = phase
                checkpoint_state["updated_at"] = utc_now()
                self._save_checkpoint(
                    state=checkpoint_state,
                    sources=sources,
                    nodes=nodes,
                    edges=edges,
                    embeddings=embeddings,
                    usage=checkpoint_usage(),
                )
                self.logger.info(
                    "Evidence Graph checkpoint saved",
                    graph_id=self.graph_id,
                    revision=revision,
                    phase=phase,
                    normalized_nodes=len(normalized_node_ids),
                    embedded_nodes=len(embedded_node_ids),
                    embedding_pending_nodes=len(embedding_pending_node_ids),
                )

            # Save the merged inputs before the first paid model call.
            if checkpoint is None:
                save_checkpoint(phase="normalization")

            def normalization_batch_completed(
                completed_node_ids: list[int],
            ) -> None:
                normalized_node_ids.update(completed_node_ids)
                embedding_pending_node_ids.update(completed_node_ids)
                save_checkpoint(phase="normalization")

            current_normalized_node_ids = self.normalizer.normalize_pending(
                nodes=nodes,
                edges=edges,
                batch_size=self.checkpoint_interval,
                on_batch_completed=normalization_batch_completed,
            )
            normalized_node_ids.update(current_normalized_node_ids)
            embedding_pending_node_ids.update(current_normalized_node_ids)

            def embedding_batch_completed(
                updated_embeddings: np.ndarray,
                completed_node_ids: list[int],
                remaining_node_ids: list[int],
            ) -> None:
                nonlocal embeddings
                embeddings = updated_embeddings
                embedded_node_ids.update(completed_node_ids)
                embedding_pending_node_ids.clear()
                embedding_pending_node_ids.update(remaining_node_ids)
                save_checkpoint(phase="embedding")

            embeddings, current_embedded_node_ids = (
                self.embedding_builder.update_matrix(
                    nodes=nodes,
                    existing_matrix=embeddings,
                    recompute_node_ids=sorted(embedding_pending_node_ids),
                    pending_node_ids=sorted(embedding_pending_node_ids),
                    batch_size=self.checkpoint_interval,
                    on_batch_completed=embedding_batch_completed,
                )
            )
            embedded_node_ids.update(current_embedded_node_ids)
            embedding_pending_node_ids.difference_update(current_embedded_node_ids)
            if embedding_pending_node_ids:
                raise RuntimeError(
                    "Embedding completed with unresolved checkpoint nodes: "
                    + ", ".join(
                        str(node_id) for node_id in sorted(embedding_pending_node_ids)
                    )
                )

            now = utc_now()
            normalization_metadata = dict(self.normalizer.metadata)
            normalization_metadata["node_count"] = len(nodes)
            embedding_metadata = dict(self.embedding_builder.metadata)
            embedding_metadata["node_count"] = len(nodes)
            new_manifest: dict[str, Any] = {
                "graph_type": "evidence",
                "schema_version": SCHEMA_VERSION,
                "graph_id": self.graph_id,
                "name": clean_whitespace(
                    str(manifest.get("name", "") or self.graph_id)
                ),
                "revision": revision,
                "created_at": clean_whitespace(
                    str(manifest.get("created_at", "") or now)
                ),
                "updated_at": now,
                "counts": {
                    "sources": len(sources),
                    "nodes": len(nodes),
                    "edges": len(edges),
                },
                "files": dict(DATA_FILES),
                "normalization": normalization_metadata,
                "embedding": embedding_metadata,
            }
            description = clean_whitespace(str(manifest.get("description", "") or ""))
            if description:
                new_manifest["description"] = description

            usage = self._merge_usage(
                previous=checkpoint_base_usage,
                current=self.usage_tracker.summary(),
                revision=revision,
            )
            errors = validate_evidence_graph(
                manifest=new_manifest,
                sources=sources,
                nodes=nodes,
                edges=edges,
                embeddings=embeddings,
                usage=usage,
                expected_graph_id=self.graph_id,
            )
            if errors:
                raise ValueError(
                    "Merged Evidence Graph is invalid: " + "; ".join(errors)
                )

            self._save_transactionally(
                manifest=new_manifest,
                sources=sources,
                nodes=nodes,
                edges=edges,
                embeddings=embeddings,
                usage=usage,
            )
            self._delete_checkpoint()

        summary = {
            "graph_id": self.graph_id,
            "graph_dir": str(self.graph_dir),
            "created": not bool(checkpoint_state["base_existed"]),
            "revision": revision,
            "imported_documents": import_results,
            "counts": dict(new_manifest["counts"]),
            "normalized_nodes": len(normalized_node_ids),
            "embedded_nodes": len(embedded_node_ids),
            "embedding_shape": list(embeddings.shape),
        }
        self.logger.info("Evidence Graph update completed", **summary)
        return summary

    def _load_single_document_graph(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Single-document graph does not exist: {path}")
        graph = read_json(path)
        if not isinstance(graph, dict):
            raise ValueError(f"Single-document graph must be an object: {path}")
        if graph.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported single-document schema_version in {path}: "
                f"{graph.get('schema_version')!r}"
            )
        errors = validate_single_document_graph(graph)
        if errors:
            raise ValueError(
                f"Invalid single-document graph {path}: " + "; ".join(errors)
            )
        source = graph.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"Single-document graph has no source object: {path}")
        if source.get("type") not in {"paper", "textbook"}:
            raise ValueError(f"Invalid source type in {path}")
        if not clean_whitespace(str(source.get("title", ""))):
            raise ValueError(f"Source title is empty in {path}")
        for edge in graph.get("edges", []):
            if not clean_whitespace(str(edge.get("semantics", ""))):
                raise ValueError(f"Edge semantics is empty in {path}")
        return graph

    @staticmethod
    def _validate_batch_source_ids(graphs: list[dict[str, Any]]) -> None:
        source_ids = [
            clean_whitespace(str(graph["source"].get("id", ""))) for graph in graphs
        ]
        if any(not source_id for source_id in source_ids):
            raise ValueError("Every single-document graph needs a non-empty source.id.")
        duplicates = sorted(
            source_id
            for source_id in set(source_ids)
            if source_ids.count(source_id) > 1
        )
        if duplicates:
            raise ValueError(
                "The import batch contains duplicate source IDs: "
                + ", ".join(duplicates)
            )

    def _input_fingerprints(self) -> list[dict[str, str]]:
        fingerprints: list[dict[str, str]] = []
        for path in self.input_paths:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            fingerprints.append(
                {
                    "path": str(path),
                    "sha256": digest.hexdigest(),
                }
            )
        return fingerprints

    def _normalization_signature(self) -> dict[str, Any]:
        metadata = dict(self.normalizer.metadata)
        configured_name = self.config.get_str("models.normalization.name", "")
        default_alias = self.config.get_str("models.default.alias", "")
        model_identity = (
            f"name:{configured_name}"
            if configured_name
            else f"alias:{default_alias}"
            if default_alias
            else f"name:{clean_whitespace(str(metadata.get('model', '')))}"
        )
        return {
            # Provider-specific API model names may differ even though they map
            # to the same configured conceptual model alias.
            "model_identity": model_identity,
            "prompt_file": clean_whitespace(str(metadata.get("prompt_file", ""))),
            "prompt_sha256": clean_whitespace(str(metadata.get("prompt_sha256", ""))),
        }

    def _embedding_signature(self) -> dict[str, Any]:
        metadata = dict(self.embedding_builder.metadata)
        return {
            "model": clean_whitespace(str(metadata.get("model", ""))),
            "dimensions": int(metadata.get("dimensions", 0) or 0),
            "dtype": clean_whitespace(str(metadata.get("dtype", ""))),
        }

    def _load_checkpoint(
        self,
        *,
        input_fingerprints: list[dict[str, str]],
        base_revision: int,
        base_existed: bool,
    ) -> dict[str, Any] | None:
        if not self.checkpoint_dir.exists():
            return None
        if not self.checkpoint_dir.is_dir():
            raise NotADirectoryError(
                f"Evidence Graph checkpoint is not a directory: {self.checkpoint_dir}"
            )

        pointer_path = self.checkpoint_dir / CHECKPOINT_POINTER_FILE
        if not pointer_path.is_file():
            # No committed pointer means a process stopped before its first
            # checkpoint became durable. Such temporary data is not reusable.
            self._delete_checkpoint()
            return None
        snapshot_name = pointer_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"snapshot-[0-9a-f]{32}", snapshot_name):
            raise ValueError(
                f"Evidence Graph checkpoint pointer is invalid: {pointer_path}"
            )
        snapshot_dir = (self.checkpoint_dir / snapshot_name).resolve()
        if snapshot_dir.parent != self.checkpoint_dir.resolve():
            raise ValueError("Unsafe Evidence Graph checkpoint snapshot path.")
        if not snapshot_dir.is_dir():
            raise ValueError(
                f"Evidence Graph checkpoint snapshot is missing: {snapshot_dir}"
            )

        required_files = {
            "state.json",
            DATA_FILES["sources"],
            DATA_FILES["nodes"],
            DATA_FILES["edges"],
            DATA_FILES["embeddings"],
            DATA_FILES["usage"],
        }
        existing_files = {
            path.name for path in snapshot_dir.iterdir() if path.is_file()
        }
        if existing_files != required_files:
            missing = sorted(required_files - existing_files)
            unexpected = sorted(existing_files - required_files)
            raise ValueError(
                "Evidence Graph checkpoint files are invalid: "
                f"missing={missing}, unexpected={unexpected}"
            )

        state = read_json(snapshot_dir / "state.json")
        if not isinstance(state, dict):
            raise ValueError("Evidence Graph checkpoint state must be an object.")
        if state.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported Evidence Graph checkpoint version: "
                f"{state.get('checkpoint_version')!r}"
            )
        if state.get("graph_id") != self.graph_id:
            raise ValueError(
                "Evidence Graph checkpoint graph_id does not match its directory."
            )

        target_revision = int(state.get("target_revision", 0) or 0)
        if target_revision <= base_revision:
            # The formal revision was committed before a process stopped while
            # deleting its recovery data.
            self._delete_checkpoint()
            return None
        if int(state.get("base_revision", -1)) != base_revision:
            raise ValueError(
                "Evidence Graph changed after this checkpoint was created. "
                "Restore the matching formal revision or remove the checkpoint "
                "only if its paid intermediate results may be discarded."
            )
        if bool(state.get("base_existed")) != base_existed:
            raise ValueError(
                "Evidence Graph checkpoint base state does not match the formal graph."
            )
        if state.get("input_fingerprints") != input_fingerprints:
            raise ValueError(
                "The requested single-document graph batch differs from the "
                "saved checkpoint. Run again with the same paths and unchanged "
                "input files, or explicitly discard the checkpoint."
            )
        if state.get("normalization_signature") != self._normalization_signature():
            raise ValueError(
                "The normalization model or prompt changed after the checkpoint "
                "was created. Restore the matching configuration before resuming."
            )
        if state.get("embedding_signature") != self._embedding_signature():
            raise ValueError(
                "The Embedding model or vector format changed after the "
                "checkpoint was created. Restore the matching configuration "
                "before resuming."
            )

        checkpoint = {
            "state": state,
            "sources": read_json(snapshot_dir / DATA_FILES["sources"]),
            "nodes": read_json(snapshot_dir / DATA_FILES["nodes"]),
            "edges": read_json(snapshot_dir / DATA_FILES["edges"]),
            "embeddings": np.load(
                snapshot_dir / DATA_FILES["embeddings"],
                allow_pickle=False,
            ),
            "usage": read_json(snapshot_dir / DATA_FILES["usage"]),
        }
        self._validate_checkpoint(checkpoint)
        return checkpoint

    def _validate_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint["state"]
        sources = checkpoint["sources"]
        nodes = checkpoint["nodes"]
        edges = checkpoint["edges"]
        embeddings = checkpoint["embeddings"]
        usage = checkpoint["usage"]
        if not isinstance(sources, list):
            raise ValueError("Checkpoint sources must be a list.")
        if not isinstance(nodes, list):
            raise ValueError("Checkpoint nodes must be a list.")
        if not isinstance(edges, list):
            raise ValueError("Checkpoint edges must be a list.")
        if not isinstance(usage, dict):
            raise ValueError("Checkpoint usage must be an object.")

        source_ids = {
            clean_whitespace(str(source.get("id", "")))
            for source in sources
            if isinstance(source, dict)
        }
        if len(source_ids) != len(sources) or "" in source_ids:
            raise ValueError("Checkpoint sources contain empty or duplicate IDs.")
        expected_in: list[list[int]] = [[] for _ in nodes]
        expected_out: list[list[int]] = [[] for _ in nodes]
        for expected_id, node in enumerate(nodes, start=1):
            if not isinstance(node, dict) or node.get("id") != expected_id:
                raise ValueError(f"Checkpoint node ID is invalid at {expected_id}.")
            if clean_whitespace(str(node.get("source_id", ""))) not in source_ids:
                raise ValueError(
                    f"Checkpoint node source_id is invalid at {expected_id}."
                )
        for expected_id, edge in enumerate(edges, start=1):
            if not isinstance(edge, dict) or edge.get("id") != expected_id:
                raise ValueError(f"Checkpoint edge ID is invalid at {expected_id}.")
            source = edge.get("source")
            target = edge.get("target")
            if (
                not isinstance(source, int)
                or isinstance(source, bool)
                or not 1 <= source <= len(nodes)
                or not isinstance(target, int)
                or isinstance(target, bool)
                or not 1 <= target <= len(nodes)
            ):
                raise ValueError(
                    f"Checkpoint edge endpoints are invalid at {expected_id}."
                )
            expected_out[source - 1].append(expected_id)
            expected_in[target - 1].append(expected_id)
        for index, node in enumerate(nodes):
            if node.get("in_edges") != expected_in[index]:
                raise ValueError(f"Checkpoint in_edges mismatch at node {index + 1}.")
            if node.get("out_edges") != expected_out[index]:
                raise ValueError(f"Checkpoint out_edges mismatch at node {index + 1}.")

        if (
            not isinstance(embeddings, np.ndarray)
            or embeddings.ndim != 2
            or embeddings.shape[1] != EMBEDDING_DIMENSIONS
            or embeddings.shape[0] > len(nodes)
            or embeddings.dtype != np.float32
            or not np.all(np.isfinite(embeddings))
        ):
            raise ValueError(
                "Checkpoint Embedding matrix has an invalid shape, dtype, or value."
            )
        for key in (
            "normalized_node_ids",
            "embedded_node_ids",
            "embedding_pending_node_ids",
        ):
            values = state.get(key)
            if not isinstance(values, list):
                raise ValueError(f"Checkpoint {key} must be a list.")
            if any(
                not isinstance(node_id, int)
                or isinstance(node_id, bool)
                or not 1 <= node_id <= len(nodes)
                for node_id in values
            ):
                raise ValueError(f"Checkpoint {key} contains an invalid node ID.")

    def _save_checkpoint(
        self,
        *,
        state: dict[str, Any],
        sources: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        embeddings: np.ndarray,
        usage: dict[str, Any],
    ) -> None:
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        transaction_id = uuid.uuid4().hex
        temporary_dir = self.checkpoint_dir / (f".snapshot-{transaction_id}.tmp")
        snapshot_name = f"snapshot-{transaction_id}"
        snapshot_dir = self.checkpoint_dir / snapshot_name
        pointer_temporary_path = self.checkpoint_dir / (
            f".{CHECKPOINT_POINTER_FILE}.{transaction_id}.tmp"
        )
        temporary_dir.mkdir()
        try:
            json_payloads = {
                "state.json": state,
                DATA_FILES["sources"]: sources,
                DATA_FILES["nodes"]: nodes,
                DATA_FILES["edges"]: edges,
                DATA_FILES["usage"]: usage,
            }
            for filename, payload in json_payloads.items():
                self._write_json_fsynced(
                    temporary_dir / filename,
                    payload,
                )
            self._write_npy_fsynced(
                temporary_dir / DATA_FILES["embeddings"],
                embeddings,
            )
            self._fsync_directory(temporary_dir)
            os.replace(temporary_dir, snapshot_dir)
            self._fsync_directory(self.checkpoint_dir)

            with pointer_temporary_path.open("x", encoding="utf-8") as file:
                file.write(snapshot_name + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(
                pointer_temporary_path,
                self.checkpoint_dir / CHECKPOINT_POINTER_FILE,
            )
            self._fsync_directory(self.checkpoint_dir)

            for path in self.checkpoint_dir.iterdir():
                if path == snapshot_dir or path.name == CHECKPOINT_POINTER_FILE:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            self._fsync_directory(self.checkpoint_dir)
        except Exception:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            if pointer_temporary_path.exists():
                pointer_temporary_path.unlink()
            raise

    def _delete_checkpoint(self) -> None:
        if not self.checkpoint_dir.exists():
            return
        if not self.checkpoint_dir.is_dir():
            raise NotADirectoryError(
                f"Evidence Graph checkpoint is not a directory: {self.checkpoint_dir}"
            )
        shutil.rmtree(self.checkpoint_dir)
        if self.graph_dir.is_dir():
            self._fsync_directory(self.graph_dir)

    def _load_existing(
        self,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        np.ndarray,
        dict[str, Any],
        bool,
    ]:
        if not self.graph_dir.exists():
            return (
                {},
                [],
                [],
                [],
                np.empty(
                    (0, EMBEDDING_DIMENSIONS),
                    dtype=np.float32,
                ),
                {},
                False,
            )
        if not self.graph_dir.is_dir():
            raise NotADirectoryError(
                f"Evidence Graph path is not a directory: {self.graph_dir}"
            )

        entries = [
            path
            for path in self.graph_dir.iterdir()
            if path.name != CHECKPOINT_DIR_NAME
        ]
        existing_files = {path.name for path in entries if path.is_file()}
        core_files = set(ALL_GRAPH_FILES)
        present_core = existing_files & core_files
        if not present_core:
            if entries:
                raise ValueError(
                    "Existing Evidence Graph directory is not empty but contains "
                    f"no graph files: {self.graph_dir}"
                )
            return (
                {},
                [],
                [],
                [],
                np.empty(
                    (0, EMBEDDING_DIMENSIONS),
                    dtype=np.float32,
                ),
                {},
                False,
            )
        if present_core != core_files:
            missing = sorted(core_files - present_core)
            raise ValueError(
                "Existing Evidence Graph is incomplete; missing: " + ", ".join(missing)
            )
        unexpected = sorted(
            path.name for path in entries if path.name not in core_files
        )
        if unexpected:
            raise ValueError(
                "Evidence Graph directory contains unexpected entries: "
                + ", ".join(unexpected)
            )

        manifest = read_json(self.graph_dir / MANIFEST_FILE)
        sources = read_json(self.graph_dir / DATA_FILES["sources"])
        nodes = read_json(self.graph_dir / DATA_FILES["nodes"])
        edges = read_json(self.graph_dir / DATA_FILES["edges"])
        embeddings = np.load(
            self.graph_dir / DATA_FILES["embeddings"],
            allow_pickle=False,
        )
        usage = read_json(self.graph_dir / DATA_FILES["usage"])
        errors = validate_evidence_graph(
            manifest=manifest,
            sources=sources,
            nodes=nodes,
            edges=edges,
            embeddings=embeddings,
            usage=usage,
            expected_graph_id=self.graph_id,
        )
        if errors:
            raise ValueError("Existing Evidence Graph is invalid: " + "; ".join(errors))
        return manifest, sources, nodes, edges, embeddings, usage, True

    @staticmethod
    def _merge_single_document(
        *,
        graph: dict[str, Any],
        sources: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        source = deepcopy(graph["source"])
        source_id = clean_whitespace(str(source["id"]))
        node_offset = len(nodes)
        edge_offset = len(edges)

        for local_node in graph["nodes"]:
            node = {
                "id": node_offset + int(local_node["id"]),
                "type": local_node["type"],
                "title": local_node["title"],
                "statement": local_node["statement"],
                "field": local_node["field"],
                "subfield": local_node["subfield"],
                "source_id": source_id,
                "in_edges": [],
                "out_edges": [],
            }
            for optional_key in ("label", "proof", "location", "semantic"):
                if optional_key in local_node:
                    node[optional_key] = deepcopy(local_node[optional_key])
            nodes.append(node)

        for local_edge in graph["edges"]:
            edge_id = edge_offset + int(local_edge["id"])
            source_node_id = node_offset + int(local_edge["source"])
            target_node_id = node_offset + int(local_edge["target"])
            edge = {
                "id": edge_id,
                "type": local_edge["type"],
                "source": source_node_id,
                "target": target_node_id,
                "semantics": local_edge["semantics"],
            }
            edges.append(edge)
            nodes[source_node_id - 1]["out_edges"].append(edge_id)
            nodes[target_node_id - 1]["in_edges"].append(edge_id)

        sources.append(source)
        return {
            "source_id": source_id,
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "node_id_start": node_offset + 1 if graph["nodes"] else None,
            "node_id_end": len(nodes) if graph["nodes"] else None,
            "edge_id_start": edge_offset + 1 if graph["edges"] else None,
            "edge_id_end": len(edges) if graph["edges"] else None,
        }

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save_transactionally(
        self,
        *,
        manifest: dict[str, Any],
        sources: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        embeddings: np.ndarray,
        usage: dict[str, Any],
    ) -> None:
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        json_payloads = {
            DATA_FILES["sources"]: sources,
            DATA_FILES["nodes"]: nodes,
            DATA_FILES["edges"]: edges,
            DATA_FILES["usage"]: usage,
            MANIFEST_FILE: manifest,
        }
        filenames = [
            DATA_FILES["sources"],
            DATA_FILES["nodes"],
            DATA_FILES["edges"],
            DATA_FILES["embeddings"],
            DATA_FILES["usage"],
            MANIFEST_FILE,
        ]
        transaction_id = uuid.uuid4().hex
        temporary_paths: dict[str, Path] = {}
        backup_paths: dict[str, Path] = {}
        committed: list[str] = []

        try:
            for filename, payload in json_payloads.items():
                temporary_path = self.graph_dir / (f".{filename}.{transaction_id}.tmp")
                self._write_json_fsynced(temporary_path, payload)
                temporary_paths[filename] = temporary_path

            embedding_filename = DATA_FILES["embeddings"]
            embedding_temporary_path = self.graph_dir / (
                f".{embedding_filename}.{transaction_id}.tmp"
            )
            self._write_npy_fsynced(
                embedding_temporary_path,
                embeddings,
            )
            temporary_paths[embedding_filename] = embedding_temporary_path

            for filename in filenames:
                final_path = self.graph_dir / filename
                if final_path.exists():
                    backup_path = self.graph_dir / (f".{filename}.{transaction_id}.bak")
                    try:
                        os.link(final_path, backup_path)
                    except OSError:
                        shutil.copy2(final_path, backup_path)
                    backup_paths[filename] = backup_path

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
            self._fsync_directory(self.graph_dir)
            raise
        finally:
            for path in temporary_paths.values():
                if path.exists():
                    path.unlink()
            for path in backup_paths.values():
                if path.exists():
                    path.unlink()

    @staticmethod
    def _write_json_fsynced(path: Path, payload: Any) -> None:
        with path.open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _write_npy_fsynced(
        path: Path,
        matrix: np.ndarray,
    ) -> None:
        with path.open("xb") as file:
            np.save(file, matrix, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())

    def _merge_usage(
        self,
        *,
        previous: dict[str, Any],
        current: dict[str, Any],
        revision: int,
    ) -> dict[str, Any]:
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

        previous_calls = (
            previous.get("calls", []) if isinstance(previous.get("calls"), list) else []
        )
        current_calls = (
            current.get("calls", []) if isinstance(current.get("calls"), list) else []
        )
        total = {
            field: sum(
                int(item.get(field, 0) or 0)
                for item in by_model.values()
                if isinstance(item, dict)
            )
            for field in numeric_fields
        }
        return {
            "started_at": clean_whitespace(str(previous.get("started_at", "") or ""))
            or clean_whitespace(str(current.get("started_at", "") or ""))
            or utc_now(),
            "updated_at": utc_now(),
            "graph_id": self.graph_id,
            "revision": revision,
            "total": total,
            "by_model": by_model,
            "calls": [*previous_calls, *current_calls],
        }

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
