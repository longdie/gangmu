from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from src.common.config import GangmuConfig, get_config
from src.common.files import (
    ProjectLogger,
    TokenUsageTracker,
    read_json,
    utc_now,
    write_json,
    write_json_atomic,
)
from src.common.paths import (
    PAPER_SINGLE_DOC_KG_OUTPUT_DIR,
    PROMPT_DIR,
    TEXTBOOK_SINGLE_DOC_KG_OUTPUT_DIR,
)
from src.single_doc.extraction import (
    DependencyExtractor,
    JSONClient,
    NodeExtractor,
    NodeQualityChecker,
)
from src.single_doc.llm import (
    LLMClient,
    ProviderCircuitOpenError,
    ProviderRequestError,
)
from src.single_doc.pacing import SingleDocumentRequestPacer
from src.single_doc.preprocessing import MarkdownPreprocessor
from src.single_doc.schema import (
    build_single_document_graph,
    compact_source_metadata,
    validate_single_document_graph,
)


COLLECTIONS = {"textbooks", "papers"}
DEPENDENCY_PROMPT_PATHS = {
    "textbooks": PROMPT_DIR / "single_doc_dependencies_textbook.md",
    "papers": PROMPT_DIR / "single_doc_dependencies_paper.md",
}


class SingleDocumentPipeline:
    """Build one local-ID document graph without constructing the Evidence Graph."""

    def __init__(
        self,
        *,
        ocr_dir: str | Path,
        collection: str,
        output_dir: str | Path | None = None,
        config: GangmuConfig | None = None,
        llm_client: JSONClient | None = None,
        logger: ProjectLogger | None = None,
        node_workers: int | None = None,
        dependency_workers: int | None = None,
        request_pacer: SingleDocumentRequestPacer | None = None,
    ) -> None:
        if collection not in COLLECTIONS:
            raise ValueError(f"collection must be one of {sorted(COLLECTIONS)}.")

        self.ocr_dir = Path(ocr_dir).expanduser().resolve()
        self.collection = collection
        self.document_id = self.ocr_dir.name
        self.config = config or get_config()
        default_output_root = (
            TEXTBOOK_SINGLE_DOC_KG_OUTPUT_DIR
            if collection == "textbooks"
            else PAPER_SINGLE_DOC_KG_OUTPUT_DIR
        )
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else default_output_root / self.document_id
        )
        self.graph_output_path = self.output_dir / f"{self.document_id}.json"
        self.usage_output_path = self.output_dir / "usage.json"
        self.quality_report_path = self.output_dir / "quality_report.json"
        self.candidate_report_path = self.output_dir / "candidate_report.json"
        self.checkpoint_dir = self.output_dir / ".checkpoint"
        self.node_segments_checkpoint_path = self.checkpoint_dir / "node_segments.json"
        self.nodes_checkpoint_path = self.checkpoint_dir / "nodes.json"
        self.dependencies_checkpoint_path = self.checkpoint_dir / "dependencies.json"
        self.checkpoint_usage_path = self.checkpoint_dir / "usage.json"

        self.logger = logger or ProjectLogger(
            self.config.log_dir("single_doc"),
            run_name=self.document_id,
        )
        self.usage_tracker = TokenUsageTracker()
        self.request_pacer = request_pacer
        if llm_client is None:
            self.request_pacer = (
                self.request_pacer
                or SingleDocumentRequestPacer.from_config(self.config)
            )
            self.llm_client = LLMClient(
                config=self.config,
                logger=self.logger,
                usage_tracker=self.usage_tracker,
                request_pacer=self.request_pacer,
            )
        else:
            self.llm_client = llm_client
        self.node_workers = node_workers or self.config.get_int(
            "node_extraction.parallel_workers",
            default=5,
            minimum=1,
        )
        self.dependency_workers = dependency_workers or self.config.get_int(
            "dependency_extraction.parallel_workers",
            default=5,
            minimum=1,
        )
        if self.node_workers < 1 or self.dependency_workers < 1:
            raise ValueError("Worker counts must be positive.")
        checkpoint_usage = read_json(self.checkpoint_usage_path, default={})
        self._checkpoint_usage_base = (
            checkpoint_usage if isinstance(checkpoint_usage, dict) else {}
        )

    def run(self) -> dict[str, Any]:
        if not self.ocr_dir.is_dir():
            raise NotADirectoryError(f"OCR directory does not exist: {self.ocr_dir}")
        markdown_path = locate_document_markdown(self.ocr_dir)
        source = self._load_source()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        markdown_fingerprint = self._file_fingerprint(markdown_path)

        self.logger.info(
            "Single-document graph construction started",
            document_id=self.document_id,
            collection=self.collection,
            ocr_dir=str(self.ocr_dir),
            markdown_path=str(markdown_path),
            output_path=str(self.graph_output_path),
            node_workers=self.node_workers,
            dependency_workers=self.dependency_workers,
        )

        completed_node_checkpoint = self._load_nodes_checkpoint(markdown_fingerprint)
        if completed_node_checkpoint is not None:
            nodes = completed_node_checkpoint["nodes"]
            quality_report = completed_node_checkpoint["quality_report"]
            segment_count = int(completed_node_checkpoint["segment_count"])
            self.logger.info(
                "Resuming from completed node checkpoint",
                document_id=self.document_id,
                node_count=len(nodes),
                segment_count=segment_count,
                checkpoint_path=str(self.nodes_checkpoint_path),
            )
        else:
            preprocessor = MarkdownPreprocessor(
                max_tokens=self.config.get_int(
                    "node_extraction.token_limit",
                    minimum=1,
                ),
                retry_overlap_tokens=self.config.get_int(
                    "node_extraction.retry_overlap_tokens",
                    minimum=0,
                ),
            )
            segments = preprocessor.segment_file(markdown_path)
            segment_count = len(segments)
            node_extractor = NodeExtractor(
                llm_client=self.llm_client,
                preprocessor=preprocessor,
                prompt_path=PROMPT_DIR / "single_doc_nodes.md",
                source_id=self.document_id,
                default_field=self.config.get_str("knowledge.default_field"),
                default_subfield=self.config.get_str("knowledge.default_subfield"),
                max_split_depth=self.config.get_int(
                    "node_extraction.max_split_depth",
                    minimum=0,
                ),
                logger=self.logger,
            )
            completed_segments = self._load_indexed_checkpoint(
                self.node_segments_checkpoint_path,
                fingerprint_key="markdown_fingerprint",
                fingerprint=markdown_fingerprint,
                expected_count=segment_count,
            )
            node_checkpoint_interval = self.config.get_int(
                "node_extraction.checkpoint_every_segments",
                minimum=1,
            )

            def save_node_progress(
                results: dict[int, list[dict[str, Any]]],
                force: bool,
            ) -> None:
                if force or len(results) % node_checkpoint_interval == 0:
                    self._save_indexed_checkpoint(
                        self.node_segments_checkpoint_path,
                        fingerprint_key="markdown_fingerprint",
                        fingerprint=markdown_fingerprint,
                        expected_count=segment_count,
                        results=results,
                    )

            raw_nodes = self._extract_nodes_parallel(
                segments=segments,
                extractor=node_extractor,
                completed_results=completed_segments,
                progress_callback=save_node_progress,
            )
            node_extractor.assign_ids(raw_nodes)
            nodes, quality_report = NodeQualityChecker().clean_nodes(raw_nodes)
            node_extractor.assign_ids(nodes)
            self._save_nodes_checkpoint(
                markdown_fingerprint=markdown_fingerprint,
                segment_count=segment_count,
                nodes=nodes,
                quality_report=quality_report,
            )

        dependency_extractor = self._create_dependency_extractor()
        node_fingerprint = self._nodes_fingerprint(nodes)
        completed_dependencies = self._load_indexed_checkpoint(
            self.dependencies_checkpoint_path,
            fingerprint_key="node_fingerprint",
            fingerprint=node_fingerprint,
            expected_count=len(nodes),
        )
        dependency_checkpoint_interval = self.config.get_int(
            "dependency_extraction.checkpoint_every_targets",
            minimum=1,
        )

        def save_dependency_progress(
            results: dict[int, dict[str, Any]],
            force: bool,
        ) -> None:
            if force or len(results) % dependency_checkpoint_interval == 0:
                self._save_indexed_checkpoint(
                    self.dependencies_checkpoint_path,
                    fingerprint_key="node_fingerprint",
                    fingerprint=node_fingerprint,
                    expected_count=len(nodes),
                    results=results,
                )

        raw_edges = dependency_extractor.extract_edges(
            nodes,
            completed_results=completed_dependencies,
            progress_callback=save_dependency_progress,
        )
        graph = build_single_document_graph(
            source=source,
            nodes=nodes,
            edges=raw_edges,
        )
        write_json(self.graph_output_path, graph)
        self._save_quality_report(
            report=quality_report,
            graph=graph,
            markdown_path=markdown_path,
            segment_count=segment_count,
        )
        self._save_candidate_report(
            diagnostics=dependency_extractor.candidate_diagnostics,
            graph=graph,
        )
        self._save_usage(
            graph=graph,
            markdown_path=markdown_path,
            operation="full_extraction",
        )
        self._save_checkpoint_usage()

        self.logger.info(
            "Single-document graph construction completed",
            document_id=self.document_id,
            node_count=len(graph["nodes"]),
            edge_count=len(graph["edges"]),
            output_path=str(self.graph_output_path),
        )
        return graph

    def rebuild_dependencies(self) -> dict[str, Any]:
        """Rebuild only dependency edges while preserving extracted nodes."""

        if not self.graph_output_path.is_file():
            raise FileNotFoundError(
                "Existing single-document graph is required for dependency-only "
                f"rebuild: {self.graph_output_path}"
            )
        existing_graph = read_json(self.graph_output_path)
        if not isinstance(existing_graph, dict):
            raise ValueError(
                f"Existing graph must be a JSON object: {self.graph_output_path}"
            )
        validation_errors = validate_single_document_graph(existing_graph)
        if validation_errors:
            raise ValueError(
                "Cannot rebuild dependencies for an invalid graph: "
                + "; ".join(validation_errors)
            )

        markdown_path = locate_document_markdown(self.ocr_dir)
        source = existing_graph["source"]
        nodes = existing_graph["nodes"]
        self.logger.info(
            "Single-document dependency rebuild started",
            document_id=self.document_id,
            collection=self.collection,
            node_count=len(nodes),
            previous_edge_count=len(existing_graph["edges"]),
            output_path=str(self.graph_output_path),
            dependency_workers=self.dependency_workers,
        )

        dependency_extractor = self._create_dependency_extractor()
        raw_edges = dependency_extractor.extract_edges(nodes)
        graph = build_single_document_graph(
            source=source,
            nodes=nodes,
            edges=raw_edges,
        )
        write_json(self.graph_output_path, graph)
        self._save_candidate_report(
            diagnostics=dependency_extractor.candidate_diagnostics,
            graph=graph,
        )
        self._save_usage(
            graph=graph,
            markdown_path=markdown_path,
            operation="dependency_rebuild",
            merge_existing=True,
        )

        self.logger.info(
            "Single-document dependency rebuild completed",
            document_id=self.document_id,
            node_count=len(graph["nodes"]),
            previous_edge_count=len(existing_graph["edges"]),
            edge_count=len(graph["edges"]),
            output_path=str(self.graph_output_path),
        )
        return graph

    def _create_dependency_extractor(self) -> DependencyExtractor:
        return DependencyExtractor(
            llm_client=self.llm_client,
            prompt_path=DEPENDENCY_PROMPT_PATHS[self.collection],
            position_window=self.config.get_int(
                "dependency_extraction.position_window",
                minimum=0,
            ),
            title_bm25_top_k=self.config.get_int(
                "dependency_extraction.title_bm25_top_k",
                minimum=0,
            ),
            proof_bm25_top_k=self.config.get_int(
                "dependency_extraction.proof_bm25_top_k",
                minimum=0,
            ),
            title_score_weight=self.config.get_float(
                "dependency_extraction.title_score_weight",
                minimum=0,
            ),
            statement_score_weight=self.config.get_float(
                "dependency_extraction.statement_score_weight",
                minimum=0,
            ),
            batch_size=self.config.get_int(
                "dependency_extraction.batch_size",
                minimum=1,
                maximum=5,
            ),
            parallel_workers=self.dependency_workers,
            logger=self.logger,
        )

    def _extract_nodes_parallel(
        self,
        *,
        segments: list[dict[str, Any]],
        extractor: NodeExtractor,
        completed_results: dict[int, list[dict[str, Any]]] | None = None,
        progress_callback: (
            Callable[[dict[int, list[dict[str, Any]]], bool], None] | None
        ) = None,
    ) -> list[dict[str, Any]]:
        if not segments:
            return []
        completed = {
            index: result
            for index, result in (completed_results or {}).items()
            if 0 <= index < len(segments) and isinstance(result, list)
        }
        pending_indices = [
            index for index in range(len(segments)) if index not in completed
        ]
        max_workers = max(
            1,
            min(self.node_workers, len(pending_indices) or 1),
        )
        fatal_error: Exception | None = None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    extractor.extract_from_segment,
                    segments[index],
                ): index
                for index in pending_indices
            }
            for future in as_completed(futures):
                index = futures[future]
                segment = segments[index]
                try:
                    completed[index] = future.result()
                except Exception as exc:
                    fatal_error = exc
                    for pending_future in futures:
                        pending_future.cancel()
                    break
                if progress_callback is not None:
                    progress_callback(completed, False)

        if fatal_error is not None:
            for future, index in futures.items():
                if index in completed or future.cancelled():
                    continue
                try:
                    completed[index] = future.result()
                except Exception:
                    # Failed or circuit-open segments remain pending for the next run.
                    continue
        if progress_callback is not None:
            progress_callback(completed, True)
        if fatal_error is not None:
            raise fatal_error

        nodes: list[dict[str, Any]] = []
        for index in sorted(completed):
            nodes.extend(completed[index])
        return nodes

    def _load_nodes_checkpoint(
        self,
        markdown_fingerprint: str,
    ) -> dict[str, Any] | None:
        checkpoint = read_json(self.nodes_checkpoint_path, default={})
        if not isinstance(checkpoint, dict):
            return None
        if (
            checkpoint.get("checkpoint_version") != 1
            or checkpoint.get("document_id") != self.document_id
            or checkpoint.get("collection") != self.collection
            or checkpoint.get("markdown_fingerprint") != markdown_fingerprint
            or not isinstance(checkpoint.get("nodes"), list)
            or not isinstance(checkpoint.get("quality_report"), dict)
            or not isinstance(checkpoint.get("segment_count"), int)
        ):
            return None
        return checkpoint

    def _save_nodes_checkpoint(
        self,
        *,
        markdown_fingerprint: str,
        segment_count: int,
        nodes: list[dict[str, Any]],
        quality_report: dict[str, Any],
    ) -> None:
        write_json_atomic(
            self.nodes_checkpoint_path,
            {
                "checkpoint_version": 1,
                "document_id": self.document_id,
                "collection": self.collection,
                "markdown_fingerprint": markdown_fingerprint,
                "segment_count": segment_count,
                "nodes": nodes,
                "quality_report": quality_report,
                "saved_at": utc_now(),
            },
        )
        self._save_checkpoint_usage()
        self.logger.info(
            "Completed node checkpoint saved",
            node_count=len(nodes),
            checkpoint_path=str(self.nodes_checkpoint_path),
        )

    def _load_indexed_checkpoint(
        self,
        path: Path,
        *,
        fingerprint_key: str,
        fingerprint: str,
        expected_count: int,
    ) -> dict[int, Any]:
        checkpoint = read_json(path, default={})
        if not isinstance(checkpoint, dict):
            return {}
        raw_results = checkpoint.get("results")
        if (
            checkpoint.get("checkpoint_version") != 1
            or checkpoint.get("document_id") != self.document_id
            or checkpoint.get("collection") != self.collection
            or checkpoint.get(fingerprint_key) != fingerprint
            or checkpoint.get("expected_count") != expected_count
            or not isinstance(raw_results, dict)
        ):
            return {}
        results: dict[int, Any] = {}
        for raw_index, result in raw_results.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 <= index < expected_count:
                results[index] = result
        if results:
            self.logger.info(
                "Checkpoint loaded",
                checkpoint_path=str(path),
                completed_count=len(results),
                expected_count=expected_count,
            )
        return results

    def _save_indexed_checkpoint(
        self,
        path: Path,
        *,
        fingerprint_key: str,
        fingerprint: str,
        expected_count: int,
        results: dict[int, Any],
    ) -> None:
        write_json_atomic(
            path,
            {
                "checkpoint_version": 1,
                "document_id": self.document_id,
                "collection": self.collection,
                fingerprint_key: fingerprint,
                "expected_count": expected_count,
                "completed_count": len(results),
                "results": {str(index): results[index] for index in sorted(results)},
                "saved_at": utc_now(),
            },
        )
        self._save_checkpoint_usage()

    def _save_checkpoint_usage(self) -> None:
        current = self.usage_tracker.summary()
        usage = (
            self._merge_usage(self._checkpoint_usage_base, current)
            if self._checkpoint_usage_base
            else current
        )
        write_json_atomic(self.checkpoint_usage_path, usage)

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _nodes_fingerprint(nodes: list[dict[str, Any]]) -> str:
        encoded = json.dumps(
            nodes,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_source(self) -> dict[str, Any]:
        metadata_path = self.ocr_dir / "source_metadata.json"
        raw_metadata: dict[str, Any] = {}
        if metadata_path.exists():
            loaded = read_json(metadata_path, default={})
            if isinstance(loaded, dict):
                raw_metadata = loaded
            else:
                self.logger.warning(
                    "Source metadata is not a JSON object; using defaults",
                    metadata_path=str(metadata_path),
                )
        else:
            self.logger.warning(
                "Source metadata is missing; using document directory name",
                metadata_path=str(metadata_path),
            )
        return compact_source_metadata(
            raw_metadata,
            document_id=self.document_id,
            collection=self.collection,
        )

    def _save_quality_report(
        self,
        *,
        report: dict[str, Any],
        graph: dict[str, Any],
        markdown_path: Path,
        segment_count: int,
    ) -> None:
        report = {
            **report,
            "document_id": self.document_id,
            "collection": self.collection,
            "markdown_path": str(markdown_path),
            "segment_count": segment_count,
            "final_node_count": len(graph["nodes"]),
            "saved_at": utc_now(),
        }
        write_json(self.quality_report_path, report)

    def _save_candidate_report(
        self,
        *,
        diagnostics: list[dict[str, Any]],
        graph: dict[str, Any],
    ) -> None:
        report = {
            "document_id": self.document_id,
            "collection": self.collection,
            "created_at": utc_now(),
            "candidate_strategy": {
                "position_window": self.config.get_int(
                    "dependency_extraction.position_window"
                ),
                "statement_bm25_top_k": self.config.get_int(
                    "dependency_extraction.title_bm25_top_k"
                ),
                "proof_bm25_top_k": self.config.get_int(
                    "dependency_extraction.proof_bm25_top_k"
                ),
                "title_score_weight": self.config.get_float(
                    "dependency_extraction.title_score_weight"
                ),
                "statement_score_weight": self.config.get_float(
                    "dependency_extraction.statement_score_weight"
                ),
                "batch_size": self.config.get_int("dependency_extraction.batch_size"),
            },
            "summary": {
                "node_count": len(graph["nodes"]),
                "edge_count": len(graph["edges"]),
                "dependency_edge_count": sum(
                    edge["type"] == "dependency" for edge in graph["edges"]
                ),
                "proof_edge_count": sum(
                    edge["type"] == "proof" for edge in graph["edges"]
                ),
                "candidate_count": sum(
                    len(item.get("candidate_ids", [])) for item in diagnostics
                ),
            },
            "per_target": diagnostics,
        }
        write_json(self.candidate_report_path, report)

    def _save_usage(
        self,
        *,
        graph: dict[str, Any],
        markdown_path: Path,
        operation: str,
        merge_existing: bool = False,
    ) -> None:
        usage = self.usage_tracker.summary()
        if self._checkpoint_usage_base:
            usage = self._merge_usage(self._checkpoint_usage_base, usage)
        if merge_existing and self.usage_output_path.is_file():
            existing = read_json(self.usage_output_path)
            if isinstance(existing, dict):
                usage = self._merge_usage(existing, usage)

        operations = usage.get("operations", [])
        if not isinstance(operations, list):
            operations = []
        operations.append(
            {
                "operation": operation,
                "started_at": self.usage_tracker.started_at,
                "completed_at": utc_now(),
            }
        )
        usage.update(
            {
                "operations": operations,
                "document_id": self.document_id,
                "collection": self.collection,
                "markdown_path": str(markdown_path),
                "graph_output_path": str(self.graph_output_path),
                "node_count": len(graph["nodes"]),
                "edge_count": len(graph["edges"]),
            }
        )
        write_json(self.usage_output_path, usage)

    @staticmethod
    def _merge_usage(
        existing: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        merged_by_model: dict[str, dict[str, Any]] = {}
        for report in (existing, current):
            by_model = report.get("by_model", {})
            if not isinstance(by_model, dict):
                continue
            for key, item in by_model.items():
                if not isinstance(item, dict):
                    continue
                merged = merged_by_model.setdefault(
                    key,
                    {
                        "provider": item.get("provider", ""),
                        "model": item.get("model", ""),
                        "calls": 0,
                        "successful_calls": 0,
                        "failed_calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                )
                for field in (
                    "calls",
                    "successful_calls",
                    "failed_calls",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ):
                    merged[field] += int(item.get(field, 0) or 0)

        total_prompt = sum(item["prompt_tokens"] for item in merged_by_model.values())
        total_completion = sum(
            item["completion_tokens"] for item in merged_by_model.values()
        )
        old_calls = existing.get("calls", [])
        new_calls = current.get("calls", [])
        return {
            **existing,
            "started_at": existing.get("started_at") or current.get("started_at"),
            "updated_at": current.get("updated_at") or utc_now(),
            "total": {
                "calls": sum(item["calls"] for item in merged_by_model.values()),
                "successful_calls": sum(
                    item["successful_calls"] for item in merged_by_model.values()
                ),
                "failed_calls": sum(
                    item["failed_calls"] for item in merged_by_model.values()
                ),
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            },
            "by_model": merged_by_model,
            "calls": (list(old_calls) if isinstance(old_calls, list) else [])
            + (list(new_calls) if isinstance(new_calls, list) else []),
        }


def locate_document_markdown(ocr_dir: str | Path) -> Path:
    directory = Path(ocr_dir)
    preferred = directory / f"{directory.name}.md"
    if preferred.is_file():
        return preferred

    candidates = sorted(path for path in directory.glob("*.md") if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No top-level Markdown file found in OCR directory: {directory}"
        )
    if len(candidates) > 1:
        raise FileNotFoundError(
            "Multiple top-level Markdown files found; cannot choose automatically: "
            f"{directory}"
        )
    return candidates[0]
