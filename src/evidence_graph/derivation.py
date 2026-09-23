from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.common.config import GangmuConfig, get_config
from src.common.files import (
    ProjectLogger,
    TokenUsageTracker,
    clean_whitespace,
    read_text,
)
from src.common.paths import PROMPT_DIR
from src.single_doc.llm import LLMClient


EMBEDDING_DIMENSIONS = 2048
EMBEDDING_DTYPE = np.dtype(np.float32)
NORMALIZATION_PROMPT_PATH = PROMPT_DIR / "normalize_statement.md"
NORMALIZED_STATEMENT_KEY = "normalized_statement"
SEMANTIC_KEYS = {
    NORMALIZED_STATEMENT_KEY,
    "conditions",
    "conclusions",
    "symbols",
}


def clean_model_text(text: str) -> str:
    """Preserve recoverable LaTeX characters and normalize model whitespace."""

    replacements = {
        "\x08": r"\b",
        "\x0c": r"\f",
        "\x0b": r"\v",
        "\x07": r"\a",
    }
    for bad, replacement in replacements.items():
        text = text.replace(bad, replacement)
    text = re.sub(r"\\[nrt](?![A-Za-z])", " ", text)
    return clean_whitespace(text.replace("\ufeff", ""))


def normalized_statement(node: dict[str, Any]) -> str:
    semantic = node.get("semantic")
    if not isinstance(semantic, dict):
        return ""
    return clean_model_text(str(semantic.get(NORMALIZED_STATEMENT_KEY, "") or ""))


def has_valid_semantic(node: dict[str, Any]) -> bool:
    semantic = node.get("semantic")
    if not isinstance(semantic, dict) or set(semantic) != SEMANTIC_KEYS:
        return False
    if not normalized_statement(node):
        return False
    return all(
        isinstance(semantic.get(key), list)
        for key in ("conditions", "conclusions", "symbols")
    )


def validate_normalization_result(
    result: dict[str, Any],
) -> tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "Normalization output must be one JSON object."
    if set(result) != {NORMALIZED_STATEMENT_KEY}:
        return (
            False,
            f"Normalization output must contain exactly '{NORMALIZED_STATEMENT_KEY}'.",
        )
    value = result.get(NORMALIZED_STATEMENT_KEY)
    if not isinstance(value, str) or not clean_model_text(value):
        return False, f"'{NORMALIZED_STATEMENT_KEY}' must be a non-empty string."
    if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
        return (
            False,
            "The normalized statement contains JSON control characters; "
            "LaTeX backslashes must be escaped.",
        )
    return True, ""


def dependency_context(
    *,
    node: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only dependency predecessors; proof edges are intentionally ignored."""

    dependencies: list[dict[str, Any]] = []
    for raw_edge_id in node.get("in_edges", []) or []:
        try:
            edge_id = int(raw_edge_id)
        except (TypeError, ValueError):
            continue
        if not 1 <= edge_id <= len(edges):
            continue
        edge = edges[edge_id - 1]
        if edge.get("id") != edge_id or edge.get("type") != "dependency":
            continue
        try:
            source_id = int(edge.get("source"))
        except (TypeError, ValueError):
            continue
        if not 1 <= source_id <= len(nodes):
            continue
        source_node = nodes[source_id - 1]
        statement = clean_model_text(str(source_node.get("statement", "") or ""))
        if not statement:
            continue
        dependencies.append(
            {
                "id": source_id,
                "type": clean_model_text(str(source_node.get("type", "") or "")),
                "title": clean_model_text(str(source_node.get("title", "") or "")),
                "statement": statement,
            }
        )
    return dependencies


class EvidenceNodeNormalizer:
    """Generate EvidenceNode.semantic.normalized_statement in parallel."""

    def __init__(
        self,
        *,
        config: GangmuConfig | None = None,
        logger: ProjectLogger | None = None,
        usage_tracker: TokenUsageTracker | None = None,
        prompt_path: str | Path = NORMALIZATION_PROMPT_PATH,
    ) -> None:
        self.config = config or get_config()
        self.logger = logger
        self.usage_tracker = usage_tracker or TokenUsageTracker()
        self.prompt_path = Path(prompt_path).expanduser().resolve()
        self.prompt = read_text(self.prompt_path)
        self.parallel_workers = self.config.get_int(
            "normalization.parallel_workers",
            minimum=1,
        )
        self.model = self.config.model("normalization")
        self.llm_client = LLMClient(
            config=self.config,
            logger=self.logger,
            usage_tracker=self.usage_tracker,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model.name,
            "provider": self.model.provider,
            "prompt_file": self.prompt_path.name,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
        }

    def normalize_pending(
        self,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        batch_size: int | None = None,
        on_batch_completed: Callable[[list[int]], None] | None = None,
    ) -> list[int]:
        sanitized_ids: list[int] = []
        pending_ids: list[int] = []
        for node in nodes:
            if not has_valid_semantic(node):
                pending_ids.append(int(node["id"]))
                continue
            semantic = node["semantic"]
            raw_statement = str(semantic.get(NORMALIZED_STATEMENT_KEY, "") or "")
            cleaned_statement = clean_model_text(raw_statement)
            if cleaned_statement != raw_statement:
                semantic[NORMALIZED_STATEMENT_KEY] = cleaned_statement
                sanitized_ids.append(int(node["id"]))
        if sanitized_ids and on_batch_completed is not None:
            on_batch_completed(sorted(sanitized_ids))
        if not pending_ids:
            return sanitized_ids

        effective_batch_size = len(pending_ids) if batch_size is None else batch_size
        if (
            isinstance(effective_batch_size, bool)
            or not isinstance(effective_batch_size, int)
            or effective_batch_size < 1
        ):
            raise ValueError("Normalization batch_size must be positive.")
        max_workers = max(1, min(self.parallel_workers, len(pending_ids)))
        completed = 0
        successful_ids: list[int] = []
        print(
            "[evidence-normalization] "
            f"pending={len(pending_ids)}, workers={max_workers}, "
            f"checkpoint_batch={effective_batch_size}"
        )

        for batch_start in range(0, len(pending_ids), effective_batch_size):
            batch_ids = pending_ids[batch_start : batch_start + effective_batch_size]
            batch_workers = max(1, min(max_workers, len(batch_ids)))
            results: dict[int, dict[str, Any]] = {}
            failures: list[tuple[int, str]] = []
            with ThreadPoolExecutor(max_workers=batch_workers) as executor:
                futures = {
                    executor.submit(
                        self._normalize_one,
                        node=nodes[node_id - 1],
                        nodes=nodes,
                        edges=edges,
                    ): node_id
                    for node_id in batch_ids
                }
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        results[node_id] = future.result()
                    except Exception as exc:
                        failures.append((node_id, str(exc)))
                        if self.logger:
                            self.logger.exception(
                                "Evidence node normalization failed",
                                exc,
                                node_id=node_id,
                            )
                    completed += 1
                    print(
                        "[evidence-normalization] "
                        f"completed {completed}/{len(pending_ids)}: "
                        f"node_id={node_id}"
                    )

            batch_successful_ids = sorted(results)
            for node_id in batch_successful_ids:
                nodes[node_id - 1]["semantic"] = results[node_id]
            successful_ids.extend(batch_successful_ids)
            if on_batch_completed is not None:
                # Save usage even when the provider failed every request in the
                # batch; completed_node_ids may therefore be empty.
                on_batch_completed(batch_successful_ids)

            if failures:
                details = "; ".join(
                    f"node {node_id}: {error}" for node_id, error in sorted(failures)
                )
                raise RuntimeError(
                    f"Normalization failed for {len(failures)} node(s): {details}"
                )

        return sorted([*sanitized_ids, *successful_ids])

    def _normalize_one(
        self,
        *,
        node: dict[str, Any],
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "node": {
                "id": node.get("id"),
                "type": clean_model_text(str(node.get("type", "") or "")),
                "title": clean_model_text(str(node.get("title", "") or "")),
                "statement": clean_model_text(str(node.get("statement", "") or "")),
            },
            "dependencies": dependency_context(
                node=node,
                nodes=nodes,
                edges=edges,
            ),
        }
        result = self.llm_client.call_json(
            model_task="normalization",
            prompt=self.prompt,
            input_text=json.dumps(payload, ensure_ascii=False, indent=2),
            usage_task="evidence_node_normalization",
            validator=validate_normalization_result,
            preserve_latex_escapes=True,
        )
        if result is None:
            raise RuntimeError("The model did not return a valid normalized statement.")
        statement = clean_model_text(str(result[NORMALIZED_STATEMENT_KEY]))
        return {
            NORMALIZED_STATEMENT_KEY: statement,
            "conditions": [],
            "conclusions": [],
            "symbols": [],
        }


def normalize_vector(
    vector: Any,
    *,
    expected_dimensions: int = EMBEDDING_DIMENSIONS,
) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(
            f"Embedding vector must be one-dimensional; got {array.shape}."
        )
    if array.shape[0] != expected_dimensions:
        raise ValueError(
            f"Expected {expected_dimensions} dimensions; got {array.shape[0]}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Embedding vector contains non-finite values.")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError("Embedding vector has zero norm.")
    return (array / norm).astype(np.float32, copy=False)


def validate_embedding_matrix(
    matrix: np.ndarray,
    *,
    expected_rows: int,
    expected_dimensions: int = EMBEDDING_DIMENSIONS,
) -> None:
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        shape = getattr(matrix, "shape", None)
        raise ValueError(f"Embedding matrix must be two-dimensional: {shape}.")
    if matrix.shape != (expected_rows, expected_dimensions):
        raise ValueError(
            "Embedding matrix shape mismatch: "
            f"expected {(expected_rows, expected_dimensions)}, "
            f"got {matrix.shape}."
        )
    if matrix.dtype != EMBEDDING_DTYPE:
        raise ValueError(f"Embedding matrix must use float32; got {matrix.dtype}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Embedding matrix contains non-finite values.")
    if expected_rows:
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError("Embedding matrix contains non-normalized rows.")


class EvidenceEmbeddingBuilder:
    """Generate and append node embeddings while preserving node-id row mapping."""

    def __init__(
        self,
        *,
        config: GangmuConfig | None = None,
        logger: ProjectLogger | None = None,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OpenAI SDK is required. Install dependencies with pip install -r requirements.txt."
            ) from exc

        self.config = config or get_config()
        self.logger = logger
        self.usage_tracker = usage_tracker or TokenUsageTracker()
        self.model = self.config.model("embedding")
        if self.model.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                "models.embedding.dimensions must be "
                f"{EMBEDDING_DIMENSIONS}; got {self.model.dimensions}."
            )
        self.dimensions = EMBEDDING_DIMENSIONS
        self.parallel_workers = self.config.get_int(
            "embedding.parallel_workers",
            minimum=1,
        )
        self.retry_delay_seconds = self.config.get_float(
            "embedding.retry_delay_seconds",
            minimum=0,
        )
        self.call_attempts = self.config.get_int(
            "requests.call_attempts",
            minimum=1,
        )
        provider = self.config.provider(
            self.model.provider,
            require_api_key=True,
        )
        client_kwargs: dict[str, Any] = {
            "api_key": provider.api_key,
            "timeout": self.config.get_float(
                "requests.timeout_seconds",
                minimum=0,
            ),
        }
        if provider.base_url:
            client_kwargs["base_url"] = provider.base_url
        self.provider = provider
        self.client = OpenAI(**client_kwargs)
        self._client_lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model.name,
            "provider": self.model.provider,
            "dimensions": self.dimensions,
            "dtype": "float32",
            "l2_normalized": True,
            "row_mapping": "node_id_minus_1",
        }

    def update_matrix(
        self,
        *,
        nodes: list[dict[str, Any]],
        existing_matrix: np.ndarray,
        recompute_node_ids: list[int],
        pending_node_ids: list[int] | None = None,
        batch_size: int | None = None,
        on_batch_completed: Callable[
            [np.ndarray, list[int], list[int]],
            None,
        ]
        | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        existing_rows = int(existing_matrix.shape[0])
        if existing_rows > len(nodes):
            raise ValueError("Existing embedding rows exceed Evidence Node count.")

        if pending_node_ids is None:
            pending_ids = sorted(
                set(recompute_node_ids) | set(range(existing_rows + 1, len(nodes) + 1))
            )
        else:
            pending_ids = sorted(set(pending_node_ids))
        if any(not 1 <= node_id <= len(nodes) for node_id in pending_ids):
            raise ValueError("Embedding pending node ID is out of range.")
        pending_id_set = set(pending_ids)
        if pending_node_ids is None:
            validate_embedding_matrix(
                existing_matrix,
                expected_rows=existing_rows,
                expected_dimensions=self.dimensions,
            )
        else:
            if (
                not isinstance(existing_matrix, np.ndarray)
                or existing_matrix.ndim != 2
                or existing_matrix.shape != (existing_rows, self.dimensions)
                or existing_matrix.dtype != EMBEDDING_DTYPE
                or not np.all(np.isfinite(existing_matrix))
            ):
                raise ValueError(
                    "Checkpoint Embedding matrix has an invalid shape, dtype, or value."
                )
            completed_row_indices = [
                node_id - 1
                for node_id in range(1, existing_rows + 1)
                if node_id not in pending_id_set
            ]
            completed_matrix = existing_matrix[completed_row_indices]
            validate_embedding_matrix(
                completed_matrix,
                expected_rows=len(completed_row_indices),
                expected_dimensions=self.dimensions,
            )
        matrix = np.zeros(
            (len(nodes), self.dimensions),
            dtype=np.float32,
        )
        if existing_rows:
            matrix[:existing_rows] = existing_matrix
        if not pending_ids:
            validate_embedding_matrix(
                matrix,
                expected_rows=len(nodes),
                expected_dimensions=self.dimensions,
            )
            return matrix, []

        effective_batch_size = len(pending_ids) if batch_size is None else batch_size
        if (
            isinstance(effective_batch_size, bool)
            or not isinstance(effective_batch_size, int)
            or effective_batch_size < 1
        ):
            raise ValueError("Embedding batch_size must be positive.")
        max_workers = max(1, min(self.parallel_workers, len(pending_ids)))
        completed = 0
        successful_ids: list[int] = []
        remaining_ids = set(pending_id_set)
        print(
            "[evidence-embedding] "
            f"pending={len(pending_ids)}, workers={max_workers}, "
            f"dimensions={self.dimensions}, "
            f"checkpoint_batch={effective_batch_size}"
        )

        for batch_start in range(0, len(pending_ids), effective_batch_size):
            batch_ids = pending_ids[batch_start : batch_start + effective_batch_size]
            batch_workers = max(1, min(max_workers, len(batch_ids)))
            vectors: dict[int, np.ndarray] = {}
            failures: list[tuple[int, str]] = []
            with ThreadPoolExecutor(max_workers=batch_workers) as executor:
                futures = {
                    executor.submit(
                        self.embed,
                        normalized_statement(nodes[node_id - 1]),
                    ): node_id
                    for node_id in batch_ids
                }
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        vectors[node_id] = future.result()
                    except Exception as exc:
                        failures.append((node_id, str(exc)))
                        if self.logger:
                            self.logger.exception(
                                "Evidence node embedding failed",
                                exc,
                                node_id=node_id,
                            )
                    completed += 1
                    print(
                        "[evidence-embedding] "
                        f"completed {completed}/{len(pending_ids)}: "
                        f"node_id={node_id}"
                    )

            batch_successful_ids = sorted(vectors)
            for node_id in batch_successful_ids:
                matrix[node_id - 1] = vectors[node_id]
            successful_ids.extend(batch_successful_ids)
            remaining_ids.difference_update(batch_successful_ids)
            if on_batch_completed is not None:
                on_batch_completed(
                    matrix,
                    batch_successful_ids,
                    sorted(remaining_ids),
                )

            if failures:
                details = "; ".join(
                    f"node {node_id}: {error}" for node_id, error in sorted(failures)
                )
                raise RuntimeError(
                    f"Embedding failed for {len(failures)} node(s): {details}"
                )

        validate_embedding_matrix(
            matrix,
            expected_rows=len(nodes),
            expected_dimensions=self.dimensions,
        )
        return matrix, successful_ids

    def embed(self, text: str) -> np.ndarray:
        text = clean_model_text(text)
        if not text:
            raise ValueError("Normalized statement is empty.")

        last_error: Exception | None = None
        for attempt in range(1, self.call_attempts + 1):
            prompt_tokens = 0
            success = False
            try:
                # The OpenAI client is safe for concurrent requests. The lock is
                # used only to keep future client replacement isolated.
                with self._client_lock:
                    client = self.client
                response = client.embeddings.create(
                    model=self.model.name,
                    input=text,
                    dimensions=self.dimensions,
                )
                usage = getattr(response, "usage", None)
                prompt_tokens = int(
                    getattr(usage, "prompt_tokens", 0)
                    or getattr(usage, "total_tokens", 0)
                    or 0
                )
                vector = normalize_vector(
                    response.data[0].embedding,
                    expected_dimensions=self.dimensions,
                )
                success = True
                return vector
            except Exception as exc:
                last_error = exc
                if self.logger:
                    details = {
                        "task": "evidence_node_embedding",
                        "provider": self.provider.name,
                        "model": self.model.name,
                        "attempt": attempt,
                        "max_attempts": self.call_attempts,
                    }
                    if attempt < self.call_attempts:
                        self.logger.warning(
                            "Embedding call failed; retrying",
                            error=str(exc),
                            **details,
                        )
                    else:
                        self.logger.exception(
                            "Embedding call failed",
                            exc,
                            **details,
                        )
            finally:
                self.usage_tracker.record(
                    provider=self.provider.name,
                    model=self.model.name,
                    task="evidence_node_embedding",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    success=success,
                    extra={
                        "dimensions": self.dimensions,
                        "attempt": attempt,
                        "max_attempts": self.call_attempts,
                    },
                )
                if self.logger:
                    self.logger.model_call(
                        task="evidence_node_embedding",
                        provider=self.provider.name,
                        model=self.model.name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        success=success,
                        dimensions=self.dimensions,
                        attempt=attempt,
                        max_attempts=self.call_attempts,
                    )
            if attempt < self.call_attempts and self.retry_delay_seconds > 0:
                time.sleep(self.retry_delay_seconds * attempt)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Embedding failed without an exception.")
