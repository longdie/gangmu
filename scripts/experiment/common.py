from __future__ import annotations

from src.common.security import redact

import hashlib
import json
import re
import threading
import unicodedata
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.common.config import GangmuConfig
from src.common.files import (
    ProjectLogger,
    TokenUsageTracker,
    append_jsonl,
    read_json,
    utc_now,
    write_json_atomic,
)
from src.cross_doc.clients import RerankerClient
from src.single_doc.llm import LLMClient

from scripts.experiment.dataset import ExperimentDataset


TOKEN_PATTERN = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9_]+|[^\s]")


def identity_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"\s+", "", normalized)


def mathematical_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return TOKEN_PATTERN.findall(normalized)


def top_scored_candidates(
    *,
    candidate_ids: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    score_key: str,
) -> list[dict[str, Any]]:
    if candidate_ids.ndim != 1 or scores.ndim != 1:
        raise ValueError("Candidate IDs and scores must be one-dimensional.")
    if candidate_ids.size != scores.size:
        raise ValueError("Candidate ID and score lengths differ.")
    if not candidate_ids.size:
        return []
    if not np.all(np.isfinite(scores)):
        raise ValueError("Candidate scores contain non-finite values.")
    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate IDs must be unique.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    count = min(int(top_k), int(candidate_ids.size))
    if count == int(candidate_ids.size):
        selected = np.arange(candidate_ids.size)
    else:
        boundary_score = float(np.partition(scores, -count)[-count])
        higher = np.flatnonzero(scores > boundary_score)
        tied = np.flatnonzero(scores == boundary_score)
        tied = tied[np.argsort(candidate_ids[tied], kind="stable")]
        selected = np.concatenate((higher, tied[: count - len(higher)]))
    ordered = selected[
        np.lexsort((candidate_ids[selected], -scores[selected]))
    ]
    return [
        {
            "rank": rank,
            "canonical_node_id": int(candidate_ids[index]),
            score_key: float(scores[index]),
        }
        for rank, index in enumerate(ordered.tolist(), start=1)
    ]


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def usage_for_task_prefix(
    tracker: TokenUsageTracker,
    task_prefix: str,
) -> dict[str, int]:
    calls = [
        item
        for item in tracker.summary()["calls"]
        if str(item.get("task", "")).startswith(task_prefix)
    ]
    return {
        "calls": len(calls),
        "successful_calls": sum(bool(item.get("success")) for item in calls),
        "failed_calls": sum(not bool(item.get("success")) for item in calls),
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in calls),
        "completion_tokens": sum(
            int(item.get("completion_tokens", 0)) for item in calls
        ),
        "total_tokens": sum(int(item.get("total_tokens", 0)) for item in calls),
    }


class ExperimentStore:
    def __init__(
        self,
        *,
        output_dir: Path,
        method_name: str,
        input_signature: str,
        overwrite: bool,
        checkpoint_root: Path | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.method_name = method_name
        self.input_signature = input_signature
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_root = checkpoint_root or output_dir / ".checkpoints"
        checkpoint_root = checkpoint_root.resolve()
        self.runtime_root = checkpoint_root
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._active_run_path = checkpoint_root / "active_run.json"
        active_run = (
            read_json(self._active_run_path)
            if self._active_run_path.exists()
            else None
        )
        if (not overwrite and isinstance(active_run, dict)
                and active_run.get("input_signature") != input_signature
                and (output_dir / "predictions.json").exists()):
            raise ValueError("Existing output belongs to different settings or inputs; use a new --output-dir or --overwrite.")
        export_names = ("predictions.json", "metrics.json", "threshold_analysis.json")
        existing_exports = [output_dir / name for name in export_names if (output_dir / name).is_file()]
        if not overwrite and not isinstance(active_run, dict) and existing_exports:
            raise ValueError("Existing exports have no matching local checkpoint; use a new --output-dir or --overwrite.")
        resume_active = bool(
            not overwrite
            and isinstance(active_run, dict)
            and active_run.get("method") == method_name
            and active_run.get("input_signature") == input_signature
            and (
                active_run.get("status") == "running"
                or (not overwrite and active_run.get("status") == "completed")
            )
            and isinstance(active_run.get("checkpoint_directory"), str)
            and (checkpoint_root / active_run["checkpoint_directory"]).is_dir()
        )
        if resume_active:
            self.run_id = str(active_run["run_id"])
            self.checkpoint_dir = checkpoint_root / str(
                active_run["checkpoint_directory"]
            )
        else:
            self.run_id = (
                re.sub(r"[^0-9A-Za-z]+", "_", utc_now()) + "_" + uuid.uuid4().hex[:8]
            )
            self.checkpoint_dir = checkpoint_root / input_signature / f"run_{self.run_id}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if overwrite and existing_exports:
            archive = self.checkpoint_dir / "previous_exports"
            archive.mkdir(parents=True, exist_ok=True)
            for exported in existing_exports:
                exported.replace(archive / exported.name)


        checkpoint_manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
        checkpoint_manifest = (
            read_json(checkpoint_manifest_path)
            if checkpoint_manifest_path.exists()
            else {
                "schema_version": "1.1",
                "method": method_name,
                "input_signature": input_signature,
                "run_id": self.run_id,
                "started_at": utc_now(),
            }
        )
        if (
            checkpoint_manifest.get("method") != method_name
            or checkpoint_manifest.get("input_signature") != input_signature
        ):
            raise ValueError("Experiment checkpoint metadata is incompatible.")
        self.started_at = str(checkpoint_manifest.get("started_at") or utc_now())
        checkpoint_manifest["started_at"] = self.started_at
        checkpoint_manifest.update({"status": "running", "updated_at": utc_now()})
        write_json_atomic(checkpoint_manifest_path, checkpoint_manifest)
        write_json_atomic(
            self._active_run_path,
            {
                "schema_version": "1.0",
                "method": method_name,
                "input_signature": input_signature,
                "run_id": self.run_id,
                "status": "running",
                "checkpoint_directory": str(
                    self.checkpoint_dir.relative_to(checkpoint_root)
                ),
                "updated_at": utc_now(),
            },
        )

    @property
    def usage_journal_path(self) -> Path:
        return self.checkpoint_dir / "token_usage.jsonl"

    def load_query(self, query_id: int) -> dict[str, Any] | None:
        path = self._query_path(query_id)
        if not path.exists():
            return None
        record = read_json(path)
        if record.get("input_signature") != self.input_signature:
            return None
        return record

    def save_query(self, query_id: int, record: dict[str, Any]) -> None:
        payload = redact(dict(record))
        payload["query_id"] = int(query_id)
        payload["input_signature"] = self.input_signature
        payload["updated_at"] = utc_now()
        with self._lock:
            write_json_atomic(self._query_path(query_id), payload)

    def completed_records(
        self,
        queries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for query in queries:
            query_id = int(query["query_id"])
            record = self.load_query(query_id)
            if record is None or record.get("status") != "completed":
                raise RuntimeError(
                    f"Method {self.method_name} did not complete query {query_id}."
                )
            records.append(record)
        return records

    def mark_complete(self) -> None:
        with self._lock:
            manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
            manifest = read_json(manifest_path)
            manifest.update({"status": "completed", "updated_at": utc_now()})
            write_json_atomic(manifest_path, manifest)
            active = read_json(self._active_run_path)
            if (
                active.get("input_signature") == self.input_signature
                and active.get("run_id") == self.run_id
            ):
                active.update({"status": "completed", "updated_at": utc_now()})
                write_json_atomic(self._active_run_path, active)

    def _query_path(self, query_id: int) -> Path:
        return self.checkpoint_dir / f"query_{int(query_id):06d}.json"


class PersistentTokenUsageTracker(TokenUsageTracker):
    """Persist every model attempt so resumed runs retain complete usage."""

    def __init__(self, journal_path: Path) -> None:
        super().__init__()
        self.journal_path = journal_path
        self._journal_lock = threading.Lock()
        self._persistence_errors: list[str] = []
        if journal_path.exists():
            for line_number, line in enumerate(
                journal_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Token usage checkpoint is invalid at line "
                        f"{line_number}: {journal_path}"
                    ) from exc
                self._record_without_persisting(record)

    def record(
        self,
        *,
        provider: str,
        model: str,
        task: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "provider": provider,
            "model": model,
            "task": task,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "success": bool(success),
            "extra": dict(extra or {}),
        }
        self._record_without_persisting(payload)
        with self._journal_lock:
            try:
                append_jsonl(self.journal_path, redact(payload))
            except Exception as exc:
                # Usage persistence must never turn an already successful paid
                # provider call into a retry.  Surface the failure in the final
                # manifest instead.
                self._persistence_errors.append(f"{type(exc).__name__}: {exc}")

    def _record_without_persisting(self, payload: dict[str, Any]) -> None:
        super().record(
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            task=str(payload["task"]),
            prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
            completion_tokens=int(payload.get("completion_tokens", 0) or 0),
            success=bool(payload.get("success", True)),
            extra=dict(payload.get("extra") or {}),
        )

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        with self._journal_lock:
            errors = list(self._persistence_errors)
        result["persistence"] = {
            "journal_path": str(self.journal_path),
            "complete": not errors,
            "errors": errors,
        }
        return result


@dataclass
class ExperimentRuntime:
    dataset: ExperimentDataset
    queries: list[dict[str, Any]]
    config: GangmuConfig
    method_name: str
    display_name: str
    output_dir: Path
    settings: dict[str, Any]
    input_signature: str
    store: ExperimentStore
    logger: ProjectLogger
    usage_tracker: TokenUsageTracker
    options: dict[str, Any]
    started_at: str

    _llm_client: LLMClient | None = None
    _reranker_client: RerankerClient | None = None

    def llm_client(self, *, request_pacer: Any | None = None) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient(
                config=self.config,
                logger=self.logger,
                usage_tracker=self.usage_tracker,
                consecutive_failures_before_abort=self.config.get_int(
                    "cross_document.provider_failures_before_abort",
                    minimum=1,
                ),
                call_attempts=self.config.get_int(
                    "cross_document.llm_call_attempts",
                    minimum=1,
                ),
                retry_delay_seconds=self.config.get_float(
                    "cross_document.llm_retry_delay_seconds",
                    minimum=0,
                ),
                request_pacer=request_pacer,
            )
        elif (
            request_pacer is not None
            and self._llm_client.request_pacer is not request_pacer
        ):
            raise RuntimeError(
                "Experiment LLM client was already created with another pacer."
            )
        return self._llm_client

    def reranker_client(self) -> RerankerClient:
        if self._reranker_client is None:
            self._reranker_client = RerankerClient(
                config=self.config,
                logger=self.logger,
                usage_tracker=self.usage_tracker,
            )
        return self._reranker_client


def run_parallel_queries(
    runtime: ExperimentRuntime,
    worker: Callable[[dict[str, Any], dict[str, Any] | None], None],
    *,
    workers: int,
) -> None:
    pending_queries = [
        query
        for query in runtime.queries
        if (runtime.store.load_query(int(query["query_id"])) or {}).get("status")
        != "completed"
    ]
    completed_before = len(runtime.queries) - len(pending_queries)
    if not pending_queries:
        print(
            f"[experiment:{runtime.method_name}] checkpoint already complete: "
            f"{completed_before}/{len(runtime.queries)}"
        )
        return
    print(
        f"[experiment:{runtime.method_name}] pending={len(pending_queries)}, "
        f"workers={workers}, resumed={completed_before}"
    )

    iterator = iter(pending_queries)
    future_to_query: dict[Future[None], dict[str, Any]] = {}
    completed = completed_before
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        for _ in range(min(workers, len(pending_queries))):
            query = next(iterator)
            future = executor.submit(
                worker,
                query,
                runtime.store.load_query(int(query["query_id"])),
            )
            future_to_query[future] = query

        while future_to_query:
            done, _ = wait(future_to_query, return_when=FIRST_COMPLETED)
            for future in done:
                query = future_to_query.pop(future)
                try:
                    future.result()
                except BaseException:
                    for remaining in future_to_query:
                        remaining.cancel()
                    raise
                completed += 1
                if completed % 10 == 0 or completed == len(runtime.queries):
                    print(
                        f"[experiment:{runtime.method_name}] completed "
                        f"{completed}/{len(runtime.queries)}"
                    )
                try:
                    next_query = next(iterator)
                except StopIteration:
                    continue
                next_future = executor.submit(
                    worker,
                    next_query,
                    runtime.store.load_query(int(next_query["query_id"])),
                )
                future_to_query[next_future] = next_query
