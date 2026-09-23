from __future__ import annotations

from src.common.security import redact

from pathlib import Path
from typing import Any

from src.common.files import read_json, utc_now, write_json_atomic

from scripts.experiment.agent.claude_runner import (
    ClaudeAttemptResult,
    ClaudeCodeRunner,
    ClaudeInvocationError,
)
from scripts.experiment.agent.retrieval import AgentRetrievalService
from scripts.experiment.agent.schemas import sanitized_query, validate_prediction
from scripts.experiment.agent.sessions import AgentSessionRegistry
from scripts.experiment.common import (
    ExperimentRuntime,
    usage_for_task_prefix,
)
from scripts.experiment.dataset import sha256_file


class AgentQueryController:
    def __init__(
        self,
        *,
        runtime: ExperimentRuntime,
        retrieval: AgentRetrievalService,
        sessions: AgentSessionRegistry,
        runner: ClaudeCodeRunner,
        maximum_attempts: int,
        maximum_searches: int = 3,
    ) -> None:
        self.runtime = runtime
        self.retrieval = retrieval
        self.sessions = sessions
        self.runner = runner
        self.maximum_attempts = int(maximum_attempts)
        self.maximum_searches = int(maximum_searches)
        if not 1 <= self.maximum_attempts <= 3:
            raise ValueError("Agent maximum_attempts must be between 1 and 3.")
        if self.maximum_searches != 3:
            raise ValueError("Agent maximum_searches must be exactly 3.")

    def process(
        self,
        query: dict[str, Any],
        existing: dict[str, Any] | None,
    ) -> None:
        query_id = int(query["query_id"])
        query_payload = sanitized_query(query)
        query_dir = self.runtime.store.checkpoint_dir / "queries" / f"{query_id:06d}"
        query_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(query_dir / "query.json", query_payload)
        write_json_atomic(
            query_dir / "run_binding.json",
            {
                "input_signature": self.runtime.input_signature,
                "query_id": query_id,
                "run_mode": self.runtime.settings["run_mode"],
            },
        )

        started_at = utc_now()
        trace = dict((existing or {}).get("trace", {}))
        if existing is not None:
            started_at = str(existing.get("started_at", started_at))
        usage_prefix = f"experiment.agent.q{query_id}"

        initial_result = self._load_initial_result(trace, query_dir)
        if initial_result is None:
            initial_result, _ = self.retrieval.original_search(
                query=query_payload,
                output_dir=query_dir,
                usage_prefix=usage_prefix,
            )
            trace["original_search"] = {
                "result_path": "search_01_original.json",
                "result_sha256": sha256_file(
                    query_dir / "search_01_original.json"
                ),
                "audit_path": "search_01_original_retrieval_trace.json",
                "audit_sha256": sha256_file(
                    query_dir / "search_01_original_retrieval_trace.json"
                ),
                "candidate_ids": [
                    int(item["canonical_node_id"])
                    for item in initial_result["candidates"]
                ],
            }
            trace.setdefault("attempts", [])
            self._save_partial(
                query_id=query_id,
                started_at=started_at,
                internal_status="original_retrieval_ready",
                trace=trace,
            )

        attempts: list[dict[str, Any]] = list(trace.get("attempts", []))
        for local_attempt in range(1, self.maximum_attempts + 1):
            attempt_number = len(attempts) + 1
            attempt_dir = query_dir / "attempts" / f"attempt_{attempt_number:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            session = self.sessions.register(
                query=query_payload,
                initial_result=initial_result,
                attempt=attempt_number,
                output_dir=attempt_dir,
                usage_prefix=usage_prefix,
                retrieval=self.retrieval,
                maximum_searches=self.maximum_searches,
            )
            task_prefix = f"{usage_prefix}.agent.attempt_{attempt_number}"
            result: ClaudeAttemptResult | None = None
            try:
                result = self.runner.run(
                    user_payload={
                        "query_evidence_node": query_payload,
                        "retrieval_result": initial_result,
                    },
                    session_token=session.token,
                    task_prefix=task_prefix,
                )
                prediction = validate_prediction(
                    result.final_output,
                    candidate_ids=session.all_candidate_ids(),
                )
                session_snapshot = session.snapshot()
                attempt_record = _attempt_record(
                    attempt=attempt_number,
                    status="completed",
                    result=result,
                    session_snapshot=session_snapshot,
                )
                attempts.append(attempt_record)
                trace["attempts"] = attempts
                write_json_atomic(attempt_dir / "attempt.json", attempt_record)
                self._promote_successful_searches(
                    attempt_dir=attempt_dir,
                    query_dir=query_dir,
                    search_count=int(session_snapshot["search_count"]),
                )
                self._complete_query(
                    query=query,
                    query_payload=query_payload,
                    query_dir=query_dir,
                    started_at=started_at,
                    trace=trace,
                    prediction=prediction,
                    session_snapshot=session_snapshot,
                    final_result=result,
                    usage_prefix=usage_prefix,
                )
                return
            except ClaudeInvocationError as exc:
                session_snapshot = session.snapshot()
                attempt_record = _attempt_record(
                    attempt=attempt_number,
                    status="error",
                    result=exc.result,
                    session_snapshot=session_snapshot,
                    error=exc,
                )
                attempts.append(attempt_record)
                trace["attempts"] = attempts
                write_json_atomic(attempt_dir / "attempt.json", attempt_record)
                terminal = exc.fatal or local_attempt >= self.maximum_attempts
                self.runtime.store.save_query(
                    query_id,
                    {
                        "status": "error" if terminal else "partial",
                        "started_at": started_at,
                        "predicted_canonical_node_id": None,
                        "internal_status": (
                            "fatal_agent_error"
                            if exc.fatal
                            else "retryable_agent_error"
                        ),
                        "error_type": type(exc).__name__,
                        "error_message": redact(str(exc)),
                        "trace": trace,
                    },
                )
                if terminal:
                    raise RuntimeError(
                        f"Agent query {query_id} failed on attempt "
                        f"{local_attempt}/{self.maximum_attempts}: {redact(str(exc))}"
                    ) from exc
            except Exception as exc:
                session_snapshot = session.snapshot()
                fallback = result or ClaudeAttemptResult(
                    final_output=None,
                    usage=None,
                    trace={"tool_events": [], "result": {}},
                    exit_code=None,
                    latency_seconds=0.0,
                    stderr="",
                )
                attempt_record = _attempt_record(
                    attempt=attempt_number,
                    status="error",
                    result=fallback,
                    session_snapshot=session_snapshot,
                    error=exc,
                )
                attempts.append(attempt_record)
                trace["attempts"] = attempts
                write_json_atomic(attempt_dir / "attempt.json", attempt_record)
                terminal = local_attempt >= self.maximum_attempts
                self.runtime.store.save_query(
                    query_id,
                    {
                        "status": "error" if terminal else "partial",
                        "started_at": started_at,
                        "predicted_canonical_node_id": None,
                        "internal_status": "agent_output_validation_error",
                        "error_type": type(exc).__name__,
                        "error_message": redact(str(exc)),
                        "trace": trace,
                    },
                )
                if terminal:
                    raise RuntimeError(
                        f"Agent query {query_id} failed output validation after "
                        f"{self.maximum_attempts} attempts: {redact(str(exc))}"
                    ) from exc
            finally:
                self.sessions.remove(session.token)

        raise AssertionError("Agent retry loop ended without a result.")

    def _load_initial_result(
        self,
        trace: dict[str, Any],
        query_dir: Path,
    ) -> dict[str, Any] | None:
        metadata = trace.get("original_search")
        if not isinstance(metadata, dict):
            return None
        result_path = query_dir / str(metadata.get("result_path", ""))
        audit_path = query_dir / str(metadata.get("audit_path", ""))
        if (
            not result_path.is_file()
            or not audit_path.is_file()
            or sha256_file(result_path) != metadata.get("result_sha256")
            or sha256_file(audit_path) != metadata.get("audit_sha256")
        ):
            raise ValueError("The saved original Agent retrieval is incomplete.")
        result = read_json(result_path)
        if int(result.get("search_id", -1)) != 1:
            raise ValueError("The saved original Agent retrieval is invalid.")
        return result

    def _save_partial(
        self,
        *,
        query_id: int,
        started_at: str,
        internal_status: str,
        trace: dict[str, Any],
    ) -> None:
        self.runtime.store.save_query(
            query_id,
            {
                "status": "partial",
                "started_at": started_at,
                "predicted_canonical_node_id": None,
                "internal_status": internal_status,
                "trace": trace,
            },
        )

    @staticmethod
    def _promote_successful_searches(
        *,
        attempt_dir: Path,
        query_dir: Path,
        search_count: int,
    ) -> None:
        for search_id in range(2, search_count + 1):
            base = f"search_{search_id:02d}_reformulated"
            for suffix in (".json", "_retrieval_trace.json"):
                source = attempt_dir / f"{base}{suffix}"
                if not source.is_file():
                    raise ValueError(
                        f"Successful Agent search file is missing: {source}"
                    )
                write_json_atomic(query_dir / source.name, read_json(source))

    def _complete_query(
        self,
        *,
        query: dict[str, Any],
        query_payload: dict[str, Any],
        query_dir: Path,
        started_at: str,
        trace: dict[str, Any],
        prediction: dict[str, Any],
        session_snapshot: dict[str, Any],
        final_result: ClaudeAttemptResult,
        usage_prefix: str,
    ) -> None:
        query_id = int(query["query_id"])
        search_summaries = [
            {
                "search_id": int(search["search_id"]),
                "search_type": str(search["search_type"]),
                "candidate_count": int(search["candidate_count"]),
                "candidate_ids": [
                    int(item["canonical_node_id"])
                    for item in search.get("candidates", [])
                ],
            }
            for search in session_snapshot["searches"]
        ]
        prediction_record = {
            "query_id": query_id,
            **prediction,
        }
        total_usage = usage_for_task_prefix(
            self.runtime.usage_tracker,
            f"{usage_prefix}.",
        )
        usage_record = {
            "query_id": query_id,
            "run_mode": self.runtime.settings["run_mode"],
            "expected_model": self.runtime.settings["expected_model"],
            "agent_usage": final_result.usage,
            "all_model_usage": total_usage,
            "execution": {
                "total_searches": int(session_snapshot["search_count"]),
                "latency_seconds": float(final_result.latency_seconds),
                "status": "completed",
            },
        }
        structured_trace = {
            "query_id": query_id,
            "attempts": trace.get("attempts", []),
            "searches": search_summaries,
            "final_agent_trace": final_result.trace,
        }
        write_json_atomic(query_dir / "prediction.json", prediction_record)
        write_json_atomic(query_dir / "usage.json", usage_record)
        write_json_atomic(query_dir / "agent_trace.json", structured_trace)

        trace.update(
            {
                "query_directory": str(query_dir),
                "prediction": prediction,
                "searches": search_summaries,
                "total_searches": int(session_snapshot["search_count"]),
                "token_usage": total_usage,
                "retrieval_stages": {
                    f"search_{item['search_id']:02d}": item["candidate_ids"]
                    for item in search_summaries
                },
                "artifacts": {
                    "prediction.json": sha256_file(
                        query_dir / "prediction.json"
                    ),
                    "usage.json": sha256_file(query_dir / "usage.json"),
                    "agent_trace.json": sha256_file(
                        query_dir / "agent_trace.json"
                    ),
                },
            }
        )
        predicted_id = prediction["canonical_node_id"]
        self.runtime.store.save_query(
            query_id,
            {
                "status": "completed",
                "started_at": started_at,
                "completed_at": utc_now(),
                "predicted_canonical_node_id": predicted_id,
                "internal_status": str(prediction["prediction"]).upper(),
                "trace": trace,
            },
        )


def _attempt_record(
    *,
    attempt: int,
    status: str,
    result: ClaudeAttemptResult,
    session_snapshot: dict[str, Any],
    error: BaseException | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt": int(attempt),
        "status": str(status),
        "search_count": int(session_snapshot["search_count"]),
        "latency_seconds": float(result.latency_seconds),
        "exit_code": result.exit_code,
        "usage": result.usage,
        "agent_trace": result.trace,
    }
    if result.stderr:
        record["stderr"] = redact(result.stderr)
    if error is not None:
        record.update(
            {
                "error_type": type(error).__name__,
                "error_message": redact(str(error)),
            }
        )
    return record
