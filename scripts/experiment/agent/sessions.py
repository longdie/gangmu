from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.experiment.agent.retrieval import AgentRetrievalService, candidate_ids


@dataclass
class AgentQuerySession:
    token: str
    query: dict[str, Any]
    initial_result: dict[str, Any]
    attempt: int
    output_dir: Path
    usage_prefix: str
    retrieval: AgentRetrievalService
    maximum_searches: int = 3
    searches: list[dict[str, Any]] = field(default_factory=list)
    audit_traces: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        if self.maximum_searches != 3:
            raise ValueError("Agent maximum_searches must be exactly 3.")
        self.searches = [self.initial_result]

    @property
    def search_count(self) -> int:
        return len(self.searches)

    def search_again(self, query_text: str) -> dict[str, Any]:
        with self._lock:
            if self.search_count >= self.maximum_searches:
                raise RuntimeError(
                    "The maximum of three searches has been reached. "
                    "Return a final merge or nil prediction now."
                )
            search_id = self.search_count + 1
            result, audit = self.retrieval.reformulated_search(
                query=self.query,
                query_text=query_text,
                search_id=search_id,
                output_dir=self.output_dir,
                usage_prefix=self.usage_prefix,
            )
            self.searches.append(result)
            self.audit_traces.append(audit)
            return result

    def all_candidate_ids(self) -> set[int]:
        with self._lock:
            return candidate_ids(self.searches)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "attempt": int(self.attempt),
                "search_count": self.search_count,
                "maximum_searches": int(self.maximum_searches),
                "searches": list(self.searches),
                "audit_traces": list(self.audit_traces),
            }


class AgentSessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, AgentQuerySession] = {}
        self._lock = threading.RLock()

    def register(
        self,
        *,
        query: dict[str, Any],
        initial_result: dict[str, Any],
        attempt: int,
        output_dir: Path,
        usage_prefix: str,
        retrieval: AgentRetrievalService,
        maximum_searches: int,
    ) -> AgentQuerySession:
        token = uuid.uuid4().hex
        session = AgentQuerySession(
            token=token,
            query=query,
            initial_result=initial_result,
            attempt=attempt,
            output_dir=output_dir,
            usage_prefix=usage_prefix,
            retrieval=retrieval,
            maximum_searches=maximum_searches,
        )
        with self._lock:
            self._sessions[token] = session
        return session

    def get(self, token: str) -> AgentQuerySession:
        with self._lock:
            session = self._sessions.get(str(token))
        if session is None:
            raise ValueError("The Agent MCP session is invalid or expired.")
        return session

    def remove(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(str(token), None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
