from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator


PACING_STATE_VERSION = 1


class CrossDocumentRequestPacer:
    """Coordinate recoverable work/rest windows for cross-document LLM calls."""

    def __init__(
        self,
        *,
        enabled: bool,
        work_seconds: float,
        short_rest_seconds: float,
        windows_per_cycle: int,
        cycle_extra_rest_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        announce: Callable[[str], None] | None = print,
    ) -> None:
        if work_seconds <= 0:
            raise ValueError("work_seconds must be positive.")
        if short_rest_seconds < 0 or cycle_extra_rest_seconds < 0:
            raise ValueError("Rest durations must be nonnegative.")
        if windows_per_cycle < 1:
            raise ValueError("windows_per_cycle must be positive.")
        self.enabled = bool(enabled)
        self.work_seconds = float(work_seconds)
        self.short_rest_seconds = float(short_rest_seconds)
        self.windows_per_cycle = int(windows_per_cycle)
        self.cycle_extra_rest_seconds = float(cycle_extra_rest_seconds)
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleeper = sleeper
        self._announce = announce
        self._condition = threading.Condition()
        self._phase = "active"
        self._window_started_at: float | None = None
        self._active_requests = 0
        self._completed_windows = 0
        self._rest_until: float | None = None
        self._rest_leader = False

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        announce: Callable[[str], None] | None = print,
    ) -> "CrossDocumentRequestPacer":
        return cls(
            enabled=config.get_bool("cross_document_pacing.enabled"),
            work_seconds=config.get_float(
                "cross_document_pacing.work_minutes",
                minimum=0,
            )
            * 60.0,
            short_rest_seconds=config.get_float(
                "cross_document_pacing.short_rest_minutes",
                minimum=0,
            )
            * 60.0,
            windows_per_cycle=config.get_int(
                "cross_document_pacing.windows_per_cycle",
                minimum=1,
            ),
            cycle_extra_rest_seconds=config.get_float(
                "cross_document_pacing.cycle_extra_rest_minutes",
                minimum=0,
            )
            * 60.0,
            announce=announce,
        )

    @contextmanager
    def request_slot(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._acquire_request_slot()
        try:
            yield
        finally:
            self._release_request_slot()

    def snapshot_state(self) -> dict[str, Any]:
        """Return process-independent pacing state for a graph checkpoint."""
        with self._condition:
            now = self._clock()
            if self._phase == "active" and self._window_started_at is not None:
                elapsed = max(0.0, now - self._window_started_at)
            elif self._phase == "draining":
                elapsed = self.work_seconds
            else:
                elapsed = 0.0

            rest_until_unix: float | None = None
            if self._phase == "resting" and self._rest_until is not None:
                remaining = max(0.0, self._rest_until - now)
                rest_until_unix = self._wall_clock() + remaining
            elif self._phase == "draining":
                # A restarted process has no in-flight requests. It should
                # therefore enter the rest belonging to this completed window.
                rest_until_unix = self._wall_clock() + self._rest_seconds_for(
                    self._completed_windows + 1
                )

            return {
                "state_version": PACING_STATE_VERSION,
                "phase": self._phase,
                "completed_windows": self._completed_windows,
                "active_window_elapsed_seconds": elapsed,
                "rest_until_unix": rest_until_unix,
            }

    def restore_state(self, state: dict[str, Any] | None) -> None:
        """Restore a checkpoint without counting process downtime as work."""
        if state is None:
            return
        normalized = self.validated_state(state)
        with self._condition:
            if self._active_requests:
                raise RuntimeError("Cannot restore pacing with active requests.")
            self._phase = "active"
            self._window_started_at = None
            self._completed_windows = int(normalized["completed_windows"])
            self._rest_until = None
            self._rest_leader = False
            if not self.enabled:
                return

            phase = str(normalized["phase"])
            if phase == "active":
                elapsed = min(
                    float(normalized["active_window_elapsed_seconds"]),
                    self.work_seconds,
                )
                if elapsed > 0:
                    self._window_started_at = self._clock() - elapsed
                    self._emit(
                        "[canonical-graph] pacing: resumed active work window "
                        f"with {elapsed / 60.0:g} minute(s) already elapsed."
                    )
                return

            if phase == "draining":
                self._completed_windows += 1
            rest_until_unix = normalized["rest_until_unix"]
            remaining = (
                max(0.0, float(rest_until_unix) - self._wall_clock())
                if rest_until_unix is not None
                else 0.0
            )
            if remaining <= 0:
                return
            self._phase = "resting"
            self._rest_until = self._clock() + remaining
            self._emit(
                "[canonical-graph] pacing: resumed checkpoint rest; "
                f"{remaining / 60.0:g} minute(s) remaining."
            )

    @staticmethod
    def validated_state(state: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(state, dict):
            raise ValueError("Cross-document pacing state must be an object.")
        if state.get("state_version") != PACING_STATE_VERSION:
            raise ValueError("Unsupported cross-document pacing state version.")
        phase = state.get("phase")
        if phase not in {"active", "draining", "resting"}:
            raise ValueError("Cross-document pacing phase is invalid.")
        completed = state.get("completed_windows")
        elapsed = state.get("active_window_elapsed_seconds")
        rest_until = state.get("rest_until_unix")
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError("Cross-document completed window count is invalid.")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or float(elapsed) < 0
        ):
            raise ValueError("Cross-document active window elapsed time is invalid.")
        if rest_until is not None and (
            isinstance(rest_until, bool)
            or not isinstance(rest_until, (int, float))
            or float(rest_until) < 0
        ):
            raise ValueError("Cross-document rest deadline is invalid.")
        if phase == "active" and rest_until is not None:
            raise ValueError("Active pacing state cannot contain a rest deadline.")
        if phase in {"draining", "resting"} and rest_until is None:
            raise ValueError("Resting pacing state requires a rest deadline.")
        return {
            "state_version": PACING_STATE_VERSION,
            "phase": phase,
            "completed_windows": completed,
            "active_window_elapsed_seconds": float(elapsed),
            "rest_until_unix": (
                None if rest_until is None else float(rest_until)
            ),
        }

    def _acquire_request_slot(self) -> None:
        while True:
            sleep_seconds: float | None = None
            with self._condition:
                now = self._clock()
                if self._phase == "active":
                    if self._window_started_at is None:
                        self._window_started_at = now
                    elapsed = now - self._window_started_at
                    if elapsed < self.work_seconds:
                        self._active_requests += 1
                        return
                    self._phase = "draining"
                    self._emit(
                        "[canonical-graph] pacing: work window ended; "
                        f"waiting for {self._active_requests} in-flight "
                        "GPT request(s)."
                    )

                if self._phase == "draining":
                    if self._active_requests:
                        self._condition.wait()
                        continue
                    self._completed_windows += 1
                    rest_seconds = self._rest_seconds_for(self._completed_windows)
                    self._phase = "resting"
                    self._rest_until = self._clock() + rest_seconds
                    self._emit(
                        "[canonical-graph] pacing: all in-flight GPT requests "
                        f"completed; resting for {rest_seconds / 60.0:g} minute(s)."
                    )
                    self._condition.notify_all()

                if self._phase == "resting":
                    if self._rest_until is None:
                        raise RuntimeError("Pacing rest deadline is missing.")
                    remaining = self._rest_until - self._clock()
                    if remaining <= 0:
                        self._finish_rest_locked()
                        continue
                    if self._rest_leader:
                        self._condition.wait()
                        continue
                    self._rest_leader = True
                    sleep_seconds = remaining

            if sleep_seconds is not None:
                try:
                    self._sleeper(sleep_seconds)
                finally:
                    with self._condition:
                        self._rest_leader = False
                        if (
                            self._phase == "resting"
                            and self._rest_until is not None
                            and self._clock() >= self._rest_until
                        ):
                            self._finish_rest_locked()
                        self._condition.notify_all()

    def _release_request_slot(self) -> None:
        with self._condition:
            self._active_requests -= 1
            if self._active_requests < 0:
                self._active_requests = 0
                raise RuntimeError("Cross-document pacing request count underflow.")
            if self._phase == "draining" and self._active_requests == 0:
                self._condition.notify_all()

    def _finish_rest_locked(self) -> None:
        self._phase = "active"
        self._window_started_at = None
        self._rest_until = None
        next_window = self._completed_windows % self.windows_per_cycle + 1
        self._emit(
            "[canonical-graph] pacing: rest completed; "
            f"starting work window {next_window}/{self.windows_per_cycle}."
        )
        self._condition.notify_all()

    def _rest_seconds_for(self, completed_windows: int) -> float:
        rest_seconds = self.short_rest_seconds
        if completed_windows % self.windows_per_cycle == 0:
            rest_seconds += self.cycle_extra_rest_seconds
        return rest_seconds

    def _emit(self, message: str) -> None:
        if self._announce is not None:
            self._announce(message)
