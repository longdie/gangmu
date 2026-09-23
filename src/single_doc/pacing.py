from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator


class SingleDocumentRequestPacer:
    """Coordinate work/rest windows across concurrent single-document calls."""

    def __init__(
        self,
        *,
        enabled: bool,
        work_seconds: float,
        short_rest_seconds: float,
        windows_per_cycle: int,
        cycle_extra_rest_seconds: float,
        clock: Callable[[], float] = time.monotonic,
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
        self._sleeper = sleeper
        self._announce = announce
        self._condition = threading.Condition()
        self._phase = "active"
        self._window_started_at: float | None = None
        self._active_requests = 0
        self._completed_windows = 0

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        announce: Callable[[str], None] | None = print,
    ) -> "SingleDocumentRequestPacer":
        return cls(
            enabled=config.get_bool("single_doc_pacing.enabled"),
            work_seconds=config.get_float(
                "single_doc_pacing.work_minutes",
                minimum=0,
            )
            * 60.0,
            short_rest_seconds=config.get_float(
                "single_doc_pacing.short_rest_minutes",
                minimum=0,
            )
            * 60.0,
            windows_per_cycle=config.get_int(
                "single_doc_pacing.windows_per_cycle",
                minimum=1,
            ),
            cycle_extra_rest_seconds=config.get_float(
                "single_doc_pacing.cycle_extra_rest_minutes",
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

    def _acquire_request_slot(self) -> None:
        while True:
            rest_seconds: float | None = None
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
                        "[single-doc] pacing: work window ended; "
                        f"waiting for {self._active_requests} in-flight request(s)."
                    )

                if self._phase == "draining":
                    if self._active_requests:
                        self._condition.wait()
                        continue
                    self._completed_windows += 1
                    rest_seconds = self.short_rest_seconds
                    if self._completed_windows % self.windows_per_cycle == 0:
                        rest_seconds += self.cycle_extra_rest_seconds
                    self._phase = "resting"
                    self._emit(
                        "[single-doc] pacing: all in-flight requests completed; "
                        f"resting for {rest_seconds / 60.0:g} minute(s)."
                    )
                elif self._phase == "resting":
                    self._condition.wait()
                    continue

            if rest_seconds is not None:
                self._sleeper(rest_seconds)
                with self._condition:
                    self._phase = "active"
                    self._window_started_at = None
                    self._condition.notify_all()
                next_window = self._completed_windows % self.windows_per_cycle + 1
                self._emit(
                    "[single-doc] pacing: rest completed; "
                    f"starting work window {next_window}/{self.windows_per_cycle}."
                )

    def _release_request_slot(self) -> None:
        with self._condition:
            self._active_requests -= 1
            if self._active_requests < 0:
                self._active_requests = 0
                raise RuntimeError("Single-document pacing request count underflow.")
            if self._phase == "draining" and self._active_requests == 0:
                self._condition.notify_all()

    def _emit(self, message: str) -> None:
        if self._announce is not None:
            self._announce(message)
