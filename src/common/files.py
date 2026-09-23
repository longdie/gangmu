from __future__ import annotations

import json
import os
import re
import threading
import tempfile
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def read_text(file_path: str | Path) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def write_text(file_path: str | Path, text: str) -> None:
    ensure_parent_dir(file_path).write_text(text, encoding="utf-8")


def read_json(file_path: str | Path, default: Any = None) -> Any:
    path = Path(file_path)
    if not path.exists():
        if default is not None:
            return deepcopy(default)
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(file_path: str | Path, data: Any) -> None:
    path = ensure_parent_dir(file_path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_json_atomic(file_path: str | Path, data: Any) -> None:
    """Atomically replace a JSON file, keeping interruptions from corrupting it."""

    path = ensure_parent_dir(file_path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
            temporary_path = Path(file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def append_jsonl(file_path: str | Path, data: dict[str, Any]) -> None:
    path = ensure_parent_dir(file_path)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class ProjectLogger:
    """Write human-readable logs and structured JSONL events."""

    def __init__(self, log_dir: str | Path, run_name: str = "default"):
        self.log_dir = ensure_dir(log_dir)
        self.run_name = run_name
        self.text_log_path = self.log_dir / f"{run_name}.log"
        self.jsonl_log_path = self.log_dir / f"{run_name}.jsonl"
        self._lock = threading.Lock()
        if not self.text_log_path.exists():
            self.text_log_path.write_text(
                f"--- log created at {utc_now()} ---\n",
                encoding="utf-8",
            )

    def info(self, message: str, **extra: Any) -> None:
        self.event("info", message, extra)

    def warning(self, message: str, **extra: Any) -> None:
        self.event("warning", message, extra)

    def error(self, message: str, **extra: Any) -> None:
        self.event("error", message, extra)

    def exception(self, message: str, exc: BaseException, **extra: Any) -> None:
        details = dict(extra)
        details["exception_type"] = type(exc).__name__
        details["exception"] = str(exc)
        details["traceback"] = traceback.format_exc()
        self.event("exception", message, details)

    def model_call(
        self,
        task: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool,
        **extra: Any,
    ) -> None:
        details = {
            "task": task,
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "success": success,
            **extra,
        }
        self.event("model_call", f"{task} with {model} via {provider}", details)

    def event(
        self, level: str, message: str, extra: dict[str, Any] | None = None
    ) -> None:
        from .security import redact
        message = redact(message)
        details = redact(extra or {})
        timestamp = utc_now()
        record = {
            "time": timestamp,
            "level": level,
            "message": message,
            **details,
        }
        line = f"[{timestamp}] [{level.upper()}] {message}"
        if details:
            line += " " + json.dumps(details, ensure_ascii=False)

        with self._lock:
            with self.text_log_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()
                os.fsync(file.fileno())
            append_jsonl(self.jsonl_log_path, record)


class TokenUsageTracker:
    """Thread-safe token usage aggregation shared by one pipeline run."""

    def __init__(self) -> None:
        self.started_at = utc_now()
        self._by_model: dict[str, dict[str, Any]] = {}
        self._calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

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
        prompt_tokens = self._safe_int(prompt_tokens)
        completion_tokens = self._safe_int(completion_tokens)
        key = f"{provider}:{model}"
        with self._lock:
            model_usage = self._by_model.setdefault(
                key,
                {
                    "provider": provider,
                    "model": model,
                    "calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            model_usage["calls"] += 1
            outcome_key = "successful_calls" if success else "failed_calls"
            model_usage[outcome_key] += 1
            model_usage["prompt_tokens"] += prompt_tokens
            model_usage["completion_tokens"] += completion_tokens
            model_usage["total_tokens"] += prompt_tokens + completion_tokens

            call = {
                "time": utc_now(),
                "provider": provider,
                "model": model,
                "task": task,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "success": success,
            }
            if extra:
                call.update(extra)
            self._calls.append(call)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_model = deepcopy(self._by_model)
            calls = deepcopy(self._calls)
        total_prompt = sum(item["prompt_tokens"] for item in by_model.values())
        total_completion = sum(item["completion_tokens"] for item in by_model.values())
        return {
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "total": {
                "calls": sum(item["calls"] for item in by_model.values()),
                "successful_calls": sum(
                    item["successful_calls"] for item in by_model.values()
                ),
                "failed_calls": sum(item["failed_calls"] for item in by_model.values()),
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            },
            "by_model": by_model,
            "calls": calls,
        }

    def save(
        self,
        file_path: str | Path,
        extra: dict[str, Any] | None = None,
    ) -> None:
        data = self.summary()
        if extra:
            data.update(extra)
        write_json(file_path, data)

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
