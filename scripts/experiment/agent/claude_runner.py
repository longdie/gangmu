from __future__ import annotations

from src.common.security import redact

import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from scripts.experiment.agent.mcp_server import (
    MCP_SERVER_NAME,
    MCP_TOOL_NAME,
    query_mcp_config,
)
from scripts.experiment.agent.usage import (
    AgentModelMismatchError,
    aggregate_model_usage,
    compact_agent_trace,
    parse_stream_json,
    record_agent_usage,
    structured_output,
    validate_expected_model,
)
from scripts.experiment.common import PersistentTokenUsageTracker


@dataclass(frozen=True)
class ClaudeRunnerSettings:
    executable: str
    provider_settings_path: Path
    run_mode: str
    expected_model: str
    effort: str
    timeout_seconds: float
    maximum_budget_usd: float


@dataclass
class ClaudeAttemptResult:
    final_output: dict[str, Any] | None
    usage: dict[str, Any] | None
    trace: dict[str, Any]
    exit_code: int | None
    latency_seconds: float
    stderr: str


class ClaudeInvocationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        result: ClaudeAttemptResult,
        fatal: bool = False,
    ) -> None:
        super().__init__(redact(message))
        self.result = result
        self.fatal = bool(fatal)


class ClaudeCodeRunner:
    def __init__(
        self,
        *,
        settings: ClaudeRunnerSettings,
        system_prompt: str,
        output_schema: dict[str, Any],
        mcp_endpoint: str,
        usage_tracker: PersistentTokenUsageTracker,
    ) -> None:
        self.settings = settings
        self.system_prompt = str(system_prompt)
        self.output_schema = output_schema
        self.mcp_endpoint = str(mcp_endpoint)
        self.usage_tracker = usage_tracker
        self.provider_environment, self.provider_name = (
            _provider_environment(settings.provider_settings_path)
        )
        self._processes: set[subprocess.Popen[str]] = set()
        self._process_lock = threading.RLock()

    def run(
        self,
        *,
        user_payload: dict[str, Any],
        session_token: str,
        task_prefix: str,
    ) -> ClaudeAttemptResult:
        started = time.monotonic()
        stdout = ""
        stderr = ""
        exit_code: int | None = None
        usage: dict[str, Any] | None = None
        events: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="mather-agent-claude-") as raw_dir:
            temporary_dir = Path(raw_dir)
            settings_path = temporary_dir / "settings.json"
            mcp_path = temporary_dir / "mcp.json"
            config_home = temporary_dir / "config"
            config_home.mkdir()
            settings_path.write_text(
                json.dumps(
                    {
                        "model": "sonnet",
                        "effortLevel": self.settings.effort,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            mcp_path.write_text(
                json.dumps(
                    query_mcp_config(
                        endpoint=self.mcp_endpoint,
                        session_token=session_token,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            environment = os.environ.copy()
            environment.pop("ANTHROPIC_API_KEY", None)
            environment.pop("ANTHROPIC_AUTH_TOKEN", None)
            environment.update(self.provider_environment)
            environment.update(
                {
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": self.settings.expected_model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": self.settings.expected_model,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": self.settings.expected_model,
                    "CLAUDE_CONFIG_DIR": str(config_home),
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                }
            )
            # Keep local MCP/transport credentials away from inherited proxies.
            for proxy_key in ("NO_PROXY", "no_proxy"):
                bypass = [item.strip() for item in environment.get(proxy_key, "").split(",") if item.strip()]
                for host in ("127.0.0.1", "localhost", "::1"):
                    if host not in bypass:
                        bypass.append(host)
                environment[proxy_key] = ",".join(bypass)
            tool_name = f"mcp__{MCP_SERVER_NAME}__{MCP_TOOL_NAME}"
            command = [
                self.settings.executable,
                "--print",
                "--bare",
                "--settings",
                str(settings_path),
                "--model",
                "sonnet",
                "--effort",
                self.settings.effort,
                "--mcp-config",
                str(mcp_path),
                "--strict-mcp-config",
                "--system-prompt",
                self.system_prompt,
                "--json-schema",
                json.dumps(
                    self.output_schema,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "--output-format",
                "stream-json",
                "--input-format",
                "text",
                "--verbose",
                "--no-session-persistence",
                "--prompt-suggestions",
                "false",
                "--disable-slash-commands",
                "--no-chrome",
                "--permission-mode",
                "dontAsk",
                "--tools",
                tool_name,
                "--allowedTools",
                tool_name,
                "--max-budget-usd",
                str(self.settings.maximum_budget_usd),
            ]
            user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    command,
                    cwd=temporary_dir,
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                self._register_process(process)
                stdout, stderr = process.communicate(
                    input=user_prompt,
                    timeout=self.settings.timeout_seconds
                )
                exit_code = int(process.returncode)
            except subprocess.TimeoutExpired:
                if process is not None:
                    self._terminate_process(process)
                    stdout, stderr = process.communicate()
                    exit_code = process.returncode
                result = ClaudeAttemptResult(
                    final_output=None,
                    usage=None,
                    trace={"tool_events": [], "result": {}},
                    exit_code=exit_code,
                    latency_seconds=time.monotonic() - started,
                    stderr=_safe_stderr(stderr),
                )
                raise ClaudeInvocationError(
                    "Claude Code query timed out after "
                    f"{self.settings.timeout_seconds:g} seconds.",
                    result=result,
                )
            except KeyboardInterrupt:
                if process is not None:
                    self._terminate_process(process)
                raise
            except OSError as exc:
                result = ClaudeAttemptResult(
                    final_output=None,
                    usage=None,
                    trace={"tool_events": [], "result": {}},
                    exit_code=None,
                    latency_seconds=time.monotonic() - started,
                    stderr="",
                )
                raise ClaudeInvocationError(
                    f"Claude Code process could not start: {exc}",
                    result=result,
                    fatal=isinstance(exc, FileNotFoundError),
                ) from exc
            finally:
                if process is not None:
                    self._unregister_process(process)

        try:
            events = parse_stream_json(stdout)
            trace = compact_agent_trace(events)
        except Exception as exc:
            detail = _safe_stderr(stderr)
            message = f"Claude Code event parsing failed: {exc}"
            if detail:
                message += f" | {detail}"
            result = ClaudeAttemptResult(
                final_output=None,
                usage=None,
                trace={"tool_events": [], "result": {}},
                exit_code=exit_code,
                latency_seconds=time.monotonic() - started,
                stderr=_safe_stderr(stderr),
            )
            raise ClaudeInvocationError(
                message,
                result=result,
                fatal=_is_fatal_configuration_error(detail),
            ) from exc

        try:
            usage = aggregate_model_usage(events)
        except Exception as exc:
            # API failures can be present only in the terminal result event,
            # with no usage or stderr. Preserve that error instead of hiding it.
            detail = _failure_message(exit_code, stderr, events)
            message = f"Claude Code usage parsing failed: {exc}"
            if detail:
                message += f" | {detail}"
            result = ClaudeAttemptResult(
                final_output=None,
                usage=None,
                trace=trace,
                exit_code=exit_code,
                latency_seconds=time.monotonic() - started,
                stderr=_safe_stderr(stderr),
            )
            raise ClaudeInvocationError(
                message,
                result=result,
                fatal=(
                    _is_fatal_provider_error(detail)
                    or _is_fatal_configuration_error(detail)
                ),
            ) from exc

        if exit_code != 0:
            record_agent_usage(
                self.usage_tracker,
                usage=usage,
                provider=self.provider_name,
                task_prefix=task_prefix,
                success=False,
            )
            message = _failure_message(exit_code, stderr, events)
            result = ClaudeAttemptResult(
                final_output=None,
                usage=usage,
                trace=trace,
                exit_code=exit_code,
                latency_seconds=time.monotonic() - started,
                stderr=_safe_stderr(stderr),
            )
            raise ClaudeInvocationError(
                message,
                result=result,
                fatal=_is_fatal_provider_error(message),
            )

        try:
            validate_expected_model(
                usage,
                expected_model=self.settings.expected_model,
            )
            final = structured_output(events)
        except AgentModelMismatchError as exc:
            record_agent_usage(
                self.usage_tracker,
                usage=usage,
                provider=self.provider_name,
                task_prefix=task_prefix,
                success=False,
            )
            result = ClaudeAttemptResult(
                final_output=None,
                usage=usage,
                trace=trace,
                exit_code=exit_code,
                latency_seconds=time.monotonic() - started,
                stderr=_safe_stderr(stderr),
            )
            raise ClaudeInvocationError(
                str(exc),
                result=result,
                fatal=True,
            ) from exc
        except Exception as exc:
            record_agent_usage(
                self.usage_tracker,
                usage=usage,
                provider=self.provider_name,
                task_prefix=task_prefix,
                success=False,
            )
            result = ClaudeAttemptResult(
                final_output=None,
                usage=usage,
                trace=trace,
                exit_code=exit_code,
                latency_seconds=time.monotonic() - started,
                stderr=_safe_stderr(stderr),
            )
            raise ClaudeInvocationError(
                f"Claude Code final output parsing failed: {exc}",
                result=result,
            ) from exc

        record_agent_usage(
            self.usage_tracker,
            usage=usage,
            provider=self.provider_name,
            task_prefix=task_prefix,
            success=True,
        )
        return ClaudeAttemptResult(
            final_output=final,
            usage=usage,
            trace=trace,
            exit_code=exit_code,
            latency_seconds=time.monotonic() - started,
            stderr=_safe_stderr(stderr),
        )

    def terminate_all(self) -> None:
        with self._process_lock:
            processes = list(self._processes)
        for process in processes:
            self._terminate_process(process)

    def _register_process(self, process: subprocess.Popen[str]) -> None:
        with self._process_lock:
            self._processes.add(process)

    def _unregister_process(self, process: subprocess.Popen[str]) -> None:
        with self._process_lock:
            self._processes.discard(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def claude_version(executable: str) -> str:
    result = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    version = result.stdout.strip() or result.stderr.strip()
    if not version:
        raise RuntimeError("Claude Code returned an empty version string.")
    return version


def provider_metadata(settings_path: Path) -> dict[str, str]:
    _, provider_name = _provider_environment(settings_path)
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    environment = data.get("env") if isinstance(data, dict) else None
    base_url = str((environment or {}).get("ANTHROPIC_BASE_URL", "")).rstrip("/")
    return {
        "credential_source": str(settings_path),
        "provider": provider_name,
        "base_url": base_url,
    }


def _provider_environment(settings_path: Path) -> tuple[dict[str, str], str]:
    if not settings_path.is_file():
        raise FileNotFoundError(f"Claude settings do not exist: {settings_path}")
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    environment = data.get("env") if isinstance(data, dict) else None
    if not isinstance(environment, dict):
        raise ValueError("Claude settings contain no env object.")
    base_url = str(environment.get("ANTHROPIC_BASE_URL", "")).strip()
    auth_token = str(environment.get("ANTHROPIC_AUTH_TOKEN", "")).strip()
    api_key = str(environment.get("ANTHROPIC_API_KEY", "")).strip()
    if not base_url:
        raise ValueError("Claude settings contain no ANTHROPIC_BASE_URL.")
    if not auth_token and not api_key:
        raise ValueError("Claude settings contain no Anthropic credential.")
    child = {"ANTHROPIC_BASE_URL": base_url}
    if auth_token:
        child["ANTHROPIC_AUTH_TOKEN"] = auth_token
    if api_key:
        child["ANTHROPIC_API_KEY"] = api_key
    hostname = urlparse(base_url).hostname or "custom-anthropic-provider"
    return child, "agent"


def _failure_message(
    exit_code: int | None,
    stderr: str,
    events: list[dict[str, Any]],
) -> str:
    details: list[str] = []
    if stderr.strip():
        details.append(_safe_stderr(stderr))
    for event in reversed(events):
        if event.get("type") != "result":
            continue
        error = event.get("error") or event.get("result")
        if isinstance(error, str) and error.strip():
            details.append(error.strip()[:2000])
        break
    suffix = " | ".join(details) if details else "no error detail"
    return redact(f"Claude Code exited with code {exit_code}: {suffix}")


def _safe_stderr(value: str) -> str:
    cleaned = redact(str(value or "").strip())
    return cleaned if len(cleaned) <= 4000 else cleaned[-4000:]


def _is_fatal_provider_error(message: str) -> bool:
    lowered = str(message).lower()
    fatal_markers = (
        "invalid api key",
        "authentication_error",
        "unauthorized",
        "forbidden",
        "blacklist",
        "black list",
        "黑名单",
        "insufficient_user_quota",
        "insufficient quota",
        "quota is not enough",
        "额度",
        "pre_consume",
        "model not found",
        "unknown model",
        "does not support tool",
        "tool_use is not supported",
        "upstream model mismatch",
    )
    return any(marker in lowered for marker in fatal_markers)


def _is_fatal_configuration_error(message: str) -> bool:
    lowered = str(message).lower()
    markers = (
        "not a valid json schema",
        "unknown option",
        "invalid value for",
        "settings file",
        "mcp config",
    )
    return any(marker in lowered for marker in markers)
