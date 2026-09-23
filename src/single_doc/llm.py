from __future__ import annotations

import json
import re
import threading
import time
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Protocol

from src.common.config import GangmuConfig, get_config
from src.common.files import ProjectLogger, TokenUsageTracker, read_text
from src.common.paths import PROMPT_DIR


JSONValidator = Callable[[dict[str, Any]], tuple[bool, str]]


class RequestPacer(Protocol):
    def request_slot(self) -> ContextManager[None]: ...


class ProviderCircuitOpenError(RuntimeError):
    """Raised after repeated provider failures make the current run unsafe."""


class ProviderRequestError(RuntimeError):
    """Raised when a provider request exhausts its retries without a response."""


class EmptyModelResponseError(RuntimeError):
    """Raised when a provider reports success without model text."""


class LLMClient:
    """LLM client supporting provider-specific protocols from Gangmu config."""

    def __init__(
        self,
        *,
        config: GangmuConfig | None = None,
        logger: ProjectLogger | None = None,
        usage_tracker: TokenUsageTracker | None = None,
        consecutive_failures_before_abort: int | None = None,
        call_attempts: int | None = None,
        retry_delay_seconds: float | None = None,
        request_pacer: RequestPacer | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OpenAI SDK is required. Install dependencies with pip install -r requirements.txt."
            ) from exc
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "HTTPX is required. Install dependencies with pip install -r requirements.txt."
            ) from exc

        self.config = config or get_config()
        self.logger = logger
        self.usage_tracker = usage_tracker or TokenUsageTracker()
        self.timeout_seconds = self.config.get_float(
            "requests.timeout_seconds",
            minimum=0,
        )
        self.connect_timeout_seconds = self.config.get_float(
            "requests.connect_timeout_seconds",
            minimum=0,
        )
        self.pool_timeout_seconds = self.config.get_float(
            "requests.pool_timeout_seconds",
            minimum=0,
        )
        self.max_connections = self.config.get_int(
            "requests.max_connections",
            minimum=1,
        )
        self.max_keepalive_connections = self.config.get_int(
            "requests.max_keepalive_connections",
            minimum=0,
        )
        if consecutive_failures_before_abort is None:
            self.consecutive_failures_before_abort = self.config.get_int(
                "requests.consecutive_failures_before_abort",
                minimum=1,
            )
        elif (
            isinstance(consecutive_failures_before_abort, bool)
            or not isinstance(consecutive_failures_before_abort, int)
            or consecutive_failures_before_abort < 1
        ):
            raise ValueError(
                "consecutive_failures_before_abort must be a positive integer."
            )
        else:
            self.consecutive_failures_before_abort = consecutive_failures_before_abort
        self.max_completion_tokens = self.config.get_int(
            "requests.max_completion_tokens",
            minimum=1,
        )
        if call_attempts is None:
            self.call_attempts = self.config.get_int(
                "requests.call_attempts",
                minimum=1,
            )
        elif (
            isinstance(call_attempts, bool)
            or not isinstance(call_attempts, int)
            or call_attempts < 1
        ):
            raise ValueError("call_attempts must be a positive integer.")
        else:
            self.call_attempts = call_attempts
        self.json_repair_attempts = self.config.get_int(
            "requests.json_repair_attempts",
            minimum=0,
        )
        self.concurrency_retry_delay_seconds = self.config.get_float(
            "requests.concurrency_retry_delay_seconds",
            minimum=0,
        )
        if retry_delay_seconds is None:
            self.retry_delay_seconds = self.config.get_float(
                "requests.repair_retry_delay_seconds",
                minimum=0,
            )
        elif (
            isinstance(retry_delay_seconds, bool)
            or not isinstance(retry_delay_seconds, (int, float))
            or retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be nonnegative.")
        else:
            self.retry_delay_seconds = float(retry_delay_seconds)
        self.request_pacer = request_pacer
        self._openai_cls = OpenAI
        self._httpx = httpx
        self._clients: dict[tuple[str, str, str], Any] = {}
        self._message_clients: dict[tuple[str, str, str, bool], Any] = {}
        self._retired_message_clients: list[Any] = []
        self._client_lock = threading.Lock()
        self._provider_failures: dict[str, int] = {}
        self._open_provider_circuits: dict[str, str] = {}
        self._provider_cooldowns: dict[str, float] = {}
        self._circuit_lock = threading.Lock()

    def call(
        self,
        *,
        model_task: str,
        prompt: str,
        input_text: str = "",
        usage_task: str | None = None,
        json_mode: bool = False,
    ) -> str:
        model = self.config.model(model_task)
        provider = self.config.provider(model.provider, require_api_key=True)
        content = prompt if not input_text else f"{prompt.rstrip()}\n\n{input_text}"
        request: dict[str, Any] = {
            "model": model.name,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": self.max_completion_tokens,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        task_name = usage_task or model_task
        last_error: Exception | None = None
        for attempt in range(1, self.call_attempts + 1):
            self._raise_if_circuit_open(provider.name)
            self._wait_for_provider_cooldown(provider.name)
            self._raise_if_circuit_open(provider.name)
            prompt_tokens = 0
            completion_tokens = 0
            success = False
            circuit_error: ProviderCircuitOpenError | None = None
            message_client: Any = None
            message_cache_key: tuple[str, str, str, bool] | None = None
            try:
                request_slot = (
                    self.request_pacer.request_slot()
                    if self.request_pacer is not None
                    else nullcontext()
                )
                with request_slot:
                    if provider.protocol == "openai_chat_completions":
                        client = self._get_client(
                            provider_name=provider.name,
                            api_key=provider.api_key,
                            base_url=provider.base_url,
                        )
                        completion = client.chat.completions.create(**request)
                        response = completion.choices[0].message.content or ""
                        usage = getattr(completion, "usage", None)
                        prompt_tokens = int(
                            getattr(usage, "prompt_tokens", 0) or 0
                        )
                        completion_tokens = int(
                            getattr(usage, "completion_tokens", 0) or 0
                        )
                    elif provider.protocol == "claude_messages":
                        message_cache_key = (
                            provider.name,
                            provider.api_key,
                            provider.base_url,
                            provider.trust_env,
                        )
                        client = self._get_message_client(
                            provider_name=provider.name,
                            api_key=provider.api_key,
                            base_url=provider.base_url,
                            trust_env=provider.trust_env,
                        )
                        message_client = client
                        (
                            response,
                            prompt_tokens,
                            completion_tokens,
                        ) = self._call_claude_messages(
                            client=client,
                            base_url=provider.base_url,
                            model_name=model.name,
                            content=content,
                        )
                    else:
                        raise RuntimeError(
                            f"Unsupported provider protocol: {provider.protocol}"
                        )
                if not isinstance(response, str) or not response.strip():
                    raise EmptyModelResponseError(
                        f"Provider '{provider.name}' returned an empty "
                        f"response for model '{model.name}'."
                    )
                self._record_provider_success(provider.name)
                success = True
                return response
            except ProviderCircuitOpenError:
                raise
            except Exception as exc:
                concurrency_limited = self._is_concurrency_limit_error(exc)
                if concurrency_limited:
                    self._start_provider_cooldown(provider.name)
                if (
                    message_cache_key is not None
                    and message_client is not None
                    and (
                        self._is_transport_error(exc)
                        or isinstance(exc, EmptyModelResponseError)
                    )
                ):
                    self._discard_message_client(
                        message_cache_key,
                        message_client,
                    )
                circuit_error = self._record_provider_failure(
                    provider.name,
                    exc,
                )
                last_error = circuit_error or exc
                if self.logger:
                    details = {
                        "task": task_name,
                        "provider": provider.name,
                        "model": model.name,
                        "attempt": attempt,
                        "max_attempts": self.call_attempts,
                    }
                    if concurrency_limited:
                        details["retry_after_seconds"] = (
                            self.concurrency_retry_delay_seconds
                        )
                    if circuit_error is not None:
                        self.logger.error(
                            "LLM provider circuit opened; aborting current run",
                            error=str(exc),
                            consecutive_failures=self.consecutive_failures_before_abort,
                            **details,
                        )
                    elif attempt < self.call_attempts:
                        self.logger.warning(
                            "LLM call failed; retrying",
                            error=str(exc),
                            **details,
                        )
                    else:
                        self.logger.exception("LLM call failed", exc, **details)
            finally:
                self.usage_tracker.record(
                    provider=provider.name,
                    model=model.name,
                    task=task_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    success=success,
                    extra={
                        "json_mode": json_mode,
                        "attempt": attempt,
                        "max_attempts": self.call_attempts,
                    },
                )
                if self.logger:
                    self.logger.model_call(
                        task=task_name,
                        provider=provider.name,
                        model=model.name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        success=success,
                        json_mode=json_mode,
                        attempt=attempt,
                        max_attempts=self.call_attempts,
                    )
            if circuit_error is not None:
                raise circuit_error
            if attempt < self.call_attempts:
                self._sleep(self.retry_delay_seconds * attempt)

        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM call failed without an exception.")

    def call_json(
        self,
        *,
        model_task: str,
        prompt: str,
        input_text: str,
        usage_task: str,
        validator: JSONValidator | None = None,
        fallback_model_task: str | None = None,
        preserve_latex_escapes: bool = False,
        repair_instructions: str = "",
    ) -> dict[str, Any] | None:
        raw_output = ""
        try:
            raw_output = self.call(
                model_task=model_task,
                prompt=prompt,
                input_text=input_text,
                usage_task=usage_task,
                json_mode=True,
            )
        except ProviderCircuitOpenError:
            raise
        except Exception as exc:
            raise ProviderRequestError(
                f"Provider request failed before JSON validation: {exc}"
            ) from exc

        try:
            return self._parse_and_validate(
                raw_output,
                validator,
                preserve_latex_escapes=preserve_latex_escapes,
            )
        except Exception as exc:
            if self.logger:
                self.logger.warning(
                    "Initial JSON output was invalid",
                    task=usage_task,
                    error=str(exc),
                )

        repaired = self._repair_json(
            raw_output=raw_output,
            model_task=model_task,
            usage_task=f"{usage_task}.repair",
            validator=validator,
            preserve_latex_escapes=preserve_latex_escapes,
            repair_instructions=repair_instructions,
        )
        if repaired is not None:
            return repaired

        if fallback_model_task:
            for attempt in range(1, self.json_repair_attempts + 1):
                self._sleep(self.retry_delay_seconds)
                try:
                    fallback_output = self.call(
                        model_task=fallback_model_task,
                        prompt=prompt,
                        input_text=input_text,
                        usage_task=f"{usage_task}.fallback",
                        json_mode=True,
                    )
                except ProviderCircuitOpenError:
                    raise
                except Exception as exc:
                    raise ProviderRequestError(
                        "Provider request failed during fallback JSON "
                        f"generation: {exc}"
                    ) from exc
                try:
                    return self._parse_and_validate(
                        fallback_output,
                        validator,
                        preserve_latex_escapes=preserve_latex_escapes,
                    )
                except Exception as exc:
                    if self.logger:
                        self.logger.warning(
                            "Fallback JSON generation failed",
                            task=usage_task,
                            attempt=attempt,
                            error=str(exc),
                        )
        return None

    def _repair_json(
        self,
        *,
        raw_output: str,
        model_task: str,
        usage_task: str,
        validator: JSONValidator | None,
        preserve_latex_escapes: bool,
        repair_instructions: str,
    ) -> dict[str, Any] | None:
        if not raw_output:
            return None
        repair_prompt = (
            read_text(PROMPT_DIR / "json_repair.md")
        )
        repair_input_parts = []
        if repair_instructions.strip():
            repair_input_parts.extend(
                [
                    "Validation requirements:",
                    repair_instructions.strip(),
                    "",
                ]
            )
        repair_input_parts.extend(
            [
                "Invalid model output:",
                raw_output,
            ]
        )
        repair_input = "\n".join(repair_input_parts)
        for attempt in range(1, self.json_repair_attempts + 1):
            self._sleep(self.retry_delay_seconds)
            try:
                repaired_output = self.call(
                    model_task=model_task,
                    prompt=repair_prompt,
                    input_text=repair_input,
                    usage_task=usage_task,
                    json_mode=True,
                )
            except ProviderCircuitOpenError:
                raise
            except Exception as exc:
                raise ProviderRequestError(
                    f"Provider request failed during JSON repair: {exc}"
                ) from exc
            try:
                return self._parse_and_validate(
                    repaired_output,
                    validator,
                    preserve_latex_escapes=preserve_latex_escapes,
                )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        "JSON repair failed",
                        task=usage_task,
                        attempt=attempt,
                        error=str(exc),
                    )
        return None

    def _get_client(
        self,
        *,
        provider_name: str,
        api_key: str,
        base_url: str,
    ) -> Any:
        cache_key = (provider_name, api_key, base_url)
        with self._client_lock:
            client = self._clients.get(cache_key)
            if client is not None:
                return client
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": self.timeout_seconds,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            client = self._openai_cls(**client_kwargs)
            self._clients[cache_key] = client
            return client

    def _get_message_client(
        self,
        *,
        provider_name: str,
        api_key: str,
        base_url: str,
        trust_env: bool,
    ) -> Any:
        if not base_url:
            raise ValueError(
                f"Provider '{provider_name}' requires a Claude Messages URL."
            )
        cache_key = (provider_name, api_key, base_url, trust_env)
        with self._client_lock:
            client = self._message_clients.get(cache_key)
            if client is not None:
                return client
            client = self._httpx.Client(
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=self._httpx.Timeout(
                    self.timeout_seconds,
                    connect=self.connect_timeout_seconds,
                    pool=self.pool_timeout_seconds,
                ),
                limits=self._httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_keepalive_connections,
                ),
                trust_env=trust_env,
            )
            self._message_clients[cache_key] = client
            return client

    def _discard_message_client(
        self,
        cache_key: tuple[str, str, str, bool],
        client: Any,
    ) -> None:
        with self._client_lock:
            if self._message_clients.get(cache_key) is client:
                self._message_clients.pop(cache_key, None)
                # Claude Messages calls may share this HTTPX client across
                # workers. Do not close it while sibling requests are still
                # in flight; a fresh client will serve subsequent attempts.
                self._retired_message_clients.append(client)

    def _is_transport_error(self, exc: Exception) -> bool:
        return isinstance(exc, self._httpx.TransportError)

    @staticmethod
    def _is_concurrency_limit_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "http 430" in message
            or "concurrency" in message and "limit" in message
            or "并发数已达到上限" in message
        )

    def _start_provider_cooldown(self, provider_name: str) -> None:
        if self.concurrency_retry_delay_seconds <= 0:
            return
        deadline = time.monotonic() + self.concurrency_retry_delay_seconds
        with self._circuit_lock:
            self._provider_cooldowns[provider_name] = max(
                deadline,
                self._provider_cooldowns.get(provider_name, 0.0),
            )

    def _wait_for_provider_cooldown(self, provider_name: str) -> None:
        while True:
            with self._circuit_lock:
                deadline = self._provider_cooldowns.get(provider_name, 0.0)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._sleep(remaining)

    def _record_provider_success(self, provider_name: str) -> None:
        with self._circuit_lock:
            if provider_name not in self._open_provider_circuits:
                self._provider_failures[provider_name] = 0

    def _record_provider_failure(
        self,
        provider_name: str,
        exc: Exception,
    ) -> ProviderCircuitOpenError | None:
        with self._circuit_lock:
            failures = self._provider_failures.get(provider_name, 0) + 1
            self._provider_failures[provider_name] = failures
            if failures < self.consecutive_failures_before_abort:
                return None
            reason = (
                f"Provider '{provider_name}' failed {failures} consecutive "
                f"requests; the current run was stopped to preserve checkpoints "
                f"and avoid repeated charges. Last error: {exc}"
            )
            self._open_provider_circuits[provider_name] = reason
            return ProviderCircuitOpenError(reason)

    def _raise_if_circuit_open(self, provider_name: str) -> None:
        with self._circuit_lock:
            reason = self._open_provider_circuits.get(provider_name)
        if reason:
            raise ProviderCircuitOpenError(reason)

    def _call_claude_messages(
        self,
        *,
        client: Any,
        base_url: str,
        model_name: str,
        content: str,
    ) -> tuple[str, int, int]:
        response = client.post(
            base_url,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self.max_completion_tokens,
            },
        )
        if response.is_error:
            error_body = response.text.strip()
            if len(error_body) > 2000:
                error_body = error_body[:2000] + "..."
            raise RuntimeError(
                "Claude Messages request failed with "
                f"HTTP {response.status_code}: {error_body}"
            )

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Claude Messages response must be a JSON object.")
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise ValueError("Claude Messages response must contain a content list.")
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and block.get("text") is not None
        )
        usage = payload.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = int(
            usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        )
        completion_tokens = int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
        return text, prompt_tokens, completion_tokens

    @staticmethod
    def _parse_and_validate(
        text: str,
        validator: JSONValidator | None,
        *,
        preserve_latex_escapes: bool = False,
    ) -> dict[str, Any]:
        parsed = parse_json_lenient(
            text,
            preserve_latex_escapes=preserve_latex_escapes,
        )
        if not isinstance(parsed, dict):
            raise ValueError("Model output must be one JSON object.")
        if validator:
            valid, error = validator(parsed)
            if not valid:
                raise ValueError(error)
        return parsed

    @staticmethod
    def _sleep(seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


def parse_json_lenient(
    text: str,
    *,
    preserve_latex_escapes: bool = False,
) -> Any:
    json_text = extract_json_substring(text)
    if preserve_latex_escapes:
        json_text = protect_latex_backslashes(json_text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            repaired = repair_json(json_text)
            return json.loads(repaired)
        except Exception as exc:
            raise ValueError(f"Could not parse JSON output: {exc}") from exc


def protect_latex_backslashes(text: str) -> str:
    r"""Escape single backslashes before letters inside math-heavy JSON.

    Mathematical models sometimes emit valid-looking JSON with LaTeX such as
    ``\theta`` instead of ``\\theta``. JSON would interpret the leading
    ``\t`` as a tab and irreversibly damage the formula. This protects invalid
    JSON escapes and common LaTeX commands that begin with a JSON escape
    letter, while leaving actual ``\n``, ``\r``, and ``\t`` whitespace escapes
    available for normal JSON decoding. Already doubled backslashes are left
    unchanged.
    """

    return re.sub(
        r"(?<!\\)\\(?=(?:[acdeghijklmopqsvwxyzA-Z]|[bfnrtu][a-z]|Re))",
        lambda _match: "\\\\",
        text,
    )


def extract_json_substring(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates: list[tuple[int, int, str]] = []
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start >= 0 and end > start:
            candidates.append((start, end, cleaned[start : end + 1]))
    if not candidates:
        return cleaned
    return max(candidates, key=lambda item: item[1] - item[0])[2]
