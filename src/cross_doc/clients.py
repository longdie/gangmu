from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from src.common.config import GangmuConfig, get_config
from src.common.files import ProjectLogger, TokenUsageTracker
from src.evidence_graph.derivation import EMBEDDING_DIMENSIONS, normalize_vector


class RerankerClient:
    """Call the configured OpenAI-compatible rerank endpoint."""

    def __init__(
        self,
        *,
        config: GangmuConfig | None = None,
        logger: ProjectLogger | None = None,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "HTTPX is required; install dependencies from requirements.txt."
            ) from exc

        self.config = config or get_config()
        self.logger = logger
        self.usage_tracker = usage_tracker or TokenUsageTracker()
        self.model = self.config.model("reranker")
        self.provider = self.config.provider(
            self.model.provider,
            require_api_key=True,
        )
        if not self.provider.base_url:
            raise ValueError("The Reranker provider requires a base URL.")
        base_url = self.provider.base_url.rstrip("/")
        self.url = base_url if base_url.endswith("/rerank") else f"{base_url}/rerank"
        self.timeout_seconds = self.config.get_float(
            "reranker.timeout_seconds",
            minimum=0,
        )
        self.call_attempts = self.config.get_int(
            "reranker.call_attempts",
            minimum=1,
        )
        self.retry_delay_seconds = self.config.get_float(
            "reranker.retry_delay_seconds",
            minimum=0,
        )
        self._httpx = httpx
        self._client_lock = threading.Lock()
        self._retired_clients: list[Any] = []
        self.client = self._create_client()

    def _create_client(self) -> Any:
        return self._httpx.Client(
            headers={
                "Authorization": f"Bearer {self.provider.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_seconds,
            trust_env=self.provider.trust_env,
        )

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int,
        usage_task: str = "cross_document_reranker",
    ) -> list[dict[str, int | float]]:
        if not documents:
            return []
        top_n = max(1, min(int(top_n), len(documents)))
        payload = {
            "model": self.model.name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.call_attempts + 1):
            prompt_tokens = 0
            success = False
            try:
                with self._client_lock:
                    client = self.client
                response = client.post(self.url, json=payload)
                if response.is_error:
                    body = response.text.strip()
                    if len(body) > 2000:
                        body = body[:2000] + "..."
                    raise RuntimeError(
                        "Reranker request failed with "
                        f"HTTP {response.status_code}: {body}"
                    )
                response_payload = response.json()
                usage = (
                    response_payload.get("usage", {})
                    if isinstance(response_payload, dict)
                    else {}
                )
                if not isinstance(usage, dict):
                    usage = {}
                prompt_tokens = int(
                    usage.get("prompt_tokens")
                    or usage.get("input_tokens")
                    or usage.get("total_tokens")
                    or 0
                )
                result = self._parse_results(
                    response_payload,
                    document_count=len(documents),
                )
                success = True
                return result[:top_n]
            except Exception as exc:
                last_error = exc
                if isinstance(exc, self._httpx.TransportError):
                    self._replace_client(client)
                if self.logger:
                    details = {
                        "provider": self.provider.name,
                        "model": self.model.name,
                        "attempt": attempt,
                        "max_attempts": self.call_attempts,
                        "document_count": len(documents),
                        "top_n": top_n,
                    }
                    if attempt < self.call_attempts:
                        self.logger.warning(
                            "Cross-document Reranker call failed; retrying",
                            error=str(exc),
                            **details,
                        )
                    else:
                        self.logger.exception(
                            "Cross-document Reranker call failed",
                            exc,
                            **details,
                        )
            finally:
                self.usage_tracker.record(
                    provider=self.provider.name,
                    model=self.model.name,
                    task=usage_task,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    success=success,
                    extra={
                        "attempt": attempt,
                        "max_attempts": self.call_attempts,
                        "document_count": len(documents),
                        "top_n": top_n,
                    },
                )
                if self.logger:
                    self.logger.model_call(
                        task=usage_task,
                        provider=self.provider.name,
                        model=self.model.name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        success=success,
                        attempt=attempt,
                        max_attempts=self.call_attempts,
                        document_count=len(documents),
                        top_n=top_n,
                    )
            if attempt < self.call_attempts and self.retry_delay_seconds > 0:
                time.sleep(self.retry_delay_seconds * attempt)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Reranker call failed without an exception.")

    def _replace_client(self, failed_client: Any) -> None:
        with self._client_lock:
            if self.client is not failed_client:
                return
            previous_client = self.client
            self.client = self._create_client()
            # Another worker may still be completing an in-flight request on
            # this thread-safe client. Keep it alive for the remainder of the
            # run instead of closing the transport underneath that request.
            self._retired_clients.append(previous_client)

    @staticmethod
    def _parse_results(
        payload: Any,
        *,
        document_count: int,
    ) -> list[dict[str, int | float]]:
        if not isinstance(payload, dict):
            raise ValueError("Reranker response must be a JSON object.")
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("Reranker response must contain a results list.")

        parsed: list[dict[str, int | float]] = []
        seen: set[int] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
                score = float(
                    item.get("relevance_score", item.get("score", 0.0)) or 0.0
                )
            except (TypeError, ValueError):
                continue
            if not 0 <= index < document_count or index in seen:
                continue
            seen.add(index)
            parsed.append({"index": index, "score": score})
        if not parsed:
            raise ValueError("Reranker response did not contain any valid result.")
        return parsed


class CanonicalStatementEmbedder:
    """Embed final Canonical titles or statements with the fixed model."""

    def __init__(
        self,
        *,
        config: GangmuConfig | None = None,
        logger: ProjectLogger | None = None,
        usage_tracker: TokenUsageTracker | None = None,
    ) -> None:
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI and HTTPX are required; install dependencies from requirements.txt."
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
        self.provider = self.config.provider(
            self.model.provider,
            require_api_key=True,
        )
        self.call_attempts = self.config.get_int(
            "requests.call_attempts",
            minimum=1,
        )
        self.retry_delay_seconds = self.config.get_float(
            "embedding.retry_delay_seconds",
            minimum=0,
        )
        timeout_seconds = self.config.get_float(
            "requests.timeout_seconds",
            minimum=0,
        )
        request_timeout = httpx.Timeout(
            timeout_seconds,
            connect=self.config.get_float(
                "requests.connect_timeout_seconds",
                minimum=0,
            ),
            pool=self.config.get_float(
                "requests.pool_timeout_seconds",
                minimum=0,
            ),
        )
        http_client = httpx.Client(
            timeout=request_timeout,
            limits=httpx.Limits(
                max_connections=self.config.get_int(
                    "requests.max_connections",
                    minimum=1,
                ),
                max_keepalive_connections=self.config.get_int(
                    "requests.max_keepalive_connections",
                    minimum=0,
                ),
            ),
            trust_env=self.provider.trust_env,
        )
        client_kwargs: dict[str, Any] = {
            "api_key": self.provider.api_key,
            "timeout": request_timeout,
            "max_retries": 0,
            "http_client": http_client,
        }
        if self.provider.base_url:
            client_kwargs["base_url"] = self.provider.base_url
        self.client = OpenAI(**client_kwargs)
        self._client_lock = threading.Lock()

    def embed(
        self,
        text: str,
        *,
        usage_task: str = "canonical_statement_embedding",
    ) -> np.ndarray:
        text = str(text or "").strip()
        if not text:
            raise ValueError("Canonical title or statement is empty.")

        last_error: Exception | None = None
        for attempt in range(1, self.call_attempts + 1):
            prompt_tokens = 0
            success = False
            try:
                with self._client_lock:
                    client = self.client
                response = client.embeddings.create(
                    model=self.model.name,
                    input=text,
                    dimensions=EMBEDDING_DIMENSIONS,
                )
                usage = getattr(response, "usage", None)
                prompt_tokens = int(
                    getattr(usage, "prompt_tokens", 0)
                    or getattr(usage, "total_tokens", 0)
                    or 0
                )
                vector = normalize_vector(
                    response.data[0].embedding,
                    expected_dimensions=EMBEDDING_DIMENSIONS,
                )
                success = True
                return vector
            except Exception as exc:
                last_error = exc
                if self.logger:
                    details = {
                        "provider": self.provider.name,
                        "model": self.model.name,
                        "attempt": attempt,
                        "max_attempts": self.call_attempts,
                    }
                    if attempt < self.call_attempts:
                        self.logger.warning(
                            "Canonical content Embedding failed; retrying",
                            error=str(exc),
                            **details,
                        )
                    else:
                        self.logger.exception(
                            "Canonical content Embedding failed",
                            exc,
                            **details,
                        )
            finally:
                self.usage_tracker.record(
                    provider=self.provider.name,
                    model=self.model.name,
                    task=usage_task,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    success=success,
                    extra={
                        "dimensions": EMBEDDING_DIMENSIONS,
                        "attempt": attempt,
                        "max_attempts": self.call_attempts,
                    },
                )
                if self.logger:
                    self.logger.model_call(
                        task=usage_task,
                        provider=self.provider.name,
                        model=self.model.name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        success=success,
                        dimensions=EMBEDDING_DIMENSIONS,
                        attempt=attempt,
                        max_attempts=self.call_attempts,
                    )
            if attempt < self.call_attempts and self.retry_delay_seconds > 0:
                time.sleep(self.retry_delay_seconds * attempt)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Canonical content Embedding failed without an exception.")
