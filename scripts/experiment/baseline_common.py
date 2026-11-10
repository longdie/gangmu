"""Responses transport and resumable pair judgments for published baselines."""
from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Callable

import httpx

from scripts.experiment.common import ExperimentRuntime, run_parallel_queries


class ModelMismatchError(ValueError):
    """A different served model must not silently enter the experiment."""


def response_text(response: Any, model: str) -> str:
    if not isinstance(response, dict):
        raise ValueError("Responses API output is not an object.")
    if (response.get("status") != "completed" or response.get("error")
            or response.get("incomplete_details")):
        raise ValueError("Responses API output is incomplete or failed.")
    if response.get("model") != model:
        raise ModelMismatchError("Responses API returned a different model ID.")
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses API output items are missing.")
    chunks = []
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("Invalid Responses API output item.")
        if item.get("type") == "reasoning":
            continue
        if (item.get("type") != "message" or item.get("role") != "assistant"
                or item.get("status") not in (None, "completed")):
            raise ValueError("Unexpected or incomplete Responses API message.")
        if not isinstance(item.get("content"), list):
            raise ValueError("Responses API message content is missing.")
        for content in item["content"]:
            if (not isinstance(content, dict) or content.get("type") != "output_text"
                    or not isinstance(content.get("text"), str)):
                raise ValueError("Responses API returned refusal or non-text content.")
            chunks.append(content["text"])
    text = "".join(chunks).strip()
    if not text:
        raise ValueError("Responses API returned no answer text.")
    return text


def llm_settings(config: Any, task: str, max_output_tokens: int) -> dict[str, Any]:
    model = config.model(task)
    provider = config.provider(model.provider, require_api_key=True)
    if provider.protocol != "openai_responses":
        raise ValueError(f"{task} requires an openai_responses connection.")
    return {
        "model_task": task,
        "model": model.name,
        "provider": model.provider,
        "protocol": provider.protocol,
        # These experiments used serial calls, including retries.
        "workers": 1,
        "llm_parameters": {
            "reasoning": {"effort": "low"},
            "max_output_tokens": max_output_tokens,
            "store": False,
        },
        "call_attempts": config.get_int("requests.call_attempts", minimum=1),
    }


class ResponsesClient:
    def __init__(self, runtime: ExperimentRuntime, *, transport=None):
        self.runtime = runtime
        self.settings = runtime.settings
        provider = runtime.config.provider(self.settings["provider"], require_api_key=True)
        base = provider.base_url.rstrip("/")
        self.endpoint = base if base.endswith("/responses") else base + "/responses"
        self.http = httpx.Client(
            headers={"Authorization": f"Bearer {provider.api_key}"},
            timeout=httpx.Timeout(
                runtime.config.get_float("requests.timeout_seconds", minimum=1),
                connect=runtime.config.get_float("requests.connect_timeout_seconds", minimum=1),
            ),
            trust_env=provider.trust_env,
            follow_redirects=False,
            transport=transport,
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.http.close()

    def judge(self, prompt: str, parser: Callable, *, task: str) -> dict[str, Any]:
        payload = {
            "model": self.settings["model"],
            **deepcopy(self.settings["llm_parameters"]),
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        }
        attempts = int(self.settings["call_attempts"])
        for attempt in range(1, attempts + 1):
            usage: dict[str, Any] = {}
            success = False
            retryable = True
            error = "unknown_error"
            try:
                response = self.http.post(self.endpoint, json=payload)
                if response.status_code != 200:
                    retryable = response.status_code in {408, 409, 429, 500, 502, 503, 504}
                    raise ValueError(f"Responses API HTTP {response.status_code}.")
                body = response.json()
                if isinstance(body, dict) and isinstance(body.get("usage"), dict):
                    usage = body["usage"]
                text = response_text(body, self.settings["model"])
                decision = parser(text)
                success = True
                return {"decision": decision, "answer": text}
            except ModelMismatchError:
                retryable = False
                error = "returned_model_mismatch"
            except httpx.RequestError as exc:
                # Network exception strings may include credentials or endpoints.
                error = type(exc).__name__
            except ValueError:
                error = "invalid_response_or_http_error"
            finally:
                self.runtime.usage_tracker.record(
                    provider=self.settings["provider"], model=self.settings["model"],
                    task=task, prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0), success=success,
                    extra={"attempt": attempt, "error": None if success else error},
                )
            if not retryable or attempt == attempts:
                raise RuntimeError(f"Baseline model call failed ({error}); query remains unfinished.") from None
            time.sleep(min(2 ** (attempt - 1), 4))
        raise AssertionError("Unreachable request loop.")


def has_pending_queries(runtime: ExperimentRuntime) -> bool:
    return any(
        (runtime.store.load_query(int(query["query_id"])) or {}).get("status") != "completed"
        for query in runtime.queries
    )


def run_pairwise(
    runtime: ExperimentRuntime, *, candidates_for_query: Callable,
    pair_prompt: Callable, parse_pair: Callable, retrieval_stage: str,
    select_prompt: Callable | None = None, parse_selection: Callable | None = None,
) -> None:
    """Judge every candidate; persist each successful pair before the next call."""
    if not has_pending_queries(runtime):
        return
    with ResponsesClient(runtime) as client:
        def worker(query: dict[str, Any], existing: dict[str, Any] | None) -> None:
            query_id = int(query["query_id"])
            candidates = candidates_for_query(query)
            ids = [int(item["canonical_node_id"]) for item in candidates]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("Baseline retrieval returned empty or duplicate candidates.")
            trace = deepcopy(existing.get("trace", {})) if existing else {
                "candidates": candidates,
                "candidate_ids_snapshot": ids,
                "candidate_judgments": [],
                "retrieval_stages": {retrieval_stage: ids},
            }
            if trace.get("candidate_ids_snapshot") != ids:
                raise ValueError(f"Baseline candidates changed while resuming query {query_id}.")
            judgments = trace["candidate_judgments"]
            completed = {item["canonical_node_id"]: item for item in judgments}
            if (len(completed) != len(judgments) or not set(completed).issubset(ids)
                    or any(type(item.get("decision")) is not bool for item in judgments)):
                raise ValueError("Invalid baseline pair checkpoint.")

            def save_partial():
                runtime.store.save_query(query_id, {"status": "partial", "trace": trace})

            save_partial()
            for node_id in ids:
                if node_id in completed:
                    continue
                judgment = client.judge(
                    pair_prompt(query, node_id), parse_pair,
                    task=f"experiment.{runtime.method_name}.q{query_id}.pair.{node_id}",
                )
                if type(judgment["decision"]) is not bool:
                    raise ValueError("Pair parser must return a boolean judgment.")
                judgment["canonical_node_id"] = node_id
                judgments.append(judgment)
                completed[node_id] = judgment
                save_partial()

            yes_ids = [node_id for node_id in ids if completed[node_id]["decision"]]
            predicted = yes_ids[0] if yes_ids else None
            status = "all_no" if not yes_ids else "unique_yes" if len(yes_ids) == 1 else "first_ranked_yes"
            if len(yes_ids) > 1 and select_prompt is not None:
                if parse_selection is None:
                    raise ValueError("Missing multiple-match selection parser.")
                if "selection" not in trace:
                    trace["selection"] = client.judge(
                        select_prompt(query, yes_ids), lambda text: parse_selection(text, yes_ids),
                        task=f"experiment.{runtime.method_name}.q{query_id}.selection",
                    )
                    save_partial()
                predicted = trace["selection"]["decision"]
                if type(predicted) is not int or predicted not in yes_ids:
                    raise ValueError("Selected ID is not an affirmative candidate.")
                status = "selected_from_multiple_yes"
            runtime.store.save_query(query_id, {
                "status": "completed",
                "predicted_canonical_node_id": predicted,
                "predicted_canonical_node_ids": [] if predicted is None else [predicted],
                "all_yes_candidate_ids": yes_ids,
                "internal_status": status,
                "trace": trace,
            })

        run_parallel_queries(runtime, worker, workers=1)
