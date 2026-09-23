from __future__ import annotations

import json
from typing import Any

from scripts.experiment.common import PersistentTokenUsageTracker


def parse_stream_json(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(str(stdout or "").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Claude Code emitted invalid stream-json at line {line_number}."
            ) from exc
        if not isinstance(value, dict):
            raise ValueError("Claude Code stream-json events must be objects.")
        events.append(value)
    if not events:
        raise ValueError("Claude Code returned an empty event stream.")
    return events


def structured_output(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        value = event.get("structured_output")
        if isinstance(value, dict):
            return value
        result = event.get("result")
        if isinstance(result, dict) and {
            "prediction",
            "canonical_node_id",
            "brief_reason",
        }.issubset(result):
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("Claude Code returned no structured final output.")


def aggregate_model_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract the largest cumulative modelUsage record for each model."""

    by_model: dict[str, dict[str, Any]] = {}
    blocks = 0
    for event in events:
        raw = event.get("modelUsage")
        if not isinstance(raw, dict):
            continue
        blocks += 1
        for model_key, value in raw.items():
            if not isinstance(value, dict):
                continue
            normalized = _normalized_usage(str(model_key), value)
            current = by_model.get(str(model_key))
            if current is None or _usage_weight(normalized) >= _usage_weight(current):
                by_model[str(model_key)] = normalized
    if not by_model:
        raise ValueError("Claude Code stream contained no modelUsage records.")
    observed_turns = max(
        (
            _safe_int(event.get("num_turns"))
            for event in events
            if event.get("type") == "result"
        ),
        default=0,
    )
    if len(by_model) == 1:
        only_usage = next(iter(by_model.values()))
        if not only_usage.pop("call_count_reported", False) and observed_turns > 0:
            only_usage["calls"] = observed_turns
            only_usage["call_count_source"] = "result.num_turns"
        else:
            only_usage["call_count_source"] = "modelUsage"
    else:
        for item in by_model.values():
            reported = bool(item.pop("call_count_reported", False))
            item["call_count_source"] = (
                "modelUsage" if reported else "unavailable"
            )
    return {
        "model_usage_blocks": blocks,
        "models": by_model,
        "total": {
            "calls": sum(int(item["calls"]) for item in by_model.values()),
            "input_tokens": sum(
                int(item["input_tokens"]) for item in by_model.values()
            ),
            "cache_creation_input_tokens": sum(
                int(item["cache_creation_input_tokens"])
                for item in by_model.values()
            ),
            "cache_read_input_tokens": sum(
                int(item["cache_read_input_tokens"])
                for item in by_model.values()
            ),
            "total_input_tokens": sum(
                int(item["total_input_tokens"]) for item in by_model.values()
            ),
            "output_tokens": sum(
                int(item["output_tokens"]) for item in by_model.values()
            ),
            "cost_usd": sum(float(item["cost_usd"]) for item in by_model.values()),
        },
    }


def validate_expected_model(
    usage: dict[str, Any],
    *,
    expected_model: str,
) -> None:
    actual: set[str] = set()
    for model_key, item in usage["models"].items():
        canonical = str(item.get("canonical_model") or model_key).strip()
        actual.add(canonical)
    if actual != {str(expected_model)}:
        raise AgentModelMismatchError(
            "Claude Code used unexpected model(s): "
            f"expected {expected_model!r}, observed {sorted(actual)!r}."
        )


def record_agent_usage(
    tracker: PersistentTokenUsageTracker,
    *,
    usage: dict[str, Any],
    provider: str,
    task_prefix: str,
    success: bool,
) -> None:
    """Persist aggregate tokens while preserving the actual request count."""

    for model_key, item in usage["models"].items():
        calls = max(1, int(item["calls"]))
        for call_index in range(1, calls + 1):
            first = call_index == 1
            tracker.record(
                provider=provider,
                model=str(item.get("canonical_model") or model_key),
                task=f"{task_prefix}.model_call",
                prompt_tokens=(int(item["total_input_tokens"]) if first else 0),
                completion_tokens=(int(item["output_tokens"]) if first else 0),
                success=success,
                extra={
                    "agent_model_key": str(model_key),
                    "agent_call_index": call_index,
                    "agent_model_calls": calls,
                    "usage_is_model_aggregate": first,
                    "uncached_input_tokens": (
                        int(item["input_tokens"]) if first else 0
                    ),
                    "cache_creation_input_tokens": (
                        int(item["cache_creation_input_tokens"]) if first else 0
                    ),
                    "cache_read_input_tokens": (
                        int(item["cache_read_input_tokens"]) if first else 0
                    ),
                    "cost_usd": float(item["cost_usd"]) if first else 0.0,
                },
            )


def compact_agent_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep tool/control events without retaining hidden reasoning text."""

    tool_calls: list[dict[str, Any]] = []
    result_metadata: dict[str, Any] = {}
    for event in events:
        message = event.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "event": "tool_use",
                                "tool_use_id": block.get("id"),
                                "name": block.get("name"),
                                "input": block.get("input"),
                            }
                        )
                    elif block.get("type") == "tool_result":
                        tool_calls.append(
                            {
                                "event": "tool_result",
                                "tool_use_id": block.get("tool_use_id"),
                                "is_error": bool(block.get("is_error", False)),
                            }
                        )
        if event.get("type") == "result":
            result_metadata = {
                key: event.get(key)
                for key in (
                    "subtype",
                    "is_error",
                    "duration_ms",
                    "duration_api_ms",
                    "num_turns",
                    "total_cost_usd",
                    "stop_reason",
                )
                if key in event
            }
    return {"tool_events": tool_calls, "result": result_metadata}


class AgentModelMismatchError(RuntimeError):
    pass


def _normalized_usage(model_key: str, value: dict[str, Any]) -> dict[str, Any]:
    input_tokens = _int_value(value, "inputTokens", "input_tokens")
    output_tokens = _int_value(value, "outputTokens", "output_tokens")
    cache_creation = _int_value(
        value,
        "cacheCreationInputTokens",
        "cache_creation_input_tokens",
    )
    cache_read = _int_value(
        value,
        "cacheReadInputTokens",
        "cache_read_input_tokens",
    )
    call_keys = ("numRequests", "requests", "calls")
    calls = _int_value(
        value,
        *call_keys,
        default=1,
    )
    canonical = str(
        value.get("canonicalModel")
        or value.get("canonical_model")
        or model_key
    )
    return {
        "canonical_model": canonical,
        "calls": max(1, calls),
        "call_count_reported": any(key in value for key in call_keys),
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "total_input_tokens": input_tokens + cache_creation + cache_read,
        "output_tokens": output_tokens,
        "cost_usd": _float_value(value, "costUSD", "cost_usd"),
        "context_window": _int_value(
            value,
            "contextWindow",
            "context_window",
        ),
    }


def _usage_weight(value: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(value["total_input_tokens"]) + int(value["output_tokens"]),
        int(value["calls"]),
        float(value["cost_usd"]),
    )


def _int_value(
    value: dict[str, Any],
    *keys: str,
    default: int = 0,
) -> int:
    for key in keys:
        if key in value:
            try:
                return int(value[key] or 0)
            except (TypeError, ValueError):
                return default
    return default


def _float_value(value: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in value:
            try:
                return float(value[key] or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
