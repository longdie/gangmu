from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260819


def evaluate_results(
    *,
    queries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    canonical_node_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(queries) != len(records):
        raise ValueError("Query and experiment record counts differ.")
    record_by_query = {int(record["query_id"]): record for record in records}
    expected_query_ids = {int(query["query_id"]) for query in queries}
    if len(record_by_query) != len(records) or set(record_by_query) != expected_query_ids:
        raise ValueError("Experiment records do not map one-to-one to queries.")
    predictions: list[dict[str, Any]] = []
    for query in queries:
        query_id = int(query["query_id"])
        record = record_by_query[query_id]
        predicted_id = record.get("predicted_canonical_node_id")
        if predicted_id is not None:
            if isinstance(predicted_id, bool) or not isinstance(predicted_id, int):
                raise ValueError("A predicted Canonical Node ID is not an integer.")
            if predicted_id < 1 or (
                canonical_node_count is not None
                and predicted_id > canonical_node_count
            ):
                raise ValueError("A predicted Canonical Node ID is out of range.")
        predicted_ids = _predicted_target_ids(
            record,
            canonical_node_count=canonical_node_count,
        )
        # Checkpoints written before accepted target sets were introduced may
        # only contain the model trace.  Recover the deterministic resolution
        # so historical runs are scored with the same semantics as new runs.
        if predicted_id is None and predicted_ids:
            predicted_id = predicted_ids[0]
        if predicted_id is not None and predicted_id not in predicted_ids:
            raise ValueError(
                "The representative predicted Canonical Node ID is not in "
                "the accepted target IDs."
            )
        gold_label = str(query["gold_label"])
        gold_id = query.get("gold_canonical_node_id")
        error_type, correct = _classify(
            gold_label=gold_label,
            gold_id=gold_id,
            predicted_ids=predicted_ids,
        )
        predictions.append(
            {
                "query_id": query_id,
                "source_id": str(query.get("source_id", "")),
                "type": str(query.get("type", "")),
                "subfield": str(query.get("subfield", "")),
                "gold_label": gold_label,
                "gold_canonical_node_id": gold_id,
                "prediction_label": "NIL" if not predicted_ids else "MERGE",
                "predicted_canonical_node_id": predicted_id,
                "predicted_canonical_node_ids": predicted_ids,
                "correct": correct,
                "error_type": error_type,
                "internal_status": record.get("internal_status"),
            }
        )

    metrics = _metrics_from_predictions(predictions, include_confidence=True)
    metrics["grouped"] = {
        field: _grouped_metrics(predictions, field)
        for field in ("type", "subfield", "source_id")
    }
    metrics["candidate_recall"] = _candidate_recall(
        queries=queries,
        records=records,
    )
    return predictions, metrics


def _metrics_from_predictions(
    predictions: list[dict[str, Any]],
    *,
    include_confidence: bool,
) -> dict[str, Any]:
    merge_rows = [row for row in predictions if row["gold_label"] == "MERGE"]
    nil_rows = [row for row in predictions if row["gold_label"] == "NIL"]
    correct_merge = sum(row["error_type"] == "correct_merge" for row in predictions)
    correct_nil = sum(row["error_type"] == "correct_nil" for row in predictions)
    wrong_target = sum(
        row["error_type"] == "wrong_target_merge" for row in predictions
    )
    missed_merge = sum(row["error_type"] == "missed_merge" for row in predictions)
    false_merge = sum(row["error_type"] == "false_merge" for row in predictions)

    benchmark_accuracy = _safe_rate(
        sum(bool(row["correct"]) for row in predictions),
        len(predictions),
    )
    merge_accuracy = _safe_rate(correct_merge, len(merge_rows))
    nil_accuracy = _safe_rate(correct_nil, len(nil_rows))
    merge_precision = _safe_rate(
        correct_merge,
        correct_merge + wrong_target + false_merge,
    )
    nil_precision = _safe_rate(correct_nil, correct_nil + missed_merge)
    result: dict[str, Any] = {
        "counts": {
            "queries": len(predictions),
            "gold_merge": len(merge_rows),
            "gold_nil": len(nil_rows),
            "predicted_merge": sum(
                row["prediction_label"] == "MERGE" for row in predictions
            ),
            "predicted_nil": sum(
                row["prediction_label"] == "NIL" for row in predictions
            ),
            "correct_merge": correct_merge,
            "correct_nil": correct_nil,
        },
        "core": {
            "benchmark_accuracy": benchmark_accuracy,
            "merge_accuracy": merge_accuracy,
            "nil_accuracy": nil_accuracy,
        },
        "secondary": {
            "merge_precision": merge_precision,
            "nil_precision": nil_precision,
            "merge_f1": _f1(merge_precision, merge_accuracy),
            "nil_f1": _f1(nil_precision, nil_accuracy),
        },
        "errors": {
            "false_merge": false_merge,
            "missed_merge": missed_merge,
            "wrong_target_merge": wrong_target,
            "rates": {
                "false_merge_among_gold_nil": _safe_rate(
                    false_merge,
                    len(nil_rows),
                ),
                "missed_merge_among_gold_merge": _safe_rate(
                    missed_merge,
                    len(merge_rows),
                ),
                "wrong_target_merge_among_gold_merge": _safe_rate(
                    wrong_target,
                    len(merge_rows),
                ),
            },
        },
    }
    if include_confidence:
        result["confidence_intervals_95"] = {
            "method": "query_level_bootstrap",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "benchmark_accuracy": _bootstrap_ci(
                [bool(row["correct"]) for row in predictions],
                seed=BOOTSTRAP_SEED,
            ),
            "merge_accuracy": _bootstrap_ci(
                [bool(row["correct"]) for row in merge_rows],
                seed=BOOTSTRAP_SEED + 1,
            ),
            "nil_accuracy": _bootstrap_ci(
                [bool(row["correct"]) for row in nil_rows],
                seed=BOOTSTRAP_SEED + 2,
            ),
        }
    return result


def _classify(
    *,
    gold_label: str,
    gold_id: Any,
    predicted_ids: list[int],
) -> tuple[str, bool]:
    if gold_label == "NIL":
        if not predicted_ids:
            return "correct_nil", True
        return "false_merge", False
    if not predicted_ids:
        return "missed_merge", False
    if int(gold_id) in predicted_ids:
        return "correct_merge", True
    return "wrong_target_merge", False


def _predicted_target_ids(
    record: dict[str, Any],
    *,
    canonical_node_count: int | None,
) -> list[int]:
    """Return accepted targets, including legacy Gangmu trace recovery."""

    raw_ids = record.get("predicted_canonical_node_ids")
    if raw_ids is None:
        single = record.get("predicted_canonical_node_id")
        if single is not None:
            raw_ids = [single]
        else:
            raw_ids = _trace_resolved_target_ids(record)
    if not isinstance(raw_ids, list):
        raise ValueError("Predicted Canonical Node IDs must be a list.")
    result: list[int] = []
    for value in raw_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("A predicted Canonical Node ID is not an integer.")
        if value < 1 or (
            canonical_node_count is not None and value > canonical_node_count
        ):
            raise ValueError("A predicted Canonical Node ID is out of range.")
        result.append(int(value))
    if len(result) != len(set(result)):
        raise ValueError("Predicted Canonical Node IDs must be unique.")
    return result


def _trace_resolved_target_ids(record: dict[str, Any]) -> list[int]:
    """Recover accepted targets from a Gangmu trace written by any version.

    New runs persist ``predicted_canonical_node_ids`` on the result record.
    Older runs persisted only ``trace.model_prediction.query_prediction``;
    their strict groups are still sufficient to reconstruct the same frozen-KB
    decision and must not silently become NIL during re-evaluation.
    """

    trace = record.get("trace")
    if not isinstance(trace, dict):
        return []
    model_prediction = trace.get("model_prediction")
    if not isinstance(model_prediction, dict):
        return []
    prediction = model_prediction.get("query_prediction")
    if not isinstance(prediction, dict):
        return []

    resolved = prediction.get("resolved_prediction")
    if isinstance(resolved, dict):
        raw_ids = resolved.get("accepted_canonical_node_ids")
        if isinstance(raw_ids, list):
            if all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in raw_ids
            ):
                return [int(value) for value in raw_ids]
            return []
        if resolved.get("label") == "MERGE":
            representative = resolved.get("canonical_node_id")
            if isinstance(representative, int) and not isinstance(
                representative, bool
            ):
                return [representative]

    label = prediction.get("label")
    if label == "MERGE":
        representative = prediction.get("canonical_node_id")
        if isinstance(representative, int) and not isinstance(representative, bool):
            return [representative]
        return []
    if label != "MULTIPLE_TARGETS":
        return []

    groups = prediction.get("strictly_mergeable_canonical_groups")
    if not isinstance(groups, list):
        return []
    normalized_groups: list[list[int]] = []
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            continue
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in group
        ):
            continue
        ordered = sorted(int(value) for value in group)
        if len(ordered) != len(set(ordered)):
            continue
        normalized_groups.append(ordered)
    equivalent_ids = prediction.get("equivalent_canonical_node_ids")
    if (
        len(normalized_groups) != 1
        or not isinstance(equivalent_ids, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in equivalent_ids
        )
        or set(normalized_groups[0]) != set(int(value) for value in equivalent_ids)
    ):
        return []
    return normalized_groups[0]


def _grouped_metrics(
    predictions: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get(field) or "unknown")].append(row)
    return {
        key: _metrics_from_predictions(rows, include_confidence=False)
        for key, rows in sorted(grouped.items())
    }


def _candidate_recall(
    *,
    queries: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    query_by_id = {int(query["query_id"]): query for query in queries}
    gold_merge_queries = {
        int(query["query_id"]): int(query["gold_canonical_node_id"])
        for query in queries
        if query["gold_label"] == "MERGE"
    }
    stages: dict[str, dict[int, list[int]]] = defaultdict(dict)
    for record in records:
        query = query_by_id[int(record["query_id"])]
        if query["gold_label"] != "MERGE":
            continue
        query_id = int(query["query_id"])
        retrieval_stages = record.get("trace", {}).get("retrieval_stages", {})
        if not isinstance(retrieval_stages, dict):
            continue
        for stage, raw_ids in retrieval_stages.items():
            if isinstance(raw_ids, list):
                stages[str(stage)][query_id] = [int(node_id) for node_id in raw_ids]

    result: dict[str, Any] = {}
    for stage, candidates_by_query in sorted(stages.items()):
        ks = (5, 10, 20, 50) if "50" in stage else (5, 10, 20)
        result[stage] = {
            f"recall_at_{k}": _safe_rate(
                sum(
                    gold_id in candidates_by_query.get(query_id, [])[:k]
                    for query_id, gold_id in gold_merge_queries.items()
                ),
                len(gold_merge_queries),
            )
            for k in ks
        }
        result[stage]["gold_merge_queries"] = len(gold_merge_queries)
        result[stage]["stage_present_queries"] = len(candidates_by_query)
        result[stage]["missing_stage_queries"] = (
            len(gold_merge_queries) - len(candidates_by_query)
        )
    return result


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _bootstrap_ci(values: list[bool], *, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    array = np.asarray(values, dtype=np.float32)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        array,
        size=(BOOTSTRAP_SAMPLES, len(array)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]
