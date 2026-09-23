"""Benchmark-tuned threshold analysis used by the two reported baselines."""
from __future__ import annotations
import math
from typing import Any
from scripts.experiment.metrics import evaluate_results

def optimize_records(*, queries, records, method_name, canonical_node_count, default_threshold=0.5):
    if method_name not in {"token_jaccard", "embedding"}:
        raise ValueError(f"Unsupported threshold method: {method_name}")
    by_id = {int(record["query_id"]): record for record in records}
    if len(by_id) != len(queries) or set(by_id) != {int(q["query_id"]) for q in queries}:
        raise ValueError("Threshold records must map one-to-one to queries.")
    rows = []
    for query in queries:
        trace = by_id[int(query["query_id"])].get("trace", {})
        score, node_id = trace.get("best_score"), trace.get("best_canonical_node_id")
        if isinstance(score, bool) or not isinstance(score, (int,float)) or not math.isfinite(score):
            raise ValueError("Threshold analysis requires a finite Top-1 score per query.")
        if isinstance(node_id, bool) or not isinstance(node_id,int) or not 1 <= node_id <= canonical_node_count:
            raise ValueError("Threshold analysis requires a valid Top-1 target per query.")
        rows.append({"query": query, "score": float(score), "canonical_node_id": node_id})
    points = [_evaluate_threshold(rows=rows, threshold=t) for t in _candidate_thresholds([r["score"] for r in rows])]
    best = min(points, key=lambda p: (-p["benchmark_accuracy"], -p["macro_f1"], p["recall_balance_gap"], p["distance_from_score_midpoint"], p["threshold"]))
    selected = []
    for row in rows:
        merge = row["score"] >= best["threshold"]
        selected.append({"query_id": row["query"]["query_id"], "predicted_canonical_node_id": row["canonical_node_id"] if merge else None, "internal_status": "oracle_threshold_merge" if merge else "oracle_threshold_nil", "trace": {}})
    predictions, metrics = evaluate_results(queries=queries, records=selected, canonical_node_count=canonical_node_count)
    scores = sorted({r["score"] for r in rows})
    report = {
        "schema_version": "1.0", "analysis_type": "benchmark_oracle_accuracy_threshold",
        "method": method_name, "benchmark": {"query_count":len(queries)},
        "decision_rule": "MERGE Top-1 iff best_score >= threshold; otherwise NIL",
        "selection_rule": ["maximize benchmark_accuracy", "maximize macro_f1", "minimize absolute difference between Merge Recall and NIL Recall", "prefer threshold nearest the observed score-range midpoint", "prefer the lower threshold"],
        "score_summary": {"minimum":min(scores), "maximum":max(scores), "unique_values":len(scores)},
        "default_threshold": _evaluate_threshold(rows=rows, threshold=float(default_threshold)),
        "optimal_threshold": best, "operating_point_count": len(points), "operating_points": points,
        "warning": "The optimal threshold uses benchmark gold labels; it is a benchmark-tuned result, not a held-out estimate.",
    }
    return predictions, metrics, report

def _candidate_thresholds(scores: list[float]) -> list[float]:
    if not scores:
        raise ValueError("Threshold optimization requires at least one score.")
    unique = sorted(set(float(score) for score in scores))
    thresholds = [math.nextafter(unique[0], -math.inf)]
    thresholds.extend(
        (left + right) / 2.0 for left, right in zip(unique, unique[1:])
    )
    thresholds.append(math.nextafter(unique[-1], math.inf))
    return thresholds


def _evaluate_threshold(
    *,
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    gold_merge = sum(row["query"]["gold_label"] == "MERGE" for row in rows)
    gold_nil = len(rows) - gold_merge
    correct_merge = correct_nil = false_merge = missed_merge = wrong_target = 0
    predicted_merge = 0
    for row in rows:
        query = row["query"]
        merge = float(row["score"]) >= threshold
        if merge:
            predicted_merge += 1
            if query["gold_label"] == "NIL":
                false_merge += 1
            elif int(query["gold_canonical_node_id"]) == int(
                row["canonical_node_id"]
            ):
                correct_merge += 1
            else:
                wrong_target += 1
        elif query["gold_label"] == "NIL":
            correct_nil += 1
        else:
            missed_merge += 1

    merge_recall = _rate(correct_merge, gold_merge)
    nil_recall = _rate(correct_nil, gold_nil)
    merge_precision = _rate(correct_merge, predicted_merge)
    predicted_nil = len(rows) - predicted_merge
    nil_precision = _rate(correct_nil, predicted_nil)
    merge_f1 = _f1(merge_precision, merge_recall)
    nil_f1 = _f1(nil_precision, nil_recall)
    minimum = min(float(row["score"]) for row in rows)
    maximum = max(float(row["score"]) for row in rows)
    return {
        "threshold": float(threshold),
        "benchmark_accuracy": _rate(correct_merge + correct_nil, len(rows)),
        "merge_recall": merge_recall,
        "nil_recall": nil_recall,
        "merge_precision": merge_precision,
        "nil_precision": nil_precision,
        "merge_f1": merge_f1,
        "nil_f1": nil_f1,
        "macro_f1": (merge_f1 + nil_f1) / 2.0,
        "recall_balance_gap": abs(merge_recall - nil_recall),
        "predicted_merge": predicted_merge,
        "predicted_nil": predicted_nil,
        "correct_merge": correct_merge,
        "correct_nil": correct_nil,
        "false_merge": false_merge,
        "missed_merge": missed_merge,
        "wrong_target_merge": wrong_target,
        "degenerate_all_merge": predicted_merge == len(rows),
        "degenerate_all_nil": predicted_nil == len(rows),
        "distance_from_score_midpoint": abs(
            float(threshold) - (minimum + maximum) / 2.0
        ),
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

