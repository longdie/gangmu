from __future__ import annotations

from collections import defaultdict
from typing import Any

from scripts.experiment.common import (
    ExperimentRuntime,
    identity_text,
    run_parallel_queries,
)


METHOD_NAME = "identity_match"
DISPLAY_NAME = "Exact String Match"


def method_settings(*_: Any, **__: Any) -> dict[str, Any]:
    return {
        "algorithm_version": 1,
        "normalization": ["unicode_nfkc", "lowercase", "remove_whitespace"],
        "same_type_only": True,
        "tie_breaker": "smallest_canonical_node_id",
    }


def run(runtime: ExperimentRuntime) -> None:
    index: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for node in runtime.dataset.canonical_nodes:
        index[str(node["type"])][identity_text(str(node["statement"]))].append(
            int(node["id"])
        )

    def worker(query: dict[str, Any], existing: dict[str, Any] | None) -> None:
        del existing
        normalized_query = identity_text(str(query["normalized_statement"]))
        matches = sorted(index[str(query["type"])].get(normalized_query, []))
        predicted_id = matches[0] if matches else None
        runtime.store.save_query(
            int(query["query_id"]),
            {
                "status": "completed",
                "predicted_canonical_node_id": predicted_id,
                "internal_status": (
                    "multiple_exact_matches"
                    if len(matches) > 1
                    else "single_exact_match"
                    if matches
                    else "no_exact_match"
                ),
                "trace": {
                    "normalized_query": normalized_query,
                    "matching_canonical_node_ids": matches,
                    "multiple_exact_matches": len(matches) > 1,
                },
            },
        )

    run_parallel_queries(runtime, worker, workers=1)

