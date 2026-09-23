from __future__ import annotations

from typing import Any

import numpy as np

from scripts.experiment.common import (
    ExperimentRuntime,
    mathematical_tokens,
    run_parallel_queries,
    top_scored_candidates,
)


METHOD_NAME = "token_jaccard"
DISPLAY_NAME = "Token Jaccard"
TOP_K = 20
THRESHOLD = 0.5


def method_settings(*_: Any, **__: Any) -> dict[str, Any]:
    return {
        "algorithm_version": 2,
        "top_k": TOP_K,
        "threshold": THRESHOLD,
        "same_type_only": True,
        "tokenization": "unicode_nfkc_lower_latex_alnum_symbols",
        "token_collection": "set",
    }


def run(runtime: ExperimentRuntime) -> None:
    tokens_by_type: dict[str, list[set[str]]] = {}
    for node_type in sorted(
        {str(node["type"]) for node in runtime.dataset.canonical_nodes}
    ):
        ids = runtime.dataset.canonical_ids_for_type(node_type)
        tokens_by_type[node_type] = [
            set(
                mathematical_tokens(
                    str(runtime.dataset.canonical_nodes[int(node_id) - 1]["statement"])
                )
            )
            for node_id in ids
        ]

    def worker(query: dict[str, Any], existing: dict[str, Any] | None) -> None:
        del existing
        node_type = str(query["type"])
        candidate_ids = runtime.dataset.canonical_ids_for_type(node_type)
        query_tokens = set(mathematical_tokens(str(query["normalized_statement"])))
        scores = np.fromiter(
            (
                _jaccard(query_tokens, candidate_tokens)
                for candidate_tokens in tokens_by_type.get(node_type, [])
            ),
            dtype=np.float32,
            count=len(candidate_ids),
        )
        top_20 = top_scored_candidates(
            candidate_ids=candidate_ids,
            scores=scores,
            top_k=TOP_K,
            score_key="jaccard_score",
        )
        best = top_20[0] if top_20 else None
        predicted_id = (
            int(best["canonical_node_id"])
            if best is not None and float(best["jaccard_score"]) >= THRESHOLD
            else None
        )
        runtime.store.save_query(
            int(query["query_id"]),
            {
                "status": "completed",
                "predicted_canonical_node_id": predicted_id,
                "internal_status": (
                    "threshold_merge" if predicted_id is not None else "threshold_nil"
                ),
                "trace": {
                    "query_tokens": sorted(query_tokens),
                    "top_20": top_20,
                    "best_canonical_node_id": (
                        None if best is None else int(best["canonical_node_id"])
                    ),
                    "best_score": (
                        None if best is None else float(best["jaccard_score"])
                    ),
                    "threshold": THRESHOLD,
                    "retrieval_stages": {
                        "jaccard_top_20": [
                            int(item["canonical_node_id"]) for item in top_20
                        ]
                    },
                },
            },
        )

    run_parallel_queries(runtime, worker, workers=1)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0
