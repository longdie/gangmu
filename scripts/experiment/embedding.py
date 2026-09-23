from __future__ import annotations

from typing import Any

import numpy as np

from scripts.experiment.common import (
    ExperimentRuntime,
    run_parallel_queries,
    top_scored_candidates,
)


METHOD_NAME = "embedding"
DISPLAY_NAME = "Dense Retrieval"
TOP_K = 20
THRESHOLD = 0.5


def method_settings(dataset: Any, config: Any, *_: Any, **__: Any) -> dict[str, Any]:
    del dataset
    embedding = config.model("embedding")
    return {
        "algorithm_version": 2,
        "top_k": TOP_K,
        "threshold": THRESHOLD,
        "same_type_only": True,
        "similarity": "cosine",
        "model": embedding.name,
        "dimensions": embedding.dimensions,
        "l2_normalized": True,
        "canonical_embedding_file": "canonical_statement_embeddings.npy",
    }


def run(runtime: ExperimentRuntime) -> None:
    def worker(query: dict[str, Any], existing: dict[str, Any] | None) -> None:
        del existing
        node_type = str(query["type"])
        candidate_ids = runtime.dataset.canonical_ids_for_type(node_type)
        vectors = runtime.dataset.canonical_vectors_for_type(node_type)
        query_vector = runtime.dataset.query_embedding(query)
        scores = np.asarray(vectors @ query_vector, dtype=np.float32)
        top_20 = top_scored_candidates(
            candidate_ids=candidate_ids,
            scores=scores,
            top_k=TOP_K,
            score_key="cosine_similarity",
        )
        best = top_20[0] if top_20 else None
        predicted_id = (
            int(best["canonical_node_id"])
            if best is not None and float(best["cosine_similarity"]) >= THRESHOLD
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
                    "top_20": top_20,
                    "best_canonical_node_id": (
                        None if best is None else int(best["canonical_node_id"])
                    ),
                    "best_score": (
                        None if best is None else float(best["cosine_similarity"])
                    ),
                    "threshold": THRESHOLD,
                    "retrieval_stages": {
                        "embedding_top_20": [
                            int(item["canonical_node_id"]) for item in top_20
                        ]
                    },
                },
            },
        )

    run_parallel_queries(runtime, worker, workers=1)
