from __future__ import annotations

from typing import Any

from scripts.experiment.common import ExperimentRuntime, run_parallel_queries


METHOD_NAME = "always_nil"
DISPLAY_NAME = "Always-NIL"


def method_settings(*_: Any, **__: Any) -> dict[str, Any]:
    return {"algorithm_version": 1, "prediction": "NIL"}


def run(runtime: ExperimentRuntime) -> None:
    def worker(query: dict[str, Any], existing: dict[str, Any] | None) -> None:
        del existing
        runtime.store.save_query(
            int(query["query_id"]),
            {
                "status": "completed",
                "predicted_canonical_node_id": None,
                "internal_status": "always_nil",
                "trace": {},
            },
        )

    run_parallel_queries(runtime, worker, workers=1)

