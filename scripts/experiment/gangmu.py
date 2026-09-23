from __future__ import annotations

from copy import deepcopy
import threading
from typing import Any

from src.common.files import read_json, read_text, utc_now, write_json_atomic
from src.cross_doc.pacing import CrossDocumentRequestPacer

from scripts.benchmark.model_judging import (
    MODEL_SPECS,
    BenchmarkModelSpec,
    QueryModelJudge,
)
from scripts.experiment.common import (
    ExperimentRuntime,
    run_parallel_queries,
    sha256_json,
    usage_for_task_prefix,
)
from scripts.experiment.dataset import PROJECT_ROOT, sha256_file


METHOD_NAME = "gangmu"
DISPLAY_NAME = "Gangmu"
EVIDENCE_TOP_K = 50
SIMILARITY_THRESHOLD = 0.5
RERANKER_TOP_K = 20
MODEL_TASK = "sev"
ALIGNMENT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "cross_doc_alignment.md"
STRICT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "cross_doc_strict_cluster_pair.md"
IMPLEMENTATION_DEPENDENCIES = (
    "scripts/benchmark/model_judging.py",
    "scripts/benchmark/schema.py",
    "src/cross_doc/clients.py",
    "src/cross_doc/resolution.py",
    "src/single_doc/llm.py",
)




def method_settings(
    dataset: Any,
    config: Any,
    options: dict[str, Any] | None = None,
    **__: Any,
) -> dict[str, Any]:
    active_options = dict(options or {})
    model_task = str(active_options.get("model_task", MODEL_TASK)).strip()
    if not model_task:
        raise ValueError("Gangmu model_task cannot be empty.")
    model = config.model(model_task)
    provider = config.provider(model.provider, require_api_key=True)
    workers = int(
        active_options.get(
            "workers",
            config.get_int(
                "experiment.parallel_workers",
                minimum=1,
                maximum=10,
            ),
        )
    )
    if not 1 <= workers <= 10:
        raise ValueError("Gangmu workers must be between 1 and 10.")
    reranker = config.model("reranker")
    embedding = config.model("embedding")
    return {
        "algorithm_version": 4,
        "evidence_top_k": EVIDENCE_TOP_K,
        "minimum_similarity": SIMILARITY_THRESHOLD,
        "reranker_top_k": RERANKER_TOP_K,
        "same_type_only": True,
        "model_task": model_task,
        "model": model.name,
        "provider": model.provider,
        "workers": workers,
        "llm_parameters": {
            "json_mode": True,
            "max_completion_tokens": config.get_int(
                "requests.max_completion_tokens",
                minimum=1,
            ),
            "call_attempts": config.get_int(
                "cross_document.llm_call_attempts",
                minimum=1,
            ),
            "json_repair_attempts": config.get_int(
                "requests.json_repair_attempts",
                minimum=0,
            ),
        },
        "reranker_model": reranker.name,
        "reranker_provider": reranker.provider,
        "embedding_model": embedding.name,
        "embedding_dimensions": embedding.dimensions,
        "alignment_prompt_sha256": sha256_file(ALIGNMENT_PROMPT_PATH),
        "strict_prompt_sha256": sha256_file(STRICT_PROMPT_PATH),
        # MULTIPLE_TARGETS is retained as an audit status.  The evaluated
        # decision is resolved to one strict group or NIL after pairwise checks.
        "multiple_targets_final_prediction": "strict_group_or_nil",
        "llm_pacing": {
            "enabled": config.get_bool("cross_document_pacing.enabled"),
            "work_minutes": config.get_float(
                "cross_document_pacing.work_minutes",
                minimum=0,
            ),
            "short_rest_minutes": config.get_float(
                "cross_document_pacing.short_rest_minutes",
                minimum=0,
            ),
            "windows_per_cycle": config.get_int(
                "cross_document_pacing.windows_per_cycle",
                minimum=1,
            ),
            "cycle_extra_rest_minutes": config.get_float(
                "cross_document_pacing.cycle_extra_rest_minutes",
                minimum=0,
            ),
        },
    }


def run(runtime: ExperimentRuntime) -> None:
    variant = str(runtime.options.get("variant", "gangmu"))
    from scripts.ablations.strategies import build_strategy
    strategy = build_strategy(runtime, variant)
    alignment_prompt = read_text(strategy["alignment_prompt"])
    strict_prompt = read_text(strategy["strict_prompt"])
    request_pacer = CrossDocumentRequestPacer.from_config(
        runtime.config,
        announce=lambda message: print(
            message.replace("[canonical-graph]", "[experiment:gangmu]")
        ),
    )
    pacing_state_path = runtime.store.checkpoint_dir / "llm_pacing_state.json"
    pacing_state_lock = threading.Lock()
    if pacing_state_path.is_file():
        request_pacer.restore_state(read_json(pacing_state_path))

    def save_pacing_state() -> None:
        with pacing_state_lock:
            write_json_atomic(pacing_state_path, request_pacer.snapshot_state())

    resolver = strategy["resolver"]
    reranker_client = runtime.reranker_client() if variant != "without_reranking" else None
    judge = strategy["judge_class"](
        llm_client=runtime.llm_client(request_pacer=request_pacer),
        alignment_prompt=alignment_prompt,
        strict_cluster_prompt=strict_prompt,
        **strategy["judge_options"],
    )
    model_task = str(runtime.settings["model_task"])
    spec = next((spec for spec in MODEL_SPECS if spec.config_task == model_task), None)
    if spec is None:
        spec = BenchmarkModelSpec(
            label=str(runtime.settings["model"]),
            config_task=model_task,
            workers_config_key="experiment.parallel_workers",
        )
    model_metadata = spec.metadata(runtime.config)
    workers = int(runtime.settings["workers"])

    def worker(query: dict[str, Any], existing: dict[str, Any] | None) -> None:
        query_id = int(query["query_id"])
        usage_prefix = f"experiment.{runtime.method_name}.q{query_id}"
        started_at = utc_now()
        trace = dict((existing or {}).get("trace", {}))
        if existing is not None:
            started_at = str(existing.get("started_at", started_at))

        candidate_record = trace.get("candidate_record")
        if not isinstance(candidate_record, dict):
            query_node = strategy["query_node"](query)
            candidates, embedding_trace = resolver.generate_candidates(
                query_node=query_node,
                query_embedding=strategy["query_embedding"](query),
            )
            if variant == "without_reranking":
                retained = deepcopy(candidates[:RERANKER_TOP_K])
                reranker_trace = {"applied": False, "policy": "removed_by_ablation"}
            else:
                retained, reranker_trace = resolver.rerank_candidates(
                    query_node=query_node,
                    candidates=candidates,
                    reranker_client=reranker_client,
                    usage_task=f"{usage_prefix}.reranker",
                    retain_all_scores=True,
                )
            shared_candidates = [
                resolver.shared_candidate_snapshot(candidate)
                for candidate in retained
            ]
            query_payload = strategy["query_payload"](query)
            candidate_snapshot_sha256 = sha256_json(
                {
                    "query_payload": query_payload,
                    "shared_candidates": shared_candidates,
                }
            )
            candidate_record = {
                "query_id": int(query["query_id"]),
                "source_id": str(query.get("source_id", "")),
                "query_payload": query_payload,
                "shared_candidates": shared_candidates,
                "candidate_snapshot_sha256": candidate_snapshot_sha256,
                "usage_task_prefix": usage_prefix,
            }
            trace.update(
                {
                    "candidate_record": candidate_record,
                    "embedding_top_50": embedding_trace["embedding_hits"],
                    "canonical_candidates_before_reranker": embedding_trace[
                        "canonical_candidates_before_reranker"
                    ],
                    "reranker": reranker_trace,
                    "retrieval_stages": {
                        "evidence_top_50": [
                            int(item["canonical_node_id"])
                            for item in embedding_trace["embedding_hits"]
                        ],
                        "gangmu_top_20": [
                            int(item["canonical_node_id"])
                            for item in retained
                        ],
                    },
                }
            )
            runtime.store.save_query(
                query_id,
                {
                    "status": "partial",
                    "started_at": started_at,
                    "predicted_canonical_node_id": None,
                    "internal_status": "candidates_ready",
                    "trace": trace,
                },
            )
        candidate_record.setdefault("usage_task_prefix", usage_prefix)

        existing_prediction = trace.get("model_prediction")

        def save_prediction(prediction: dict[str, Any]) -> None:
            trace["model_prediction"] = prediction
            runtime.store.save_query(
                query_id,
                {
                    "status": "partial",
                    "started_at": started_at,
                    "predicted_canonical_node_id": None,
                    "internal_status": "judging_candidates",
                    "trace": trace,
                },
            )

        prediction_record = judge.complete(
            spec=spec,
            model_metadata=model_metadata,
            candidate_record=candidate_record,
            existing=(
                existing_prediction
                if isinstance(existing_prediction, dict)
                else None
            ),
            save=save_prediction,
        )
        trace["model_prediction"] = prediction_record
        trace["token_usage"] = usage_for_task_prefix(
            runtime.usage_tracker,
            f"{usage_prefix}.",
        )
        query_prediction = prediction_record["query_prediction"]
        resolved_prediction = query_prediction.get("resolved_prediction")
        if not isinstance(resolved_prediction, dict):
            resolved_prediction = query_prediction
        label = str(resolved_prediction["label"])
        accepted_target_ids = [
            int(value)
            for value in resolved_prediction.get(
                "accepted_canonical_node_ids", []
            )
        ]
        predicted_id = (
            int(resolved_prediction["canonical_node_id"])
            if label == "MERGE"
            else None
        )
        runtime.store.save_query(
            query_id,
            {
                "status": "completed",
                "started_at": started_at,
                "completed_at": utc_now(),
                "predicted_canonical_node_id": predicted_id,
                "predicted_canonical_node_ids": accepted_target_ids,
                "internal_status": str(query_prediction["label"]),
                "resolution_status": label,
                "trace": trace,
            },
        )
        save_pacing_state()

    pacing = runtime.settings["llm_pacing"]
    third_rest = float(pacing["short_rest_minutes"]) + float(
        pacing["cycle_extra_rest_minutes"]
    )
    print(
        "[experiment:gangmu] LLM pacing: "
        f"enabled={pacing['enabled']}, "
        f"work={float(pacing['work_minutes']):g} min, "
        f"short_rest={float(pacing['short_rest_minutes']):g} min, "
        f"windows_per_cycle={int(pacing['windows_per_cycle'])}, "
        f"third_rest={third_rest:g} min"
    )
    try:
        run_parallel_queries(runtime, worker, workers=workers)
    finally:
        save_pacing_state()
