from __future__ import annotations

import hashlib
import threading
from typing import Any

from rank_bm25 import BM25Okapi

from src.common.files import read_json, read_text, utc_now, write_json_atomic
from src.cross_doc.pacing import CrossDocumentRequestPacer
from src.cross_doc.resolution import judge_evidence_to_canonical

from scripts.experiment.representations import CanonicalTextView

from scripts.experiment.common import (
    ExperimentRuntime,
    mathematical_tokens,
    run_parallel_queries,
    usage_for_task_prefix,
)
from scripts.experiment.dataset import PROJECT_ROOT, sha256_file


METHOD_NAME = "bm25_llm"
DISPLAY_NAME = "BM25 + SEV"
TOP_K = 20
K1 = 1.5
B = 0.75
PROMPT_PATH = PROJECT_ROOT / "prompts" / "cross_doc_alignment.md"
MODEL_TASK = "sev"
IMPLEMENTATION_DEPENDENCIES = (
    "src/cross_doc/resolution.py",
    "scripts/experiment/representations.py",
    "src/single_doc/llm.py",
)


def method_settings(
    dataset: Any,
    config: Any,
    options: dict[str, Any] | None = None,
    **__: Any,
) -> dict[str, Any]:
    del dataset
    active_options = dict(options or {})
    model_task = str(active_options.get("model_task", MODEL_TASK)).strip()
    if not model_task:
        raise ValueError("BM25 + SEV model_task cannot be empty.")
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
        raise ValueError("BM25 + SEV workers must be between 1 and 10.")
    return {
        "algorithm_version": 3,
        "top_k": TOP_K,
        "same_type_only": True,
        "bm25": {"implementation": "rank_bm25.BM25Okapi", "k1": K1, "b": B},
        "tokenization": "unicode_nfkc_lower_latex_alnum_symbols",
        "decision": "rank_order_first_equivalent",
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
        "alignment_prompt_sha256": sha256_file(PROMPT_PATH),
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
    prompt = read_text(PROMPT_PATH)
    request_pacer = CrossDocumentRequestPacer.from_config(
        runtime.config,
        announce=lambda message: print(
            message.replace("[canonical-graph]", "[experiment:bm25_llm]")
        ),
    )
    pacing_state_path = runtime.store.checkpoint_dir / "llm_pacing_state.json"
    pacing_state_lock = threading.Lock()
    if pacing_state_path.is_file():
        request_pacer.restore_state(read_json(pacing_state_path))

    def save_pacing_state() -> None:
        with pacing_state_lock:
            write_json_atomic(pacing_state_path, request_pacer.snapshot_state())

    resolver = CanonicalTextView(runtime.dataset)
    indexes: dict[str, BM25Okapi] = {}
    ids_by_type: dict[str, Any] = {}
    for node_type in sorted(
        {str(node["type"]) for node in runtime.dataset.canonical_nodes}
    ):
        ids = runtime.dataset.canonical_ids_for_type(node_type)
        corpus = [
            mathematical_tokens(
                str(runtime.dataset.canonical_nodes[int(node_id) - 1]["statement"])
            )
            for node_id in ids
        ]
        indexes[node_type] = BM25Okapi(corpus, k1=K1, b=B)
        ids_by_type[node_type] = ids

    model_task = str(runtime.settings["model_task"])
    llm_client = runtime.llm_client(request_pacer=request_pacer)
    workers = int(runtime.settings["workers"])

    def worker(query: dict[str, Any], existing: dict[str, Any] | None) -> None:
        query_id = int(query["query_id"])
        usage_prefix = f"experiment.bm25_llm.q{query_id}."
        node_type = str(query["type"])
        candidate_ids = ids_by_type[node_type]
        scores = indexes[node_type].get_scores(
            mathematical_tokens(str(query["normalized_statement"]))
        )
        ordered_indices = sorted(
            range(len(candidate_ids)),
            key=lambda index: (-float(scores[index]), int(candidate_ids[index])),
        )[:TOP_K]
        top_20 = [
            {
                "rank": rank,
                "canonical_node_id": int(candidate_ids[index]),
                "bm25_score": float(scores[index]),
            }
            for rank, index in enumerate(ordered_indices, start=1)
        ]
        candidate_ids_snapshot = [
            int(item["canonical_node_id"]) for item in top_20
        ]

        trace: dict[str, Any]
        started_at = utc_now()
        if existing is not None:
            trace = dict(existing.get("trace", {}))
            started_at = str(existing.get("started_at", started_at))
            if trace.get("candidate_ids_snapshot") != candidate_ids_snapshot:
                raise ValueError(
                    f"BM25 candidates changed while resuming query {query_id}."
                )
        else:
            trace = {
                "top_20": top_20,
                "candidate_ids_snapshot": candidate_ids_snapshot,
                "candidate_payloads": [
                    resolver.canonical_payload(
                        int(item["canonical_node_id"]),
                        include_query_match=False,
                    )
                    for item in top_20
                ],
                "candidate_judgments": [],
                "retrieval_stages": {"bm25_top_20": candidate_ids_snapshot},
            }

        judgments = trace.setdefault("candidate_judgments", [])
        completed_by_id = {
            int(item["canonical_node_id"]): item for item in judgments
        }
        equivalent = next(
            (
                int(item["canonical_node_id"])
                for item in judgments
                if item.get("relation") == "equivalent"
            ),
            None,
        )
        if equivalent is None:
            query_payload = runtime.dataset.query_payload(query)
            payload_by_id = {
                int(payload["canonical_node_id"]): payload
                for payload in trace["candidate_payloads"]
            }
            for item in top_20:
                canonical_node_id = int(item["canonical_node_id"])
                if canonical_node_id in completed_by_id:
                    continue
                relation = judge_evidence_to_canonical(
                    llm_client=llm_client,
                    model_task=model_task,
                    prompt=prompt,
                    query_payload=query_payload,
                    candidate_payload=payload_by_id[canonical_node_id],
                    usage_task=f"{usage_prefix}alignment",
                )
                judgment = {
                    "rank": int(item["rank"]),
                    "canonical_node_id": canonical_node_id,
                    "relation": relation,
                }
                judgments.append(judgment)
                completed_by_id[canonical_node_id] = judgment
                trace["stop_position"] = int(item["rank"])
                trace["stop_reason"] = (
                    "first_equivalent" if relation == "equivalent" else "continue"
                )
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
                save_pacing_state()
                if relation == "equivalent":
                    equivalent = canonical_node_id
                    break

        if equivalent is None:
            trace["stop_position"] = len(judgments)
            trace["stop_reason"] = "no_equivalent_in_top_20"
        trace["token_usage"] = usage_for_task_prefix(
            runtime.usage_tracker,
            usage_prefix,
        )
        runtime.store.save_query(
            query_id,
            {
                "status": "completed",
                "started_at": started_at,
                "completed_at": utc_now(),
                "predicted_canonical_node_id": equivalent,
                "internal_status": (
                    "first_equivalent" if equivalent is not None else "no_equivalent"
                ),
                "trace": trace,
            },
        )
        save_pacing_state()

    pacing = runtime.settings["llm_pacing"]
    third_rest = float(pacing["short_rest_minutes"]) + float(
        pacing["cycle_extra_rest_minutes"]
    )
    print(
        "[experiment:bm25_llm] LLM pacing: "
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
