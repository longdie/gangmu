"""Official LLMForEM zero-shot matcher adapted to frozen-KB MathER.

Prompt source: wbsg-uni-mannheim/MatchGPT, LLMForEM,
commit ba1691935c744bb0eacc9abf772f39271e318e4b.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from scripts.experiment.baseline_common import llm_settings, run_pairwise
from scripts.experiment.common import top_scored_candidates


DISPLAY_NAME = "LLMForEM"
TOP_K = 20
UPSTREAM_COMMIT = "ba1691935c744bb0eacc9abf772f39271e318e4b"
IMPLEMENTATION_DEPENDENCIES = ("scripts/experiment/baseline_common.py",)
PROMPT_TEMPLATE = (
    "Do the two entity descriptions refer to the same real-world entity? "
    "Answer with 'Yes' if they do and 'No' if they do not.\n"
    "Entity 1: '{entity1}'\nEntity 2: '{entity2}'"
)


def serialize_entity(entity: dict[str, Any], *, query: bool) -> str:
    statement_key = "normalized_statement" if query else "statement"
    values = [entity.get("type"), entity.get("title"), entity.get(statement_key)]
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("LLMForEM requires nonempty type, title and statement values.")
    return " ".join(values)


def pair_prompt(query: dict[str, Any], canonical: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        entity1=serialize_entity(query, query=True),
        entity2=serialize_entity(canonical, query=False),
    )


def parse_equivalence(text: str) -> bool:
    match = re.fullmatch(r"(Yes|No)[.!]?", text.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError("LLMForEM expected an unambiguous Yes/No answer.")
    return match.group(1).lower() == "yes"


def method_settings(dataset: Any, config: Any, *_: Any, **__: Any) -> dict[str, Any]:
    del dataset
    embedding = config.model("embedding")
    return {
        "algorithm_version": 1,
        **llm_settings(config, "llmforem", 1024),
        "top_k": TOP_K,
        "same_type_only": True,
        "embedding_model": embedding.name,
        "dimensions": embedding.dimensions,
        "upstream_commit": UPSTREAM_COMMIT,
        "serialization": "type_title_statement_values_only",
        "decision": "judge_all_then_first_ranked_yes",
    }


def retrieve(dataset: Any, query: dict[str, Any]) -> list[dict[str, Any]]:
    ids = dataset.canonical_ids_for_type(query["type"])
    scores = np.asarray(
        dataset.canonical_vectors_for_type(query["type"]) @ dataset.query_embedding(query),
        dtype=np.float32,
    )
    return top_scored_candidates(
        candidate_ids=ids, scores=scores, top_k=TOP_K, score_key="cosine_similarity",
    )


def run(runtime) -> None:
    nodes = {int(node["id"]): node for node in runtime.dataset.canonical_nodes}
    run_pairwise(
        runtime,
        candidates_for_query=lambda query: retrieve(runtime.dataset, query),
        pair_prompt=lambda query, node_id: pair_prompt(query, nodes[node_id]),
        parse_pair=parse_equivalence,
        retrieval_stage="embedding_top_20",
    )
