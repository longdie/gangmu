from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import numpy as np

from src.common.files import read_text
from src.common.paths import PROMPT_DIR
from src.evidence_graph.derivation import (
    EMBEDDING_DIMENSIONS,
    clean_model_text,
    normalized_statement,
)


RELATIONS = {
    "equivalent",
    "possible_equivalent",
    "not_equivalent",
}


def top_k_indices(
    scores: np.ndarray,
    top_k: int,
    *,
    tie_breaker_ids: np.ndarray | None = None,
) -> list[int]:
    """Return deterministic descending score indices."""

    if scores.ndim != 1:
        raise ValueError("Similarity scores must be one-dimensional.")
    if scores.size == 0:
        return []
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Similarity scores contain non-finite values.")
    tie_ids = (
        np.arange(scores.size)
        if tie_breaker_ids is None
        else np.asarray(tie_breaker_ids)
    )
    if tie_ids.shape != scores.shape or len(np.unique(tie_ids)) != len(tie_ids):
        raise ValueError("Similarity tie-breaker IDs are invalid.")
    count = min(top_k, int(scores.size))
    if count == int(scores.size):
        selected = np.arange(scores.size)
    else:
        boundary_score = float(np.partition(scores, -count)[-count])
        higher = np.flatnonzero(scores > boundary_score)
        tied = np.flatnonzero(scores == boundary_score)
        tied = tied[np.argsort(tie_ids[tied], kind="stable")]
        selected = np.concatenate((higher, tied[: count - len(higher)]))
    ordered = selected[np.lexsort((tie_ids[selected], -scores[selected]))]
    return ordered.tolist()


def candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(candidate[key])
        for key in (
            "canonical_node_id",
            "evidence_similarity",
            "evidence_hit_count",
            "query_match_evidence_node_id",
            "embedding_rank",
            "reranker_score",
            "reranker_rank",
        )
        if key in candidate
    }


def evidence_payload(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_node_id": int(node["id"]),
        "type": str(node["type"]),
        "title": str(node["title"]),
        "field": str(node["field"]),
        "subfield": str(node["subfield"]),
        "normalized_statement": normalized_statement(node),
    }


class FrozenCandidateResolver:
    """Resolve queries against an immutable Canonical Graph snapshot.

    The Canonical Graph is the target namespace. Its uniquely bound Evidence
    Graph supplies the Evidence-level vectors and representative statements
    required by Gangmu's candidate policy.
    """

    def __init__(
        self,
        *,
        canonical_nodes: list[dict[str, Any]],
        evidence_nodes: list[dict[str, Any]],
        evidence_embeddings: np.ndarray,
        evidence_node_to_canonical: np.ndarray,
        evidence_top_k: int,
        minimum_similarity: float,
        reranker_top_k: int,
        require_complete_mapping: bool = True,
    ) -> None:
        self.canonical_nodes = canonical_nodes
        self.evidence_nodes = evidence_nodes
        self.evidence_embeddings = evidence_embeddings
        self.mapping = np.asarray(evidence_node_to_canonical, dtype=np.int64)
        self.evidence_top_k = _positive_int(evidence_top_k, "evidence_top_k")
        self.reranker_top_k = _positive_int(reranker_top_k, "reranker_top_k")
        self.minimum_similarity = float(minimum_similarity)
        if not -1.0 <= self.minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between -1 and 1.")
        expected_shape = (len(evidence_nodes), EMBEDDING_DIMENSIONS)
        if evidence_embeddings.shape != expected_shape:
            raise ValueError(
                "Evidence embedding shape does not match Evidence Nodes: "
                f"expected {expected_shape}, got {evidence_embeddings.shape}."
            )
        if self.mapping.shape != (len(evidence_nodes),):
            raise ValueError("Evidence-to-Canonical mapping length is invalid.")
        minimum_mapping = 1 if require_complete_mapping else 0
        if self.mapping.size and (
            int(self.mapping.min()) < minimum_mapping
            or int(self.mapping.max()) > len(canonical_nodes)
        ):
            raise ValueError("Evidence-to-Canonical mapping contains invalid IDs.")

        self._ids_by_type: dict[str, np.ndarray] = {}
        # Frozen-KB queries repeatedly search the same type partition. Cache
        # each partition's matrix on first use instead of copying the memmap
        # for every query. Mutable Canonical construction passes an explicit
        # Evidence prefix and therefore never uses this cache.
        self._vectors_by_type: dict[str, np.ndarray] = {}
        for node_type in sorted({str(node["type"]) for node in evidence_nodes}):
            self._ids_by_type[node_type] = np.fromiter(
                (
                    int(node["id"])
                    for node in evidence_nodes
                    if str(node["type"]) == node_type
                ),
                dtype=np.int64,
            )

    def generate_candidates(
        self,
        *,
        query_node: dict[str, Any],
        query_embedding: np.ndarray,
        candidate_evidence_ids: np.ndarray | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        query_type = str(query_node["type"])
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        if query_vector.shape != (EMBEDDING_DIMENSIONS,):
            raise ValueError("Query embedding has an invalid shape.")
        if not np.all(np.isfinite(query_vector)):
            raise ValueError("Query embedding contains non-finite values.")

        trace: dict[str, Any] = {
            "same_type_only": True,
            "evidence_top_k": self.evidence_top_k,
            "minimum_similarity": self.minimum_similarity,
            "embedding_hits": [],
            "canonical_candidates_before_reranker": [],
        }
        if candidate_evidence_ids is None:
            candidate_ids = self._ids_by_type.get(
                query_type,
                np.empty((0,), dtype=np.int64),
            )
            vectors = self._vectors_by_type.get(query_type)
            if vectors is None:
                vectors = np.asarray(
                    self.evidence_embeddings[candidate_ids - 1],
                    dtype=np.float32,
                )
                self._vectors_by_type[query_type] = vectors
        else:
            raw_ids = np.asarray(candidate_evidence_ids, dtype=np.int64)
            if raw_ids.ndim != 1 or (
                raw_ids.size
                and (int(raw_ids.min()) < 1 or int(raw_ids.max()) > len(self.evidence_nodes))
            ):
                raise ValueError("Candidate Evidence Node IDs are invalid.")
            candidate_ids = np.asarray(
                [
                    int(node_id)
                    for node_id in raw_ids
                    if str(self.evidence_nodes[int(node_id) - 1]["type"])
                    == query_type
                ],
                dtype=np.int64,
            )
            vectors = np.asarray(
                self.evidence_embeddings[candidate_ids - 1],
                dtype=np.float32,
            )
        candidates: dict[int, dict[str, Any]] = {}
        if candidate_ids.size:
            scores = np.asarray(vectors @ query_vector, dtype=np.float32)
            selected = top_k_indices(
                scores,
                self.evidence_top_k,
                tie_breaker_ids=candidate_ids,
            )
            for rank, local_index in enumerate(selected, start=1):
                score = float(scores[local_index])
                evidence_node_id = int(candidate_ids[local_index])
                canonical_node_id = int(self.mapping[evidence_node_id - 1])
                if canonical_node_id <= 0:
                    raise ValueError(
                        "A candidate Evidence Node has no Canonical mapping."
                    )
                retained = score >= self.minimum_similarity
                trace["embedding_hits"].append(
                    {
                        "rank": rank,
                        "evidence_node_id": evidence_node_id,
                        "canonical_node_id": canonical_node_id,
                        "similarity": score,
                        "retained": retained,
                    }
                )
                if not retained:
                    continue
                candidate = candidates.setdefault(
                    canonical_node_id,
                    {
                        "canonical_node_id": canonical_node_id,
                        "evidence_similarity": score,
                        "evidence_hit_count": 0,
                        "query_match_evidence_node_id": evidence_node_id,
                    },
                )
                candidate["evidence_hit_count"] += 1
                if score > float(candidate["evidence_similarity"]):
                    candidate["evidence_similarity"] = score
                    candidate["query_match_evidence_node_id"] = evidence_node_id

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -float(item["evidence_similarity"]),
                int(item["canonical_node_id"]),
            ),
        )
        trace["canonical_candidates_before_reranker"] = [
            candidate_record(candidate) for candidate in ordered
        ]
        return ordered, trace

    def rerank_candidates(
        self,
        *,
        query_node: dict[str, Any],
        candidates: list[dict[str, Any]],
        reranker_client: Any,
        usage_task: str,
        retain_all_scores: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        input_records = [candidate_record(item) for item in candidates]
        if len(candidates) <= self.reranker_top_k:
            retained = deepcopy(candidates)
            return retained, {
                "applied": False,
                "top_k": self.reranker_top_k,
                "input_candidates": input_records,
                "scored_candidates": [],
                "canonical_candidates_after_reranker": [
                    candidate_record(item) for item in retained
                ],
            }

        documents = [self.format_reranker_document(item) for item in candidates]
        requested_top_n = len(documents) if retain_all_scores else min(
            self.reranker_top_k,
            len(documents),
        )
        results = reranker_client.rerank(
            query=self.format_reranker_query(query_node),
            documents=documents,
            top_n=requested_top_n,
            usage_task=usage_task,
        )
        scored: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        for result in results:
            index = int(result["index"])
            score = float(result["score"])
            if (
                not 0 <= index < len(candidates)
                or index in seen_indices
                or not np.isfinite(score)
            ):
                continue
            seen_indices.add(index)
            candidate = deepcopy(candidates[index])
            candidate["reranker_score"] = score
            candidate["_reranker_input_rank"] = index + 1
            scored.append(candidate)
        if not scored:
            raise RuntimeError("Reranker returned no usable Canonical candidates.")
        if retain_all_scores and len(scored) != len(candidates):
            raise RuntimeError(
                "Reranker did not return a score for every Canonical candidate."
            )
        # Do not rely on a provider-specific result order.  The original
        # candidate rank (Embedding rank) is the documented secondary key.
        scored.sort(
            key=lambda item: (
                -float(item["reranker_score"]),
                int(item["_reranker_input_rank"]),
                int(item["canonical_node_id"]),
            )
        )
        for rank, candidate in enumerate(scored, start=1):
            candidate["reranker_rank"] = rank
            candidate.pop("_reranker_input_rank", None)
        retained = scored[: self.reranker_top_k]
        return retained, {
            "applied": True,
            "top_k": self.reranker_top_k,
            "input_candidates": input_records,
            "scored_candidates": [candidate_record(item) for item in scored],
            "canonical_candidates_after_reranker": [
                candidate_record(item) for item in retained
            ],
        }

    def candidate_representatives(
        self,
        candidate: dict[str, Any],
        *,
        include_query_match: bool = True,
    ) -> list[dict[str, Any]]:
        canonical_node_id = int(candidate["canonical_node_id"])
        node = self.canonical_nodes[canonical_node_id - 1]
        specs: list[tuple[str, int | None]] = [
            ("center_evidence", int(node["center_evidence_node_id"])),
            *[
                (f"diverse_evidence_{index}", int(evidence_node_id))
                for index, evidence_node_id in enumerate(
                    node.get("diverse_evidence_node_ids", []),
                    start=1,
                )
            ],
        ]
        if include_query_match:
            raw_query_match = candidate.get("query_match_evidence_node_id")
            specs.append(
                (
                    "query_matched_evidence",
                    None if raw_query_match is None else int(raw_query_match),
                )
            )

        representatives: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        seen_statements: set[str] = set()
        for role, evidence_node_id in specs:
            if evidence_node_id is None or evidence_node_id in seen_ids:
                continue
            if not 1 <= evidence_node_id <= len(self.evidence_nodes):
                raise ValueError("Representative Evidence Node ID is invalid.")
            seen_ids.add(evidence_node_id)
            statement = clean_model_text(
                normalized_statement(self.evidence_nodes[evidence_node_id - 1])
            )
            key = " ".join(statement.split())
            if not key or key in seen_statements:
                continue
            seen_statements.add(key)
            representatives.append(
                {
                    "role": role,
                    "evidence_node_id": evidence_node_id,
                    "normalized_statement": statement,
                }
            )
        return representatives

    def canonical_payload(
        self,
        canonical_node_id: int,
        *,
        query_match_evidence_node_id: int | None = None,
        include_query_match: bool = True,
    ) -> dict[str, Any]:
        node = self.canonical_nodes[canonical_node_id - 1]
        candidate = {
            "canonical_node_id": canonical_node_id,
            "query_match_evidence_node_id": query_match_evidence_node_id,
        }
        return {
            "canonical_node_id": canonical_node_id,
            "type": str(node["type"]),
            "title": str(node["title"]),
            "field": str(node["field"]),
            "subfield": str(node["subfield"]),
            "representative_evidence_nodes": self.candidate_representatives(
                candidate,
                include_query_match=include_query_match,
            ),
        }

    def shared_candidate_snapshot(
        self,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        canonical_node_id = int(candidate["canonical_node_id"])
        payload = self.canonical_payload(
            canonical_node_id,
            query_match_evidence_node_id=int(
                candidate["query_match_evidence_node_id"]
            ),
            include_query_match=True,
        )
        return {
            **candidate_record(candidate),
            "candidate_canonical_node": payload,
            "cluster_representative_evidence_nodes": (
                self.candidate_representatives(
                    {"canonical_node_id": canonical_node_id},
                    include_query_match=False,
                )
            ),
        }

    def format_reranker_query(self, query_node: dict[str, Any]) -> str:
        return "\n".join(
            (
                f"Type: {query_node['type']}",
                f"Title: {query_node['title']}",
                f"Field: {query_node['field']}",
                f"Subfield: {query_node['subfield']}",
                "Normalized statement: " + normalized_statement(query_node),
            )
        )

    def format_reranker_document(self, candidate: dict[str, Any]) -> str:
        lines: list[str] = []
        for representative in self.candidate_representatives(candidate):
            role = str(representative["role"]).replace("_", " ").title()
            lines.append(
                f"{role} statement: {representative['normalized_statement']}"
            )
        return "\n".join(lines)


def validate_single_relation(result: dict[str, Any]) -> tuple[bool, str]:
    if (
        not isinstance(result, dict)
        or set(result) != {"relation"}
        or result.get("relation") not in RELATIONS
    ):
        return False, "Output must contain exactly one valid relation."
    return True, ""


def judge_evidence_to_canonical(
    *,
    llm_client: Any,
    model_task: str,
    prompt: str,
    query_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    usage_task: str,
) -> str:
    result = llm_client.call_json(
        model_task=model_task,
        prompt=prompt,
        input_text=json.dumps(
            {
                "current_evidence_node": query_payload,
                "candidate_canonical_node": candidate_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        usage_task=usage_task,
        validator=validate_single_relation,
        preserve_latex_escapes=True,
        repair_instructions=(
            read_text(PROMPT_DIR / "ternary_relation_output.md")
        ),
    )
    if result is None:
        raise RuntimeError("Semantic alignment returned no valid result.")
    return str(result["relation"])


def judge_canonical_pair(
    *,
    llm_client: Any,
    model_task: str,
    prompt: str,
    canonical_node_a: dict[str, Any],
    canonical_node_b: dict[str, Any],
    usage_task: str,
) -> str:
    result = llm_client.call_json(
        model_task=model_task,
        prompt=prompt,
        input_text=json.dumps(
            {
                "canonical_node_a": canonical_node_a,
                "canonical_node_b": canonical_node_b,
            },
            ensure_ascii=False,
            indent=2,
        ),
        usage_task=usage_task,
        validator=validate_single_relation,
        preserve_latex_escapes=True,
        repair_instructions=(
            read_text(PROMPT_DIR / "ternary_relation_output.md")
        ),
    )
    if result is None:
        raise RuntimeError("Strict Canonical comparison returned no valid result.")
    return str(result["relation"])


def connected_components(graph: dict[int, set[int]]) -> list[set[int]]:
    unseen = set(graph)
    components: list[set[int]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            stack.extend(graph[node] - component)
        components.append(component)
    return components


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value
