from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.common.files import write_json_atomic
from src.cross_doc.clients import CanonicalStatementEmbedder, RerankerClient
from src.cross_doc.resolution import FrozenCandidateResolver

from scripts.experiment.dataset import ExperimentDataset


EVIDENCE_TOP_K = 50
SIMILARITY_THRESHOLD = 0.5
RERANKER_TOP_K = 20


class AgentRetrievalService:
    def __init__(
        self,
        *,
        dataset: ExperimentDataset,
        resolver: FrozenCandidateResolver,
        reranker_client: RerankerClient,
        query_embedder: CanonicalStatementEmbedder,
    ) -> None:
        self.dataset = dataset
        self.resolver = resolver
        self.reranker_client = reranker_client
        self.query_embedder = query_embedder
        self._candidate_lock = threading.Lock()

    def original_search(
        self,
        *,
        query: dict[str, Any],
        output_dir: Path,
        usage_prefix: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._search(
            query=query,
            query_text=str(query["normalized_statement"]),
            query_embedding=self.dataset.query_embedding(query),
            search_id=1,
            search_type="original",
            output_dir=output_dir,
            usage_prefix=usage_prefix,
        )

    def reformulated_search(
        self,
        *,
        query: dict[str, Any],
        query_text: str,
        search_id: int,
        output_dir: Path,
        usage_prefix: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cleaned = " ".join(str(query_text or "").split())
        if not cleaned:
            raise ValueError("search_again query_text must be non-empty.")
        if len(cleaned) > 20_000:
            raise ValueError("search_again query_text exceeds 20000 characters.")
        vector = self.query_embedder.embed(
            cleaned,
            usage_task=f"{usage_prefix}.search_{search_id}.embedding",
        )
        return self._search(
            query=query,
            query_text=cleaned,
            query_embedding=vector,
            search_id=search_id,
            search_type="reformulated",
            output_dir=output_dir,
            usage_prefix=usage_prefix,
        )

    def _search(
        self,
        *,
        query: dict[str, Any],
        query_text: str,
        query_embedding: np.ndarray,
        search_id: int,
        search_type: str,
        output_dir: Path,
        usage_prefix: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        query_node = self.dataset.query_node(query)
        if search_type == "reformulated":
            query_node = deepcopy(query_node)
            query_node["semantic"] = {"normalized_statement": query_text}

        # FrozenCandidateResolver lazily builds same-type vector caches. Keep
        # that short section serialized; external Reranker calls remain
        # concurrent and use their own thread-safe client.
        with self._candidate_lock:
            candidates, embedding_trace = self.resolver.generate_candidates(
                query_node=query_node,
                query_embedding=query_embedding,
            )
        retained, reranker_trace = self.resolver.rerank_candidates(
            query_node=query_node,
            candidates=candidates,
            reranker_client=self.reranker_client,
            usage_task=f"{usage_prefix}.search_{search_id}.reranker",
            retain_all_scores=True,
        )
        snapshots = [
            self.resolver.shared_candidate_snapshot(candidate)
            for candidate in retained
        ]
        agent_candidates = [
            _compact_candidate(snapshot, rank=rank)
            for rank, snapshot in enumerate(snapshots, start=1)
        ]
        ranking_source = "reranker" if reranker_trace["applied"] else "embedding"
        agent_result = {
            "query_id": int(query["query_id"]),
            "search_id": int(search_id),
            "search_type": str(search_type),
            "search_query": str(query_text),
            "ranking_source": ranking_source,
            "candidate_count": len(agent_candidates),
            "retrieval_policy": {
                "same_type_only": True,
                "evidence_top_k": EVIDENCE_TOP_K,
                "minimum_similarity": SIMILARITY_THRESHOLD,
                "reranker_top_k": RERANKER_TOP_K,
            },
            "candidates": agent_candidates,
        }
        if len(agent_candidates) > RERANKER_TOP_K:
            raise AssertionError("Agent-visible retrieval exceeded Top-20.")
        audit_trace = {
            "query_id": int(query["query_id"]),
            "search_id": int(search_id),
            "search_type": str(search_type),
            "search_query": str(query_text),
            "embedding_top_50": embedding_trace["embedding_hits"],
            "canonical_candidates_before_reranker": embedding_trace[
                "canonical_candidates_before_reranker"
            ],
            "reranker": reranker_trace,
            "final_candidate_snapshots": snapshots,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = (
            "search_01_original"
            if search_id == 1
            else f"search_{search_id:02d}_reformulated"
        )
        write_json_atomic(output_dir / f"{base_name}.json", agent_result)
        write_json_atomic(
            output_dir / f"{base_name}_retrieval_trace.json",
            audit_trace,
        )
        return agent_result, audit_trace


def candidate_ids(searches: list[dict[str, Any]]) -> set[int]:
    return {
        int(candidate["canonical_node_id"])
        for search in searches
        for candidate in search.get("candidates", [])
    }


def _compact_candidate(
    snapshot: dict[str, Any],
    *,
    rank: int,
) -> dict[str, Any]:
    payload = snapshot["candidate_canonical_node"]
    representatives = [
        {
            "role": str(item["role"]),
            "normalized_statement": str(item["normalized_statement"]),
        }
        for item in payload["representative_evidence_nodes"]
    ]
    return {
        "rank": int(rank),
        "canonical_node_id": int(payload["canonical_node_id"]),
        "type": str(payload["type"]),
        "title": str(payload["title"]),
        "representative_evidence_nodes": representatives,
    }
