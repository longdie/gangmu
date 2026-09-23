"""Candidate resolver for the Center-only Evidence ablation.

Evidence retrieval and Canonical aggregation are inherited unchanged from the
production resolver.  Only the representative Evidence payloads used by the
Reranker and the semantic judges are reduced to the fixed Center Evidence.
"""

from __future__ import annotations

from typing import Any

from src.cross_doc.resolution import FrozenCandidateResolver
from src.evidence_graph.derivation import clean_model_text, normalized_statement


class CenterOnlyResolver(FrozenCandidateResolver):
    """Use exactly one center Evidence for every candidate representation."""

    def candidate_representatives(
        self,
        candidate: dict[str, Any],
        *,
        include_query_match: bool = True,
    ) -> list[dict[str, Any]]:
        del include_query_match
        canonical_node_id = int(candidate["canonical_node_id"])
        canonical = self.canonical_nodes[canonical_node_id - 1]
        evidence_node_id = int(canonical["center_evidence_node_id"])
        if not 1 <= evidence_node_id <= len(self.evidence_nodes):
            raise ValueError("Center Evidence Node ID is invalid.")
        evidence = self.evidence_nodes[evidence_node_id - 1]
        statement = clean_model_text(normalized_statement(evidence))
        if not statement:
            raise ValueError(
                f"Center Evidence Node {evidence_node_id} has an empty statement."
            )
        return [
            {
                "role": "center_evidence",
                "evidence_node_id": evidence_node_id,
                "normalized_statement": statement,
            }
        ]


