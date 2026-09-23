"""Evidence-level candidate resolver for the raw-statement ablation.

The production resolver deliberately reads ``semantic.normalized_statement``.
This module changes only the text representation used by retrieval, reranking,
and the candidate snapshots passed to the LLM. Candidate aggregation and all
ranking rules remain inherited from :class:`FrozenCandidateResolver`.
"""

from __future__ import annotations

from typing import Any

from src.cross_doc.resolution import FrozenCandidateResolver
from src.evidence_graph.derivation import clean_model_text


class RawStatementResolver(FrozenCandidateResolver):
    """Resolve candidates using raw Evidence Node statements everywhere."""

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
                str(self.evidence_nodes[evidence_node_id - 1].get("statement", ""))
            )
            key = " ".join(statement.split())
            if not key or key in seen_statements:
                continue
            seen_statements.add(key)
            representatives.append(
                {
                    "role": role,
                    "evidence_node_id": evidence_node_id,
                    "statement": statement,
                }
            )
        return representatives

    def format_reranker_query(self, query_node: dict[str, Any]) -> str:
        return "\n".join(
            (
                f"Type: {query_node['type']}",
                f"Title: {query_node['title']}",
                f"Field: {query_node['field']}",
                f"Subfield: {query_node['subfield']}",
                "Statement: " + str(query_node.get("statement", "")),
            )
        )

    def format_reranker_document(self, candidate: dict[str, Any]) -> str:
        lines: list[str] = []
        for representative in self.candidate_representatives(candidate):
            role = str(representative["role"]).replace("_", " ").title()
            lines.append(f"{role} statement: {representative['statement']}")
        return "\n".join(lines)

