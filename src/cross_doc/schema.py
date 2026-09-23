from __future__ import annotations

import re
from typing import Any

import numpy as np

from src.common.files import clean_whitespace
from src.evidence_graph.derivation import EMBEDDING_DIMENSIONS
from src.single_doc.schema import EDGE_TYPES, NODE_TYPES


SCHEMA_VERSION = "2.1"
CONTENT_ALGORITHM_VERSION = 4
LEGACY_SCHEMA_VERSION = "2.0"
LEGACY_DATA_FILES = {
    "nodes": "nodes.json",
    "edges": "edges.json",
    "possible_equivalence_edges": "possible_equivalence_edges.json",
    "evidence_node_to_canonical": "evidence_node_to_canonical.npy",
    "embedding_sums": "canonical_embedding_sums.npy",
    "centroid_embeddings": "canonical_centroid_embeddings.npy",
    "statement_embeddings": "canonical_statement_embeddings.npy",
    "diagnostics": "diagnostics.json",
    "usage": "usage.json",
}
DATA_FILES = {
    **LEGACY_DATA_FILES,
    "title_embeddings": "canonical_title_embeddings.npy",
    "center_evidence_embeddings": ("canonical_center_evidence_embeddings.npy"),
}


def validate_canonical_graph(
    *,
    manifest: Any,
    nodes: Any,
    edges: Any,
    possible_equivalence_edges: Any,
    evidence_node_to_canonical: Any,
    embedding_sums: Any,
    centroid_embeddings: Any,
    statement_embeddings: Any,
    title_embeddings: Any = None,
    center_evidence_embeddings: Any = None,
    diagnostics: Any,
    usage: Any,
    evidence_manifest: Any,
    evidence_nodes: Any,
    evidence_edges: Any,
    evidence_embeddings: Any,
    expected_graph_id: str | None = None,
) -> list[str]:
    """Return all structural errors in one submitted Canonical Graph."""

    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if not isinstance(nodes, list):
        return ["nodes must be a list"]
    if not isinstance(edges, list):
        return ["edges must be a list"]
    if not isinstance(possible_equivalence_edges, list):
        return ["possible_equivalence_edges must be a list"]
    if not isinstance(evidence_manifest, dict):
        return ["evidence_manifest must be an object"]
    if not isinstance(evidence_nodes, list):
        return ["evidence_nodes must be a list"]
    if not isinstance(evidence_edges, list):
        return ["evidence_edges must be a list"]

    if manifest.get("graph_type") != "canonical":
        errors.append("manifest.graph_type must be 'canonical'")
    schema_version = manifest.get("schema_version")
    if schema_version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        errors.append(
            "manifest.schema_version must be "
            f"{LEGACY_SCHEMA_VERSION!r} or {SCHEMA_VERSION!r}"
        )
    graph_id = clean_whitespace(str(manifest.get("graph_id", "") or ""))
    if not graph_id:
        errors.append("manifest.graph_id must be non-empty")
    elif expected_graph_id is not None and graph_id != expected_graph_id:
        errors.append("manifest.graph_id does not match its directory")
    if manifest.get("evidence_graph_id") != evidence_manifest.get("graph_id"):
        errors.append("manifest.evidence_graph_id does not match Evidence Graph")
    if manifest.get("evidence_graph_revision") != evidence_manifest.get("revision"):
        errors.append("manifest.evidence_graph_revision does not match Evidence Graph")
    if not _positive_int(manifest.get("revision")):
        errors.append("manifest.revision must be a positive integer")
    expected_files = (
        LEGACY_DATA_FILES if schema_version == LEGACY_SCHEMA_VERSION else DATA_FILES
    )
    if manifest.get("files") != expected_files:
        errors.append("manifest.files does not match the canonical file schema")
    construction_records = manifest.get("construction_records")
    if construction_records is not None:
        revision = manifest.get("revision")
        expected_revision_directory = (
            f"construction_records/revision_{revision:04d}"
            if _positive_int(revision)
            else ""
        )
        if (
            not isinstance(construction_records, dict)
            or construction_records.get("schema_version") != "1.0"
            or construction_records.get("directory") != "construction_records"
            or construction_records.get("revision_directory")
            != expected_revision_directory
        ):
            errors.append("manifest.construction_records is invalid")

    counts = manifest.get("counts")
    expected_counts = {
        "evidence_nodes": len(evidence_nodes),
        "evidence_edges": len(evidence_edges),
        "canonical_nodes": len(nodes),
        "canonical_edges": len(edges),
        "possible_equivalence_edges": len(possible_equivalence_edges),
    }
    if counts != expected_counts:
        errors.append("manifest.counts does not match the materialized graph")

    construction = manifest.get("construction")
    if not isinstance(construction, dict):
        errors.append("manifest.construction must be an object")
    elif construction.get("processed_evidence_node_count") != len(evidence_nodes):
        errors.append(
            "construction.processed_evidence_node_count must equal Evidence Node count"
        )
    enforce_meaningful_titles = bool(
        schema_version == SCHEMA_VERSION
        and isinstance(construction, dict)
        and construction.get("content_algorithm_version") == CONTENT_ALGORITHM_VERSION
    )
    if enforce_meaningful_titles:
        for history_key in (
            "alignment_model_history",
            "strict_verification_model_history",
        ):
            if not _valid_model_history(
                construction.get(history_key),
                processed_count=len(evidence_nodes),
            ):
                errors.append(
                    f"construction.{history_key} must cover every processed "
                    "Evidence Node exactly once"
                )

    expected_in: list[list[int]] = [[] for _ in nodes]
    expected_out: list[list[int]] = [[] for _ in nodes]
    expected_possible: list[list[int]] = [[] for _ in nodes]
    membership = np.zeros(len(evidence_nodes), dtype=np.int64)

    for expected_id, node in enumerate(nodes, start=1):
        if not isinstance(node, dict):
            errors.append(f"node {expected_id} must be an object")
            continue
        if node.get("id") != expected_id:
            errors.append(f"node ID is not continuous at row {expected_id}")
        node_type = node.get("type")
        if node_type not in NODE_TYPES:
            errors.append(f"node {expected_id} has invalid type")
        for key in ("title", "statement", "field", "subfield"):
            if not clean_whitespace(str(node.get(key, "") or "")):
                errors.append(f"node {expected_id}.{key} must be non-empty")
        if enforce_meaningful_titles and not _valid_canonical_title(node.get("title")):
            errors.append(
                f"node {expected_id}.title must be meaningful and non-generic"
            )

        member_ids = node.get("evidence_node_ids")
        if (
            not isinstance(member_ids, list)
            or not member_ids
            or member_ids != sorted(set(member_ids))
        ):
            errors.append(
                f"node {expected_id}.evidence_node_ids must be a non-empty "
                "sorted unique list"
            )
            member_ids = []
        for evidence_node_id in member_ids:
            if not _valid_id(evidence_node_id, len(evidence_nodes)):
                errors.append(f"node {expected_id} contains invalid Evidence Node ID")
                continue
            if membership[evidence_node_id - 1] != 0:
                errors.append(
                    f"Evidence Node {evidence_node_id} belongs to multiple "
                    "Canonical Nodes"
                )
            membership[evidence_node_id - 1] = expected_id
            if evidence_nodes[evidence_node_id - 1].get("type") != node_type:
                errors.append(f"node {expected_id} mixes different Evidence Node types")

        center_id = node.get("center_evidence_node_id")
        diverse_ids = node.get("diverse_evidence_node_ids")
        if center_id not in member_ids:
            errors.append(f"node {expected_id}.center_evidence_node_id is not a member")
        if (
            not isinstance(diverse_ids, list)
            or len(diverse_ids) > 2
            or len(set(diverse_ids)) != len(diverse_ids)
            or center_id in diverse_ids
            or any(item not in member_ids for item in diverse_ids)
        ):
            errors.append(f"node {expected_id}.diverse_evidence_node_ids is invalid")
        for key in (
            "in_edges",
            "out_edges",
            "possible_equivalence_edge_ids",
        ):
            value = node.get(key)
            if not isinstance(value, list) or value != sorted(set(value)):
                errors.append(f"node {expected_id}.{key} must be a sorted unique list")

    supported_evidence_edges: set[int] = set()
    for expected_id, edge in enumerate(edges, start=1):
        if not isinstance(edge, dict):
            errors.append(f"edge {expected_id} must be an object")
            continue
        if edge.get("id") != expected_id:
            errors.append(f"edge ID is not continuous at row {expected_id}")
        source = edge.get("source")
        target = edge.get("target")
        if not _valid_id(source, len(nodes)) or not _valid_id(
            target,
            len(nodes),
        ):
            errors.append(f"edge {expected_id} has invalid endpoints")
            continue
        if source == target:
            errors.append(f"edge {expected_id} is a self-loop")
        expected_out[source - 1].append(expected_id)
        expected_in[target - 1].append(expected_id)

        evidence_edge_ids = edge.get("evidence_edge_ids")
        if (
            not isinstance(evidence_edge_ids, list)
            or not evidence_edge_ids
            or evidence_edge_ids != sorted(set(evidence_edge_ids))
        ):
            errors.append(
                f"edge {expected_id}.evidence_edge_ids must be non-empty, "
                "sorted, and unique"
            )
            evidence_edge_ids = []
        actual_types: set[str] = set()
        for evidence_edge_id in evidence_edge_ids:
            if not _valid_id(evidence_edge_id, len(evidence_edges)):
                errors.append(f"edge {expected_id} contains invalid Evidence Edge ID")
                continue
            if evidence_edge_id in supported_evidence_edges:
                errors.append(
                    f"Evidence Edge {evidence_edge_id} supports multiple "
                    "Canonical Edges"
                )
            supported_evidence_edges.add(evidence_edge_id)
            evidence_edge = evidence_edges[evidence_edge_id - 1]
            actual_types.add(str(evidence_edge.get("type", "")))
            if (
                membership[int(evidence_edge["source"]) - 1] != source
                or membership[int(evidence_edge["target"]) - 1] != target
            ):
                errors.append(
                    f"edge {expected_id} endpoints do not match Evidence Edge "
                    f"{evidence_edge_id}"
                )
        types = edge.get("types")
        if (
            not isinstance(types, list)
            or not types
            or len(types) != len(set(types))
            or any(item not in EDGE_TYPES for item in types)
            or set(types) != actual_types
        ):
            errors.append(f"edge {expected_id}.types is inconsistent")

    seen_possible_pairs: set[tuple[int, int]] = set()
    for expected_id, edge in enumerate(
        possible_equivalence_edges,
        start=1,
    ):
        if not isinstance(edge, dict):
            errors.append(f"possible equivalence edge {expected_id} must be an object")
            continue
        if edge.get("id") != expected_id:
            errors.append(
                f"possible equivalence edge ID is not continuous at row {expected_id}"
            )
        node_a = edge.get("node_a")
        node_b = edge.get("node_b")
        if (
            not _valid_id(node_a, len(nodes))
            or not _valid_id(node_b, len(nodes))
            or node_a >= node_b
        ):
            errors.append(
                f"possible equivalence edge {expected_id} has invalid endpoints"
            )
            continue
        pair = (node_a, node_b)
        if pair in seen_possible_pairs:
            errors.append(f"duplicate possible equivalence pair {pair}")
        seen_possible_pairs.add(pair)
        if edge.get("type") != "possible_equivalence":
            errors.append(f"possible equivalence edge {expected_id} has invalid type")
        if edge.get("status") != "needs_review":
            errors.append(f"possible equivalence edge {expected_id} has invalid status")
        expected_possible[node_a - 1].append(expected_id)
        expected_possible[node_b - 1].append(expected_id)

    if np.any(membership == 0):
        errors.append("every Evidence Node must belong to one Canonical Node")
    if not _valid_mapping(
        evidence_node_to_canonical,
        expected=membership,
        canonical_node_count=len(nodes),
    ):
        errors.append("evidence_node_to_canonical does not match canonical memberships")

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        if node.get("in_edges") != expected_in[index]:
            errors.append(f"node {index + 1}.in_edges is inconsistent")
        if node.get("out_edges") != expected_out[index]:
            errors.append(f"node {index + 1}.out_edges is inconsistent")
        if node.get("possible_equivalence_edge_ids") != expected_possible[index]:
            errors.append(
                f"node {index + 1}.possible_equivalence_edge_ids is inconsistent"
            )

    _validate_matrix(
        embedding_sums,
        rows=len(nodes),
        normalized=False,
        name="canonical_embedding_sums",
        errors=errors,
    )
    _validate_matrix(
        centroid_embeddings,
        rows=len(nodes),
        normalized=True,
        name="canonical_centroid_embeddings",
        errors=errors,
    )
    _validate_matrix(
        statement_embeddings,
        rows=len(nodes),
        normalized=True,
        name="canonical_statement_embeddings",
        errors=errors,
    )
    if schema_version == SCHEMA_VERSION:
        _validate_matrix(
            title_embeddings,
            rows=len(nodes),
            normalized=True,
            name="canonical_title_embeddings",
            errors=errors,
        )
        _validate_matrix(
            center_evidence_embeddings,
            rows=len(nodes),
            normalized=True,
            name="canonical_center_evidence_embeddings",
            errors=errors,
        )
        if (
            isinstance(center_evidence_embeddings, np.ndarray)
            and isinstance(evidence_embeddings, np.ndarray)
            and evidence_embeddings.shape == (len(evidence_nodes), EMBEDDING_DIMENSIONS)
            and all(
                isinstance(node, dict)
                and _valid_id(
                    node.get("center_evidence_node_id"),
                    len(evidence_nodes),
                )
                for node in nodes
            )
        ):
            center_rows = np.asarray(
                [int(node["center_evidence_node_id"]) - 1 for node in nodes],
                dtype=np.int64,
            )
            if not np.allclose(
                center_evidence_embeddings,
                evidence_embeddings[center_rows],
                rtol=1e-5,
                atol=1e-6,
            ):
                errors.append(
                    "canonical_center_evidence_embeddings does not match "
                    "the selected Evidence embeddings"
                )

    if (
        isinstance(embedding_sums, np.ndarray)
        and isinstance(
            centroid_embeddings,
            np.ndarray,
        )
        and embedding_sums.shape == centroid_embeddings.shape
    ):
        norms = np.linalg.norm(embedding_sums, axis=1, keepdims=True)
        expected_centroids = embedding_sums / np.maximum(norms, 1e-12)
        if not np.allclose(
            expected_centroids,
            centroid_embeddings,
            rtol=1e-4,
            atol=1e-4,
        ):
            errors.append("canonical_centroid_embeddings does not match embedding sums")

    if not isinstance(diagnostics, dict):
        errors.append("diagnostics must be an object")
    else:
        self_loops = diagnostics.get("projected_self_loop_evidence_edge_ids")
        if (
            not isinstance(self_loops, list)
            or self_loops != sorted(set(self_loops))
            or any(not _valid_id(item, len(evidence_edges)) for item in self_loops)
        ):
            errors.append(
                "diagnostics.projected_self_loop_evidence_edge_ids is invalid"
            )
            self_loops = []
        if supported_evidence_edges | set(self_loops) != set(
            range(1, len(evidence_edges) + 1)
        ):
            errors.append(
                "every Evidence Edge must be projected or recorded as a self-loop"
            )
        validation = diagnostics.get("validation")
        if not isinstance(validation, dict) or validation.get("passed") is not True:
            errors.append("diagnostics.validation.passed must be true")
        if enforce_meaningful_titles:
            content_validation = diagnostics.get("canonical_content_validation")
            if (
                not isinstance(content_validation, dict)
                or set(content_validation)
                != {"max_attempts", "accepted_after_failed_validation"}
                or content_validation.get("max_attempts") != 3
                or not isinstance(
                    content_validation.get("accepted_after_failed_validation"),
                    list,
                )
            ):
                errors.append(
                    "diagnostics.canonical_content_validation is invalid"
                )
            else:
                failures = content_validation[
                    "accepted_after_failed_validation"
                ]
                failure_ids: list[int] = []
                for item in failures:
                    if (
                        not isinstance(item, dict)
                        or set(item)
                        != {"canonical_node_id", "attempts", "final_feedback"}
                        or not _valid_id(item.get("canonical_node_id"), len(nodes))
                        or item.get("attempts") != 3
                        or not isinstance(item.get("final_feedback"), str)
                        or not clean_whitespace(item["final_feedback"])
                    ):
                        errors.append(
                            "diagnostics canonical-content failure entry is invalid"
                        )
                        continue
                    failure_ids.append(int(item["canonical_node_id"]))
                if failure_ids != sorted(set(failure_ids)):
                    errors.append(
                        "diagnostics canonical-content failure IDs must be "
                        "sorted and unique"
                    )
    if not isinstance(usage, dict):
        errors.append("usage must be an object")

    return errors


def _valid_canonical_title(value: Any) -> bool:
    title = clean_whitespace(str(value or ""))
    if not title or len(title) > 240:
        return False
    normalized = title.strip(" .:;,-_").lower()
    generic = re.fullmatch(
        r"(?:main\s+)?(?:theorem|lemma|proposition|corollary|definition|"
        r"claim|property|result)(?:\s+(?:\d+(?:\.\d+)*|[ivxlcdm]+))?",
        normalized,
    )
    return generic is None


def _valid_model_history(value: Any, *, processed_count: int) -> bool:
    if not isinstance(value, list) or not value:
        return processed_count == 0 and value == []
    expected_start = 1
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "model",
            "provider",
            "evidence_node_id_start",
            "evidence_node_id_end",
        }:
            return False
        start = item.get("evidence_node_id_start")
        end = item.get("evidence_node_id_end")
        if (
            not clean_whitespace(str(item.get("model", "") or ""))
            or not clean_whitespace(str(item.get("provider", "") or ""))
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start != expected_start
            or end < start
            or end > processed_count
        ):
            return False
        expected_start = end + 1
    return expected_start == processed_count + 1


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _valid_id(value: Any, upper: int) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= upper
    )


def _valid_mapping(
    value: Any,
    *,
    expected: np.ndarray,
    canonical_node_count: int,
) -> bool:
    return (
        isinstance(value, np.ndarray)
        and value.dtype == np.int64
        and value.shape == expected.shape
        and np.array_equal(value, expected)
        and (
            value.size == 0
            or (int(value.min()) >= 1 and int(value.max()) <= canonical_node_count)
        )
    )


def _validate_matrix(
    value: Any,
    *,
    rows: int,
    normalized: bool,
    name: str,
    errors: list[str],
) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != (rows, EMBEDDING_DIMENSIONS)
        or not np.all(np.isfinite(value))
    ):
        errors.append(
            f"{name} must be a finite float32 matrix with shape "
            f"{(rows, EMBEDDING_DIMENSIONS)}"
        )
        return
    if rows == 0:
        return
    norms = np.linalg.norm(value, axis=1)
    if normalized:
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            errors.append(f"{name} contains non-normalized rows")
    elif np.any(norms <= 0):
        errors.append(f"{name} contains a zero row")
