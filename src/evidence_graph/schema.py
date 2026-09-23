from __future__ import annotations

from typing import Any

import numpy as np

from src.common.files import clean_whitespace
from src.evidence_graph.derivation import (
    EMBEDDING_DIMENSIONS,
    has_valid_semantic,
    validate_embedding_matrix,
)
from src.single_doc.schema import EDGE_TYPES, NODE_TYPES


SCHEMA_VERSION = "1.0"
DATA_FILES = {
    "sources": "sources.json",
    "nodes": "nodes.json",
    "edges": "edges.json",
    "embeddings": "embeddings.npy",
    "usage": "usage.json",
}


def validate_evidence_graph(
    *,
    manifest: Any,
    sources: Any,
    nodes: Any,
    edges: Any,
    embeddings: Any = None,
    usage: Any = None,
    expected_graph_id: str | None = None,
) -> list[str]:
    """Return structural errors for one fully materialized Evidence Graph."""

    errors: list[str] = []

    if not isinstance(manifest, dict):
        return ["manifest must be an object"]

    graph_id = clean_whitespace(str(manifest.get("graph_id", "")))
    if manifest.get("graph_type") != "evidence":
        errors.append("manifest.graph_type must be 'evidence'")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"manifest.schema_version must be '{SCHEMA_VERSION}'")
    if not graph_id:
        errors.append("manifest.graph_id must be non-empty")
    if expected_graph_id is not None and graph_id != expected_graph_id:
        errors.append("manifest.graph_id must equal the Evidence Graph directory name")
    if not clean_whitespace(str(manifest.get("name", ""))):
        errors.append("manifest.name must be non-empty")

    revision = manifest.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        errors.append("manifest.revision must be a positive integer")
    for key in ("created_at", "updated_at"):
        if not clean_whitespace(str(manifest.get(key, ""))):
            errors.append(f"manifest.{key} must be non-empty")
    if manifest.get("files") != DATA_FILES:
        errors.append("manifest.files does not match the required data files")

    normalization = manifest.get("normalization")
    if not isinstance(normalization, dict):
        errors.append("manifest.normalization must be an object")
        normalization = {}
    for key in ("model", "provider", "prompt_file", "prompt_sha256"):
        if not clean_whitespace(str(normalization.get(key, ""))):
            errors.append(f"manifest.normalization.{key} must be non-empty")

    embedding = manifest.get("embedding")
    if not isinstance(embedding, dict):
        errors.append("manifest.embedding must be an object")
        embedding = {}
    for key in ("model", "provider", "dtype", "row_mapping"):
        if not clean_whitespace(str(embedding.get(key, ""))):
            errors.append(f"manifest.embedding.{key} must be non-empty")
    if embedding.get("dimensions") != EMBEDDING_DIMENSIONS:
        errors.append(f"manifest.embedding.dimensions must be {EMBEDDING_DIMENSIONS}")
    if embedding.get("dtype") != "float32":
        errors.append("manifest.embedding.dtype must be 'float32'")
    if embedding.get("l2_normalized") is not True:
        errors.append("manifest.embedding.l2_normalized must be true")
    if embedding.get("row_mapping") != "node_id_minus_1":
        errors.append("manifest.embedding.row_mapping must be 'node_id_minus_1'")

    if not isinstance(sources, list):
        errors.append("sources must be a list")
        sources = []
    if not isinstance(nodes, list):
        errors.append("nodes must be a list")
        nodes = []
    if not isinstance(edges, list):
        errors.append("edges must be a list")
        edges = []

    expected_counts = {
        "sources": len(sources),
        "nodes": len(nodes),
        "edges": len(edges),
    }
    if manifest.get("counts") != expected_counts:
        errors.append("manifest.counts does not match the data files")

    source_ids: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            errors.append(f"source at index {index} must be an object")
            continue
        source_id = clean_whitespace(str(source.get("id", "")))
        if not source_id:
            errors.append(f"source at index {index} has an empty id")
        elif source_id in source_ids:
            errors.append(f"duplicate source id: {source_id}")
        else:
            source_ids.add(source_id)
        if source.get("type") not in {"paper", "textbook"}:
            errors.append(f"invalid source type at index {index}")
        if not clean_whitespace(str(source.get("title", ""))):
            errors.append(f"empty source title at index {index}")

    expected_in: list[list[int]] = [[] for _ in nodes]
    expected_out: list[list[int]] = [[] for _ in nodes]
    node_source_ids: list[str] = []
    for expected_id, node in enumerate(nodes, start=1):
        if not isinstance(node, dict):
            errors.append(f"node {expected_id} must be an object")
            node_source_ids.append("")
            continue
        if node.get("id") != expected_id:
            errors.append(f"node ID must equal list index + 1 at {expected_id}")
        if node.get("type") not in NODE_TYPES:
            errors.append(f"invalid node type at {expected_id}")
        for key in ("title", "statement", "field", "subfield"):
            if not clean_whitespace(str(node.get(key, ""))):
                errors.append(f"empty node {key} at {expected_id}")
        source_id = clean_whitespace(str(node.get("source_id", "")))
        node_source_ids.append(source_id)
        if source_id not in source_ids:
            errors.append(f"unknown node source_id at {expected_id}: {source_id}")
        if not isinstance(node.get("in_edges"), list):
            errors.append(f"in_edges must be a list at node {expected_id}")
        if not isinstance(node.get("out_edges"), list):
            errors.append(f"out_edges must be a list at node {expected_id}")
        if "location" in node and not isinstance(node["location"], dict):
            errors.append(f"location must be an object at node {expected_id}")
        if not has_valid_semantic(node):
            errors.append(f"invalid node semantic at {expected_id}")
        if node.get("type") == "definition" and "proof" in node:
            errors.append(f"definition node cannot contain proof at {expected_id}")

    seen_edges: set[tuple[int, int, str]] = set()
    for expected_id, edge in enumerate(edges, start=1):
        if not isinstance(edge, dict):
            errors.append(f"edge {expected_id} must be an object")
            continue
        if edge.get("id") != expected_id:
            errors.append(f"edge ID must equal list index + 1 at {expected_id}")
        edge_type = edge.get("type")
        if edge_type not in EDGE_TYPES:
            errors.append(f"invalid edge type at {expected_id}")
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, int) or isinstance(source, bool):
            errors.append(f"invalid edge source at {expected_id}")
            continue
        if not isinstance(target, int) or isinstance(target, bool):
            errors.append(f"invalid edge target at {expected_id}")
            continue
        if not 1 <= source <= len(nodes):
            errors.append(f"edge source out of range at {expected_id}")
            continue
        if not 1 <= target <= len(nodes):
            errors.append(f"edge target out of range at {expected_id}")
            continue
        if source == target:
            errors.append(f"self-loop is not allowed at edge {expected_id}")
        edge_key = (source, target, str(edge_type))
        if edge_key in seen_edges:
            errors.append(f"duplicate edge at {expected_id}")
        else:
            seen_edges.add(edge_key)
        if not clean_whitespace(str(edge.get("semantics", ""))):
            errors.append(f"empty edge semantics at {expected_id}")
        if (
            node_source_ids[source - 1]
            and node_source_ids[target - 1]
            and node_source_ids[source - 1] != node_source_ids[target - 1]
        ):
            errors.append(f"cross-document Evidence Edge at {expected_id}")
        expected_out[source - 1].append(expected_id)
        expected_in[target - 1].append(expected_id)

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        if node.get("in_edges") != expected_in[index]:
            errors.append(f"in_edges mismatch at node {index + 1}")
        if node.get("out_edges") != expected_out[index]:
            errors.append(f"out_edges mismatch at node {index + 1}")

    if normalization.get("node_count") != len(nodes):
        errors.append("manifest.normalization.node_count must equal node count")
    if embedding.get("node_count") != len(nodes):
        errors.append("manifest.embedding.node_count must equal node count")

    if not isinstance(embeddings, np.ndarray):
        errors.append("embeddings must be a NumPy array")
    else:
        try:
            validate_embedding_matrix(
                embeddings,
                expected_rows=len(nodes),
                expected_dimensions=EMBEDDING_DIMENSIONS,
            )
        except ValueError as exc:
            errors.append(str(exc))

    if not isinstance(usage, dict):
        errors.append("usage must be an object")
    else:
        if not isinstance(usage.get("total"), dict):
            errors.append("usage.total must be an object")
        if not isinstance(usage.get("by_model"), dict):
            errors.append("usage.by_model must be an object")
        if not isinstance(usage.get("calls"), list):
            errors.append("usage.calls must be a list")

    return errors
