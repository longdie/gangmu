from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from src.common.files import clean_whitespace


NODE_TYPES = {"definition", "theorem"}
EDGE_TYPES = {"dependency", "proof"}

FIELD_VOCABULARY: dict[str, tuple[str, ...]] = {
    "Analysis": (
        "Calculus",
        "Real Analysis",
        "Complex Analysis",
        "Functional Analysis",
        "Measure Theory",
        "Harmonic Analysis",
        "Differential Equations",
        "Partial Differential Equations",
        "Others",
    ),
    "Algebra": (
        "Linear Algebra",
        "Group Theory",
        "Ring Theory",
        "Field Theory",
        "Commutative Algebra",
        "Homological Algebra",
        "Representation Theory",
        "Algebraic Geometry",
        "Others",
    ),
    "Geometry": (
        "Euclidean Geometry",
        "Differential Geometry",
        "Riemannian Geometry",
        "Algebraic Geometry",
        "Convex Geometry",
        "Symplectic Geometry",
        "Others",
    ),
    "Topology": (
        "General Topology",
        "Algebraic Topology",
        "Differential Topology",
        "Geometric Topology",
        "Others",
    ),
    "Number Theory": (
        "Elementary Number Theory",
        "Algebraic Number Theory",
        "Analytic Number Theory",
        "Arithmetic Geometry",
        "Modular Forms",
        "Others",
    ),
    "Discrete Mathematics": (
        "Combinatorics",
        "Graph Theory",
        "Coding Theory",
        "Cryptography",
        "Others",
    ),
    "Logic and Foundations": (
        "Mathematical Logic",
        "Set Theory",
        "Model Theory",
        "Proof Theory",
        "Category Theory",
        "Type Theory",
        "Others",
    ),
    "Applied Mathematics": (
        "Probability",
        "Statistics",
        "Optimization",
        "Numerical Analysis",
        "Dynamical Systems",
        "Mathematical Physics",
        "Information Theory",
        "Operations Research",
        "Others",
    ),
}


def normalize_authors(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            clean_whitespace(str(item)) for item in value if clean_whitespace(str(item))
        ]
    if isinstance(value, str):
        pieces = [value]
        for separator in (" and ", ";", "；"):
            pieces = [part for piece in pieces for part in piece.split(separator)]
        return [clean_whitespace(piece) for piece in pieces if clean_whitespace(piece)]
    return []


def compact_source_metadata(
    raw_metadata: dict[str, Any] | None,
    *,
    document_id: str,
    collection: str,
) -> dict[str, Any]:
    raw = raw_metadata if isinstance(raw_metadata, dict) else {}
    source_type = "textbook" if collection == "textbooks" else "paper"
    title = clean_whitespace(
        str(
            raw.get("title")
            or raw.get("book_title")
            or raw.get("paper_title")
            or document_id
        )
    )
    result: dict[str, Any] = {
        "id": document_id,
        "type": source_type,
        "title": title or document_id,
    }

    authors = normalize_authors(raw.get("authors", []))
    if authors:
        result["authors"] = authors

    year = _normalize_year(raw.get("year"))
    if year is not None:
        result["year"] = year

    optional_values = {
        "publication": raw.get("publication") or raw.get("publication_status"),
        "url": raw.get("url") or raw.get("source_url"),
        "license": raw.get("license"),
    }
    for key, value in optional_values.items():
        normalized = clean_whitespace(str(value or ""))
        if normalized:
            result[key] = normalized
    return result


def normalize_field_pair(
    field: str,
    subfield: str,
    *,
    default_field: str,
    default_subfield: str,
) -> tuple[str, str]:
    if field not in FIELD_VOCABULARY:
        field = default_field
    if field not in FIELD_VOCABULARY:
        field = "Algebra"

    if subfield not in FIELD_VOCABULARY[field]:
        subfield = default_subfield
    if subfield not in FIELD_VOCABULARY[field]:
        subfield = "Others"
    return field, subfield


def build_single_document_graph(
    *,
    source: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    source_id = clean_whitespace(str(source.get("id", "")))
    if not source_id:
        raise ValueError("Single-document graph source.id cannot be empty.")

    normalized_nodes: list[dict[str, Any]] = []
    node_id_map: dict[int, int] = {}
    for new_id, raw_node in enumerate(nodes, start=1):
        try:
            old_id = int(raw_node.get("id", new_id))
        except (TypeError, ValueError):
            old_id = new_id
        if old_id in node_id_map:
            raise ValueError(f"Duplicate local node ID: {old_id}")
        node_id_map[old_id] = new_id

        node = _copy_graph_node(raw_node, node_id=new_id, source_id=source_id)
        normalized_nodes.append(node)

    normalized_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[int, int, str]] = set()
    for raw_edge in edges:
        if not isinstance(raw_edge, dict):
            continue
        try:
            old_source = int(raw_edge.get("source"))
            old_target = int(raw_edge.get("target"))
        except (TypeError, ValueError):
            continue
        if old_source not in node_id_map or old_target not in node_id_map:
            continue
        edge_source = node_id_map[old_source]
        edge_target = node_id_map[old_target]
        edge_type = clean_whitespace(str(raw_edge.get("type", "")).lower())
        edge_key = (edge_source, edge_target, edge_type)
        if (
            edge_type not in EDGE_TYPES
            or edge_source == edge_target
            or edge_key in seen_edges
        ):
            continue
        seen_edges.add(edge_key)
        edge_id = len(normalized_edges) + 1
        normalized_edge = {
            "id": edge_id,
            "type": edge_type,
            "source": edge_source,
            "target": edge_target,
            "semantics": clean_whitespace(str(raw_edge.get("semantics", "") or "")),
        }
        normalized_edges.append(normalized_edge)
        normalized_nodes[edge_source - 1]["out_edges"].append(edge_id)
        normalized_nodes[edge_target - 1]["in_edges"].append(edge_id)

    graph = {
        "graph_type": "single_document",
        "schema_version": "1.0",
        "source": deepcopy(source),
        "nodes": normalized_nodes,
        "edges": normalized_edges,
    }
    errors = validate_single_document_graph(graph)
    if errors:
        raise ValueError("Invalid single-document graph: " + "; ".join(errors))
    return graph


def validate_single_document_graph(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if graph.get("graph_type") != "single_document":
        errors.append("graph_type must be 'single_document'")

    source = graph.get("source")
    if not isinstance(source, dict) or not clean_whitespace(str(source.get("id", ""))):
        errors.append("source must contain a non-empty id")
        source_id = ""
    else:
        source_id = clean_whitespace(str(source["id"]))

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list):
        errors.append("nodes must be a list")
        nodes = []
    if not isinstance(edges, list):
        errors.append("edges must be a list")
        edges = []

    for expected_id, node in enumerate(nodes, start=1):
        if not isinstance(node, dict):
            errors.append(f"node {expected_id} must be an object")
            continue
        if node.get("id") != expected_id:
            errors.append(f"node ID must equal list index + 1 at {expected_id}")
        if node.get("type") not in NODE_TYPES:
            errors.append(f"invalid node type at {expected_id}")
        if not clean_whitespace(str(node.get("title", ""))):
            errors.append(f"empty node title at {expected_id}")
        if not clean_whitespace(str(node.get("statement", ""))):
            errors.append(f"empty node statement at {expected_id}")
        if not clean_whitespace(str(node.get("field", ""))):
            errors.append(f"empty node field at {expected_id}")
        if not clean_whitespace(str(node.get("subfield", ""))):
            errors.append(f"empty node subfield at {expected_id}")
        if node.get("source_id") != source_id:
            errors.append(f"source_id mismatch at node {expected_id}")
        if not isinstance(node.get("in_edges"), list):
            errors.append(f"in_edges must be a list at node {expected_id}")
        if not isinstance(node.get("out_edges"), list):
            errors.append(f"out_edges must be a list at node {expected_id}")

    expected_in: list[list[int]] = [[] for _ in nodes]
    expected_out: list[list[int]] = [[] for _ in nodes]
    for expected_id, edge in enumerate(edges, start=1):
        if not isinstance(edge, dict):
            errors.append(f"edge {expected_id} must be an object")
            continue
        if edge.get("id") != expected_id:
            errors.append(f"edge ID must equal list index + 1 at {expected_id}")
        if edge.get("type") not in EDGE_TYPES:
            errors.append(f"invalid edge type at {expected_id}")
        edge_source = edge.get("source")
        edge_target = edge.get("target")
        if not isinstance(edge_source, int) or not 1 <= edge_source <= len(nodes):
            errors.append(f"invalid edge source at {expected_id}")
            continue
        if not isinstance(edge_target, int) or not 1 <= edge_target <= len(nodes):
            errors.append(f"invalid edge target at {expected_id}")
            continue
        expected_out[edge_source - 1].append(expected_id)
        expected_in[edge_target - 1].append(expected_id)

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        if node.get("in_edges") != expected_in[index]:
            errors.append(f"in_edges mismatch at node {index + 1}")
        if node.get("out_edges") != expected_out[index]:
            errors.append(f"out_edges mismatch at node {index + 1}")
    return errors


def _copy_graph_node(
    raw_node: dict[str, Any],
    *,
    node_id: int,
    source_id: str,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": node_id,
        "type": clean_whitespace(str(raw_node.get("type", "")).lower()),
        "title": clean_whitespace(str(raw_node.get("title", ""))),
        "statement": clean_whitespace(str(raw_node.get("statement", ""))),
        "field": clean_whitespace(str(raw_node.get("field", ""))),
        "subfield": clean_whitespace(str(raw_node.get("subfield", ""))),
        "source_id": source_id,
        "in_edges": [],
        "out_edges": [],
    }
    label = clean_whitespace(str(raw_node.get("label", "") or ""))
    proof = clean_whitespace(str(raw_node.get("proof", "") or ""))
    if label:
        node["label"] = label
    if proof and node["type"] == "theorem":
        node["proof"] = proof

    location = raw_node.get("location")
    if isinstance(location, dict):
        cleaned_location = _clean_location(location)
        if cleaned_location:
            node["location"] = cleaned_location

    semantic = raw_node.get("semantic")
    if isinstance(semantic, dict) and semantic:
        node["semantic"] = deepcopy(semantic)
    return node


def _clean_location(location: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    segment_id = clean_whitespace(str(location.get("segment_id", "") or ""))
    if segment_id:
        result["segment_id"] = segment_id

    pages = location.get("pages")
    if isinstance(pages, list):
        normalized_pages = sorted(
            {
                int(page)
                for page in pages
                if isinstance(page, int) and not isinstance(page, bool) and page > 0
            }
        )
        if normalized_pages:
            result["pages"] = normalized_pages

    title_path = location.get("title_path")
    if isinstance(title_path, list):
        normalized_titles = [
            clean_whitespace(str(title))
            for title in title_path
            if clean_whitespace(str(title))
        ]
        if normalized_titles:
            result["title_path"] = normalized_titles
    return result


def _normalize_year(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value > 0 else None
    match = re.search(r"\b(\d{4})\b", str(value or ""))
    return int(match.group(1)) if match else None
