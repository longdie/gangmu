"""AutoMathKG MathVD1 retrieval and entity fusion on the frozen MathER-Alg KB.

Reimplementation of arXiv:2505.13406v1, Sections 5 and 6.2 and Appendix
Table 20/C5. The published fusion instructions are preserved verbatim.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from src.common.files import read_json, write_json_atomic
from scripts.experiment.baseline_common import has_pending_queries, llm_settings, run_pairwise
from scripts.experiment.dataset import PROJECT_ROOT, sha256_file


DISPLAY_NAME = "AutoMathKG"
TOP_K = 5
DIMENSIONS = 384
IMPLEMENTATION_DEPENDENCIES = ("scripts/experiment/baseline_common.py",)
PAIR_INSTRUCTION = (
    'Your task is to decide if the first math theorem mean the same thing as the '
    'second theorem. If they have the same meaning, you should answer “yes”. '
    'Otherwise, you answer “no”.'
)
SELECT_INSTRUCTION = (
    'Your task is to decide which of the candidate theorems mean the same '
    'thing as the new theorem. You should choose one candidate and output '
    'its id number as your answer.'
)


def entity(node: dict[str, Any], *, query: bool = False) -> dict[str, Any]:
    statement = node["normalized_statement" if query else "statement"]
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("AutoMathKG requires a nonempty mathematical statement.")
    return {
        "id": int(node["query_id" if query else "id"]),
        "type": {"definition": "Definition", "theorem": "Theorem"}[node["type"]],
        "title": str(node.get("title", "")),
        "field": str(node.get("field", "")),
        "contents": [statement],
        "in_refs": {},
        "out_refs": {},
    }


def canonical_entities(dataset: Any) -> list[dict[str, Any]]:
    nodes = [entity(node) for node in dataset.canonical_nodes]
    by_id = {node["id"]: node for node in nodes}
    relationships: dict[tuple[int, int], set[str]] = {}
    for edge in read_json(dataset.canonical_dir / "edges.json"):
        source, target = int(edge["source"]), int(edge["target"])
        if source not in by_id or target not in by_id:
            raise ValueError("Canonical edge references an unknown endpoint.")
        labels = edge.get("types", [])
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise ValueError("Canonical edge types must be strings.")
        relationships.setdefault((source, target), set()).update(labels)
    for (source, target), labels in sorted(relationships.items()):
        source_node, target_node = by_id[source], by_id[target]
        source_key = f"{source_node['type']}:{source_node['title']} [id={source}]"
        target_key = f"{target_node['type']}:{target_node['title']} [id={target}]"
        ordered = sorted(labels)
        value = ordered[0] if len(ordered) == 1 else ordered
        target_node["in_refs"][source_key] = value
        source_node["out_refs"][target_key] = value
    return nodes


def mathvd1_text(node: dict[str, Any]) -> str:
    parts = []
    for key, prefix in zip(
        ("title", "field", "contents", "in_refs", "out_refs"),
        ("title", "field", "content", "in references", "out references"),
    ):
        value = node[key]
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        parts.append(f"{prefix}: {value}")
    return "\n".join(parts)


def _payload(node: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(node[key]) for key in ("id", "type", "title", "field", "contents")}


def pair_prompt(query: dict[str, Any], candidate: dict[str, Any]) -> str:
    return (PAIR_INSTRUCTION + "\n\nFirst theorem:\n"
            + json.dumps(_payload(query), ensure_ascii=False) + "\n\nSecond theorem:\n"
            + json.dumps(_payload(candidate), ensure_ascii=False))


def selection_prompt(query: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    return (SELECT_INSTRUCTION + "\n\nNew theorem:\n"
            + json.dumps(_payload(query), ensure_ascii=False) + "\n\nCandidate theorems:\n"
            + json.dumps([_payload(node) for node in candidates], ensure_ascii=False))


def _unwrap(text: str) -> str:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json|text)?\s*\n?(.*?)\n?```", value, re.DOTALL)
    return fenced.group(1).strip() if fenced else value


def parse_equivalence(text: str) -> bool:
    value = _unwrap(text)
    if value.endswith("."):
        value = value[:-1].strip()
    value = value.strip('"“”').lower()
    if value not in {"yes", "no"}:
        raise ValueError("AutoMathKG expected an unambiguous yes/no answer.")
    return value == "yes"


def parse_selection(text: str, ids: list[int]) -> int:
    value = _unwrap(text)
    try:
        selected = json.loads(value)
    except json.JSONDecodeError:
        selected = value
    if type(selected) is not int or selected not in ids:
        raise ValueError("Selected ID is not one of the affirmative candidates.")
    return selected


def method_settings(dataset: Any, config: Any, *_: Any, **__: Any) -> dict[str, Any]:
    del dataset
    model = config.get_str("automathkg.embedding_model")
    if not model:
        raise ValueError("AutoMathKG embedding_model cannot be empty.")
    local = Path(model).expanduser()
    if not local.is_absolute():
        local = PROJECT_ROOT / local
    local_hashes = {}
    if local.is_dir():
        model = str(local.resolve())
        local_hashes = {
            str(path.relative_to(local)): sha256_file(path)
            for path in sorted(local.rglob("*"))
            if path.is_file() and path.suffix in {".json", ".txt", ".bin", ".safetensors"}
        }
    return {
        "algorithm_version": 1,
        **llm_settings(config, "automathkg", 4096),
        "top_k": TOP_K,
        "same_type_only": False,
        "variant": "mathvd1",
        "decision": "judge_all_then_select_among_affirmative_candidates",
        "embedding": {
            "model": model,
            "revision": config.get_str("automathkg.embedding_revision"),
            "local_files_only": config.get_bool("automathkg.local_files_only"),
            "local_model_sha256": local_hashes,
            "dimensions": DIMENSIONS,
            "device": "cpu",
            "batch_size": 32,
            "max_seq_length": 256,
        },
    }


def unit_vectors(vectors: Any, rows: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.shape != (rows, DIMENSIONS) or not np.isfinite(matrix).all():
        raise ValueError("MiniLM returned an invalid embedding matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("MiniLM returned a zero embedding.")
    return matrix / norms


def embedding_matrices(runtime, nodes, queries) -> tuple[np.ndarray, np.ndarray]:
    cache = runtime.store.checkpoint_dir / "minilm_vectors.npz"
    manifest_path = cache.with_suffix(".json")
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if (manifest.get("input_signature") != runtime.input_signature
                or manifest.get("sha256") != sha256_file(cache)):
            raise ValueError("MiniLM cache does not match the current run.")
        with np.load(cache, allow_pickle=False) as saved:
            return unit_vectors(saved["canonical"], len(nodes)), unit_vectors(saved["queries"], len(queries))
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("AutoMathKG requires: pip install -r requirements-automathkg.txt") from None
    settings = runtime.settings["embedding"]
    model = SentenceTransformer(
        settings["model"], device=settings["device"],
        revision=settings["revision"] or None,
        local_files_only=settings["local_files_only"],
    )
    model.max_seq_length = settings["max_seq_length"]

    def encode(texts):
        return np.asarray(model.encode(
            texts, batch_size=settings["batch_size"], show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        ), dtype=np.float64)

    canonical_vectors = encode([mathvd1_text(node) for node in nodes])
    # Encode all released query texts in their original order, deduplicating
    # identical inputs as in the formal run (also for a query-limit smoke test).
    query_texts = [mathvd1_text(query) for query in queries]
    unique_texts = list(dict.fromkeys(query_texts))
    unique_vectors = encode(unique_texts)
    by_text = dict(zip(unique_texts, unique_vectors))
    query_vectors = np.asarray([by_text[text] for text in query_texts])
    canonical_unit = unit_vectors(canonical_vectors, len(nodes))
    query_unit = unit_vectors(query_vectors, len(queries))
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, canonical=canonical_vectors, queries=query_vectors)
    temporary.replace(cache)
    write_json_atomic(manifest_path, {"input_signature": runtime.input_signature, "sha256": sha256_file(cache)})
    return canonical_unit, query_unit


def retrieve(nodes, canonical_unit, query_unit) -> list[dict[str, Any]]:
    scores = canonical_unit @ query_unit
    # Stable ties use frozen KB order; there is no type filter or threshold.
    indices = np.argsort(-scores, kind="stable")[:TOP_K]
    return [{"rank": rank, "canonical_node_id": int(nodes[index]["id"]),
             "cosine_similarity": float(scores[index])}
            for rank, index in enumerate(indices, start=1)]


def run(runtime) -> None:
    if not has_pending_queries(runtime):
        return
    nodes = canonical_entities(runtime.dataset)
    queries = [entity(query, query=True) for query in runtime.dataset.queries]
    by_id = {node["id"]: node for node in nodes}
    query_by_id = {query["id"]: query for query in queries}
    query_index = {query["id"]: index for index, query in enumerate(queries)}
    canonical_unit, query_unit = embedding_matrices(runtime, nodes, queries)
    run_pairwise(
        runtime,
        candidates_for_query=lambda query: retrieve(nodes, canonical_unit, query_unit[query_index[int(query["query_id"])]]),
        pair_prompt=lambda query, node_id: pair_prompt(query_by_id[int(query["query_id"])], by_id[node_id]),
        parse_pair=parse_equivalence,
        retrieval_stage="mathvd1_top_5",
        select_prompt=lambda query, ids: selection_prompt(query_by_id[int(query["query_id"])], [by_id[node_id] for node_id in ids]),
        parse_selection=parse_selection,
    )
