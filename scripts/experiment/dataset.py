from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from functools import cached_property

import numpy as np

from src.common.files import read_json
from src.common.config import GangmuConfig, get_config, resolve_project_path
from src.common.embeddings import EmbeddingCache
from src.common.paths import ROOT_DIR
from src.cross_doc.resolution import FrozenCandidateResolver
from src.evidence_graph.derivation import EMBEDDING_DIMENSIONS

PROJECT_ROOT = ROOT_DIR
# Binds the supplied 512 normalized-query vectors to their ordered text inputs.
RELEASE_QUERY_TEXT_SHA256 = "ae5ed9886ba53e57a4a0490834777ca8a49af8ccef6944ea1465eb1ab605f75b"
RELEASE_EMBEDDING_MODEL = "qwen3-embedding-8b"
RELEASE_EVIDENCE_VECTOR_SHA256 = "a4177ec749b555cf4206906d8086b00aad3637d93b666f8e98700842a7de9363"
RELEASE_CANONICAL_VECTOR_SHA256 = "d60ddec87eea4b1b9a29ae8a6f304b9ea61c70574b60581475a651c8e3c78181"
RELEASE_QUERY_VECTOR_SHA256 = "c2ff453f04fceb74c226fd7e498740c56649778eab66d31c00eb19e2fa7adb8d"
RELEASE_EVIDENCE_TEXT_SHA256 = "4eaf6a769a18e93f2a8ee4bf685b97479dc93707739556bafd1065e52ae90e9b"
RELEASE_CANONICAL_TEXT_SHA256 = "c654bd04f97ba33da143cd75ff044fc267bd313be9ccb899b077f137f523e9cc"


class ExperimentDataset:
    """Read the released two-layer KB without legacy manifests or query graphs.

    Vector matrices are lazy: text baselines and metric recomputation never
    instantiate an embedding client. IDs in published JSON remain unchanged.
    """
    @classmethod
    def load(cls, benchmark_dir: str | Path | None = None, *, config: GangmuConfig | None = None,
             cache_dir: str | Path | None = None) -> ExperimentDataset:
        directory = resolve_project_path(benchmark_dir or 'Material')
        if (directory / 'MathER-Alg').is_dir():
            directory /= 'MathER-Alg'
        if directory.name == 'Query_Set':
            directory = directory.parent
        self = cls()
        self.config = config or get_config()
        self._embedding_cache = EmbeddingCache(config=self.config, directory=cache_dir)
        self.benchmark_dir = directory
        self.canonical_dir = directory / 'Knowledge_Base' / 'Canonical_Graph'
        self.library_evidence_dir = directory / 'Knowledge_Base' / 'Evidence_Graph'
        self.query_evidence_dir = directory / 'Query_Set'
        queries_path = self.query_evidence_dir / 'queries.json'
        self.queries = read_json(queries_path)
        if not isinstance(self.queries, list) or not self.queries:
            raise ValueError('Query_Set/queries.json must be a nonempty list.')
        _validate_queries(self.queries)
        self.queries_sha256 = sha256_file(queries_path)
        self.canonical_manifest = read_json(self.canonical_dir / 'manifest.json')
        self.library_manifest = read_json(self.library_evidence_dir / 'manifest.json')
        self.canonical_nodes = read_json(self.canonical_dir / 'nodes.json')
        self.library_evidence_nodes = read_json(self.library_evidence_dir / 'nodes.json')
        _validate_sequential_nodes(self.canonical_nodes, 'Canonical')
        _validate_sequential_nodes(self.library_evidence_nodes, 'Evidence')
        _validate_canonical_nodes(self.canonical_nodes)
        _validate_gold_targets(self.queries, self.canonical_nodes)
        self.evidence_node_to_canonical = np.load(self.canonical_dir / 'evidence_node_to_canonical.npy', mmap_mode='r', allow_pickle=False)
        mapping = self.evidence_node_to_canonical
        if mapping.shape != (len(self.library_evidence_nodes),) or not np.issubdtype(mapping.dtype, np.integer):
            raise ValueError('Evidence-to-Canonical mapping shape/dtype is invalid.')
        if mapping.size and (mapping.min() < 1 or mapping.max() > len(self.canonical_nodes)):
            raise ValueError('Evidence-to-Canonical mapping contains invalid IDs.')
        _validate_canonical_evidence_bindings(canonical_nodes=self.canonical_nodes,
            evidence_nodes=self.library_evidence_nodes, mapping=mapping)
        self.query_evidence_nodes = [self.query_node(query) for query in self.queries]
        # Runtime metadata only; never recreate a removed public manifest.
        self.benchmark_manifest = {'benchmark_id':'MathER-Alg', 'query_count':len(self.queries),
            'label_counts':{label:sum(q['gold_label']==label for q in self.queries) for label in ('MERGE','NIL')}}
        core_files = ['Query_Set/queries.json', 'Knowledge_Base/Evidence_Graph/nodes.json',
            'Knowledge_Base/Evidence_Graph/edges.json', 'Knowledge_Base/Evidence_Graph/manifest.json',
            'Knowledge_Base/Canonical_Graph/nodes.json', 'Knowledge_Base/Canonical_Graph/edges.json',
            'Knowledge_Base/Canonical_Graph/manifest.json', 'Knowledge_Base/Canonical_Graph/evidence_node_to_canonical.npy']
        if (directory/'Query_Set/embeddings.npy').exists():
            core_files.append('Query_Set/embeddings.npy')
        self.benchmark_manifest_sha256 = hashlib.sha256(json.dumps(
            {name:sha256_file(directory/name) for name in core_files},sort_keys=True).encode()).hexdigest()
        self._canonical_ids_by_type = {}
        self._canonical_vectors_by_type = {}
        self._resolver = None
        self._prepare_type_indices()
        return self

    @property
    def benchmark_name(self) -> str:
        return 'MathER-Alg'

    @cached_property
    def library_evidence_embeddings(self) -> np.ndarray:
        nodes = self.library_evidence_nodes
        return self._embedding_cache.matrix('normalized_evidence', [n['id'] for n in nodes],
            [n['semantic']['normalized_statement'] for n in nodes], supplied=self.library_evidence_dir/'embeddings.npy',
            supplied_model=RELEASE_EMBEDDING_MODEL, supplied_text_hash=RELEASE_EVIDENCE_TEXT_SHA256,
            supplied_sha256=RELEASE_EVIDENCE_VECTOR_SHA256)

    @cached_property
    def canonical_statement_embeddings(self) -> np.ndarray:
        nodes = self.canonical_nodes
        return self._embedding_cache.matrix('canonical_statements', [n['id'] for n in nodes], [n['statement'] for n in nodes],
            supplied=self.canonical_dir/'canonical_statement_embeddings.npy', supplied_model=RELEASE_EMBEDDING_MODEL,
            supplied_text_hash=RELEASE_CANONICAL_TEXT_SHA256, supplied_sha256=RELEASE_CANONICAL_VECTOR_SHA256)

    @cached_property
    def query_evidence_embeddings(self) -> np.ndarray:
        return self._embedding_cache.matrix('normalized_queries', [q['query_id'] for q in self.queries],
            [q['normalized_statement'] for q in self.queries], supplied=self.query_evidence_dir/'embeddings.npy',
            supplied_model=RELEASE_EMBEDDING_MODEL, supplied_text_hash=RELEASE_QUERY_TEXT_SHA256, supplied_sha256=RELEASE_QUERY_VECTOR_SHA256)

    @cached_property
    def center_evidence_embeddings(self) -> np.ndarray:
        ids = np.asarray([n['center_evidence_node_id'] for n in self.canonical_nodes], dtype=np.int64)
        return np.asarray(self.library_evidence_embeddings[ids - 1], dtype=np.float32)

    @cached_property
    def raw_evidence_embeddings(self) -> np.ndarray:
        return self._embedding_cache.matrix('raw_evidence', [n['id'] for n in self.library_evidence_nodes],
            [n['statement'] for n in self.library_evidence_nodes])

    @cached_property
    def raw_query_embeddings(self) -> np.ndarray:
        return self._embedding_cache.matrix('raw_queries', [q['query_id'] for q in self.queries],
            [q['statement'] for q in self.queries])

    def canonical_ids_for_type(self, node_type: str) -> np.ndarray:
        return self._canonical_ids_by_type.get(str(node_type), np.empty((0,), dtype=np.int64))

    def canonical_vectors_for_type(self, node_type: str) -> np.ndarray:
        key = str(node_type)
        if key not in self._canonical_vectors_by_type:
            self._canonical_vectors_by_type[key] = np.asarray(self.canonical_statement_embeddings[self.canonical_ids_for_type(key)-1], dtype=np.float32)
        return self._canonical_vectors_by_type[key]

    def query_embedding(self, query: dict[str, Any]) -> np.ndarray:
        return np.asarray(self.query_evidence_embeddings[int(query['query_id'])-1], dtype=np.float32)

    def query_node(self, query: dict[str, Any]) -> dict[str, Any]:
        return {'id':int(query['query_id']), 'type':str(query['type']), 'title':str(query['title']),
            'field':str(query.get('field','')), 'subfield':str(query.get('subfield','')),
            'source_id':str(query.get('source_id','')), 'statement':str(query.get('statement','')),
            'semantic':{'normalized_statement':str(query['normalized_statement'])}}

    def query_payload(self, query: dict[str, Any]) -> dict[str, Any]:
        # A local query ID is enough for prompt bookkeeping. No gold fields.
        return {'evidence_node_id':int(query['query_id']), 'type':str(query['type']), 'title':str(query['title']),
            'field':str(query.get('field','')), 'subfield':str(query.get('subfield','')),
            'normalized_statement':str(query['normalized_statement'])}

    def resolver(self, *, evidence_top_k: int, minimum_similarity: float, reranker_top_k: int) -> FrozenCandidateResolver:
        if self._resolver is None:
            self._resolver = FrozenCandidateResolver(canonical_nodes=self.canonical_nodes,
                evidence_nodes=self.library_evidence_nodes, evidence_embeddings=self.library_evidence_embeddings,
                evidence_node_to_canonical=self.evidence_node_to_canonical,
                evidence_top_k=evidence_top_k, minimum_similarity=minimum_similarity, reranker_top_k=reranker_top_k)
        elif (self._resolver.evidence_top_k, self._resolver.minimum_similarity, self._resolver.reranker_top_k) != (evidence_top_k, minimum_similarity, reranker_top_k):
            raise ValueError('Experiment requested incompatible resolver settings.')
        return self._resolver

    def _prepare_type_indices(self) -> None:
        for kind in sorted({str(n['type']) for n in self.canonical_nodes}):
            self._canonical_ids_by_type[kind] = np.asarray([int(n['id']) for n in self.canonical_nodes if str(n['type']) == kind], dtype=np.int64)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sequential_nodes(nodes: Any, label: str) -> None:
    if not isinstance(nodes, list):
        raise ValueError(f"{label} nodes must be a list.")
    if [int(node.get("id", -1)) for node in nodes] != list(
        range(1, len(nodes) + 1)
    ):
        raise ValueError(f"{label} Node IDs are not sequential.")


def _validate_queries(queries: list[dict[str, Any]]) -> None:
    if [int(query.get("query_id", -1)) for query in queries] != list(
        range(1, len(queries) + 1)
    ):
        raise ValueError("Benchmark query IDs are not sequential.")
    for query in queries:
        label = str(query.get("gold_label"))
        target = query.get("gold_canonical_node_id")
        if str(query.get("type", "")) not in {"definition", "theorem"}:
            raise ValueError("A Benchmark query has an unsupported node type.")
        if not str(query.get("normalized_statement", "")).strip():
            raise ValueError("A Benchmark query has an empty normalized statement.")
        if label == "MERGE" and (
            isinstance(target, bool) or not isinstance(target, int)
        ):
            raise ValueError("A MERGE query has no integer Gold target.")
        if label == "NIL" and target is not None:
            raise ValueError("A NIL query has a non-null Gold target.")
        if label not in {"MERGE", "NIL"}:
            raise ValueError(f"Unsupported Gold label: {label}")


def _validate_canonical_nodes(nodes: list[dict[str, Any]]) -> None:
    for node in nodes:
        if str(node.get("type", "")) not in {"definition", "theorem"}:
            raise ValueError("A Canonical Node has an unsupported node type.")
        if not str(node.get("statement", "")).strip():
            raise ValueError("A Canonical Node has an empty statement.")


def _validate_gold_targets(
    queries: list[dict[str, Any]],
    canonical_nodes: list[dict[str, Any]],
) -> None:
    for query in queries:
        if query["gold_label"] != "MERGE":
            continue
        target = int(query["gold_canonical_node_id"])
        if not 1 <= target <= len(canonical_nodes):
            raise ValueError(
                f"Benchmark query {query['query_id']} has an out-of-range Gold target."
            )
        if str(canonical_nodes[target - 1]["type"]) != str(query["type"]):
            raise ValueError(
                f"Benchmark query {query['query_id']} and its Gold target have "
                "different node types."
            )


def _validate_canonical_evidence_bindings(
    *,
    canonical_nodes: list[dict[str, Any]],
    evidence_nodes: list[dict[str, Any]],
    mapping: np.ndarray,
) -> None:
    mapped_members: dict[int, list[int]] = {
        canonical_node_id: []
        for canonical_node_id in range(1, len(canonical_nodes) + 1)
    }
    for evidence_node_id, canonical_node_id in enumerate(mapping.tolist(), start=1):
        mapped_members[int(canonical_node_id)].append(evidence_node_id)

    for node in canonical_nodes:
        canonical_node_id = int(node["id"])
        members = [int(value) for value in node.get("evidence_node_ids", [])]
        if not members or len(set(members)) != len(members):
            raise ValueError("A Canonical Node has invalid Evidence members.")
        if sorted(members) != mapped_members[canonical_node_id]:
            raise ValueError(
                "Canonical Evidence members disagree with evidence_node_to_canonical."
            )
        center = int(node.get("center_evidence_node_id", 0))
        diverse = [
            int(value) for value in node.get("diverse_evidence_node_ids", [])
        ]
        if center not in members:
            raise ValueError("A Canonical center is not one of its Evidence members.")
        if (
            len(diverse) > 2
            or len(set(diverse)) != len(diverse)
            or center in diverse
            or any(value not in members for value in diverse)
        ):
            raise ValueError("A Canonical Node has invalid diverse Evidence members.")
        if any(
            str(evidence_nodes[evidence_node_id - 1]["type"])
            != str(node["type"])
            for evidence_node_id in members
        ):
            raise ValueError("A Canonical Node contains a different Evidence type.")
def _validate_l2_normalized_embeddings(
    embeddings: np.ndarray,
    label: str,
    *,
    tolerance: float = 1e-4,
    chunk_size: int = 2048,
) -> None:
    for start in range(0, len(embeddings), chunk_size):
        chunk = np.asarray(embeddings[start : start + chunk_size], dtype=np.float32)
        if not np.all(np.isfinite(chunk)):
            raise ValueError(f"{label} embeddings contain non-finite values.")
        norms = np.linalg.norm(chunk, axis=1)
        if np.any(np.abs(norms - 1.0) > tolerance):
            raise ValueError(f"{label} embeddings are not L2-normalized.")
