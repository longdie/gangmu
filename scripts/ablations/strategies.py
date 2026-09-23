"""Explicit differences between Gangmu and its four reported ablations."""
from pathlib import Path
import numpy as np
from scripts.experiment.dataset import PROJECT_ROOT
from scripts.benchmark.model_judging import QueryModelJudge
from scripts.ablations.raw_resolver import RawStatementResolver
from scripts.ablations.center_only_resolver import CenterOnlyResolver
from scripts.ablations.binary_judge import BinaryQueryModelJudge, _binary_alignment, _binary_strict_pair

def build_strategy(runtime, variant):
    if variant not in {"gangmu", "without_semantic_normalization", "center_only", "without_reranking", "binary_sev"}:
        raise ValueError(f"Unknown variant: {variant}")
    dataset = runtime.dataset
    common = dict(canonical_nodes=dataset.canonical_nodes, evidence_nodes=dataset.library_evidence_nodes, evidence_node_to_canonical=dataset.evidence_node_to_canonical, evidence_top_k=50, minimum_similarity=0.5, reranker_top_k=20)
    query_node, query_payload, query_embedding = dataset.query_node, dataset.query_payload, dataset.query_embedding
    prefix = ""
    if variant == "without_semantic_normalization":
        resolver = RawStatementResolver(evidence_embeddings=dataset.raw_evidence_embeddings, **common)
        raw_queries = dataset.raw_query_embeddings
        def query_node(query):
            return {"id":int(query["query_id"]), "type":query["type"], "title":query["title"], "field":query.get("field",""), "subfield":query.get("subfield",""), "source_id":query.get("source_id",""), "statement":query["statement"]}
        def query_payload(query):
            node = query_node(query)
            return {"evidence_node_id":node["id"], **{key:node[key] for key in ("type","title","field","subfield","statement")}}
        def query_embedding(query):
            return np.asarray(raw_queries[int(query["query_id"])-1], dtype=np.float32)
        prefix = "raw_"
    elif variant == "center_only":
        resolver = CenterOnlyResolver(evidence_embeddings=dataset.library_evidence_embeddings, **common)
    else:
        resolver = dataset.resolver(evidence_top_k=50, minimum_similarity=0.5, reranker_top_k=20)
    judge_class, judge_options = QueryModelJudge, {}
    if variant == "binary_sev":
        prefix = "binary_"
        judge_class = BinaryQueryModelJudge
        judge_options = {"alignment_judge":_binary_alignment, "strict_judge":_binary_strict_pair}
    return {"resolver":resolver, "query_node":query_node, "query_payload":query_payload, "query_embedding":query_embedding, "alignment_prompt":PROJECT_ROOT/"prompts"/(prefix+"cross_doc_alignment.md"), "strict_prompt":PROJECT_ROOT/"prompts"/(prefix+"cross_doc_strict_cluster_pair.md"), "judge_class":judge_class, "judge_options":judge_options}
