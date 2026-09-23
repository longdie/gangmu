"""Canonical representative payloads for methods without vector retrieval."""
from src.cross_doc.resolution import FrozenCandidateResolver

class CanonicalTextView:
    """Reuse production representative selection without loading any vectors."""
    candidate_representatives = FrozenCandidateResolver.candidate_representatives
    canonical_payload = FrozenCandidateResolver.canonical_payload

    def __init__(self, dataset):
        self.canonical_nodes = dataset.canonical_nodes
        self.evidence_nodes = dataset.library_evidence_nodes
