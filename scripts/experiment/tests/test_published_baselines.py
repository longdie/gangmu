"""Offline behavioral checks; no downloaded models, benchmark files or API keys."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from src.common.config import CONFIG_EXAMPLE_PATH, GangmuConfig
from src.common.files import TokenUsageTracker, write_json_atomic
from scripts.experiment import automathkg, baseline_common, llmforem
from scripts.experiment.baseline_common import ResponsesClient, response_text
from scripts.experiment.registry import METHOD_MODULES


def response(answer="No", **overrides):
    return {
        "status": "completed", "model": "test-model",
        "output": [{"type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": answer}]}],
        "usage": {"input_tokens": 12, "output_tokens": 1},
        **overrides,
    }


class MemoryStore:
    def __init__(self, root):
        self.records = {}
        self.checkpoint_dir = root

    def load_query(self, query_id):
        return deepcopy(self.records.get(query_id))

    def save_query(self, query_id, record):
        self.records[query_id] = deepcopy({**record, "query_id": query_id})


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        data = tomllib.loads(CONFIG_EXAMPLE_PATH.read_text())
        data["providers"]["baseline"].update(
            api_key="test-key", api_key_env="", base_url="https://example.invalid/v1",
            base_url_env="", trust_env=False,
        )
        for method in ("automathkg", "llmforem"):
            data["models"][method]["name"] = "test-model"
        data["requests"]["call_attempts"] = 1
        self.config = GangmuConfig(data, None)
        self.config.validate()
        nodes = [{"id": i, "type": "definition", "title": f"Object {i}",
                  "statement": f"Definition ${i}$.", "field": "Algebra"} for i in range(1, 27)]
        nodes[-1]["type"] = "theorem"
        query = {"query_id": 1, "type": "definition", "title": "Query", "field": "Algebra",
                 "normalized_statement": "Define $G$.", "gold_label": "GOLD_SENTINEL",
                 "gold_canonical_node_id": 999999}
        vectors = np.asarray([[np.cos(i / 100), np.sin(i / 100)] for i in range(26)], dtype=np.float32)
        vectors[-1] = [1, 0]
        self.dataset = SimpleNamespace(canonical_nodes=nodes, queries=[query], canonical_dir=root)
        self.dataset.canonical_ids_for_type = lambda kind: np.array([n["id"] for n in nodes if n["type"] == kind])
        self.dataset.canonical_vectors_for_type = lambda kind: vectors[self.dataset.canonical_ids_for_type(kind) - 1]
        self.dataset.query_embedding = lambda _: np.array([1, 0], dtype=np.float32)
        write_json_atomic(root / "edges.json", [
            {"source": 1, "target": 26, "types": ["proof"]},
            {"source": 1, "target": 26, "types": ["dependency"]},
        ])
        self.runtime = SimpleNamespace(
            dataset=self.dataset, queries=[query], config=self.config,
            store=MemoryStore(root), usage_tracker=TokenUsageTracker(), input_signature="test-signature",
        )

    def configure_method(self, name):
        self.runtime.method_name = name
        module = {"automathkg": automathkg, "llmforem": llmforem}[name]
        self.runtime.settings = module.method_settings(self.dataset, self.config, {"model_task": "sev", "workers": 5})
        return module

    def transport(self, answers):
        pending = iter(answers)
        self.requests = []

        def handler(request):
            self.requests.append(json.loads(request.content))
            item = next(pending)
            if isinstance(item, int):
                return httpx.Response(item, json={"error": "synthetic failure"})
            return httpx.Response(200, json=item if isinstance(item, dict) else response(item))

        transport = httpx.MockTransport(handler)
        return patch.object(baseline_common, "ResponsesClient", lambda runtime: ResponsesClient(runtime, transport=transport))

    def test_registration_and_distinct_model_tasks(self):
        self.assertEqual(len(METHOD_MODULES), 9)
        for name in ("automathkg", "llmforem"):
            self.configure_method(name)
            self.assertEqual(self.runtime.settings["model_task"], name)
            self.assertEqual(self.runtime.settings["workers"], 1)

    def test_llmforem_judges_all_twenty_and_returns_first_yes(self):
        module = self.configure_method("llmforem")
        with self.transport(["No", "Yes", "Yes"] + ["No"] * 17):
            module.run(self.runtime)
        record = self.runtime.store.load_query(1)
        self.assertEqual(record["predicted_canonical_node_id"], 2)
        self.assertEqual(record["all_yes_candidate_ids"], [2, 3])
        self.assertEqual(len(self.requests), 20)
        self.assertNotIn(26, record["trace"]["candidate_ids_snapshot"])
        first = self.requests[0]
        self.assertEqual(first["reasoning"], {"effort": "low"})
        self.assertEqual(first["max_output_tokens"], 1024)
        self.assertIs(first["store"], False)
        self.assertNotIn("temperature", first)
        self.assertNotIn("GOLD_SENTINEL", json.dumps(self.requests))
        self.assertNotIn("999999", json.dumps(self.requests))
        self.assertEqual(first["input"][0]["content"][0]["text"],
                         "Do the two entity descriptions refer to the same real-world entity? "
                         "Answer with 'Yes' if they do and 'No' if they do not.\n"
                         "Entity 1: 'definition Query Define $G$.'\nEntity 2: 'definition Object 1 Definition $1$.'")
        with self.transport([]):
            module.run(self.runtime)
        self.assertEqual(self.requests, [])

    def test_all_negative_means_nil(self):
        module = self.configure_method("llmforem")
        with self.transport(["No"] * 20):
            module.run(self.runtime)
        self.assertIsNone(self.runtime.store.load_query(1)["predicted_canonical_node_id"])

    def test_failed_call_remains_partial_and_resume_skips_successful_pairs(self):
        module = self.configure_method("llmforem")
        with self.transport(["Yes", 503]), self.assertRaises(RuntimeError):
            module.run(self.runtime)
        partial = self.runtime.store.load_query(1)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(len(partial["trace"]["candidate_judgments"]), 1)
        self.assertNotIn("predicted_canonical_node_id", partial)
        with self.transport(["No"] * 19):
            module.run(self.runtime)
        self.assertEqual(len(self.requests), 19)
        self.assertEqual(self.runtime.store.load_query(1)["predicted_canonical_node_id"], 1)

    def test_automathkg_references_and_cross_type_retrieval(self):
        nodes = automathkg.canonical_entities(self.dataset)
        self.assertEqual(nodes[0]["out_refs"], {"Theorem:Object 26 [id=26]": ["dependency", "proof"]})
        query = automathkg.entity(self.dataset.queries[0], query=True)
        self.assertEqual(query["in_refs"], {})
        self.assertEqual(query["out_refs"], {})
        self.assertTrue(automathkg.mathvd1_text(query).endswith("in references: {}\nout references: {}"))
        vectors = np.zeros((26, automathkg.DIMENSIONS))
        vectors[:, 0] = np.arange(1, 27)
        qvector = np.zeros(automathkg.DIMENSIONS)
        qvector[0] = 1
        candidates = automathkg.retrieve(nodes, vectors, qvector)
        self.assertEqual([c["canonical_node_id"] for c in candidates], [26, 25, 24, 23, 22])

    def test_automathkg_selects_only_from_yes_candidates(self):
        module = self.configure_method("automathkg")
        vectors = np.zeros((26, automathkg.DIMENSIONS))
        vectors[:, 0] = 1
        query_vector = vectors[:1]
        with patch.object(module, "embedding_matrices", return_value=(vectors, query_vector)):
            with self.transport(["yes", "no", "yes", "no", "no", "3"]):
                module.run(self.runtime)
        self.assertEqual(self.runtime.store.load_query(1)["predicted_canonical_node_id"], 3)
        self.assertEqual(len(self.requests), 6)
        self.assertEqual(self.requests[0]["max_output_tokens"], 4096)
        prompt = self.requests[-1]["input"][0]["content"][0]["text"]
        selected_payload = json.loads(prompt.split("Candidate theorems:\n")[1])
        self.assertEqual([node["id"] for node in selected_payload], [1, 3])
        self.assertNotIn("gold", prompt)

    def test_strict_output_parsers(self):
        self.assertTrue(automathkg.parse_equivalence('```text\nyes\n```'))
        self.assertTrue(llmforem.parse_equivalence("YES!"))
        for module in (automathkg, llmforem):
            for answer in ("Yes, because it is equivalent", "maybe", ""):
                with self.assertRaises(ValueError):
                    module.parse_equivalence(answer)
        for answer in ("true", '"1"', "2", "[1]"):
            with self.assertRaises(ValueError):
                automathkg.parse_selection(answer, [1])

    def test_invalid_responses_are_errors(self):
        self.assertEqual(response_text(response("Yes"), "test-model"), "Yes")
        cases = [response(model="other-model"), response(status="incomplete"),
                 response(error={"message": "error"}), response(output=[]),
                 response(output=[{"type": "message", "role": "assistant", "content": [
                     {"type": "refusal", "refusal": "refused"}]}])]
        for body in cases:
            with self.assertRaises(ValueError):
                response_text(body, "test-model")

    def test_model_mismatch_does_not_create_nil(self):
        module = self.configure_method("llmforem")
        with self.transport([response(model="other-model")]), self.assertRaises(RuntimeError):
            module.run(self.runtime)
        self.assertEqual(self.runtime.store.load_query(1)["status"], "partial")

    def test_minilm_cache_reuses_vectors_and_detects_corruption(self):
        self.configure_method("automathkg")
        nodes = automathkg.canonical_entities(self.dataset)
        queries = [automathkg.entity(q, query=True) for q in self.dataset.queries]
        calls = []

        class FakeEncoder:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, texts, **kwargs):
                calls.append(texts)
                return np.ones((len(texts), automathkg.DIMENSIONS), dtype=np.float32)

        with patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(SentenceTransformer=FakeEncoder)}):
            first = automathkg.embedding_matrices(self.runtime, nodes, queries)
            second = automathkg.embedding_matrices(self.runtime, nodes, queries)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(np.linalg.norm(first[0], axis=1), 1)
        cache = self.runtime.store.checkpoint_dir / "minilm_vectors.npz"
        cache.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            automathkg.embedding_matrices(self.runtime, nodes, queries)


if __name__ == "__main__":
    unittest.main()
