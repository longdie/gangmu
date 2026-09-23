from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Protocol

from src.common.files import ProjectLogger, clean_whitespace, utc_now
from src.single_doc.llm import ProviderCircuitOpenError, ProviderRequestError
from src.single_doc.preprocessing import MarkdownPreprocessor
from src.single_doc.schema import NODE_TYPES, normalize_field_pair


class JSONClient(Protocol):
    def call_json(
        self,
        *,
        model_task: str,
        prompt: str,
        input_text: str,
        usage_task: str,
        validator: Any = None,
        fallback_model_task: str | None = None,
        preserve_latex_escapes: bool = False,
    ) -> dict[str, Any] | None: ...


class NodeExtractor:
    def __init__(
        self,
        *,
        llm_client: JSONClient,
        preprocessor: MarkdownPreprocessor,
        prompt_path: str | Path,
        source_id: str,
        default_field: str,
        default_subfield: str,
        max_split_depth: int,
        logger: ProjectLogger | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.preprocessor = preprocessor
        self.prompt_path = Path(prompt_path)
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.source_id = source_id
        self.default_field = default_field
        self.default_subfield = default_subfield
        self.max_split_depth = max_split_depth
        self.logger = logger

    def extract_from_segment(
        self,
        segment: dict[str, Any],
        *,
        depth: int = 0,
    ) -> list[dict[str, Any]]:
        result = self.llm_client.call_json(
            model_task="node_extraction",
            prompt=self.prompt,
            input_text=self._format_input(segment),
            usage_task="node_extraction",
            validator=self._validate_output,
        )
        if result is None:
            return self._retry_split(segment, depth=depth)

        nodes: list[dict[str, Any]] = []
        for raw_node in result["nodes"]:
            normalized = self.normalize_node(raw_node, segment)
            if normalized is not None:
                nodes.append(normalized)
        return nodes

    def normalize_node(
        self,
        raw_node: Any,
        segment: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(raw_node, dict):
            return None

        node_type = clean_whitespace(str(raw_node.get("type", "")).lower())
        if node_type in {
            "lemma",
            "proposition",
            "corollary",
            "property",
            "conjecture",
            "claim",
        }:
            node_type = "theorem"
        if node_type not in NODE_TYPES:
            return None

        statement = clean_whitespace(str(raw_node.get("statement", "") or ""))
        if not statement:
            return None

        label = self._normalize_label(
            clean_whitespace(str(raw_node.get("label", "") or ""))
        )
        if self._should_skip(label=label, node_type=node_type):
            return None

        title = clean_whitespace(
            str(raw_node.get("title") or raw_node.get("name") or "")
        )
        if not title:
            title = self._derive_title(statement, node_type)

        field, subfield = normalize_field_pair(
            clean_whitespace(str(raw_node.get("field", "") or "")),
            clean_whitespace(str(raw_node.get("subfield", "") or "")),
            default_field=self.default_field,
            default_subfield=self.default_subfield,
        )
        proof = clean_whitespace(str(raw_node.get("proof", "") or ""))

        node: dict[str, Any] = {
            "id": 0,
            "type": node_type,
            "title": title,
            "statement": statement,
            "field": field,
            "subfield": subfield,
            "source_id": self.source_id,
            "in_edges": [],
            "out_edges": [],
        }
        if label:
            node["label"] = label
        if node_type == "theorem" and proof:
            node["proof"] = proof

        location: dict[str, Any] = {}
        segment_id = clean_whitespace(str(segment.get("segment_id", "") or ""))
        if segment_id:
            location["segment_id"] = segment_id
        pages = segment.get("pages")
        if isinstance(pages, list) and pages:
            location["pages"] = list(pages)
        title_path = segment.get("title_path")
        if isinstance(title_path, list) and title_path:
            location["title_path"] = list(title_path)
        if location:
            node["location"] = location
        return node

    @staticmethod
    def assign_ids(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for node_id, node in enumerate(nodes, start=1):
            node["id"] = node_id
        return nodes

    def _retry_split(
        self,
        segment: dict[str, Any],
        *,
        depth: int,
    ) -> list[dict[str, Any]]:
        if depth >= self.max_split_depth:
            if self.logger:
                self.logger.error(
                    "Node extraction failed after recursive splitting",
                    segment_id=segment.get("segment_id"),
                    depth=depth,
                )
            raise RuntimeError("Node extraction exhausted validation retries; the segment remains unfinished.")

        child_segments = self.preprocessor.split_segment_for_retry(segment)
        if len(child_segments) <= 1:
            if self.logger:
                self.logger.error(
                    "Node extraction failed and segment cannot be split further",
                    segment_id=segment.get("segment_id"),
                    depth=depth,
                )
            raise RuntimeError("Node extraction failed for an indivisible segment; retry this document.")

        nodes: list[dict[str, Any]] = []
        for child in child_segments:
            nodes.extend(self.extract_from_segment(child, depth=depth + 1))
        return nodes

    @staticmethod
    def _validate_output(result: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(result.get("nodes"), list):
            return False, "Node output must contain a nodes list."
        return True, ""

    @staticmethod
    def _should_skip(*, label: str, node_type: str) -> bool:
        normalized_label = label.lower()
        if any(
            word in normalized_label
            for word in ("question", "problem", "exercise", "example", "remark")
        ):
            return True
        return node_type == "theorem" and not label

    @staticmethod
    def _normalize_label(label: str) -> str:
        replacements = {
            "定义": "Definition",
            "定理": "Theorem",
            "引理": "Lemma",
            "命题": "Proposition",
            "推论": "Corollary",
            "性质": "Property",
            "猜想": "Conjecture",
            "例": "Example",
        }
        for chinese, english in replacements.items():
            label = label.replace(chinese, english)
        return clean_whitespace(label)

    @staticmethod
    def _derive_title(statement: str, node_type: str) -> str:
        words = tokenize_retrieval_text(statement)[:8]
        return " ".join(words).title() if words else node_type.title()

    @staticmethod
    def _format_input(segment: dict[str, Any]) -> str:
        title_path = " > ".join(segment.get("title_path", []))
        pages = ", ".join(str(page) for page in segment.get("pages", []))
        return (
            "Document segment metadata:\n"
            f"- segment_id: {segment.get('segment_id', '')}\n"
            f"- pages: {pages}\n"
            f"- title_path: {title_path}\n\n"
            "Markdown segment:\n"
            "```markdown\n"
            f"{segment.get('text', '')}\n"
            "```"
        )


class NodeQualityChecker:
    def clean_nodes(
        self,
        nodes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        deduped, duplicate_groups = self._deduplicate_exact(nodes)
        near_duplicates = self._find_near_duplicates(deduped)
        suspicious = [
            result
            for result in (self._check_node(node) for node in deduped)
            if result["flags"]
        ]
        return deduped, {
            "created_at": utc_now(),
            "input_node_count": len(nodes),
            "output_node_count": len(deduped),
            "removed_duplicate_count": len(nodes) - len(deduped),
            "exact_duplicate_groups": duplicate_groups,
            "near_duplicate_candidates": near_duplicates,
            "suspicious_nodes": suspicious,
        }

    def _deduplicate_exact(
        self,
        nodes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        duplicates: dict[tuple[str, str], list[dict[str, Any]]] = {}
        deduped: list[dict[str, Any]] = []
        for node in nodes:
            key = (
                str(node.get("type", "")),
                self._statement_key(str(node.get("statement", ""))),
            )
            if not key[1] or key not in seen:
                seen[key] = node
                duplicates[key] = [node]
                deduped.append(node)
            else:
                duplicates[key].append(node)

        groups = [
            {
                "kept_original_id": group[0].get("id"),
                "removed_original_ids": [node.get("id") for node in group[1:]],
                "title": group[0].get("title", ""),
                "segments": [
                    node.get("location", {}).get("segment_id", "") for node in group
                ],
            }
            for group in duplicates.values()
            if len(group) > 1
        ]
        return deduped, groups

    def _find_near_duplicates(
        self,
        nodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for node in nodes:
            node_type = str(node.get("type", ""))
            for reason in ("label", "title"):
                key_value = self._statement_key(str(node.get(reason, "")))
                if key_value:
                    groups.setdefault((node_type, reason, key_value), []).append(node)

        candidates: list[dict[str, Any]] = []
        seen_pairs: set[tuple[int, int, str]] = set()
        for (_, reason, _), group in groups.items():
            if len(group) <= 1 or len(group) > 50:
                continue
            for left_index, left in enumerate(group):
                for right in group[left_index + 1 :]:
                    left_id = int(left.get("id", 0))
                    right_id = int(right.get("id", 0))
                    pair = (min(left_id, right_id), max(left_id, right_id), reason)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    similarity = token_similarity(
                        str(left.get("statement", "")),
                        str(right.get("statement", "")),
                    )
                    if reason == "label" or similarity >= 0.82:
                        candidates.append(
                            {
                                "left_id": left_id,
                                "right_id": right_id,
                                "reason": f"same_{reason}",
                                "statement_token_similarity": round(similarity, 3),
                            }
                        )
                        if len(candidates) >= 200:
                            return candidates
        return candidates

    def _check_node(self, node: dict[str, Any]) -> dict[str, Any]:
        statement = str(node.get("statement", "") or "")
        flags: list[str] = []
        if statement.count("$$") % 2:
            flags.append("unbalanced_display_math")
        inline_without_display = statement.replace("$$", "")
        if inline_without_display.count("$") % 2:
            flags.append("unbalanced_inline_math")
        begins = re.findall(r"\\begin\{([^}]+)\}", statement)
        ends = re.findall(r"\\end\{([^}]+)\}", statement)
        if sorted(begins) != sorted(ends):
            flags.append("latex_environment_mismatch")
        if abs(statement.count("{") - statement.count("}")) >= 2:
            flags.append("possible_unbalanced_braces")
        if re.match(r"^(?:\([b-z]\)|[b-z]\)|=|,|;)", statement.strip()):
            flags.append("starts_mid_statement")
        return {
            "id": node.get("id"),
            "title": node.get("title", ""),
            "segment_id": node.get("location", {}).get("segment_id", ""),
            "flags": flags,
        }

    @staticmethod
    def _statement_key(statement: str) -> str:
        return (
            re.sub(r"\s+", " ", statement or "")
            .strip()
            .lower()
            .replace("’", "'")
            .replace("–", "-")
            .replace("—", "-")
        )


class DependencyExtractor:
    _GENERIC_FORMAL_LABELS = {
        "definition",
        "theorem",
        "lemma",
        "proposition",
        "corollary",
        "conjecture",
        "claim",
        "property",
        "axiom",
        "定义",
        "定理",
        "引理",
        "命题",
        "推论",
        "猜想",
        "断言",
        "性质",
        "公理",
    }

    def __init__(
        self,
        *,
        llm_client: JSONClient,
        prompt_path: str | Path,
        position_window: int,
        title_bm25_top_k: int,
        proof_bm25_top_k: int,
        title_score_weight: float,
        statement_score_weight: float,
        batch_size: int,
        parallel_workers: int,
        logger: ProjectLogger | None = None,
    ) -> None:
        if not 1 <= batch_size <= 5:
            raise ValueError("Dependency batch_size must be between 1 and 5.")
        self.llm_client = llm_client
        self.prompt_path = Path(prompt_path)
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.position_window = position_window
        self.title_bm25_top_k = title_bm25_top_k
        self.proof_bm25_top_k = proof_bm25_top_k
        self.title_score_weight = float(title_score_weight)
        self.statement_score_weight = float(statement_score_weight)
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
        self.logger = logger
        self.candidate_diagnostics: list[dict[str, Any]] = []
        self._label_token_pattern = re.compile(r"[a-z]+\d+")

    def extract_edges(
        self,
        nodes: list[dict[str, Any]],
        *,
        completed_results: dict[int, dict[str, Any]] | None = None,
        progress_callback: (
            Callable[[dict[int, dict[str, Any]], bool], None] | None
        ) = None,
    ) -> list[dict[str, Any]]:
        if not nodes:
            self.candidate_diagnostics = []
            return []

        results = {
            index: result
            for index, result in (completed_results or {}).items()
            if 0 <= index < len(nodes) and isinstance(result, dict)
        }
        pending_indices = [index for index in range(len(nodes)) if index not in results]
        max_workers = max(
            1,
            min(self.parallel_workers, len(pending_indices) or 1),
        )
        fatal_error: Exception | None = None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._judge_target,
                    index,
                    nodes[index],
                    nodes[:index],
                ): index
                for index in pending_indices
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    fatal_error = exc
                    for pending_future in futures:
                        pending_future.cancel()
                    break
                if progress_callback is not None:
                    progress_callback(results, False)

        if fatal_error is not None:
            for future, index in futures.items():
                if index in results or future.cancelled():
                    continue
                try:
                    results[index] = future.result()
                except Exception:
                    # Failed or circuit-open targets remain pending for the next run.
                    continue
        if progress_callback is not None:
            progress_callback(results, True)
        if fatal_error is not None:
            raise fatal_error

        self.candidate_diagnostics = [
            results[index]["diagnostic"] for index in sorted(results)
        ]
        edges: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str]] = set()
        for index in sorted(results):
            for edge in results[index]["edges"]:
                key = (edge["source"], edge["target"], edge["type"])
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)
        return edges

    def get_candidate_info(
        self,
        target: dict[str, Any],
        previous_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        position = (
            previous_nodes[-self.position_window :] if self.position_window > 0 else []
        )
        explicit_statement, explicit_proof = self._explicit_label_edges(
            target,
            previous_nodes,
        )
        explicit_sources = {
            edge["source"] for edge in explicit_statement + explicit_proof
        }
        statement_bm25 = self._bm25_candidates(
            target_text=str(target.get("statement", "")),
            previous_nodes=previous_nodes,
            top_k=self.title_bm25_top_k,
        )
        proof_bm25 = self._bm25_candidates(
            target_text=str(target.get("proof", "")),
            previous_nodes=previous_nodes,
            top_k=(
                self.proof_bm25_top_k
                if target.get("type") == "theorem" and target.get("proof")
                else 0
            ),
        )
        candidates = [
            node
            for node in self._deduplicate_candidates(
                position + statement_bm25 + proof_bm25
            )
            if node.get("id") not in explicit_sources
        ]
        return {
            "candidates": candidates,
            "position_ids": self._ids(position),
            "statement_bm25_ids": self._ids(statement_bm25),
            "proof_bm25_ids": self._ids(proof_bm25),
            "explicit_statement_edges": explicit_statement,
            "explicit_proof_edges": explicit_proof,
        }

    def _judge_target(
        self,
        index: int,
        target: dict[str, Any],
        previous_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidate_info = self.get_candidate_info(target, previous_nodes)
        edges = [
            *candidate_info["explicit_statement_edges"],
            *candidate_info["explicit_proof_edges"],
        ]
        seen = {(edge["source"], edge["target"], edge["type"]) for edge in edges}
        candidates = candidate_info["candidates"]
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            for edge in self._judge_batch(target, batch):
                key = (edge["source"], edge["target"], edge["type"])
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)

        return {
            "edges": edges,
            "diagnostic": {
                "target_id": target.get("id"),
                "candidate_ids": self._ids(candidates),
                "position_candidate_ids": candidate_info["position_ids"],
                "statement_bm25_candidate_ids": candidate_info["statement_bm25_ids"],
                "proof_bm25_candidate_ids": candidate_info["proof_bm25_ids"],
                "explicit_statement_source_ids": [
                    edge["source"]
                    for edge in candidate_info["explicit_statement_edges"]
                ],
                "explicit_proof_source_ids": [
                    edge["source"] for edge in candidate_info["explicit_proof_edges"]
                ],
                "accepted_edges": edges,
                "target_index": index,
            },
        }

    def _judge_batch(
        self,
        target: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        result = self.llm_client.call_json(
            model_task="dependency_extraction",
            prompt=self.prompt,
            input_text=self._format_input(target, candidates),
            usage_task="dependency_extraction",
            validator=self._validator(candidates),
            fallback_model_task="dependency_fallback",
        )
        if result is None:
            if self.logger:
                self.logger.error(
                    "Dependency batch remained invalid; no edges accepted",
                    target_id=target.get("id"),
                    candidate_ids=self._ids(candidates),
                )
            raise RuntimeError("Dependency verification exhausted validation retries; its target remains unfinished.")

        candidate_ids = set(self._ids(candidates))
        edges: list[dict[str, Any]] = []
        for item in result["edges"]:
            candidate_id = int(item["candidate_id"])
            edge_type = clean_whitespace(str(item.get("edge_type", "")).lower())
            if candidate_id not in candidate_ids or edge_type == "none":
                continue
            if edge_type not in {"dependency", "proof"}:
                continue
            if target.get("type") == "definition" and edge_type == "proof":
                continue
            edges.append(
                {
                    "source": candidate_id,
                    "target": int(target["id"]),
                    "type": edge_type,
                    "semantics": clean_whitespace(str(item.get("semantics", "") or "")),
                }
            )
        return edges

    def _validator(self, candidates: list[dict[str, Any]]):
        expected_ids = set(self._ids(candidates))

        def validate(result: dict[str, Any]) -> tuple[bool, str]:
            items = result.get("edges")
            if not isinstance(items, list):
                return False, "Dependency output must contain an edges list."
            if len(items) != len(expected_ids):
                return False, (
                    "Dependency output must contain exactly one item per candidate."
                )

            output_ids: list[int] = []
            for item in items:
                if not isinstance(item, dict):
                    return False, "Every dependency item must be an object."
                try:
                    output_ids.append(int(item.get("candidate_id")))
                except (TypeError, ValueError):
                    return False, "Every dependency item needs a valid candidate_id."
                edge_type = clean_whitespace(str(item.get("edge_type", "")).lower())
                if edge_type not in {"dependency", "proof", "none"}:
                    return False, f"Invalid edge_type: {edge_type}"
            if set(output_ids) != expected_ids or len(output_ids) != len(
                set(output_ids)
            ):
                return False, "Dependency candidate IDs do not match the input."
            return True, ""

        return validate

    def _explicit_label_edges(
        self,
        target: dict[str, Any],
        previous_nodes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        statement_text = self._normalize_label_text(str(target.get("statement", "")))
        proof_text = self._normalize_label_text(str(target.get("proof", "")))
        statement_edges: list[dict[str, Any]] = []
        proof_edges: list[dict[str, Any]] = []
        for candidate in previous_nodes:
            label = clean_whitespace(str(candidate.get("label", "") or ""))
            if not label:
                continue
            if self._is_generic_formal_label(label):
                continue
            label_token = self._normalized_label_token(label)
            if not label_token:
                continue
            candidate_id = int(candidate["id"])
            target_id = int(target["id"])
            candidate_title = str(candidate.get("title", "") or label)
            target_title = str(target.get("title", "") or target_id)
            if self._contains_label_token(statement_text, label_token):
                statement_edges.append(
                    {
                        "source": candidate_id,
                        "target": target_id,
                        "type": "dependency",
                        "semantics": (
                            f"{target_title} explicitly cites {label} "
                            f"({candidate_title}) in its statement."
                        ),
                    }
                )
            elif target.get("type") == "theorem" and self._contains_label_token(
                proof_text, label_token
            ):
                proof_edges.append(
                    {
                        "source": candidate_id,
                        "target": target_id,
                        "type": "proof",
                        "semantics": (
                            f"The proof of {target_title} explicitly cites {label} "
                            f"({candidate_title})."
                        ),
                    }
                )
        return statement_edges, proof_edges

    @classmethod
    def _is_generic_formal_label(cls, label: str) -> bool:
        """Return whether a label names only a formal-item kind.

        Labels such as ``Theorem`` or ``Definition:`` are not unique
        references. Treating them as explicit citations links every earlier
        item of that kind whenever the ordinary word occurs in later text.
        """

        return cls._normalize_label_text(label) in cls._GENERIC_FORMAL_LABELS

    def _bm25_candidates(
        self,
        *,
        target_text: str,
        previous_nodes: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not previous_nodes or top_k <= 0:
            return []
        query = tokenize_retrieval_text(target_text)
        if not query:
            return []

        title_scores = bm25_scores(
            query,
            [
                tokenize_retrieval_text(str(node.get("title", "")))
                for node in previous_nodes
            ],
        )
        statement_scores = bm25_scores(
            query,
            [
                tokenize_retrieval_text(str(node.get("statement", "")))
                for node in previous_nodes
            ],
        )
        weighted_scores = [
            self.title_score_weight * title_score
            + self.statement_score_weight * statement_score
            for title_score, statement_score in zip(
                title_scores,
                statement_scores,
            )
        ]
        ranked = sorted(
            zip(previous_nodes, weighted_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [node for node, score in ranked if score > 0][:top_k]

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen: set[int] = set()
        result: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = int(candidate["id"])
            if candidate_id not in seen:
                seen.add(candidate_id)
                result.append(candidate)
        return result

    @staticmethod
    def _ids(nodes: list[dict[str, Any]]) -> list[int]:
        return [int(node["id"]) for node in nodes]

    def _normalized_label_token(self, label: str) -> str:
        normalized = self._normalize_label_text(label)
        match = self._label_token_pattern.search(normalized)
        return match.group(0) if match else normalized

    @staticmethod
    def _contains_label_token(text: str, label_token: str) -> bool:
        if not label_token:
            return False
        suffix_guard = r"(?!\d)" if label_token[-1].isdigit() else ""
        return bool(re.search(re.escape(label_token) + suffix_guard, text))

    @staticmethod
    def _normalize_label_text(text: str) -> str:
        normalized = re.sub(r"\s+", "", (text or "").lower())
        return re.sub(r"[^a-z0-9\u3400-\u9fff]", "", normalized)

    @staticmethod
    def _format_input(
        target: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> str:
        candidate_blocks = []
        for candidate in candidates:
            candidate_blocks.append(
                "\n".join(
                    [
                        f"Candidate ID: {candidate.get('id')}",
                        f"Type: {candidate.get('type', '')}",
                        f"Title: {candidate.get('title', '')}",
                        f"Label: {candidate.get('label', '')}",
                        f"Statement: {candidate.get('statement', '')}",
                    ]
                )
            )
        return (
            "Target node:\n"
            f"ID: {target.get('id')}\n"
            f"Type: {target.get('type', '')}\n"
            f"Title: {target.get('title', '')}\n"
            f"Label: {target.get('label', '')}\n"
            f"Statement: {target.get('statement', '')}\n"
            f"Proof: {target.get('proof', '')}\n\n"
            "Candidate prerequisite nodes, at most five:\n\n"
            + "\n\n---\n\n".join(candidate_blocks)
        )

    @staticmethod
    def _empty_diagnostic(target: dict[str, Any]) -> dict[str, Any]:
        return {
            "target_id": target.get("id"),
            "candidate_ids": [],
            "position_candidate_ids": [],
            "statement_bm25_candidate_ids": [],
            "proof_bm25_candidate_ids": [],
            "explicit_statement_source_ids": [],
            "explicit_proof_source_ids": [],
            "accepted_edges": [],
        }


def normalize_retrieval_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = re.sub(r"\$+", " ", normalized)
    normalized = re.sub(r"\\[a-zA-Z]+", " ", normalized)
    normalized = re.sub(
        r"\b(theorem|definition|lemma|proposition|corollary|property)\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"[_\-]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\b([a-z]{4,})s\b", r"\1", normalized)
    return clean_whitespace(normalized)


def tokenize_retrieval_text(text: str) -> list[str]:
    return [token for token in normalize_retrieval_text(text).split() if len(token) > 1]


def token_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize_retrieval_text(left))
    right_tokens = set(tokenize_retrieval_text(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def bm25_scores(
    query_tokens: list[str],
    documents: list[list[str]],
) -> list[float]:
    if not query_tokens:
        return [0.0 for _ in documents]
    try:
        from rank_bm25 import BM25Okapi

        return [float(score) for score in BM25Okapi(documents).get_scores(query_tokens)]
    except Exception:
        query_set = set(query_tokens)
        return [
            (
                len(query_set & set(document)) / len(query_set | set(document))
                if document
                else 0.0
            )
            for document in documents
        ]
