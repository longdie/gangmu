from __future__ import annotations

from src.common.security import redact

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

from src.common.config import GangmuConfig
from src.common.files import utc_now
from src.cross_doc.resolution import (
    connected_components,
    judge_canonical_pair,
    judge_evidence_to_canonical,
)
from scripts.benchmark.schema import PREDICTION_SCHEMA_VERSION


@dataclass(frozen=True)
class BenchmarkModelSpec:
    label: str
    config_task: str
    workers_config_key: str

    def metadata(self, config: GangmuConfig) -> dict[str, Any]:
        model = config.model(self.config_task)
        metadata: dict[str, Any] = {
            "label": self.label,
            "config_task": self.config_task,
            "model": model.name,
            "provider": model.provider,
            "parallel_workers": config.get_int(
                self.workers_config_key,
                minimum=1,
                maximum=10,
            ),
        }
        return metadata


MODEL_SPECS = (
    BenchmarkModelSpec("gpt-5.6-terra", "sev", "experiment.parallel_workers"),
)


class QueryModelJudge:
    """Complete one model's prediction while preserving paid partial work."""

    def __init__(
        self,
        *,
        llm_client: Any,
        alignment_prompt: str,
        strict_cluster_prompt: str,
        alignment_judge: Callable[..., str] = judge_evidence_to_canonical,
        strict_judge: Callable[..., str] = judge_canonical_pair,
    ) -> None:
        self.llm_client = llm_client
        self.alignment_judge = alignment_judge
        self.strict_judge = strict_judge
        self.alignment_prompt = alignment_prompt
        self.strict_cluster_prompt = strict_cluster_prompt
        self.prompt_sha256 = {
            "alignment": hashlib.sha256(
                alignment_prompt.encode("utf-8")
            ).hexdigest(),
            "strict_cluster_pair": hashlib.sha256(
                strict_cluster_prompt.encode("utf-8")
            ).hexdigest(),
        }

    def complete(
        self,
        *,
        spec: BenchmarkModelSpec,
        model_metadata: dict[str, Any],
        candidate_record: dict[str, Any],
        existing: dict[str, Any] | None,
        save: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        query_id = int(candidate_record["query_id"])
        candidate_hash = str(candidate_record["candidate_snapshot_sha256"])
        shared_candidates = candidate_record["shared_candidates"]
        usage_task_prefix = str(candidate_record.get("usage_task_prefix", "")).strip()
        record = deepcopy(existing) if existing is not None else {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "status": "partial",
            "query_id": query_id,
            "source_id": candidate_record["source_id"],
            "model_label": spec.label,
            "model": model_metadata["model"],
            "provider": model_metadata["provider"],
            "config_task": spec.config_task,
            "prompt_sha256": deepcopy(self.prompt_sha256),
            "candidate_snapshot_sha256": candidate_hash,
            "candidate_judgments": [],
            "strict_cluster_comparisons": [],
            "query_prediction": None,
            "started_at": utc_now(),
            "updated_at": utc_now(),
        }
        if record.get("status") == "completed":
            return record
        record["status"] = "partial"
        record.pop("last_error", None)

        completed_candidates = {
            int(item["canonical_node_id"]): item
            for item in record["candidate_judgments"]
        }
        query_payload = candidate_record["query_payload"]
        try:
            for candidate in shared_candidates:
                canonical_node_id = int(candidate["canonical_node_id"])
                if canonical_node_id in completed_candidates:
                    continue
                relation = self.alignment_judge(
                    llm_client=self.llm_client,
                    model_task=spec.config_task,
                    prompt=self.alignment_prompt,
                    query_payload=query_payload,
                    candidate_payload=candidate["candidate_canonical_node"],
                    usage_task=(
                        f"{usage_task_prefix}.alignment"
                        if usage_task_prefix
                        else f"mather_alignment.{spec.label}"
                    ),
                )
                judgment = {
                    "canonical_node_id": canonical_node_id,
                    "relation": relation,
                    "model": str(model_metadata["model"]),
                    "provider": str(model_metadata["provider"]),
                }
                record["candidate_judgments"].append(judgment)
                completed_candidates[canonical_node_id] = judgment
                record["updated_at"] = utc_now()
                save(record)

            equivalent_ids = [
                int(candidate["canonical_node_id"])
                for candidate in shared_candidates
                if completed_candidates[int(candidate["canonical_node_id"])][
                    "relation"
                ]
                == "equivalent"
            ]
            possible_ids = [
                int(candidate["canonical_node_id"])
                for candidate in shared_candidates
                if completed_candidates[int(candidate["canonical_node_id"])][
                    "relation"
                ]
                == "possible_equivalent"
            ]
            if len(equivalent_ids) > 1:
                self._complete_strict_pairs(
                    spec=spec,
                    shared_candidates=shared_candidates,
                    equivalent_ids=equivalent_ids,
                    record=record,
                    save=save,
                    usage_task_prefix=usage_task_prefix,
                )

            record["query_prediction"] = self._query_prediction(
                equivalent_ids=equivalent_ids,
                possible_ids=possible_ids,
                strict_comparisons=record["strict_cluster_comparisons"],
                no_candidates=not shared_candidates,
            )
            record["status"] = "completed"
            record["completed_at"] = utc_now()
            record["updated_at"] = record["completed_at"]
            save(record)
            return record
        except BaseException as exc:
            record["status"] = "processing_error"
            record["last_error"] = {
                "type": type(exc).__name__,
                "message": redact(str(exc)),
                "recorded_at": utc_now(),
            }
            record["updated_at"] = utc_now()
            save(record)
            raise

    def _complete_strict_pairs(
        self,
        *,
        spec: BenchmarkModelSpec,
        shared_candidates: list[dict[str, Any]],
        equivalent_ids: list[int],
        record: dict[str, Any],
        save: Callable[[dict[str, Any]], None],
        usage_task_prefix: str,
    ) -> None:
        by_id = {
            int(candidate["canonical_node_id"]): candidate
            for candidate in shared_candidates
        }
        completed_pairs = {
            (int(item["canonical_node_a"]), int(item["canonical_node_b"]))
            for item in record["strict_cluster_comparisons"]
        }
        for node_a, node_b in combinations(sorted(equivalent_ids), 2):
            pair = (node_a, node_b)
            if pair in completed_pairs:
                continue
            payload_a = self._strict_payload(by_id[node_a])
            payload_b = self._strict_payload(by_id[node_b])
            relation = self.strict_judge(
                llm_client=self.llm_client,
                model_task=spec.config_task,
                prompt=self.strict_cluster_prompt,
                canonical_node_a=payload_a,
                canonical_node_b=payload_b,
                usage_task=(
                    f"{usage_task_prefix}.strict_cluster_pair"
                    if usage_task_prefix
                    else f"mather_strict_cluster_pair.{spec.label}"
                ),
            )
            record["strict_cluster_comparisons"].append(
                {
                    "canonical_node_a": node_a,
                    "canonical_node_b": node_b,
                    "relation": relation,
                    "model": str(record["model"]),
                    "provider": str(record["provider"]),
                }
            )
            completed_pairs.add(pair)
            record["updated_at"] = utc_now()
            save(record)

    @staticmethod
    def _strict_payload(candidate: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(candidate["candidate_canonical_node"])
        payload["representative_evidence_nodes"] = deepcopy(
            candidate["cluster_representative_evidence_nodes"]
        )
        return payload


    @staticmethod
    def _query_prediction(
        *,
        equivalent_ids: list[int],
        possible_ids: list[int],
        strict_comparisons: list[dict[str, Any]],
        no_candidates: bool,
    ) -> dict[str, Any]:
        """Resolve first-stage matches without mutating the frozen graph.

        A multi-match is accepted only when every initially equivalent
        candidate is pairwise strictly equivalent.  The top-level
        ``MULTIPLE_TARGETS`` value is retained for construction-time routing;
        ``resolved_prediction`` is the value used by experiment scoring.
        """
        if not equivalent_ids:
            resolved = {
                "label": "NIL",
                "canonical_node_id": None,
                "accepted_canonical_node_ids": [],
                "resolution_type": (
                    "no_candidate_after_embedding_threshold"
                    if no_candidates
                    else "no_equivalent_candidate"
                ),
            }
            return {
                "label": "NIL",
                "canonical_node_id": None,
                "accepted_canonical_node_ids": [],
                "possible_equivalent_canonical_node_ids": sorted(possible_ids),
                "decision_source": resolved["resolution_type"],
                "resolved_prediction": resolved,
            }
        if len(equivalent_ids) == 1:
            canonical_node_id = int(equivalent_ids[0])
            resolved = {
                "label": "MERGE",
                "canonical_node_id": canonical_node_id,
                "accepted_canonical_node_ids": [canonical_node_id],
                "resolution_type": "single_equivalent",
            }
            return {
                "label": "MERGE",
                "canonical_node_id": canonical_node_id,
                "accepted_canonical_node_ids": [canonical_node_id],
                "possible_equivalent_canonical_node_ids": sorted(possible_ids),
                "decision_source": "single_equivalent_candidate",
                "resolved_prediction": resolved,
            }

        equivalent_graph = {node_id: set() for node_id in equivalent_ids}
        possible_pairs: list[list[int]] = []
        for comparison in strict_comparisons:
            node_a = int(comparison["canonical_node_a"])
            node_b = int(comparison["canonical_node_b"])
            relation = str(comparison["relation"])
            if relation == "equivalent":
                equivalent_graph[node_a].add(node_b)
                equivalent_graph[node_b].add(node_a)
            elif relation == "possible_equivalent":
                possible_pairs.append([node_a, node_b])
        merge_groups: list[list[int]] = []
        for component in connected_components(equivalent_graph):
            ordered = sorted(component)
            if len(ordered) >= 2 and all(
                node_b in equivalent_graph[node_a]
                for index, node_a in enumerate(ordered)
                for node_b in ordered[index + 1 :]
            ):
                merge_groups.append(ordered)
        merge_groups.sort(key=lambda group: (group[0], len(group), group))
        all_equivalent_form_one_clique = (
            len(merge_groups) == 1
            and set(merge_groups[0]) == set(equivalent_ids)
        )
        if all_equivalent_form_one_clique:
            accepted_ids = list(merge_groups[0])
            resolved = {
                "label": "MERGE",
                "canonical_node_id": accepted_ids[0],
                "accepted_canonical_node_ids": accepted_ids,
                "resolution_type": "strict_equivalence_group",
            }
        else:
            if not merge_groups:
                resolution_type = "no_strictly_mergeable_group"
            elif len(merge_groups) == 1:
                resolution_type = "ambiguous_strict_verification"
            else:
                resolution_type = "multiple_strictly_mergeable_groups"
            resolved = {
                "label": "NIL",
                "canonical_node_id": None,
                "accepted_canonical_node_ids": [],
                "resolution_type": resolution_type,
            }
        return {
            "label": "MULTIPLE_TARGETS",
            "canonical_node_id": None,
            "accepted_canonical_node_ids": [],
            "equivalent_canonical_node_ids": sorted(equivalent_ids),
            "possible_equivalent_canonical_node_ids": sorted(possible_ids),
            "strictly_mergeable_canonical_groups": merge_groups,
            "possible_equivalence_pairs": sorted(possible_pairs),
            "decision_source": "multiple_equivalent_candidates",
            "resolved_prediction": resolved,
        }
