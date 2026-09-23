"""Binary SEV strategies for the relation-space ablation."""
from typing import Any
import json
from src.common.files import read_text
from scripts.experiment.dataset import PROJECT_ROOT
BINARY_REPAIR_PROMPT = PROJECT_ROOT / "prompts" / "binary_relation_output.md"
from scripts.benchmark.model_judging import QueryModelJudge
BINARY_RELATIONS = {"equivalent", "not_equivalent"}

def _validate_binary_relation(result: dict[str, Any]) -> tuple[bool, str]:
    if (
        not isinstance(result, dict)
        or set(result) != {"relation"}
        or result.get("relation") not in BINARY_RELATIONS
    ):
        return False, "Output must contain exactly one binary relation."
    return True, ""

def _binary_alignment(
    *,
    llm_client: Any,
    model_task: str,
    prompt: str,
    query_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    usage_task: str,
) -> str:
    result = llm_client.call_json(
        model_task=model_task,
        prompt=prompt,
        input_text=json.dumps(
            {
                "current_evidence_node": query_payload,
                "candidate_canonical_node": candidate_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        usage_task=usage_task,
        validator=_validate_binary_relation,
        preserve_latex_escapes=True,
        repair_instructions=read_text(BINARY_REPAIR_PROMPT),
    )
    if result is None:
        raise RuntimeError("Binary semantic alignment returned no valid result.")
    return str(result["relation"])

def _binary_strict_pair(
    *,
    llm_client: Any,
    model_task: str,
    prompt: str,
    canonical_node_a: dict[str, Any],
    canonical_node_b: dict[str, Any],
    usage_task: str,
) -> str:
    result = llm_client.call_json(
        model_task=model_task,
        prompt=prompt,
        input_text=json.dumps(
            {
                "canonical_node_a": canonical_node_a,
                "canonical_node_b": canonical_node_b,
            },
            ensure_ascii=False,
            indent=2,
        ),
        usage_task=usage_task,
        validator=_validate_binary_relation,
        preserve_latex_escapes=True,
        repair_instructions=read_text(BINARY_REPAIR_PROMPT),
    )
    if result is None:
        raise RuntimeError("Binary strict comparison returned no valid result.")
    return str(result["relation"])

def _without_possible_keys(value: Any) -> Any:
    """Remove production-only possible-equivalence fields from binary traces."""

    if isinstance(value, dict):
        return {
            str(key): _without_possible_keys(item)
            for key, item in value.items()
            if "possible" not in str(key).lower()
        }
    if isinstance(value, list):
        return [_without_possible_keys(item) for item in value]
    return value

class BinaryQueryModelJudge(QueryModelJudge):
    """Production judge with a binary relation-space result schema."""

    @staticmethod
    def _query_prediction(
        *,
        equivalent_ids: list[int],
        possible_ids: list[int],
        strict_comparisons: list[dict[str, Any]],
        no_candidates: bool,
    ) -> dict[str, Any]:
        result = QueryModelJudge._query_prediction(
            equivalent_ids=equivalent_ids,
            possible_ids=[],
            strict_comparisons=strict_comparisons,
            no_candidates=no_candidates,
        )
        return _without_possible_keys(result)
