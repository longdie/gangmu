from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.files import read_json


SCHEMA_PATH = Path(__file__).resolve().parent / "final_output.schema.json"
QUERY_FIELDS = (
    "query_id",
    "type",
    "title",
    "field",
    "subfield",
    "normalized_statement",
)


def final_output_schema() -> dict[str, Any]:
    schema = read_json(SCHEMA_PATH)
    if not isinstance(schema, dict):
        raise ValueError("Agent final output schema must be a JSON object.")
    return schema


def final_output_schema_argument() -> str:
    return json.dumps(
        final_output_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sanitized_query(query: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "query_id": int(query["query_id"]),
        "type": str(query["type"]),
        "title": str(query["title"]),
        "field": str(query.get("field", "")),
        "subfield": str(query.get("subfield", "")),
        "normalized_statement": str(query["normalized_statement"]),
    }
    if set(payload) != set(QUERY_FIELDS):
        raise AssertionError("Agent query field whitelist changed unexpectedly.")
    if payload["type"] not in {"definition", "theorem"}:
        raise ValueError("Agent query type is invalid.")
    if not payload["normalized_statement"].strip():
        raise ValueError("Agent query normalized statement is empty.")
    return payload


def validate_prediction(
    value: Any,
    *,
    candidate_ids: set[int],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Agent final output must be a JSON object.")
    if set(value) != {"prediction", "canonical_node_id", "brief_reason"}:
        raise ValueError("Agent final output contains invalid keys.")
    prediction = value.get("prediction")
    if prediction not in {"merge", "nil"}:
        raise ValueError("Agent prediction must be 'merge' or 'nil'.")
    reason = value.get("brief_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Agent brief_reason must be a non-empty string.")
    if len(reason) > 2000:
        raise ValueError("Agent brief_reason exceeds 2000 characters.")
    raw_id = value.get("canonical_node_id")
    if prediction == "nil":
        if raw_id is not None:
            raise ValueError("A nil prediction must have canonical_node_id=null.")
        canonical_node_id = None
    else:
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise ValueError("A merge prediction requires an integer Canonical ID.")
        canonical_node_id = int(raw_id)
        if canonical_node_id not in candidate_ids:
            raise ValueError(
                "The predicted Canonical Node did not appear in any search result."
            )
    return {
        "prediction": str(prediction),
        "canonical_node_id": canonical_node_id,
        "brief_reason": reason.strip(),
    }
