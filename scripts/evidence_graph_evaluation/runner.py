"""Run the released, fixed Evidence Graph correctness evaluation.

Only the frozen ``samples.json`` input objects are sent to the judge. The
human labels are used after inference, for agreement evaluation alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from src.common.config import GangmuConfig, load_config
from src.common.files import read_json, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_SUBDIR = Path("Experiments/Evidence_Graph_Correctness_Evaluation")
MODEL_TASK = "evidence_precision_judge"
PAPER_MODEL = "claude-opus-5"
KINDS = ("nodes", "edges")
LABELS = ("CORRECT", "INCORRECT")


def rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def id_key(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError("Sample kind must be nodes or edges.")
    return "node_id" if kind == "nodes" else "edge_id"


def indexed(records: Any, kind: str, *, description: str) -> dict[int, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"{description} must be a list.")
    key = id_key(kind)
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record in {description}.")
        item_id = record.get(key)
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 0:
            raise ValueError(f"Invalid {key} in {description}.")
        if item_id in result:
            raise ValueError(f"Duplicate {key} {item_id} in {description}.")
        result[item_id] = record
    return result


def load_fixed_samples(
    material_dir: Path, kind: str, *, expected_count: int = 500,
    expected_human_count: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    directory = material_dir / EVALUATION_SUBDIR / kind
    samples = read_json(directory / "samples.json")
    by_id = indexed(samples, kind, description="fixed samples")
    if len(by_id) != expected_count:
        raise ValueError(f"Expected {expected_count} fixed {kind} samples, got {len(by_id)}.")
    key = id_key(kind)
    for item_id, sample in by_id.items():
        payload = sample.get("input")
        if not isinstance(payload, dict):
            raise ValueError(f"Sample {item_id} has no fixed input object.")
        if kind == "nodes":
            valid = set(payload) == {"evidence_node"}
            node = payload.get("evidence_node")
            valid = valid and isinstance(node, dict) and node.get("id") == item_id
        else:
            valid = set(payload) == {"source_node", "target_node", "evidence_edge"}
            source, target, edge = (payload.get(name) for name in
                                    ("source_node", "target_node", "evidence_edge"))
            valid = valid and all(isinstance(item, dict) for item in (source, target, edge))
            if valid:
                valid = (edge.get("id") == item_id and edge.get("source") == source.get("id")
                         and edge.get("target") == target.get("id"))
        if not valid:
            raise ValueError(f"Sample {item_id} has an invalid or mismatched fixed input.")

    human_samples = read_json(directory / "human_review_samples.json")
    human_samples_by_id = indexed(human_samples, kind, description="human review samples")
    human_document = read_json(directory / "human_review_results.json")
    if not isinstance(human_document, dict):
        raise ValueError("Human review results must be an object.")
    human_records = human_document.get("results")
    humans = indexed(human_records, kind, description="human review labels")
    if len(humans) != expected_human_count or set(humans) != set(human_samples_by_id):
        raise ValueError("Human review sample and label IDs/counts do not match.")
    for item_id, record in humans.items():
        if item_id not in by_id:
            raise ValueError(f"Human review {key} {item_id} is outside the fixed samples.")
        if human_samples_by_id[item_id].get("input") != by_id[item_id]["input"]:
            raise ValueError(f"Human review input {item_id} differs from the frozen sample.")
        if record.get("human_final_decision") not in {"correct", "incorrect"}:
            raise ValueError(f"Invalid human final decision for {key} {item_id}.")
    return samples, human_records


def validate_judge_output(value: Any) -> str:
    if not isinstance(value, dict) or set(value) != {"label"}:
        raise ValueError("Judge output must contain exactly the label key.")
    label = value.get("label")
    if label not in LABELS:
        raise ValueError("Judge label must be CORRECT or INCORRECT.")
    return label


def aggregate(
    samples: list[dict[str, Any]], human_records: list[dict[str, Any]],
    results: list[dict[str, Any]], kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recalculate correctness and agreement without trusting stored summaries."""
    key = id_key(kind)
    by_id = indexed(samples, kind, description="evaluation samples")
    judgments = indexed(results, kind, description="judge results")
    if set(judgments) != set(by_id):
        raise ValueError("Judge result IDs must exactly match the evaluated sample IDs.")
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(LABELS, 0))
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(LABELS, 0))
    correct_count = 0
    for item_id, result in judgments.items():
        label = result.get("label")
        if label not in LABELS:
            raise ValueError(f"Missing or invalid judgment for {key} {item_id}.")
        sample = by_id[item_id]
        by_type[str(sample.get("type", "unknown"))][label] += 1
        by_source[str(sample.get("source_id", "unknown"))][label] += 1
        correct_count += label == "CORRECT"
    summary = {
        "sample_count": len(samples), "correct_count": correct_count,
        "incorrect_count": len(samples) - correct_count,
        "correctness": correct_count / len(samples) if samples else None,
        "by_type": dict(by_type), "by_source": dict(by_source),
    }
    humans = indexed(human_records, kind, description="human review labels")
    reviewed = []
    human_counts = dict.fromkeys(LABELS, 0)
    for item_id, human in humans.items():
        if item_id not in judgments:
            continue  # A bounded smoke run may contain only part of the frozen subset.
        decision = human.get("human_final_decision")
        if decision not in {"correct", "incorrect"}:
            raise ValueError(f"Invalid human final decision for {key} {item_id}.")
        model_decision = judgments[item_id]["label"].lower()
        reviewed.append({key: item_id, "model_decision": model_decision,
                         "human_final_decision": decision,
                         "agreement": model_decision == decision})
        human_counts[decision.upper()] += 1
    agreement = sum(record["agreement"] for record in reviewed)
    review = {
        "summary": {
            "sample_count": len(reviewed), "agreement_count": agreement,
            "disagreement_count": len(reviewed) - agreement,
            "agreement_rate": agreement / len(reviewed) if reviewed else None,
            "human_final_label_counts": human_counts,
        },
        "results": reviewed,
    }
    return summary, review


def run_judgments(
    samples: list[dict[str, Any]], kind: str, *, prompt: str, model: str,
    cache_dir: Path, call: Callable[[str, dict[str, Any]], str], workers: int = 5,
    parse_attempts: int = 3, request_settings: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Resume only judgments from identical inputs, prompt, model, and settings."""
    from src.single_doc.llm import parse_json_lenient

    if workers < 1 or parse_attempts < 1:
        raise ValueError("Workers and parse attempts must be positive.")
    key = id_key(kind)
    indexed(samples, kind, description="evaluation samples")
    identity = json_digest({"kind": kind, "model": model, "prompt": prompt,
                            "samples": [{key: sample[key], "input": sample["input"]}
                                        for sample in samples],
                            "request_settings": request_settings or {}})
    directory = cache_dir / kind / identity
    directory.mkdir(parents=True, exist_ok=True)

    def one(sample: dict[str, Any]) -> dict[str, Any]:
        item_id = sample[key]
        path = directory / f"{item_id}.json"
        if path.exists():
            cached = read_json(path)
            if (not isinstance(cached, dict) or cached.get("identity") != identity
                    or cached.get(key) != item_id or cached.get("label") not in LABELS):
                raise ValueError(f"Invalid checkpoint for {kind} {item_id}.")
            return {key: item_id, "label": cached["label"]}
        for attempt in range(parse_attempts):
            try:
                raw = call(prompt, sample["input"])
            except Exception as exc:
                # The underlying HTTP error may include credentials or a private URL.
                raise RuntimeError(
                    f"Judge request failed for {kind} {item_id} ({type(exc).__name__}). "
                    "Completed judgments remain cached; rerun to resume."
                ) from None
            try:
                label = validate_judge_output(parse_json_lenient(raw))
                break
            except (ValueError, TypeError):
                if attempt + 1 == parse_attempts:
                    raise RuntimeError(
                        f"Judge returned invalid labels for {kind} {item_id}; "
                        "no judgment was recorded."
                    ) from None
        result = {key: item_id, "label": label}
        write_json_atomic(path, {"identity": identity, **result})
        return result

    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, sample): sample[key] for sample in samples}
        try:
            for future in as_completed(futures):
                result = future.result()
                results[result[key]] = result
                if progress and (len(results) % 25 == 0 or len(results) == len(samples)):
                    progress(f"{kind}: {len(results)}/{len(samples)} judgments complete")
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [results[sample[key]] for sample in samples]


def save_results(
    output_dir: Path, kind: str, samples: list[dict[str, Any]],
    human_records: list[dict[str, Any]], results: list[dict[str, Any]],
    *, model: str, prompt_sha256: str,
) -> dict[str, Any]:
    summary, review = aggregate(samples, human_records, results, kind)
    write_json_atomic(output_dir / kind / "claude_results.json", {
        "model": model, "prompt_sha256": prompt_sha256,
        "summary": summary, "results": results,
    })
    write_json_atomic(output_dir / kind / "human_review_results.json", review)
    return {"correctness": summary["correctness"],
            "human_agreement": review["summary"]["agreement_rate"],
            "sample_count": summary["sample_count"],
            "human_sample_count": review["summary"]["sample_count"]}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(
        "Evaluate the released fixed 500 node and 500 edge correctness samples. "
        "No sampling or benchmark reconstruction is performed."
    ))
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--material-dir", default="Material")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default="tmp/evidence_graph_evaluation")
    parser.add_argument("--kind", choices=("both", *KINDS), default="both")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test the first N frozen samples per kind; requires --output-dir.")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute all metrics from saved labels without any API calls.")
    parser.add_argument("--judge-results-dir", default=None,
                        help="With --recompute, read labels from this directory's nodes/edges folders.")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if args.limit is not None and (not 1 <= args.limit <= 500 or args.output_dir is None):
        raise ValueError("--limit must be between 1 and 500 and requires an explicit --output-dir.")
    if args.judge_results_dir and not args.recompute:
        raise ValueError("--judge-results-dir requires --recompute.")
    material_dir = rooted(args.material_dir)
    output_dir = rooted(args.output_dir or "experiments/Evidence_Graph_Correctness_Evaluation")
    if output_dir == material_dir or material_dir in output_dir.parents:
        raise ValueError("The output directory must be outside the frozen Material directory.")
    cache_dir = rooted(args.cache_dir)
    if cache_dir == material_dir or material_dir in cache_dir.parents:
        raise ValueError("The cache directory must be outside the frozen Material directory.")
    kinds = KINDS if args.kind == "both" else (args.kind,)
    # Validate both input collections before starting any potentially costly API calls.
    datasets = {kind: load_fixed_samples(material_dir, kind) for kind in kinds}
    config: GangmuConfig | None = None
    client = None
    if not args.recompute:
        from src.common.config import set_config
        from src.single_doc.llm import LLMClient
        config = load_config(rooted(args.config))
        set_config(config)
        model = config.model(MODEL_TASK)
        if model.name != PAPER_MODEL:
            raise ValueError(f"This paper evaluation requires model {PAPER_MODEL}.")
        config.provider(model.provider, require_api_key=True)
        client = LLMClient(config=config)

    report = {}
    for kind, (all_samples, humans) in datasets.items():
        samples = all_samples if args.limit is None else all_samples[:args.limit]
        stem = "node" if kind == "nodes" else "edge"
        prompt_path = ROOT / "prompts" / f"evidence_{stem}_precision_judge.md"
        prompt = prompt_path.read_text(encoding="utf-8")
        prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        if args.recompute:
            source_dir = (rooted(args.judge_results_dir) if args.judge_results_dir
                          else material_dir / EVALUATION_SUBDIR)
            document = read_json(source_dir / kind / "claude_results.json")
            if not isinstance(document, dict):
                raise ValueError("Saved judge results must be an object.")
            if document.get("prompt_sha256") != prompt_sha256:
                raise ValueError("Saved judge results use a different prompt; refusing to relabel provenance.")
            if document.get("model") != PAPER_MODEL:
                raise ValueError("Saved judge results do not use the paper's Claude judge model.")
            records = indexed(document.get("results"), kind, description="saved judge results")
            all_ids = set(indexed(all_samples, kind, description="fixed samples"))
            selected_ids = {sample[id_key(kind)] for sample in samples}
            if not selected_ids <= set(records) or not set(records) <= all_ids:
                raise ValueError("Saved judge result IDs do not match the frozen samples.")
            results = [{id_key(kind): sample[id_key(kind)],
                        "label": records[sample[id_key(kind)]].get("label")} for sample in samples]
        else:
            assert config is not None and client is not None
            model_config = config.model(MODEL_TASK)
            provider = config.provider(model_config.provider)

            def call(template: str, payload: dict[str, Any]) -> str:
                return client.call(
                    model_task=MODEL_TASK, prompt=template,
                    input_text=json.dumps(payload, ensure_ascii=False, indent=2),
                    usage_task=f"evidence_{stem}_correctness", json_mode=True,
                )

            results = run_judgments(
                samples, kind, prompt=prompt, model=PAPER_MODEL, cache_dir=cache_dir,
                call=call, workers=args.workers,
                request_settings={"protocol": provider.protocol,
                                  "max_completion_tokens": client.max_completion_tokens},
                progress=print,
            )
        report[kind] = save_results(output_dir, kind, samples, humans, results,
                                    model=PAPER_MODEL, prompt_sha256=prompt_sha256)
        print(f"{kind}: {json.dumps(report[kind])}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        from src.common.security import redact
        print(f"Evaluation failed ({type(exc).__name__}): {redact(str(exc))}")
        return 1
    return 0
