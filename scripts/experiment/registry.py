"""Shared resumable runner and clean result export for released experiments."""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from importlib import import_module
from pathlib import Path
import json
import tempfile
from typing import Any

from src.common.config import GangmuConfig, get_config
from src.common.files import ProjectLogger, read_json, write_json_atomic
from scripts.experiment.common import ExperimentRuntime, ExperimentStore, PersistentTokenUsageTracker, sha256_json
from scripts.experiment.dataset import ExperimentDataset, PROJECT_ROOT, sha256_file
from scripts.experiment.metrics import evaluate_results
from scripts.experiment.threshold_optimization import optimize_records

METHOD_MODULES = {
    "always_nil": "scripts.experiment.always_nil",
    "identity_match": "scripts.experiment.identity_match",
    "token_jaccard": "scripts.experiment.token_jaccard",
    "bm25_llm": "scripts.experiment.bm25_llm",
    "embedding": "scripts.experiment.embedding",
    "agent": "scripts.experiment.agent",
    "gangmu": "scripts.experiment.gangmu",
}
METHOD_DIRECTORIES = {
    "always_nil": "Always_NIL", "identity_match": "Exact_String_Match",
    "token_jaccard": "Token_Jaccard", "bm25_llm": "BM25_SEV",
    "embedding": "Dense_Retrieval", "agent": "Claude_Code_Agent", "gangmu": "Gangmu",
    "without_semantic_normalization": "Without_Semantic_Normalization",
    "center_only": "With_Center_Only_Representation", "without_reranking": "Without_Reranking", "binary_sev": "With_Binary_SEV",
}

@contextmanager
def agent_settings(config):
    """Configure the declared transport without exposing upstream credentials."""
    model = config.model("agent")
    provider = config.provider(model.provider, require_api_key=True)
    with ExitStack() as stack:
        if provider.protocol == "openai_chat_completions":
            from scripts.experiment.agent.chat_bridge import ChatCompletionsBridge

            bridge = stack.enter_context(ChatCompletionsBridge(
                api_key=provider.api_key,
                base_url=provider.base_url,
                model=model.name,
                timeout_seconds=config.get_float("requests.timeout_seconds"),
                trust_env=provider.trust_env,
            ))
            base_url, auth_token = bridge.base_url, bridge.auth_token
        elif provider.protocol == "claude_messages":
            base_url = provider.base_url.rstrip("/")
            for suffix in ("/v1/messages", "/messages"):
                if base_url.endswith(suffix):
                    base_url = base_url[:-len(suffix)]
                    break
            auth_token = provider.api_key
        else:
            raise ValueError("Unsupported Agent connection protocol.")
        directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="gangmu-agent-"))
        path = Path(directory) / "connection.json"
        path.touch(mode=0o600)
        path.write_text(json.dumps({
            "env": {
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_AUTH_TOKEN": auth_token,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model.name,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model.name,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model.name,
            },
            "model": "sonnet",
            "effortLevel": "medium",
        }))
        yield path

def run_experiments(*, benchmark_dir, method_names, overwrite=False, output_dir=None, output_namespace=None, config=None, method_options=None):
    if not method_names or len(set(method_names)) != len(method_names) or any(n not in METHOD_MODULES for n in method_names):
        raise ValueError("Select unique registered MathER methods.")
    active_config = config or get_config()
    dataset = ExperimentDataset.load(benchmark_dir, config=active_config)
    root = Path(output_dir or PROJECT_ROOT/"experiments"/"MathER_Evaluation").resolve()
    if output_namespace:
        if Path(output_namespace).name != output_namespace:
            raise ValueError("Output namespace must be one directory name.")
        root /= output_namespace
    results = {}
    for name in method_names:
        options = dict((method_options or {}).get(name, {}))
        module = import_module(METHOD_MODULES[name])
        if name == "agent":
            with agent_settings(active_config) as path:
                options["provider_settings_path"] = str(path)
                options["model_task"] = "agent"
                results[name] = run_method(dataset=dataset, config=active_config, module=module, name=name, output_root=root, overwrite=overwrite, options=options)
        else:
            results[name] = run_method(dataset=dataset, config=active_config, module=module, name=name, output_root=root, overwrite=overwrite, options=options)
    return results

def run_method(*, dataset, config, module, name, output_root, overwrite=False, options=None):
    options = dict(options or {})
    output = (Path(output_root) / METHOD_DIRECTORIES[name]).resolve()
    benchmark_directory = getattr(dataset, "benchmark_dir", None)
    if benchmark_directory is not None:
        benchmark_directory = Path(benchmark_directory).resolve()
        protected_roots = [benchmark_directory]
        if benchmark_directory.parent.name.casefold() == "material":
            protected_roots.append(benchmark_directory.parent)
        if any(output == protected or output.is_relative_to(protected) for protected in protected_roots):
            raise ValueError("Experiment output must be outside the supplied Material directory.")
    queries = selected_queries(dataset.queries, options)
    options["selected_query_ids"] = [int(q["query_id"]) for q in queries]
    settings = module.method_settings(dataset, config, options)
    source_paths = [Path(__file__), Path(__file__).with_name("common.py"), Path(__file__).with_name("dataset.py"), Path(__file__).with_name("metrics.py"), Path(module.__file__), PROJECT_ROOT/"src/common/embeddings.py", PROJECT_ROOT/"src/common/config.py"]
    source_paths.extend(PROJECT_ROOT/p for p in getattr(module,"IMPLEMENTATION_DEPENDENCIES",()))
    source_paths.extend(sorted((PROJECT_ROOT / "prompts").glob("*.md")))
    signature = sha256_json({"queries_sha256":dataset.queries_sha256, "benchmark_sha256":dataset.benchmark_manifest_sha256, "method":name, "settings":settings, "selected_query_ids":options["selected_query_ids"], "source":{str(p.relative_to(PROJECT_ROOT)):sha256_file(p) for p in source_paths}})
    runtime_key = sha256_json({"output":str(output.resolve()), "method":name})[:20]
    checkpoint_root = PROJECT_ROOT/"outputs"/"experiment_runtime"/runtime_key
    store = ExperimentStore(output_dir=output, method_name=name, input_signature=signature, overwrite=overwrite, checkpoint_root=checkpoint_root)
    logger = ProjectLogger(config.log_dir("experiment"), run_name=name)
    runtime = ExperimentRuntime(dataset=dataset, queries=queries, config=config, method_name=name, display_name=getattr(module,"DISPLAY_NAME",name), output_dir=output, settings=settings, input_signature=signature, store=store, logger=logger, usage_tracker=PersistentTokenUsageTracker(store.usage_journal_path), options=options, started_at=store.started_at)
    print(f"[experiment:{name}] queries={len(queries)}, output={output}", flush=True)
    module.run(runtime)
    records = store.completed_records(queries)
    if name in {"token_jaccard", "embedding"}:
        predictions, metrics, analysis = optimize_records(queries=queries, records=records, method_name=name, canonical_node_count=len(dataset.canonical_nodes), default_threshold=float(settings["threshold"]))
        write_json_atomic(output/"threshold_analysis.json", analysis)
    else:
        predictions, metrics = evaluate_results(queries=queries, records=records, canonical_node_count=len(dataset.canonical_nodes))
    write_json_atomic(output/"predictions.json", predictions)
    write_json_atomic(output/"metrics.json", metrics)
    store.mark_complete()
    print(f"[experiment:{name}] accuracy={metrics['core']['benchmark_accuracy']:.6f}", flush=True)
    return metrics

def selected_queries(queries, options):
    selected = list(queries)
    if options.get("query_ids") is not None:
        ids = options["query_ids"]
        if not isinstance(ids,list) or not ids or any(isinstance(n,bool) or not isinstance(n,int) for n in ids) or len(set(ids)) != len(ids):
            raise ValueError("query_ids must be unique integers.")
        by_id = {int(q["query_id"]):q for q in queries}
        if any(n not in by_id for n in ids):
            raise ValueError("Unknown query ID.")
        selected = [by_id[n] for n in ids]
    limit = options.get("query_limit")
    if limit is not None:
        if isinstance(limit,bool) or not isinstance(limit,int) or limit < 1:
            raise ValueError("query_limit must be positive.")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No selected queries.")
    return selected

def rescore_directory(*, dataset, directory):
    """Recompute metrics that depend on predictions; candidate recall needs traces."""
    directory = Path(directory).resolve()
    files = [directory] if directory.is_file() else sorted(directory.rglob("predictions.json"))
    if not files:
        raise FileNotFoundError("No predictions.json found.")
    by_id = {int(q["query_id"]):q for q in dataset.queries}
    result = {}
    for path in files:
        records = read_json(path)
        queries = [by_id[int(r["query_id"])] for r in records]
        _, metrics = evaluate_results(queries=queries, records=records, canonical_node_count=len(dataset.canonical_nodes))
        result[str(path)] = metrics
        print(f"{path.parent.name}: accuracy={metrics['core']['benchmark_accuracy']:.6f}")
    return result
