from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

from src.common.files import read_json, read_text, write_json_atomic
from src.cross_doc.clients import CanonicalStatementEmbedder
from src.cross_doc.pacing import CrossDocumentRequestPacer

from scripts.experiment.agent.claude_runner import (
    ClaudeCodeRunner,
    ClaudeRunnerSettings,
    claude_version,
    provider_metadata,
)
from scripts.experiment.agent.controller import AgentQueryController
from scripts.experiment.agent.mcp_server import (
    AgentMCPServer, MCP_SERVER_NAME, MCP_TOOL_NAME, mcp_config_template,
)
from scripts.experiment.agent.retrieval import (
    EVIDENCE_TOP_K,
    RERANKER_TOP_K,
    SIMILARITY_THRESHOLD,
    AgentRetrievalService,
)
from scripts.experiment.agent.schemas import final_output_schema
from scripts.experiment.agent.sessions import AgentSessionRegistry
from scripts.experiment.common import ExperimentRuntime, run_parallel_queries, sha256_json
from scripts.experiment.dataset import PROJECT_ROOT, sha256_file


METHOD_NAME = "agent"
DISPLAY_NAME = "Claude Code Agent"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "retrieval_agent.md"
SCHEMA_PATH = Path(__file__).resolve().parent / "final_output.schema.json"
DEFAULT_CLAUDE_SETTINGS = (
    PROJECT_ROOT / ".agent_runtime" / "claude_provider.json"
)
MAXIMUM_SEARCHES = 3
IMPLEMENTATION_DEPENDENCIES = (
    "scripts/experiment/agent/method.py",
    "scripts/experiment/agent/claude_runner.py",
    "scripts/experiment/agent/controller.py",
    "scripts/experiment/agent/mcp_server.py",
    "scripts/experiment/agent/retrieval.py",
    "scripts/experiment/agent/schemas.py",
    "scripts/experiment/agent/sessions.py",
    "scripts/experiment/agent/usage.py",
    "scripts/experiment/agent/final_output.schema.json",
    "prompts/retrieval_agent.md",
    "src/cross_doc/clients.py",
    "src/cross_doc/resolution.py",
)


def method_settings(
    dataset: Any,
    config: Any,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_options = dict(options or {})
    run_mode = str(active_options.get("run_mode", "formal")).strip().lower()
    if run_mode not in {"test", "formal"}:
        raise ValueError("Agent run_mode must be 'test' or 'formal'.")
    model_task = str(
        active_options.get("model_task", "agent")
    ).strip()
    if not model_task:
        raise ValueError("Agent model_task cannot be empty.")
    expected_model = str(config.model(model_task).name)
    model_config = config.model(model_task)
    provider_config = config.provider(model_config.provider, require_api_key=True)
    effort = str(active_options.get("effort", "medium")).strip().lower()
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Agent effort is invalid.")
    settings_path = Path(
        str(
            active_options.get(
                "provider_settings_path",
                active_options.get(
                    "claude_settings_path",
                    DEFAULT_CLAUDE_SETTINGS,
                ),
            )
        )
    ).expanduser().resolve()
    executable = _claude_executable(active_options.get("claude_executable"))
    metadata = provider_metadata(settings_path)
    reranker = config.model("reranker")
    embedding = config.model("embedding")
    timeout_seconds = _positive_float(
        active_options.get("timeout_seconds", 900.0),
        "timeout_seconds",
    )
    maximum_budget_usd = _positive_float(
        active_options.get("maximum_budget_usd", 1.0),
        "maximum_budget_usd",
    )
    maximum_attempts = _bounded_int(
        active_options.get("maximum_attempts", 3),
        "maximum_attempts",
        minimum=1,
        maximum=3,
    )
    workers = _bounded_int(
        active_options.get("workers", 5),
        "workers",
        minimum=1,
        maximum=5,
    )
    return {
        "algorithm_version": 1,
        "run_mode": run_mode,
        "agent_framework": "Claude Code",
        "claude_code_version": claude_version(executable),
        "claude_executable": executable,
        "expected_model": expected_model,
        "model_task": model_task,
        "model_aliases": {
            "opus": expected_model,
            "sonnet": expected_model,
            "haiku": expected_model,
        },
        "effort": effort,
        "provider": metadata["provider"],
        # The concrete temporary path is process-specific and must not enter
        # the experiment input signature. The runner receives it through
        # runtime.options at execution time.
        "credential_source": "temporary_claude_config",
        "temporary_claude_config": True,
        "bare_mode": True,
        "session_persistence": False,
        "allowed_tools": [f"mcp__{MCP_SERVER_NAME}__{MCP_TOOL_NAME}"],
        "retrieval": {
            "same_type_only": True,
            "evidence_top_k": EVIDENCE_TOP_K,
            "minimum_similarity": SIMILARITY_THRESHOLD,
            "reranker_top_k": RERANKER_TOP_K,
            "agent_visible_candidate_limit": RERANKER_TOP_K,
            "maximum_searches": MAXIMUM_SEARCHES,
            "maximum_reformulated_searches": MAXIMUM_SEARCHES - 1,
            "embedding_model": embedding.name,
            "embedding_provider": embedding.provider,
            "embedding_dimensions": embedding.dimensions,
            "reranker_model": reranker.name,
            "reranker_provider": reranker.provider,
        },
        "timeout_seconds": timeout_seconds,
        "maximum_budget_usd_per_query": maximum_budget_usd,
        "maximum_attempts": maximum_attempts,
        "workers": workers,
        "llm_pacing": {
            "enabled": config.get_bool("cross_document_pacing.enabled"),
            "work_minutes": config.get_float(
                "cross_document_pacing.work_minutes",
                minimum=0,
            ),
            "short_rest_minutes": config.get_float(
                "cross_document_pacing.short_rest_minutes",
                minimum=0,
            ),
            "windows_per_cycle": config.get_int(
                "cross_document_pacing.windows_per_cycle",
                minimum=1,
            ),
            "cycle_extra_rest_minutes": config.get_float(
                "cross_document_pacing.cycle_extra_rest_minutes",
                minimum=0,
            ),
        },
        "prompt_sha256": sha256_file(PROMPT_PATH),
        "output_schema_sha256": sha256_file(SCHEMA_PATH),
        "mcp_config_template_sha256": sha256_json(mcp_config_template()),
        "gold_fields_visible_to_agent": False,
        "selected_query_count": len(active_options.get("selected_query_ids", [])),
    }




def run(runtime: ExperimentRuntime) -> None:
    system_prompt = read_text(PROMPT_PATH)
    resolver = runtime.dataset.resolver(
        evidence_top_k=EVIDENCE_TOP_K,
        minimum_similarity=SIMILARITY_THRESHOLD,
        reranker_top_k=RERANKER_TOP_K,
    )
    retrieval = AgentRetrievalService(
        dataset=runtime.dataset,
        resolver=resolver,
        reranker_client=runtime.reranker_client(),
        query_embedder=CanonicalStatementEmbedder(
            config=runtime.config,
            logger=runtime.logger,
            usage_tracker=runtime.usage_tracker,
        ),
    )
    sessions = AgentSessionRegistry()
    mcp_server = AgentMCPServer(sessions=sessions)
    runner: ClaudeCodeRunner | None = None
    try:
        mcp_server.start()
        runner = ClaudeCodeRunner(
            settings=ClaudeRunnerSettings(
                executable=str(runtime.settings["claude_executable"]),
                provider_settings_path=_runtime_provider_settings_path(runtime),
                run_mode=str(runtime.settings["run_mode"]),
                expected_model=str(runtime.settings["expected_model"]),
                effort=str(runtime.settings["effort"]),
                timeout_seconds=float(runtime.settings["timeout_seconds"]),
                maximum_budget_usd=float(
                    runtime.settings["maximum_budget_usd_per_query"]
                ),
            ),
            system_prompt=system_prompt,
            output_schema=final_output_schema(),
            mcp_endpoint=mcp_server.endpoint,
            usage_tracker=runtime.usage_tracker,
        )
        controller = AgentQueryController(
            runtime=runtime,
            retrieval=retrieval,
            sessions=sessions,
            runner=runner,
            maximum_attempts=int(runtime.settings["maximum_attempts"]),
            maximum_searches=MAXIMUM_SEARCHES,
        )
        request_pacer = CrossDocumentRequestPacer.from_config(
            runtime.config,
            announce=lambda message: print(
                message.replace("[canonical-graph]", "[experiment:agent]")
            ),
        )
        pacing_state_path = runtime.store.checkpoint_dir / "llm_pacing_state.json"
        pacing_state_lock = threading.Lock()
        if pacing_state_path.is_file():
            request_pacer.restore_state(read_json(pacing_state_path))

        def save_pacing_state() -> None:
            with pacing_state_lock:
                write_json_atomic(
                    pacing_state_path,
                    request_pacer.snapshot_state(),
                )

        print(
            f"[experiment:agent] MCP ready at {mcp_server.endpoint}; "
            f"mode={runtime.settings['run_mode']}, "
            f"model={runtime.settings['expected_model']}, "
            f"workers={runtime.settings['workers']}, "
            f"pacing={runtime.settings['llm_pacing']['enabled']}"
        )
        try:
            def paced_process(
                query: dict[str, Any],
                existing: dict[str, Any] | None,
            ) -> None:
                with request_pacer.request_slot():
                    controller.process(query, existing)
                save_pacing_state()

            run_parallel_queries(
                runtime,
                paced_process,
                workers=int(runtime.settings["workers"]),
            )
        finally:
            save_pacing_state()
    finally:
        if runner is not None:
            runner.terminate_all()
        sessions.clear()
        mcp_server.stop()


def _claude_executable(value: Any) -> str:
    if value:
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Claude Code executable does not exist: {path}")
        return str(path)
    discovered = shutil.which("claude")
    if not discovered:
        raise FileNotFoundError("Claude Code executable was not found in PATH.")
    return str(Path(discovered).resolve())


def _runtime_provider_settings_path(runtime: ExperimentRuntime) -> Path:
    value = runtime.options.get("provider_settings_path")
    if not value:
        raise ValueError(
            "Agent runtime is missing its temporary provider settings path."
        )
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Claude provider settings do not exist: {path}")
    return path


def _bounded_int(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed
