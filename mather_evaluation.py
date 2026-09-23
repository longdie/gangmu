"""Run Gangmu and the six reported MathER baselines."""
from scripts.experiment.cli import configure, options, parser_for
from scripts.experiment.dataset import ExperimentDataset
from scripts.experiment.registry import METHOD_MODULES, rescore_directory, run_experiments

def main():
    parser = parser_for("Run Gangmu and six baselines on the released MathER-Alg benchmark.", "MathER_Evaluation")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHOD_MODULES), default=list(METHOD_MODULES))
    parser.add_argument("--claude-executable", help="Optional Claude Code executable for the agent baseline.")
    args = parser.parse_args()
    config = configure(args)
    if args.rescore:
        return rescore_directory(dataset=ExperimentDataset.load(args.material_dir,config=config),directory=args.rescore)
    per_method = {name:options(args) for name in args.methods}
    if "agent" in per_method:
        per_method["agent"]["model_task"] = "agent"
        if args.claude_executable:
            per_method["agent"]["claude_executable"] = args.claude_executable
    return run_experiments(benchmark_dir=args.material_dir, method_names=args.methods, output_dir=args.output_dir, config=config, overwrite=args.overwrite, method_options=per_method)

if __name__ == "__main__":
    from src.common.security import redact
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"{type(exc).__name__}: {redact(str(exc))}") from None
