"""Run the four reported ablations, without re-running a Gangmu control."""
from scripts.experiment.cli import configure, options, parser_for
from scripts.experiment.dataset import ExperimentDataset
from scripts.experiment.registry import rescore_directory, run_method
from scripts.ablations import method

VARIANTS = ("without_semantic_normalization", "center_only", "without_reranking", "binary_sev")

def main():
    parser = parser_for("Run the four released Gangmu ablations.", "Ablation_Studies")
    parser.add_argument("--variants", "--methods", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate ablation variants.")
    config = configure(args)
    dataset = ExperimentDataset.load(args.material_dir,config=config)
    if args.rescore:
        return rescore_directory(dataset=dataset,directory=args.rescore)
    results = {}
    for variant in args.variants:
        variant_options = {**options(args), "variant":variant}
        results[variant] = run_method(dataset=dataset, config=config, module=method, name=variant, output_root=args.output_dir, overwrite=args.overwrite, options=variant_options)
    return results

if __name__ == "__main__":
    from src.common.security import redact
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"{type(exc).__name__}: {redact(str(exc))}") from None
