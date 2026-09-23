"""Common command-line options for reviewer-facing experiment entrypoints."""
from __future__ import annotations
import argparse
from pathlib import Path
from src.common.config import load_config, set_config
from scripts.experiment.dataset import PROJECT_ROOT

def repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT/path).resolve()

def parser_for(description, output_directory):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=repo_path, default=None, help="Private TOML configuration; defaults to config.toml, then config.example.toml.")
    parser.add_argument("--material-dir", type=repo_path, default=PROJECT_ROOT/"Material", help="Local supplementary Material directory.")
    parser.add_argument("--output-dir", type=repo_path, default=PROJECT_ROOT/"experiments"/output_directory)
    parser.add_argument("--query-limit", type=int, help="Limit to the first N selected queries for a smoke test.")
    parser.add_argument("--query-ids", type=int, nargs="+", help="Select specific stable query IDs.")
    parser.add_argument("--workers", type=int, default=5, choices=range(1,6))
    parser.add_argument("--overwrite", action="store_true", help="Start a new run instead of resuming compatible checkpoints.")
    parser.add_argument("--rescore", type=repo_path, help="Recompute prediction-based metrics in this result directory without model calls.")
    return parser

def configure(args):
    config = load_config(args.config)
    set_config(config)
    return config

def options(args):
    return {"model_task":"sev", "workers":args.workers, "query_limit":args.query_limit, "query_ids":args.query_ids}
