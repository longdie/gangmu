"""Run the PDF pipeline through the evidence_graph stage."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.pipeline.runner import main as pipeline_main


def main():
    pipeline_main(default_stop_after="evidence_graph")


if __name__ == "__main__":
    main()
