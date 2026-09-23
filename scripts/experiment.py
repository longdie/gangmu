"""Compatibility entrypoint; prefer the repository-root mather_evaluation.py."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from mather_evaluation import main
if __name__ == "__main__":
    from src.common.security import redact
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"{type(exc).__name__}: {redact(str(exc))}") from None
