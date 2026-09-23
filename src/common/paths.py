from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT_DIR / "inputs"
# Retained as an import alias for the lower-level document helpers.
DATA_DIR = INPUT_DIR
TEXTBOOK_DATA_DIR = DATA_DIR / "textbooks"
PAPER_DATA_DIR = DATA_DIR / "papers"

PROMPT_DIR = ROOT_DIR / "prompts"
OUTPUT_DIR = ROOT_DIR / "outputs"
OCR_OUTPUT_DIR = OUTPUT_DIR / "OCRs"
TEXTBOOK_OCR_OUTPUT_DIR = OCR_OUTPUT_DIR / "textbooks"
PAPER_OCR_OUTPUT_DIR = OCR_OUTPUT_DIR / "papers"

SINGLE_DOC_KG_OUTPUT_DIR = OUTPUT_DIR / "single_doc_kgs"
TEXTBOOK_SINGLE_DOC_KG_OUTPUT_DIR = SINGLE_DOC_KG_OUTPUT_DIR / "textbooks"
PAPER_SINGLE_DOC_KG_OUTPUT_DIR = SINGLE_DOC_KG_OUTPUT_DIR / "papers"

EVIDENCE_GRAPH_OUTPUT_DIR = OUTPUT_DIR / "evidence_graphs"
CANONICAL_GRAPH_OUTPUT_DIR = OUTPUT_DIR / "canonical_graphs"
