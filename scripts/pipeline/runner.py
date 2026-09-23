"""Run the existing Gangmu construction algorithms on a folder of PDFs.

Only orchestration and resume bookkeeping live here. Mathematical extraction,
normalization, verification, clustering and edge projection use src/ unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.common.config import GangmuConfig, load_config, set_config
from src.common.files import ProjectLogger, read_json, write_json_atomic
from src.common.paths import ROOT_DIR, PROMPT_DIR
from src.common.security import redact

STAGES = ("ocr", "single_doc", "evidence_graph", "canonical_graph")


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else ROOT_DIR / candidate).resolve()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_pdfs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"PDF input folder does not exist: {input_dir}")
    files = sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: (path.relative_to(input_dir).as_posix().casefold(), path.relative_to(input_dir).as_posix()),
    )
    if not files:
        raise ValueError(f"No PDF files found in {input_dir}")
    if any(not path.resolve().is_relative_to(input_dir.resolve()) for path in files):
        raise ValueError("PDF symlinks outside the supplied input folder are not supported.")
    return files


def document_id(pdf: Path, input_dir: Path) -> str:
    relative = pdf.relative_to(input_dir).as_posix()
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", pdf.stem).strip("_")[:60] or "document"
    return f"{slug}_{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:12]}"


def settings_digest(config: GangmuConfig, collection: str) -> str:
    # Connection secrets and endpoint addresses are deliberately excluded.
    sections = ("models", "ocr", "requests", "knowledge", "node_extraction", "dependency_extraction", "normalization", "embedding", "reranker", "cross_document", "single_document_pacing", "cross_document_pacing")
    settings = {section: config.get(section, {}) for section in sections}
    settings["collection"] = collection
    settings["prompts"] = {path.name: file_digest(path) for path in sorted(PROMPT_DIR.glob("*.md"))}
    encoded = json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_ocr_document(pdf: Path, output_root: Path, doc_id: str, config: GangmuConfig, collection: str) -> Path:
    from src.ocr import OCRRunConfig, PDFToMarkdownOCR

    ocr_dir = output_root / "OCRs" / doc_id
    write_json_atomic(ocr_dir / "source_metadata.json", {
        "source_id": doc_id, "title": pdf.stem,
        "source_type": "textbook" if collection == "textbooks" else "paper",
        "collection": collection,
    })
    run_config = OCRRunConfig.from_config(
        config, pdf_path=pdf, output_dir=ocr_dir.parent,
        document_id=doc_id, resume=True,
    )
    logger = ProjectLogger(output_root / "logs" / "ocr", run_name=doc_id)
    manifest = PDFToMarkdownOCR(run_config, logger=logger).run()
    usage = read_json(ocr_dir / "usage.json")
    failed = [row["page"] for row in usage.get("pages", []) if row.get("status") not in {"ok", "basic_check_failed_kept"}]
    if failed or "ocr_failure_markers" in manifest.get("quality_summary", {}).get("issues", []):
        raise RuntimeError(f"OCR did not complete for {doc_id}; failed pages: {failed}. Check the configured service and rerun the same command to resume.")
    return ocr_dir / f"{doc_id}.md"


def run_single_document(ocr_dir: Path, output_root: Path, config: GangmuConfig, collection: str) -> Path:
    from src.single_doc import SingleDocumentPipeline

    graph_dir = output_root / "single_doc_kgs" / ocr_dir.name
    pipeline = SingleDocumentPipeline(
        ocr_dir=ocr_dir, collection=collection, output_dir=graph_dir,
        config=config, logger=ProjectLogger(output_root / "logs" / "single_doc", run_name=ocr_dir.name),
    )
    graph = pipeline.run()
    if not graph.get("nodes"):
        raise RuntimeError(f"No mathematical nodes extracted from {ocr_dir.name}; inspect its extraction logs before continuing.")
    return graph_dir / f"{ocr_dir.name}.json"


def build_evidence_graph(graph_paths: list[Path], output_root: Path, config: GangmuConfig) -> dict[str, Any]:
    from src.evidence_graph import EvidenceGraphPipeline

    graph_dir = output_root / "evidence_graphs" / "documents"
    # The construction core accepts only newly imported sources. Excluding
    # existing sources also preserves their frozen numeric node IDs on resume.
    sources = read_json(graph_dir / "sources.json", default=[])
    existing_ids = {source["id"] for source in sources}
    pending = [path for path in graph_paths if read_json(path)["source"]["id"] not in existing_ids]
    if not pending and (graph_dir / "manifest.json").is_file():
        return read_json(graph_dir / "manifest.json")
    return EvidenceGraphPipeline(
        evidence_graph_dir=graph_dir, single_document_graph_paths=pending,
        config=config, logger=ProjectLogger(output_root / "logs" / "evidence_graph", run_name="documents"),
    ).run()


def build_canonical_graph(output_root: Path, config: GangmuConfig) -> dict[str, Any]:
    from src.cross_doc import CanonicalGraphPipeline

    return CanonicalGraphPipeline(
        evidence_graph_dir=output_root / "evidence_graphs" / "documents",
        canonical_graph_dir=output_root / "canonical_graphs" / "documents",
        config=config, logger=ProjectLogger(output_root / "logs" / "cross_document", run_name="documents"),
    ).run()


def _completed(record: dict[str, Any], stage: str, path: Path) -> bool:
    expected = record.get(stage)
    if not expected:
        return False
    if not path.is_file() or file_digest(path) != expected:
        raise ValueError(f"A completed {stage} artifact was removed or modified: {path}. Use a new --output-dir for a fresh construction.")
    return True


def run_pipeline(*, input_dir: str | Path, output_dir: str | Path, config: GangmuConfig, collection: str = "textbooks", stop_after: str = "canonical_graph") -> dict[str, Any]:
    input_root, output_root = resolve_path(input_dir), resolve_path(output_dir)
    if collection not in {"textbooks", "papers"} or stop_after not in STAGES:
        raise ValueError("Unsupported collection or pipeline stage.")
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise ValueError("The output directory must be outside the PDF input folder.")
    pdfs = discover_pdfs(input_root)
    signature = settings_digest(config, collection)
    state_path = output_root / "pipeline_state.json"
    if not state_path.exists() and any((output_root / name).exists() and any((output_root / name).iterdir()) for name in ("OCRs", "single_doc_kgs", "evidence_graphs", "canonical_graphs")):
        raise ValueError("This output folder has graph artifacts but no pipeline_state.json; choose a fresh --output-dir.")
    state = read_json(state_path, default={"schema_version": 1, "settings_sha256": signature, "documents": {}})
    if state.get("settings_sha256") != signature:
        raise ValueError("Models, settings, or prompts changed since this run; use a new --output-dir to avoid mixing results.")
    documents = state["documents"]
    requested_ids = {document_id(pdf, input_root) for pdf in pdfs}
    if not set(documents).issubset(requested_ids):
        raise ValueError("Previously processed PDFs were removed or renamed; use a new --output-dir for a different collection.")
    # Validate every PDF before starting any paid work.
    for pdf in pdfs:
        doc_id, fingerprint = document_id(pdf, input_root), file_digest(pdf)
        existing = documents.get(doc_id)
        if existing and existing["pdf_sha256"] != fingerprint:
            raise ValueError(f"Previously processed PDF changed: {pdf}. Use a new --output-dir.")
        documents.setdefault(doc_id, {"input_file": pdf.relative_to(input_root).as_posix(), "pdf_sha256": fingerprint})
    write_json_atomic(state_path, state)
    graph_paths = []
    for index, pdf in enumerate(pdfs, 1):
        doc_id = document_id(pdf, input_root)
        record = documents[doc_id]
        print(f"[pipeline] document {index}/{len(pdfs)}: {pdf.relative_to(input_root)}", flush=True)
        ocr_path = output_root / "OCRs" / doc_id / f"{doc_id}.md"
        if not _completed(record, "ocr", ocr_path):
            ocr_path = run_ocr_document(pdf, output_root, doc_id, config, collection)
            record["ocr"] = file_digest(ocr_path)
            write_json_atomic(state_path, state)
        if stop_after == "ocr":
            continue
        graph_path = output_root / "single_doc_kgs" / doc_id / f"{doc_id}.json"
        if not _completed(record, "single_doc", graph_path):
            graph_path = run_single_document(ocr_path.parent, output_root, config, collection)
            record["single_doc"] = file_digest(graph_path)
            write_json_atomic(state_path, state)
        graph_paths.append(graph_path)
    summaries = {"document_count": len(pdfs), "output_dir": str(output_root), "completed_through": stop_after}
    if STAGES.index(stop_after) >= STAGES.index("evidence_graph"):
        summaries["evidence_graph"] = build_evidence_graph(graph_paths, output_root, config)
    if stop_after == "canonical_graph":
        summaries["canonical_graph"] = build_canonical_graph(output_root, config)
    state["completed_through"] = stop_after
    write_json_atomic(state_path, state)
    return summaries


def main(argv: list[str] | None = None, *, default_stop_after: str = "canonical_graph") -> None:
    parser = argparse.ArgumentParser(description="Run Gangmu on a folder of PDFs; rerun the same command to resume.")
    parser.add_argument("--input-dir", default="inputs", help="PDF folder (searched recursively; default: inputs).")
    parser.add_argument("--output-dir", default="outputs", help="Construction output folder (default: outputs).")
    parser.add_argument("--config", default=None, help="Private TOML configuration path.")
    parser.add_argument("--document-type", choices=("textbooks", "papers"), default="textbooks")
    parser.add_argument("--stop-after", choices=STAGES, default=default_stop_after)
    args = parser.parse_args(argv)
    try:
        config = load_config(resolve_path(args.config) if args.config else None)
        set_config(config)
        summary = run_pipeline(input_dir=args.input_dir, output_dir=args.output_dir, config=config, collection=args.document_type, stop_after=args.stop_after)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Gangmu pipeline stopped: {redact(str(exc))}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Gangmu interrupted; checkpoints were preserved. Rerun the same command to resume.\n")
    print(f"[pipeline] completed {summary['document_count']} document(s) through {summary['completed_through']}: {summary['output_dir']}")
