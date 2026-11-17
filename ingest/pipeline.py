"""Ingestion orchestrator.

    python -m ingest.pipeline                 # ingest every PDF in data/papers/
    python -m ingest.pipeline --file x.pdf    # ingest one file
    python -m ingest.pipeline --no-describe   # skip Gemini figure cards (faster, cheaper)
    python -m ingest.pipeline --rechunk       # re-chunk + re-index without re-parsing

Stages:  PDF -> pages.jsonl + figures.jsonl (+ PNGs) -> chunks.jsonl -> index
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import config
from ingest import catalogue, chunker, describe, figures, jsonl, parse
from ingest.validate import InvalidFileError, validate_pdf_path

log = logging.getLogger("researchiq.ingest")


def ingest_pdf(path: Path, describe_figures: bool = True) -> dict:
    path = Path(path)
    validate_pdf_path(path)
    pid = catalogue.paper_id_for(path.name)
    if path.parent.resolve() != config.PAPERS_DIR.resolve():
        shutil.copy(path, config.PAPERS_DIR / path.name)

    pages, guessed_title = parse.parse_pdf(path, pid)
    figs = figures.extract_figures(path, pid)
    if describe_figures:
        for f in figs:
            describe.describe(f)
    entry = catalogue.upsert(pid, title=guessed_title, filename=path.name, pages=len(pages))
    jsonl.replace_paper(config.PAGES_PATH, pid, pages)
    jsonl.replace_paper(config.FIGURES_PATH, pid, figs)
    return {
        "paper_id": pid,
        "title": entry.get("title", pid),
        "pages": len(pages),
        "scanned_pages": sum(p["is_scanned"] for p in pages),
        "figures": sum(f["type"] == "figure" for f in figs),
        "tables": sum(f["type"] == "table" for f in figs),
    }


def build_chunks() -> list[dict]:
    pages, figs = jsonl.read(config.PAGES_PATH), jsonl.read(config.FIGURES_PATH)
    cat = catalogue.load()
    chunks = []
    for pid in sorted({p["paper_id"] for p in pages}):
        title = cat.get(pid, {}).get("title", pid)
        chunks += chunker.chunk_text([p for p in pages if p["paper_id"] == pid], pid, title)
        chunks += [chunker.figure_chunk(f, title) for f in figs if f["paper_id"] == pid]
    jsonl.write(config.CHUNKS_PATH, chunks)
    log.info("Wrote %d chunks", len(chunks))
    return chunks


def remove_paper(pid: str) -> None:
    for p in (config.PAGES_PATH, config.FIGURES_PATH):
        jsonl.replace_paper(p, pid, [])
    catalogue.remove(pid)
    for png in config.FIGURES_DIR.glob(f"{pid}_*.png"):
        png.unlink()


def run(paths: list[Path], describe_figures: bool = True, rechunk_only: bool = False) -> dict:
    from retrieval import index

    results, errors = [], []
    if not rechunk_only:
        for p in paths:
            try:
                results.append(ingest_pdf(p, describe_figures))
                log.info("Ingested %s", p.name)
            except InvalidFileError as e:
                errors.append({"file": p.name, "error": str(e)})
            except Exception as e:  # one bad paper must not stop the batch
                log.exception("Failed on %s", p.name)
                errors.append({"file": p.name, "error": f"Processing failed ({type(e).__name__})"})
    chunks = build_chunks()
    index.build(chunks)
    return {"papers": results, "errors": errors, "chunks": len(chunks)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path)
    ap.add_argument("--no-describe", action="store_true")
    ap.add_argument("--rechunk", action="store_true")
    a = ap.parse_args()
    config.check_keys()
    paths = [a.file] if a.file else sorted(config.PAPERS_DIR.glob("*.pdf"))
    if not paths and not a.rechunk:
        print(f"No PDFs found in {config.PAPERS_DIR}. Run: python scripts/download_papers.py")
        return
    out = run(paths, describe_figures=not a.no_describe, rechunk_only=a.rechunk)
    for r in out["papers"]:
        print(f"  {r['paper_id']:<28} pages={r['pages']:<3} figures={r['figures']:<3} tables={r['tables']:<3} scanned={r['scanned_pages']}")
    for e in out["errors"]:
        print(f"  ERROR {e['file']}: {e['error']}")
    print(f"Indexed {out['chunks']} chunks.")


if __name__ == "__main__":
    main()
