"""FastAPI backend.   Run:  uvicorn app.api:app --reload   then open http://127.0.0.1:8000/docs"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from app import generation
from ingest import catalogue, pipeline
from ingest.validate import InvalidFileError, validate_pdf_bytes
from retrieval import search

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("researchiq.api")
app = FastAPI(title="ResearchIQ API", description="Multimodal research assistant over Advanced-RAG papers")


@app.exception_handler(Exception)
async def safe_errors(request: Request, exc: Exception):
    """Log the full error server-side; never expose stack traces, paths or prompts to users."""
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "Something went wrong while processing the request."})


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    paper_ids: list[str] | None = None
    mode: str = Field("auto", pattern="^(auto|lookup|compare|figure|synthesis)$")


class CompareIn(BaseModel):
    paper_a: str
    paper_b: str
    aspect: str = Field("", max_length=300)


class SynthesisIn(BaseModel):
    topic: str = Field("", max_length=300)


def _check_ready():
    try:
        search.all_chunks()
    except search.IndexNotReady as e:
        raise HTTPException(409, str(e))


@app.get("/health")
def health():
    return {"status": "ok", "mock": config.MOCK_LLM, "model": config.LLM_MODEL}


@app.get("/papers")
def papers():
    return list(catalogue.load().values())


@app.post("/papers")
async def upload(files: list[UploadFile] = File(...)):
    saved, errors = [], []
    for f in files:
        name = re.sub(r"[^A-Za-z0-9_.\-]", "_", Path(f.filename or "upload.pdf").name)
        data = await f.read()
        try:
            validate_pdf_bytes(data, name)
        except InvalidFileError as e:
            errors.append({"file": name, "error": str(e)})
            continue
        dest = config.PAPERS_DIR / name
        dest.write_bytes(data)
        saved.append(dest)
    out = pipeline.run(saved) if saved else {"papers": [], "errors": [], "chunks": None}
    out["errors"] = errors + out["errors"]
    return out


@app.delete("/papers/{paper_id}")
def delete(paper_id: str):
    if paper_id not in catalogue.load():
        raise HTTPException(404, "Unknown paper id")
    pipeline.remove_paper(paper_id)
    pipeline.run([], rechunk_only=True)
    return {"deleted": paper_id}


@app.post("/ask")
def ask(body: AskIn):
    _check_ready()
    return generation.answer(body.question, paper_ids=body.paper_ids, mode=body.mode)


@app.post("/compare")
def compare(body: CompareIn):
    _check_ready()
    return generation.compare(body.paper_a, body.paper_b, body.aspect)


@app.post("/synthesis")
def synthesis(body: SynthesisIn):
    _check_ready()
    return generation.synthesize(body.topic)


@app.get("/figures/{chunk_id}")
def figure(chunk_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", chunk_id):
        raise HTTPException(400, "Bad id")
    path = (config.FIGURES_DIR / f"{chunk_id}.png").resolve()
    if config.FIGURES_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Figure not found")
    return FileResponse(path, media_type="image/png")
