"""Build the two indexes from chunks.jsonl: Chroma (dense vectors) + BM25 (keywords)."""
from __future__ import annotations

import json
import logging
import pickle
import re

import config
import llm_client

log = logging.getLogger("researchiq.index")

CHROMA_DIR = config.INDEX_DIR / "chroma"
BM25_PATH = config.INDEX_DIR / "bm25.pkl"
META_PATH = config.INDEX_DIR / "index_meta.json"
COLLECTION = "chunks"

STOP = set("a an the of and or to in on for with by is are was were be been this that these those it its as at from "
           "which what how why who whom does do did can could should would we our their they".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9@\-\.]+", text.lower()) if t not in STOP and len(t) > 1]


def _metadata(c: dict) -> dict:
    keys = ("paper_id", "paper_title", "type", "section", "page_start", "page_end", "figure_no", "image_path")
    md = {k: c[k] for k in keys if c.get(k) is not None}
    md["flagged"] = bool(c.get("injection_flags"))
    return md


def build(chunks: list[dict]) -> None:
    import chromadb
    from rank_bm25 import BM25Okapi

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if not chunks:
        log.warning("No chunks to index")
    else:
        vectors = llm_client.embed([c["text"] for c in chunks], task="RETRIEVAL_DOCUMENT")
        for i in range(0, len(chunks), 500):
            part = chunks[i : i + 500]
            col.add(
                ids=[c["chunk_id"] for c in part],
                embeddings=vectors[i : i + 500],
                documents=[c["text"] for c in part],
                metadatas=[_metadata(c) for c in part],
            )
    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "ids": [c["chunk_id"] for c in chunks],
            "bm25": BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None,
            "chunks": {c["chunk_id"]: c for c in chunks},
        }, f)
    META_PATH.write_text(json.dumps({"embed_model": "mock" if config.MOCK_LLM else config.EMBED_MODEL,
                                     "dim": config.EMBED_DIM, "chunks": len(chunks)}))
    from retrieval import search
    search.reset_cache()
    log.info("Indexed %d chunks", len(chunks))
