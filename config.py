"""Central configuration. Every setting comes from environment variables / .env.

No API key is ever hard-coded. Copy `.env.example` to `.env` and fill in the keys.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# ---- API keys (filled in by the user in .env) ----
GEMINI_API_KEY = _get("GEMINI_API_KEY")
OPENAI_API_KEY = _get("OPENAI_API_KEY")

# ---- Models ----
LLM_PROVIDER = _get("LLM_PROVIDER", "gemini")
LLM_MODEL = _get("LLM_MODEL", "gemini-3.8-flash")
VISION_MODEL = _get("VISION_MODEL", LLM_MODEL)
EMBED_MODEL = _get("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(_get("EMBED_DIM", "768"))
RERANKER = _get("RERANKER", "bge").lower()
RERANKER_MODEL = _get("RERANKER_MODEL", "BAAI/bge-reranker-base")

# ---- Retrieval ----
CHUNK_TOKENS = int(_get("CHUNK_TOKENS", "800"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "100"))
DENSE_TOP_K = int(_get("DENSE_TOP_K", "20"))
BM25_TOP_K = int(_get("BM25_TOP_K", "20"))
FINAL_TOP_K = int(_get("FINAL_TOP_K", "8"))
EVIDENCE_THRESHOLD = float(_get("EVIDENCE_THRESHOLD", "0.05"))          # reranker score (0-1)
EVIDENCE_THRESHOLD_DENSE = float(_get("EVIDENCE_THRESHOLD_DENSE", "0.45"))  # cosine, used when no reranker

# ---- Mock mode: run without any API key ----
MOCK_LLM = _get("MOCK_LLM", "0") == "1"

# ---- Paths ----
DATA_DIR = ROOT / _get("DATA_DIR", "data")
PAPERS_DIR = DATA_DIR / "papers"
FIGURES_DIR = DATA_DIR / "figures"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = ROOT / _get("INDEX_DIR", "index")
CATALOGUE_PATH = DATA_DIR / "catalogue.json"
PAGES_PATH = PROCESSED_DIR / "pages.jsonl"
FIGURES_PATH = PROCESSED_DIR / "figures.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
MAX_UPLOAD_MB = int(_get("MAX_UPLOAD_MB", "30"))

for _d in (PAPERS_DIR, FIGURES_DIR, PROCESSED_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def check_keys() -> None:
    """Fail early with a clear message if a key is missing (unless in mock mode)."""
    if MOCK_LLM:
        return
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is empty. Copy .env.example to .env and paste your key, "
            "or set MOCK_LLM=1 to test without a key."
        )
