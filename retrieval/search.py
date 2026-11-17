"""Hybrid retrieval: dense (Chroma) + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank.

Also: a question router (lookup / compare / figure / synthesis / out_of_scope),
direct lookup of named figures ("Figure 3 of PathRAG"), and per-paper retrieval
so a comparison gets evidence from every paper involved.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import re
from functools import lru_cache

import config
import llm_client
from ingest import catalogue
from retrieval.index import BM25_PATH, CHROMA_DIR, COLLECTION, META_PATH, tokenize

log = logging.getLogger("researchiq.search")
RRF_K = 60
FIG_REF = re.compile(r"\b(fig(?:ure|\.)?|table)\s*([A-Z]?\d+)\b", re.I)


class IndexNotReady(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _state():
    if not BM25_PATH.exists():
        raise IndexNotReady("No papers indexed yet. Ingest papers first.")
    meta = json.loads(META_PATH.read_text())
    expected = "mock" if config.MOCK_LLM else config.EMBED_MODEL
    if meta.get("embed_model") != expected:
        raise IndexNotReady(f"Index was built with '{meta.get('embed_model')}' but current setting is "
                            f"'{expected}'. Re-run ingestion (python -m ingest.pipeline --rechunk).")
    import chromadb

    col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return col, data


@lru_cache(maxsize=1)
def _reranker():
    if config.RERANKER != "bge" or config.MOCK_LLM:
        return None
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(config.RERANKER_MODEL, max_length=512)
    except Exception as exc:
        log.warning("Reranker unavailable (%s); continuing without it", type(exc).__name__)
        return None


def reset_cache():
    _state.cache_clear()


def chunk(chunk_id: str) -> dict | None:
    return _state()[1]["chunks"].get(chunk_id)


def all_chunks() -> dict[str, dict]:
    return _state()[1]["chunks"]


# ------------------------------------------------------------------ retrievers

def dense(question: str, k: int, paper_ids: list[str] | None = None) -> list[tuple[str, float]]:
    col, _ = _state()
    qv = llm_client.embed([question], task="RETRIEVAL_QUERY")[0]
    where = {"paper_id": {"$in": paper_ids}} if paper_ids else None
    n = min(k, col.count())
    if n == 0:
        return []
    res = col.query(query_embeddings=[qv], n_results=n, where=where)
    return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]  # cosine similarity


def sparse(question: str, k: int, paper_ids: list[str] | None = None) -> list[tuple[str, float]]:
    _, data = _state()
    if data["bm25"] is None:
        return []
    scores = data["bm25"].get_scores(tokenize(question))
    ranked = sorted(zip(data["ids"], scores), key=lambda x: -x[1])
    if paper_ids:
        ranked = [r for r in ranked if data["chunks"][r[0]]["paper_id"] in paper_ids]
    return [r for r in ranked[:k] if r[1] > 0]


def rrf(*lists: list[tuple[str, float]]) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for lst in lists:
        for rank, (cid, _) in enumerate(lst):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


def rerank(question: str, ids: list[str]) -> list[tuple[str, float]] | None:
    model = _reranker()
    if model is None or not ids:
        return None
    texts = [all_chunks()[i]["text"][:2000] for i in ids]
    scores = model.predict([(question, t) for t in texts])
    scores = [1 / (1 + math.exp(-s)) if (s < 0 or s > 1) else float(s) for s in scores]  # to 0..1
    return sorted(zip(ids, scores), key=lambda x: -x[1])


# ------------------------------------------------------------------ router

def _heuristic_route(question: str, papers: dict[str, str]) -> dict:
    q = question.lower()
    named = [pid for pid, title in papers.items()
             if pid.replace("_", " ") in q or (title and title.lower().split(":")[0].strip() in q)]
    if FIG_REF.search(question):
        qtype = "figure"
    elif len(named) >= 2 or re.search(r"\b(compare|comparison|differ|versus|vs\.?)\b", q):
        qtype = "compare"
    elif re.search(r"\b(synthes|overall|across (all|the) papers|research gaps?|future directions?|trends?)\b", q):
        qtype = "synthesis"
    else:
        qtype = "lookup"
    return {"type": qtype, "papers": named, "figure_refs": [m.group(0) for m in FIG_REF.finditer(question)]}


def route(question: str) -> dict:
    cat = catalogue.load()
    papers = {pid: (p.get("short_name") or p.get("title") or pid) for pid, p in cat.items()}
    fallback = _heuristic_route(question, papers)
    listing = "\n".join(f"- {pid}: {t}" for pid, t in papers.items())
    prompt = f"""Classify the user's question about a collection of research papers.

Papers (id: title):
{listing}

Question: {question}

Return JSON: {{"type": "lookup|compare|figure|synthesis|out_of_scope",
 "papers": ["ids of papers the question names explicitly, else []"],
 "figure_refs": ["e.g. 'Figure 3', 'Table 2' if named, else []"]}}
- compare: asks how two or more papers/methods differ or relate
- figure: asks about a figure, chart, diagram or table
- synthesis: asks about the whole collection, trends, gaps, future directions
- out_of_scope: clearly unrelated to AI/RAG research (still answerable with a refusal)"""
    try:
        out = llm_client.generate(prompt, json_output=True, temperature=0.0, mock=lambda: fallback)
        out["papers"] = [p for p in out.get("papers", []) if p in papers]
        out.setdefault("figure_refs", fallback["figure_refs"])
        if out.get("type") not in {"lookup", "compare", "figure", "synthesis", "out_of_scope"}:
            out["type"] = fallback["type"]
        return out
    except Exception:
        return fallback


# ------------------------------------------------------------------ main entry

def retrieve(question: str, paper_ids: list[str] | None = None, k: int | None = None,
             mode: str = "hybrid_rerank", figure_refs: list[str] | None = None) -> list[dict]:
    """Return the top-k chunks (dicts with 'score' and 'score_kind').

    mode: dense | bm25 | hybrid | hybrid_rerank   (used by the eval ablation)
    """
    k = k or config.FINAL_TOP_K
    d = dense(question, config.DENSE_TOP_K, paper_ids) if mode != "bm25" else []
    s = sparse(question, config.BM25_TOP_K, paper_ids) if mode != "dense" else []
    dense_sim = dict(d)

    if mode == "dense":
        ranked, kind = d, "cosine"
    elif mode == "bm25":
        ranked, kind = s, "bm25"
    else:
        ranked, kind = rrf(d, s), "rrf"
        if mode == "hybrid_rerank":
            rr = rerank(question, [cid for cid, _ in ranked[: max(k * 4, 30)]])
            if rr is not None:
                ranked, kind = rr, "rerank"

    # named figures/tables go first ("Figure 3 of PathRAG")
    pinned = []
    if figure_refs and paper_ids:
        for ref in figure_refs:
            m = FIG_REF.search(ref)
            if m:
                tag = "tab" if m.group(1).lower().startswith("tab") else "fig"
                for pid in paper_ids:
                    cid = f"{pid}_{tag}{m.group(2)}"
                    if chunk(cid):
                        pinned.append(cid)

    score_of = dict(ranked)
    out, used = [], set()
    for cid in pinned + [c for c, _ in ranked]:
        if cid in used:
            continue
        used.add(cid)
        c = dict(all_chunks()[cid])
        c["score"] = round(score_of.get(cid, 1.0), 4)
        c["score_kind"] = "pinned" if cid in pinned else kind
        c["dense_sim"] = round(dense_sim.get(cid, 0.0), 4)
        out.append(c)
        if len(out) >= k:
            break
    return out


def retrieve_per_paper(question: str, paper_ids: list[str], k_each: int = 4) -> list[dict]:
    out = []
    for pid in paper_ids:
        out += retrieve(question, paper_ids=[pid], k=k_each)
    return out
