"""Streamlit UI.   Run:  streamlit run app/ui.py"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import config
from app import generation
from ingest import catalogue, pipeline
from ingest.validate import InvalidFileError, validate_pdf_bytes
from retrieval import search

st.set_page_config(page_title="ResearchIQ", page_icon="📚", layout="wide")
st.title("ResearchIQ")
st.caption("Multimodal research assistant over Advanced-RAG papers: every answer cites paper, page and figure.")
if config.MOCK_LLM:
    st.warning("MOCK MODE: no API calls are made; answers are placeholders built from retrieved text. "
               "Set MOCK_LLM=0 and add GEMINI_API_KEY in .env for real answers.")


def paper_options() -> dict[str, str]:
    return {pid: (p.get("short_name") or p.get("title") or pid) for pid, p in catalogue.load().items()}


def index_ready() -> bool:
    try:
        search.all_chunks()
        return True
    except search.IndexNotReady as e:
        st.info(str(e))
        return False


def show_answer(res: dict):
    if res.get("insufficient_evidence"):
        st.error("Not enough evidence in the papers")
    text = res["answer"]
    num = {s["chunk_id"]: s["n"] for s in res.get("sources", [])}
    text = re.sub(r"\[([A-Za-z0-9_\-]+)\]", lambda m: f"**[{num[m.group(1)]}]**" if m.group(1) in num else "", text)
    st.markdown(text)
    if res.get("missing_information") and res.get("insufficient_evidence"):
        st.caption("Missing: " + res["missing_information"])
    if res.get("ai_suggestions"):
        st.info("**AI-generated suggestions, not findings from the papers:**\n\n" +
                "\n".join(f"- {s}" for s in res["ai_suggestions"]))
    if res.get("sources"):
        st.subheader("Sources")
        for s in res["sources"]:
            label = f"[{s['n']}] {s['paper_title'][:80]} | p.{s['page']}" + (f" | {s['figure_no']}" if s.get("figure_no") else "")
            with st.expander(label):
                if s.get("image_path") and (config.ROOT / s["image_path"]).exists():
                    st.image(str(config.ROOT / s["image_path"]), width="stretch")
                st.caption(s["excerpt"])
    with st.expander("Debug: route, retrieval scores, latency"):
        st.json({k: res.get(k) for k in ("route", "top_score", "retrieved", "invalid_citations",
                                          "refused_before_llm", "latency_s", "model")})


tab_papers, tab_ask, tab_compare, tab_synth, tab_eval = st.tabs(
    ["📄 Papers", "💬 Ask", "⚖️ Compare", "🧭 Synthesis", "📊 Evaluation"])

with tab_papers:
    up = st.file_uploader("Upload research-paper PDFs", type=["pdf"], accept_multiple_files=True)
    describe_figs = st.checkbox("Describe figures and tables with Gemini (recommended)", value=True)
    if up and st.button("Process uploaded papers", type="primary"):
        saved = []
        for f in up:
            data = f.getvalue()
            name = re.sub(r"[^A-Za-z0-9_.\-]", "_", f.name)
            try:
                validate_pdf_bytes(data, name)
            except InvalidFileError as e:
                st.error(f"{name}: {e}")
                continue
            (config.PAPERS_DIR / name).write_bytes(data)
            saved.append(config.PAPERS_DIR / name)
        if saved:
            with st.spinner("Parsing, extracting figures and indexing..."):
                try:
                    out = pipeline.run(saved, describe_figures=describe_figs)
                    st.success(f"Indexed {out['chunks']} chunks.")
                    for e in out["errors"]:
                        st.error(f"{e['file']}: {e['error']}")
                except Exception:
                    st.error("Processing failed. Check the server log for details.")
    if st.button("Re-index all papers in data/papers"):
        with st.spinner("Re-indexing..."):
            out = pipeline.run(sorted(config.PAPERS_DIR.glob("*.pdf")), describe_figures=describe_figs)
            st.success(f"Indexed {out['chunks']} chunks.")
    cat = list(catalogue.load().values())
    if cat:
        st.subheader(f"Collection: {len(cat)} papers")
        st.dataframe([{k: str(p.get(k, "")) for k in ("paper_id", "title", "year", "source", "identifier", "pages")}
                      for p in cat], width="stretch", hide_index=True)

with tab_ask:
    opts = paper_options()
    q = st.text_area("Question", placeholder="e.g. How does PathRAG reduce redundant information compared to LightRAG?")
    c1, c2 = st.columns([3, 1])
    scope = c1.multiselect("Limit to papers (optional)", list(opts), format_func=lambda p: opts[p])
    mode = c2.selectbox("Mode", ["auto", "lookup", "compare", "figure", "synthesis"])
    if st.button("Ask", type="primary") and q.strip() and index_ready():
        with st.spinner("Retrieving evidence and answering..."):
            try:
                show_answer(generation.answer(q, paper_ids=scope or None, mode=mode))
            except Exception:
                st.error("Something went wrong while answering. Please try again.")

with tab_compare:
    opts = paper_options()
    if len(opts) >= 2:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Paper A", list(opts), format_func=lambda p: opts[p], key="pa")
        b = c2.selectbox("Paper B", list(opts), index=1, format_func=lambda p: opts[p], key="pb")
        aspect = st.text_input("Aspect (optional)", placeholder="e.g. retrieval strategy, datasets, results")
        if st.button("Compare", type="primary") and a != b and index_ready():
            with st.spinner("Retrieving from both papers..."):
                show_answer(generation.compare(a, b, aspect))
    else:
        st.info("Add at least two papers first.")

with tab_synth:
    topic = st.text_input("Focus (optional)", placeholder="e.g. graph-based retrieval")
    if st.button("Generate synthesis", type="primary") and index_ready():
        with st.spinner("Synthesising the collection..."):
            show_answer(generation.synthesize(topic))

with tab_eval:
    res_dir = config.ROOT / "eval" / "results"
    summaries = sorted(res_dir.glob("summary_*.md")) if res_dir.exists() else []
    if summaries:
        st.markdown(summaries[-1].read_text())
        js = summaries[-1].with_suffix(".json")
        if js.exists():
            with st.expander("Raw summary JSON"):
                st.json(json.loads(js.read_text()))
    else:
        st.info("No evaluation results yet. Run: python -m eval.run_eval")
