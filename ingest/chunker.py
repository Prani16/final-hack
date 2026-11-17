"""Section-aware chunking.

Text: split on section headings, then pack paragraphs into ~CHUNK_TOKENS chunks
with CHUNK_OVERLAP overlap; never cross a heading. Each chunk is prefixed with
"[Paper title - Section]" (contextual retrieval). The References section is skipped.

Figures/tables: never split. One figure = one chunk, built from its caption,
Gemini description (or table rows + summary) and the paper's sentences about it.
"""
from __future__ import annotations

import re

import config
from ingest import injection

SKIP_SECTIONS = re.compile(r"^(\d+\.?\s*)?(references|bibliography)$", re.I)


def _tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)  # rough tokens-per-word for English


def chunk_text(pages: list[dict], paper_id: str, title: str) -> list[dict]:
    # 1) flatten into (section, page, paragraph)
    units, section = [], "Front matter"
    for rec in sorted(pages, key=lambda r: r["page"]):
        for b in rec["blocks"]:
            if b["is_heading"]:
                section = b["text"].strip()[:120]
                continue
            if SKIP_SECTIONS.match(section.strip(" .:")):
                continue
            units.append((section, rec["page"], b["text"]))

    # 2) pack within each section
    chunks, n = [], 0
    max_tok, overlap_words = config.CHUNK_TOKENS, int(config.CHUNK_OVERLAP / 1.3)

    def flush(sec, buf):
        nonlocal n
        body = " ".join(t for _, t in buf).strip()
        if len(body) < 40:
            return
        clean, flags = injection.redact(body)
        n += 1
        chunks.append({
            "chunk_id": f"{paper_id}_c{n:03d}",
            "type": "text",
            "paper_id": paper_id,
            "paper_title": title,
            "section": sec,
            "page_start": buf[0][0],
            "page_end": buf[-1][0],
            "text": f"[{title} - {sec}] {clean}",
            "injection_flags": flags,
        })

    i = 0
    while i < len(units):
        sec = units[i][0]
        buf, tok = [], 0
        while i < len(units) and units[i][0] == sec:
            _, page, text = units[i]
            t = _tokens(text)
            if buf and tok + t > max_tok:
                flush(sec, buf)
                tail = " ".join(x for _, x in buf).split()[-overlap_words:]
                buf, tok = [(buf[-1][0], " ".join(tail))], _tokens(" ".join(tail))
            if t > max_tok:  # one huge paragraph: hard split by words
                words = text.split()
                step = int(max_tok / 1.3)
                for s in range(0, len(words), step - overlap_words):
                    piece = " ".join(words[s : s + step])
                    if buf:
                        flush(sec, buf)
                    buf, tok = [(page, piece)], _tokens(piece)
            else:
                buf.append((page, text))
                tok += t
            i += 1
        if buf:
            flush(sec, buf)
    return chunks


def figure_chunk(fig: dict, title: str) -> dict:
    parts = [f"[{title}, p.{fig['page']}, {fig['figure_no']}]", f"Caption: {fig['caption']}"]
    if fig["type"] == "figure" and fig.get("description"):
        parts.append(f"Description: {fig['description']}")
        if fig.get("key_terms"):
            parts.append("Key terms: " + ", ".join(fig["key_terms"]))
    if fig["type"] == "table":
        if fig.get("summary"):
            parts.append(f"Table summary: {fig['summary']}")
        if fig.get("table_md"):
            parts.append("Table:\n" + fig["table_md"])
    if fig.get("nearby_text"):
        parts.append(f"Paper text: {fig['nearby_text']}")
    clean, flags = injection.redact("\n".join(parts))
    return {
        "chunk_id": fig["figure_id"],
        "type": fig["type"],
        "paper_id": fig["paper_id"],
        "paper_title": title,
        "section": fig["figure_no"],
        "page_start": fig["page"],
        "page_end": fig["page"],
        "figure_no": fig["figure_no"],
        "image_path": fig["image_path"],
        "text": clean,
        "injection_flags": flags,
    }
