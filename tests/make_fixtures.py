"""Generate two small synthetic 'papers' for offline testing (no real paper content).

They contain: two-column text, numbered headings, a vector-drawn architecture
diagram, a raster chart (matplotlib), a ruled table, and a References section.
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

BODY = 9.5
LOREM_A = ("ToyGraph builds an entity graph from document chunks and retrieves relational paths between the "
           "entities mentioned in a question. The retriever scores each path by how much relevance flows from the "
           "query nodes, and keeps only the highest-scoring paths to reduce redundant context. ")
LOREM_B = ("We evaluate on two multi-hop question answering benchmarks, HotpotQA and 2Wiki, and report Recall@5 "
           "and exact match. ToyGraph improves Recall@5 over a dense retriever baseline while sending fewer "
           "tokens to the generator. ")


def _text(page, rect, text, size=BODY, font="helv"):
    r = pymupdf.Rect(rect)
    while page.insert_textbox(r, text, fontsize=size, fontname=font) < 0:  # grow until it fits
        r.y1 += 6
        if r.y1 > page.rect.height:
            raise ValueError("fixture text does not fit")


def _chart_png() -> bytes:
    fig, ax = plt.subplots(figsize=(4, 2.6), dpi=120)
    k = [1, 2, 5, 10, 20]
    ax.plot(k, [0.31, 0.42, 0.58, 0.66, 0.70], marker="o", label="ToyGraph")
    ax.plot(k, [0.25, 0.33, 0.47, 0.57, 0.63], marker="s", label="Dense baseline")
    ax.set_xlabel("Number of retrieved passages (k)")
    ax.set_ylabel("Recall@k")
    ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def paper_a(path: Path):
    doc = pymupdf.open()
    W = 612
    L, R, MID = 54, W - 54, W / 2
    # ---- page 1
    p = doc.new_page(width=W, height=792)
    _text(p, (L, 60, R, 90), "ToyGraph: Path-Pruned Graph Retrieval for Question Answering", 15, "hebo")
    _text(p, (L, 95, R, 110), "A. Author, B. Author (synthetic test fixture)", 9)
    _text(p, (L, 120, R, 135), "Abstract", 11, "hebo")
    _text(p, (L, 138, R, 200), LOREM_A * 2, BODY)
    _text(p, (L, 210, MID - 8, 225), "1 Introduction", 11, "hebo")
    _text(p, (L, 228, MID - 8, 470), (LOREM_A + LOREM_B) * 2, BODY)
    _text(p, (MID + 8, 210, R, 470), LOREM_B * 3 + "As shown in Figure 1, the pipeline has three stages. ", BODY)
    # vector architecture diagram (full width) + caption below
    y0 = 490
    boxes = [("Chunk text", L + 10), ("Build entity graph", L + 180), ("Prune paths", L + 350)]
    for label, x in boxes:
        r = pymupdf.Rect(x, y0 + 30, x + 140, y0 + 80)
        p.draw_rect(r, color=(0.1, 0.3, 0.5), width=1.2)
        p.insert_text((x + 15, y0 + 60), label, fontsize=10)
    for x in (L + 150, L + 320):
        p.draw_line((x, y0 + 55), (x + 30, y0 + 55), color=(0, 0, 0), width=1)
    _text(p, (L, y0 + 100, R, y0 + 130), "Figure 1: Overview of the ToyGraph pipeline: chunking, entity-graph construction, and flow-based path pruning.", 9)
    _text(p, (L, y0 + 140, R, 760), LOREM_B * 3, BODY)

    # ---- page 2: method, table, chart
    p = doc.new_page(width=W, height=792)
    _text(p, (L, 60, R, 75), "2 Method", 11, "hebo")
    _text(p, (L, 78, R, 170), LOREM_A * 3 + "Table 1 lists the results on both benchmarks. ", BODY)
    _text(p, (L, 180, R, 195), "Table 1: Recall@5 and exact match (EM) on HotpotQA and 2Wiki.", 9)
    rows = [["Method", "HotpotQA R@5", "2Wiki R@5", "EM"], ["Dense baseline", "47.1", "41.0", "38.2"],
            ["ToyGraph", "58.3", "52.6", "44.9"], ["ToyGraph w/o pruning", "53.0", "47.4", "41.1"]]
    tx, ty, cw, rh = L + 40, 200, 110, 20
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            p.insert_text((tx + j * cw + 4, ty + i * rh + 14), cell, fontsize=9)
    for i in range(len(rows) + 1):
        p.draw_line((tx, ty + i * rh), (tx + cw * 4, ty + i * rh), width=0.6)
    for j in range(5):
        p.draw_line((tx + j * cw, ty), (tx + j * cw, ty + rh * len(rows)), width=0.6)
    _text(p, (L, 300, R, 315), "3 Results", 11, "hebo")
    _text(p, (L, 318, R, 400), LOREM_B * 3 + "Figure 2 shows recall as k grows. ", BODY)
    p.insert_image(pymupdf.Rect(L + 80, 410, R - 80, 640), stream=_chart_png())
    _text(p, (L, 645, R, 670), "Figure 2: Recall@k as the number of retrieved passages k increases, ToyGraph vs. dense baseline.", 9)
    _text(p, (L, 680, R, 695), "References", 11, "hebo")
    _text(p, (L, 698, R, 760), "[1] Someone et al. A dense retriever. 2020. [2] Other et al. Graph RAG. 2024.", 8)
    doc.save(path)


def paper_b(path: Path):
    doc = pymupdf.open()
    p = doc.new_page()
    _text(p, (54, 60, 558, 90), "ToyVis: Retrieving Page Images Instead of Parsed Text", 15, "hebo")
    _text(p, (54, 100, 558, 115), "Abstract", 11, "hebo")
    txt = ("ToyVis embeds whole page images with a vision-language model and retrieves pages directly, avoiding "
           "parsing errors in tables and figures. Compared with text-based retrieval, ToyVis keeps layout "
           "information but needs more storage per page. ")
    _text(p, (54, 118, 558, 260), txt * 4, BODY)
    _text(p, (54, 270, 558, 285), "1 Limitations", 11, "hebo")
    _text(p, (54, 288, 558, 400), "ToyVis is slower at indexing time and struggles with very small fonts. " * 4, BODY)
    doc.save(path)


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures")
    out.mkdir(parents=True, exist_ok=True)
    paper_a(out / "toygraph.pdf")
    paper_b(out / "toyvis.pdf")
    (out / "not_a_pdf.pdf").write_text("this is plain text pretending to be a pdf")
    print("fixtures written to", out)
