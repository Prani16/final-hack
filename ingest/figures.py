"""Figure and table extraction.

1. Find caption blocks ("Figure 3:", "Fig. 2.", "Table 1:").
2. Figures: the graphic sits ABOVE the caption. Union the image and vector-drawing
   boxes between the caption and the nearest body-text block above it.
3. Tables: the table sits BELOW the caption. Try PyMuPDF find_tables() for exact
   cells; always crop the region as an image too.
4. If a crop looks wrong (too small), fall back to the whole page image.
Each record also stores the paper's own sentences that mention the figure.
"""
from __future__ import annotations

import logging
import re

import pymupdf

import config
from ingest.parse import _split_block

log = logging.getLogger("researchiq.figures")

SECTION = re.compile(r"^(\d+(\.\d+)*|[A-Z])\.?\s+[A-Z][^.]{2,70}$")  # "3 Results", "A.1 Setup"
CAPTION = re.compile(r"^(Figure|Fig\.|Table)\s*([A-Z]?\d+)\s*[:.|]", re.I)
DPI = 150
MIN_H = 40  # points; smaller crops are treated as failed


def _is_body(text: str, width: float, page_w: float) -> bool:
    return len(text) > 150 and width > page_w * 0.3


def _overlaps_x(a, b, slack=10) -> bool:
    return a[0] < b[2] + slack and b[0] < a[2] + slack


def _column_span(cap_rect: pymupdf.Rect, page: pymupdf.Page) -> tuple[float, float]:
    w = page.rect.width
    if cap_rect.width < w * 0.55:  # caption in one column
        return (36, w / 2 + 6) if cap_rect.x1 <= w / 2 + 20 else (w / 2 - 6, w - 36)
    return 36, w - 36


def _graphics_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if r.width > 2 or r.height > 2:
            rects.append(r)
    return rects


def _table_markdown(page: pymupdf.Page, clip: pymupdf.Rect) -> str | None:
    try:
        tabs = page.find_tables(clip=clip)
    except Exception:
        return None
    best = max(tabs.tables, key=lambda t: t.row_count * t.col_count, default=None)
    if not best or best.row_count < 2 or best.col_count < 2:
        return None
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in best.extract()]
    filled = sum(1 for r in rows for c in r if c)
    if filled < 0.5 * best.row_count * best.col_count:  # mostly empty = broken detection
        return None
    # sanity check: decimal numbers on the page must survive extraction, else use the vision model
    page_nums = set(re.findall(r"\d+\.\d+", page.get_text(clip=best.bbox)))
    cell_nums = set(re.findall(r"\d+\.\d+", " ".join(c for r in rows for c in r)))
    if page_nums and len(page_nums & cell_nums) < 0.8 * len(page_nums):
        log.info("find_tables output failed the number check; table will be read by the vision model")
        return None
    head, *body = rows
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md)


def _mentions(full_text: str, label: str, number: str, caption: str) -> str:
    kind = "Fig(?:ure|\\.)" if label.lower().startswith("fig") else "Table"
    rx = re.compile(rf"[^.]*\b{kind}\s*{re.escape(number)}\b[^.]*\.", re.I)
    found = [m.group(0).strip() for m in rx.finditer(full_text) if m.group(0).strip()[:60] not in caption]
    return " ".join(found[:2])[:700]


def extract_figures(path, paper_id: str) -> list[dict]:
    doc = pymupdf.open(path)
    full_text = " ".join(p.get_text() for p in doc).replace("\n", " ")
    records, seen = [], set()

    for page in doc:
        pw, ph = page.rect.width, page.rect.height
        # text segments (x0, y0, x1, y1, text); blocks are split so merged captions are found
        blocks = [(*seg["bbox"], seg["text"]) for b in page.get_text("dict")["blocks"] if b.get("type") == 0
                  for seg in _split_block(b)]
        graphics = None
        for b in blocks:
            text = " ".join(b[4].split())
            m = CAPTION.match(text)
            if not m:
                continue
            label, number = m.group(1), m.group(2)
            is_table = label.lower() == "table"
            fid = f"{paper_id}_{'tab' if is_table else 'fig'}{number}"
            if fid in seen:  # caption text repeated (e.g. continued table)
                continue
            cap = pymupdf.Rect(b[:4])
            cx0, cx1 = _column_span(cap, page)
            col = (cx0, 0, cx1, ph)

            if not is_table:
                # nearest body text ABOVE the caption, in the same column
                above = [pymupdf.Rect(x[:4]) for x in blocks
                         if x[3] <= cap.y0 + 1 and _overlaps_x(x[:4], col) and _is_body(x[4], x[2] - x[0], pw)]
                top = max((r.y1 for r in above), default=36)
                graphics = graphics if graphics is not None else _graphics_rects(page)
                parts = [g for g in graphics if g.y0 >= top - 2 and g.y1 <= cap.y0 + 2 and _overlaps_x(g, col)]
                parts += [pymupdf.Rect(x[:4]) for x in blocks  # small labels inside the figure
                          if x[1] >= top and x[3] <= cap.y0 and _overlaps_x(x[:4], col) and not _is_body(x[4], x[2] - x[0], pw)
                          and not CAPTION.match(" ".join(x[4].split()))]
                if parts:
                    region = pymupdf.Rect(parts[0])
                    for r in parts[1:]:
                        region |= r
                else:
                    region = pymupdf.Rect(cx0, top, cx1, cap.y0)
            else:
                # table sits below its caption, up to the next body text / caption
                below = [pymupdf.Rect(x[:4]) for x in blocks
                         if x[1] >= cap.y1 - 1 and _overlaps_x(x[:4], col)
                         and (_is_body(x[4], x[2] - x[0], pw) or CAPTION.match(" ".join(x[4].split()))
                              or SECTION.match(x[4].strip()))]
                bottom = min((r.y0 for r in below), default=ph - 36)
                graphics = graphics if graphics is not None else _graphics_rects(page)
                parts = [g for g in graphics if g.y0 >= cap.y1 - 2 and g.y1 <= bottom + 2 and _overlaps_x(g, col)]
                parts += [pymupdf.Rect(x[:4]) for x in blocks
                          if x[1] >= cap.y1 - 1 and x[3] <= bottom + 1 and _overlaps_x(x[:4], col)]
                region = pymupdf.Rect(cx0, cap.y1, cx1, bottom)
                if parts:  # tables often span wider than the caption's column
                    region = pymupdf.Rect(parts[0])
                    for r in parts[1:]:
                        region |= r
            if not is_table:
                region.y0 = max(region.y0, top)  # never include the paragraph above
            region = (region + (-6, -2, 6, 2)) & page.rect
            crop_mode = "region"
            if region.height < MIN_H or region.width < 60:
                region, crop_mode = page.rect, "page_fallback"

            img_path = config.FIGURES_DIR / f"{fid}.png"
            page.get_pixmap(clip=region, dpi=DPI).save(img_path)
            rec = {
                "figure_id": fid,
                "type": "table" if is_table else "figure",
                "paper_id": paper_id,
                "page": page.number + 1,
                "figure_no": f"{'Table' if is_table else 'Figure'} {number}",
                "caption": text[:600],
                "nearby_text": _mentions(full_text, label, number, text),
                "image_path": str(img_path.relative_to(config.ROOT)),
                "bbox": [round(v, 1) for v in region],
                "crop_mode": crop_mode,
            }
            if is_table:
                md = _table_markdown(page, region)
                rec["table_md"] = md or ""
                rec["table_source"] = "find_tables" if md else "pending_vision"
            records.append(rec)
            seen.add(fid)
    doc.close()
    log.info("%s: %d figures/tables found", paper_id, len(records))
    return records
