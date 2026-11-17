"""End-to-end smoke test in MOCK mode (no API key, no network).

    python tests/smoke_test.py

Uses a separate data/index folder under tests/_run so real data is untouched.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "tests" / "_run"
shutil.rmtree(RUN, ignore_errors=True)
os.environ.update(MOCK_LLM="1", RERANKER="none", DATA_DIR=str(RUN / "data"), INDEX_DIR=str(RUN / "index"),
                  EVIDENCE_THRESHOLD_DENSE="0.12")
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from tests import make_fixtures  # noqa: E402

fx = RUN / "fixtures"
fx.mkdir(parents=True)
make_fixtures.paper_a(fx / "toygraph.pdf")
make_fixtures.paper_b(fx / "toyvis.pdf")
(fx / "not_a_pdf.pdf").write_text("plain text pretending to be a pdf")

from scripts import make_injection_pdf  # noqa: E402

make_injection_pdf.main()

from app import generation  # noqa: E402
from ingest import jsonl, pipeline  # noqa: E402


ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)


out = pipeline.run([fx / "toygraph.pdf", fx / "toyvis.pdf", fx / "not_a_pdf.pdf",
                    config.PAPERS_DIR / "test_injection.pdf"])
check(len(out["papers"]) == 3, f"3 valid papers ingested ({[p['paper_id'] for p in out['papers']]})")
check(any("not a valid PDF" in e["error"] for e in out["errors"]), "fake PDF rejected with a friendly message")

figs = {f["figure_id"]: f for f in jsonl.read(config.FIGURES_PATH)}
print("   figures:", {k: (v["page"], v["crop_mode"], v.get("table_source")) for k, v in figs.items()})
check({"toygraph_fig1", "toygraph_fig2", "toygraph_tab1"} <= set(figs), "Figure 1 (vector), Figure 2 (raster), Table 1 found")
t1 = figs.get("toygraph_tab1", {})
check(t1.get("table_source") == "find_tables" and "58.3" in t1.get("table_md", "")
      or t1.get("table_source") == "gemini_vision",
      f"table rows exact via find_tables, or garbled output rejected and sent to vision ({t1.get('table_source')})")
check(all((config.ROOT / f["image_path"]).exists() for f in figs.values()), "cropped PNGs saved")
check("Figure 1" in figs.get("toygraph_fig1", {}).get("nearby_text", ""), "paper text mentioning Figure 1 captured")

chunks = jsonl.read(config.CHUNKS_PATH)
check(not any("Someone et al" in c["text"] for c in chunks), "References section skipped")
check(any(c["injection_flags"] for c in chunks if c["paper_id"] == "test_injection"), "injection text flagged")
check(not any("PINEAPPLE" in c["text"] for c in chunks), "injection sentence redacted")
check(all(c["text"].startswith("[") for c in chunks), "chunks carry [title - section] prefix")
secs = {c["section"] for c in chunks if c["paper_id"] == "toygraph" and c["type"] == "text"}
print("   sections:", secs)
check(any("Method" in s for s in secs), "section headings detected")

r = generation.answer("How does ToyGraph prune paths to reduce redundant context?")
check(r["sources"] and not r["insufficient_evidence"], f"lookup answered with {len(r['sources'])} citations")
r = generation.answer("Explain Figure 2 of ToyGraph")
check(r["route"]["type"] == "figure" and r["retrieved"][0]["chunk_id"] == "toygraph_fig2", "named figure pinned first")
r = generation.answer("Compare ToyGraph and ToyVis", paper_ids=["toygraph", "toyvis"], mode="compare")
check({s["paper_id"] for s in r["sources"]} >= {"toygraph", "toyvis"}, "comparison cites both papers")
r = generation.answer("What is the boiling point of mercury in Kelvin under standard pressure?")
check(r["insufficient_evidence"], f"off-topic question refused (top_score={r['top_score']})")
r = generation.synthesize()
check(bool(r["ai_suggestions"]), "synthesis returns labelled AI suggestions")

from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402

c = TestClient(app)
check(c.get("/health").status_code == 200, "API /health")
check(len(c.get("/papers").json()) == 3, "API /papers lists collection")
resp = c.post("/ask", json={"question": "What benchmarks does ToyGraph use?"})
check(resp.status_code == 200 and resp.json()["sources"], "API /ask returns cited answer")
check(c.get("/figures/toygraph_fig1").status_code == 200, "API serves figure image")
check(c.get("/figures/..%2F..%2Fconfig").status_code in (400, 404), "API blocks path traversal")
bad = c.post("/papers", files={"files": ("evil.pdf", b"not a pdf", "application/pdf")})
check(bad.status_code == 200 and bad.json()["errors"], "API rejects bad upload safely")

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
