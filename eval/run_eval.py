"""Repeatable evaluation.

    python -m eval.run_eval                 # retrieval ablation + answers
    python -m eval.run_eval --judge         # + LLM-as-judge faithfulness and correctness
    python -m eval.run_eval --retrieval-only

Retrieval (questions with gold_chunk_ids): Recall@5 and MRR@10 for
dense vs hybrid (dense+BM25+RRF) vs hybrid+rerank.
Answers: key-fact coverage, citation precision vs gold, invalid citations,
refusal accuracy, injection test, latency; optional LLM-judge faithfulness.

Outputs eval/results/: per-question CSV (with empty columns for the human
check), summary JSON and a Markdown summary to paste into the README.
Report only numbers you actually ran; LLM-judge scores are a signal, not ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from datetime import datetime

import config
import llm_client
from app import generation
from ingest import jsonl
from retrieval import search

Q_PATH = config.ROOT / "eval" / "questions.jsonl"
OUT = config.ROOT / "eval" / "results"
MODES = ["dense", "hybrid", "hybrid_rerank"]


def recall_at(ids: list[str], gold: set[str], k: int) -> float:
    return len(set(ids[:k]) & gold) / len(gold)


def mrr_at(ids: list[str], gold: set[str], k: int) -> float:
    for rank, cid in enumerate(ids[:k], 1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


JUDGE = """You are grading a research assistant's answer.

Sources the answer cited:
{sources}

Answer:
{answer}

Expected key facts (may be empty): {facts}

Return JSON:
{{"total_claims": <number of factual claims in the answer>,
  "supported_claims": <how many are fully supported by the sources above>,
  "unsupported": ["claims not supported"],
  "correctness": "correct | partial | wrong | n/a (n/a if no key facts given)"}}"""


def judge(res: dict, q: dict) -> dict:
    srcs = "\n\n".join(f"[{s['chunk_id']}] {search.chunk(s['chunk_id'])['text'][:1500]}" for s in res["sources"])
    out = llm_client.generate(
        JUDGE.format(sources=srcs or "(none)", answer=res["answer"], facts=q.get("expected_key_facts")),
        json_output=True, temperature=0.0,
        mock=lambda: {"total_claims": 3, "supported_claims": 3, "unsupported": [], "correctness": "n/a"})
    tot = max(int(out.get("total_claims", 0)), 1)
    return {"faithfulness": round(int(out.get("supported_claims", 0)) / tot, 3),
            "judge_correctness": out.get("correctness", "n/a"),
            "unsupported": out.get("unsupported", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    a = ap.parse_args()
    config.check_keys()
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    questions = jsonl.read(Q_PATH)
    known = set(search.all_chunks())

    # ---------------- retrieval ablation
    ret_rows, ablation = [], {}
    labelled = [q for q in questions if q.get("gold_chunk_ids")]
    for mode in MODES:
        rs, ms = [], []
        for q in labelled:
            gold = set(q["gold_chunk_ids"]) & known
            if not gold:
                continue
            ids = [c["chunk_id"] for c in search.retrieve(q["question"], k=10, mode=mode)]
            rs.append(recall_at(ids, gold, 5))
            ms.append(mrr_at(ids, gold, 10))
            ret_rows.append({"id": q["id"], "mode": mode, "recall@5": round(rs[-1], 3), "mrr@10": round(ms[-1], 3),
                             "top5": " ".join(ids[:5])})
        if rs:
            ablation[mode] = {"n": len(rs), "recall@5": round(st.mean(rs), 3), "mrr@10": round(st.mean(ms), 3)}
    missing_gold = [g for q in labelled for g in q["gold_chunk_ids"] if g not in known]

    # ---------------- answers
    ans_rows = []
    if not a.retrieval_only:
        for q in questions:
            res = generation.answer(q["question"], paper_ids=None)
            text = res["answer"]
            cited = [s["chunk_id"] for s in res.get("sources", [])]
            gold = set(q.get("gold_chunk_ids") or [])
            facts = q.get("expected_key_facts") or []
            row = {
                "id": q["id"], "type": q["type"], "question": q["question"],
                "route": res["route"]["type"], "top_score": res.get("top_score"),
                "refused": bool(res.get("insufficient_evidence")),
                "expect_refusal": q.get("expect_refusal", False),
                "key_fact_coverage": round(sum(f.lower() in text.lower() for f in facts) / len(facts), 3) if facts else "",
                "cited": " ".join(cited), "n_cited": len(cited),
                "citation_precision_vs_gold": round(len(set(cited) & gold) / len(cited), 3) if gold and cited else "",
                "invalid_citations": len(res.get("invalid_citations", [])),
                "latency_s": res.get("latency_s"),
                "answer": text.replace("\n", " ")[:1500],
                "human_correct(Y/P/N)": "", "human_citations_ok(Y/N)": "",
            }
            if q["type"] == "injection":
                leaked = [s for s in q.get("forbidden_strings", []) if s.lower() in text.lower()]
                inj = q.get("injection_paper")
                flagged = any(c.get("injection_flags") for c in search.all_chunks().values() if c["paper_id"] == inj)
                row["injection_pass"] = not leaked
                row["injection_chunk_flagged"] = flagged
            if a.judge and res.get("sources"):
                row.update(judge(res, q))
            ans_rows.append(row)
            print(f"{q['id']} [{row['route']:<9}] refused={row['refused']!s:<5} cited={row['n_cited']} "
                  f"facts={row['key_fact_coverage']} {row['latency_s']}s")

    # ---------------- summary
    summary = {"timestamp": stamp, "mock": config.MOCK_LLM, "model": config.LLM_MODEL, "embed_model": config.EMBED_MODEL,
               "reranker": config.RERANKER, "n_questions": len(questions), "retrieval": ablation,
               "gold_ids_not_in_index": missing_gold}
    if ans_rows:
        should = [r for r in ans_rows if r["expect_refusal"]]
        shouldnt = [r for r in ans_rows if not r["expect_refusal"]]
        cov = [r["key_fact_coverage"] for r in ans_rows if r["key_fact_coverage"] != ""]
        prec = [r["citation_precision_vs_gold"] for r in ans_rows if r["citation_precision_vs_gold"] != ""]
        summary["answers"] = {
            "correct_refusals": f"{sum(r['refused'] for r in should)}/{len(should)}",
            "false_refusals": f"{sum(r['refused'] for r in shouldnt)}/{len(shouldnt)}",
            "key_fact_coverage_mean": round(st.mean(cov), 3) if cov else None,
            "citation_precision_vs_gold_mean": round(st.mean(prec), 3) if prec else None,
            "answers_with_citations": f"{sum(r['n_cited'] > 0 for r in shouldnt)}/{len(shouldnt)}",
            "invalid_citations_removed": sum(r["invalid_citations"] for r in ans_rows),
            "injection_pass": [r.get("injection_pass") for r in ans_rows if "injection_pass" in r],
            "injection_chunk_flagged": [r.get("injection_chunk_flagged") for r in ans_rows if "injection_chunk_flagged" in r],
            "mean_latency_s": round(st.mean(r["latency_s"] or 0 for r in ans_rows), 2),
        }
        if a.judge:
            f = [r["faithfulness"] for r in ans_rows if "faithfulness" in r]
            summary["answers"]["llm_judge_faithfulness_mean"] = round(st.mean(f), 3) if f else None

    with open(OUT / f"answers_{stamp}.csv", "w", newline="", encoding="utf-8") as fh:
        if ans_rows:
            keys = list(dict.fromkeys(k for r in ans_rows for k in r))
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(ans_rows)
    with open(OUT / f"retrieval_{stamp}.csv", "w", newline="", encoding="utf-8") as fh:
        if ret_rows:
            w = csv.DictWriter(fh, fieldnames=list(ret_rows[0]))
            w.writeheader()
            w.writerows(ret_rows)
    (OUT / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2))

    md = [f"## Evaluation results ({stamp})", "",
          f"Model `{summary['model']}`, embeddings `{summary['embed_model']}`, reranker `{summary['reranker']}`"
          + (" (**MOCK MODE: numbers are not meaningful**)" if config.MOCK_LLM else ""), "",
          "### Retrieval (questions with labelled gold chunks)", "",
          "| Mode | n | Recall@5 | MRR@10 |", "|---|---|---|---|"]
    md += [f"| {m} | {v['n']} | {v['recall@5']} | {v['mrr@10']} |" for m, v in ablation.items()] or ["| (no labelled questions) | | | |"]
    if missing_gold:
        md += ["", f"Gold ids not found in the index (fix labels): {', '.join(missing_gold)}"]
    if "answers" in summary:
        md += ["", "### Answers", "", "| Metric | Value |", "|---|---|"]
        md += [f"| {k} | {v} |" for k, v in summary["answers"].items()]
    md += ["", f"Per-question details: `eval/results/answers_{stamp}.csv` (fill in the human-check columns)."]
    (OUT / f"summary_{stamp}.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
