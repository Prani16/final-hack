"""Help label gold chunk ids for the eval set.

    python -m eval.label_helper            # questions with empty gold_chunk_ids
    python -m eval.label_helper --id q03   # one question

Prints the top-15 hybrid candidates (union of dense + BM25) for each question.
Read them, then paste the ids that truly answer the question into
eval/questions.jsonl -> "gold_chunk_ids". Label BEFORE tuning, and do not just
copy the system's top results: open the paper and check.
"""
from __future__ import annotations

import argparse

from ingest import jsonl
from retrieval import search

Q_PATH = "eval/questions.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id")
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args()
    for q in jsonl.read(Q_PATH):
        if a.id and q["id"] != a.id or not a.id and q["gold_chunk_ids"] or q["type"] in ("no_evidence",):
            continue
        print("=" * 100, f"\n{q['id']} [{q['type']}] {q['question']}")
        res = search.retrieve(q["question"], paper_ids=q.get("paper_ids") or None, k=a.k, mode="hybrid")
        for c in res:
            snippet = c["text"].split("] ", 1)[-1][:160].replace("\n", " ")
            print(f"  {c['chunk_id']:<26} p.{c['page_start']:<3} {c['type']:<6} {snippet}")


if __name__ == "__main__":
    main()
