from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def replace_paper(path: Path, paper_id: str, new_rows: list[dict]) -> None:
    """Rewrite a JSONL file with this paper's rows replaced (re-ingest is idempotent)."""
    rows = [r for r in read(path) if r.get("paper_id") != paper_id]
    write(path, rows + new_rows)
