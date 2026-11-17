"""Paper catalogue: one entry per paper (title, authors, year, source, id, filename)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import config


def load() -> dict[str, dict]:
    if config.CATALOGUE_PATH.exists():
        return {p["paper_id"]: p for p in json.loads(config.CATALOGUE_PATH.read_text())}
    return {}


def save(cat: dict[str, dict]) -> None:
    config.CATALOGUE_PATH.write_text(json.dumps(list(cat.values()), indent=2, ensure_ascii=False))


def paper_id_for(filename: str) -> str:
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:40]
    return slug or "paper"


def upsert(paper_id: str, force: bool = False, **fields) -> dict:
    cat = load()
    entry = cat.get(paper_id, {"paper_id": paper_id})
    for k, v in fields.items():
        if v and (force or not entry.get(k)):  # guesses never overwrite curated metadata
            entry[k] = v
    cat[paper_id] = entry
    save(cat)
    return entry


def remove(paper_id: str) -> None:
    cat = load()
    cat.pop(paper_id, None)
    save(cat)


def title_of(paper_id: str) -> str:
    return load().get(paper_id, {}).get("title") or paper_id
