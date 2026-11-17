"""Prompt-injection filter applied to paper text at ingestion time.

Paper text is untrusted. Instruction-like sentences aimed at an AI model are
redacted and the chunk is flagged, so the injection test can show the filter
caught them. The answer prompt adds a second layer (sources are data only).
"""
from __future__ import annotations

import re

PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"ignore all instructions",
    r"disregard (all |any |the )?(previous|prior|above|system) (instructions|prompts|messages)",
    r"forget (all |your )?(previous |prior )?instructions",
    r"you are now (a|an|in) ",
    r"new (system )?instructions?:",
    r"system prompt",
    r"reveal (your|the) (prompt|instructions|system)",
    r"(as an ai|as a language model)[, ]+(you must|always|never)",
    r"(always|you must) (answer|respond|say) that",
    r"do not (cite|mention) (any )?(sources|other papers)",
]
_RX = re.compile("|".join(f"(?:{p})" for p in PATTERNS), re.I)
_SENT = re.compile(r"[^.!?\n]*(?:[.!?]|\n|$)")


def scan(text: str) -> list[str]:
    return [m.group(0) for m in _RX.finditer(text)]


def redact(text: str) -> tuple[str, list[str]]:
    """Replace every sentence containing an injection pattern. Returns (clean_text, flags)."""
    flags = scan(text)
    if not flags:
        return text, []

    def _sub(m: re.Match) -> str:
        s = m.group(0)
        return "[removed: instruction-like text found in source] " if _RX.search(s) else s

    return _SENT.sub(_sub, text), flags
