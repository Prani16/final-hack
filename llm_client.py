"""Thin wrapper around the Gemini API (google-genai SDK).

Every model call in the project goes through `generate()` or `embed()`, so
switching models or adding retries happens in one place.

MOCK_LLM=1 replaces the API with deterministic local fakes, so the whole
pipeline can be tested without an API key.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from typing import Any, Callable, Sequence

import config

log = logging.getLogger("researchiq.llm")

_client = None


def _get_client():
    global _client
    if _client is None:
        config.check_keys()
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _retry(fn: Callable[[], Any], tries: int = 4, base_delay: float = 2.0) -> Any:
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # network errors, rate limits (429), 5xx
            if attempt == tries - 1:
                raise
            delay = base_delay * (2**attempt)
            log.warning("Model call failed (%s). Retrying in %.0fs", type(exc).__name__, delay)
            time.sleep(delay)


def parse_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating ```json fences and LaTeX backslashes."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    start = cleaned.find("{")
    if start > 0:
        cleaned = cleaned[start : cleaned.rfind("}") + 1]
    # \alpha, \theta etc. are invalid JSON escapes: double any backslash that is not a JSON escape
    for candidate in (cleaned, re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned)):
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
    raise ValueError("Model did not return valid JSON")


def generate(
    prompt: str,
    *,
    system: str | None = None,
    images: Sequence[bytes] = (),
    json_output: bool = False,
    model: str | None = None,
    temperature: float = 0.1,
    mock: Callable[[], Any] | None = None,
) -> Any:
    """Call the LLM. Returns a dict/list if json_output else a string.

    `mock` is used when MOCK_LLM=1: a callable returning the fake result.
    """
    if config.MOCK_LLM:
        result = mock() if mock else "[mock] " + prompt[:200]
        return result

    from google.genai import types

    parts: list[Any] = [types.Part.from_bytes(data=img, mime_type="image/png") for img in images]
    parts.append(prompt)
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json" if json_output else "text/plain",
    )

    client = _get_client()  # outside the retry loop: a missing key should fail immediately

    def call():
        resp = client.models.generate_content(
            model=model or config.LLM_MODEL, contents=parts, config=cfg
        )
        return resp.text or ""

    text = _retry(call)
    return parse_json(text) if json_output else text


# ---------------------------------------------------------------- embeddings

MOCK_DIM = 4096  # large, so hashed words rarely collide
_MOCK_STOP = set("a an the of and or to in on for with by is are was were be this that it as at from what how "
                 "which does do under than its into their we our".split())


def _mock_embed(text: str) -> list[float]:
    """Hashed bag-of-words vector: crude but gives sensible similarity for tests."""
    vec = [0.0] * MOCK_DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _MOCK_STOP or len(tok) < 3:
            continue
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % MOCK_DIM] += 1.0 if (h >> 8) % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed(texts: Sequence[str], task: str = "RETRIEVAL_DOCUMENT", batch_size: int = 50) -> list[list[float]]:
    """Embed texts. task = RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    if config.MOCK_LLM:
        return [_mock_embed(t) for t in texts]

    from google.genai import types

    client = _get_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=config.EMBED_DIM)

        def call(batch=batch, cfg=cfg):
            return client.models.embed_content(model=config.EMBED_MODEL, contents=batch, config=cfg)

        resp = _retry(call)
        for e in resp.embeddings:
            vals = list(e.values)
            norm = math.sqrt(sum(v * v for v in vals)) or 1.0  # normalise (needed when dim < 3072)
            out.append([v / norm for v in vals])
    return out
