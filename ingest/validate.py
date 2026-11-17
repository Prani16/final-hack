"""Upload validation: only real, readable, reasonably sized PDFs get through."""
from __future__ import annotations

from pathlib import Path

import pymupdf

import config


class InvalidFileError(ValueError):
    """Raised with a user-safe message (no paths, no stack traces)."""


def validate_pdf_bytes(data: bytes, filename: str = "file") -> None:
    if not filename.lower().endswith(".pdf"):
        raise InvalidFileError("Only PDF files are supported.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise InvalidFileError(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    if not data.startswith(b"%PDF"):
        raise InvalidFileError("This file is not a valid PDF (wrong file signature).")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise InvalidFileError("This PDF could not be opened. It may be damaged.") from None
    try:
        if doc.needs_pass:
            raise InvalidFileError("Password-protected PDFs are not supported.")
        if doc.page_count == 0:
            raise InvalidFileError("This PDF has no pages.")
    finally:
        doc.close()


def validate_pdf_path(path: Path) -> None:
    validate_pdf_bytes(Path(path).read_bytes(), Path(path).name)
