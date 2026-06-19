"""PDF -> texto plano.

Los PDFs de la facultad (plan de estudios, estado académico) son digitales con texto
seleccionable, así que pdfplumber alcanza. Si algún día entra un PDF escaneado habría que
sumar OCR (pytesseract), pero hoy no hace falta.
"""
from __future__ import annotations

from pathlib import Path

import pdfplumber

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def pdf_to_text(path: str | Path) -> str:
    """Devuelve todo el texto del PDF, una página por línea en blanco."""
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


if __name__ == "__main__":
    for p in sorted(RAW_DIR.glob("*.pdf")):
        txt = pdf_to_text(p)
        print(f"\n===== {p.name}  ({len(txt)} chars) =====")
        print(txt[:800])
