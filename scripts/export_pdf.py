"""Exporta informe/, slides/ y guion/ a PDF con Edge o Chrome headless (respeta @media print).

Uso: python scripts/export_pdf.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

DOCS = [
    ("informe/informe.html", "informe/Informe-Tecnico.pdf"),
    ("slides/slides.html", "slides/Presentacion.pdf"),
    ("guion/guion.html", "guion/Guion-Presentacion.pdf"),
    ("estudio/guia-estudio.html", "estudio/Guia-Estudio.pdf"),
]


def find_browser() -> str:
    for c in CANDIDATES:
        if Path(c).exists():
            return c
    sys.exit("No se encontró Edge ni Chrome. Exportá manualmente (Imprimir > Guardar como PDF).")


def main():
    browser = find_browser()
    print("Navegador:", browser)
    profile = tempfile.mkdtemp(prefix="pdf_export_")  # perfil propio: evita reusar Edge abierto
    for src, dst in DOCS:
        src_p, dst_p = ROOT / src, ROOT / dst
        subprocess.run([
            browser, "--headless=new", "--disable-gpu", "--no-sandbox",
            f"--user-data-dir={profile}", "--virtual-time-budget=8000",
            "--run-all-compositor-stages-before-draw", "--no-pdf-header-footer",
            f"--print-to-pdf={dst_p}", src_p.as_uri(),
        ], check=True, timeout=120)
        kb = (dst_p.stat().st_size // 1024) if dst_p.exists() else 0
        print(f"  {dst}  ({kb} KB)")
    print("Listo.")


if __name__ == "__main__":
    main()
