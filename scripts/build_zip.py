"""Arma el .zip de la entrega para correr el notebook en Colab (lo sube la celda de Bootstrap).

Bundle MÍNIMO autocontenido: lo que el notebook necesita para ejecutarse de punta a punta.
- src/*.py (todos los módulos, incluido db.py)
- data/raw/*.pdf (PDFs fuente, incluida la Ordenanza 1877)
- data/eval_resultados.json (resultados de la evaluación)
- notebook.ipynb, requirements.txt, README.md, INSTRUCCIONES.txt

NO incluye las bases generadas (data/chroma, data/academico.db): el notebook las reconstruye
desde los PDFs. Tampoco app/, scripts/, informe/slides/guion (no hacen falta para el notebook).

Salida: ../Entrega-Grupo14-GeneradorAcademico.zip (al lado del repo). Raíz interna:
'generador-academico-rag/', que es la carpeta a la que hace `cd` el bootstrap del notebook.

ponytail: builder one-shot del entregable; re-ejecutar tras cambiar código/datos para no entregar
un zip viejo (fue justo lo que pasó: el zip había quedado desactualizado).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCROOT = "generador-academico-rag"  # carpeta raíz dentro del zip (el bootstrap hace cd acá)
OUT = ROOT.parent / "Entrega-Grupo14-GeneradorAcademico.zip"


def archivos() -> list[Path]:
    fijos = ["notebook.ipynb", "requirements.txt", "README.md", "INSTRUCCIONES.txt",
             "data/eval_resultados.json", "scripts/run_eval.py"]
    paths = [ROOT / f for f in fijos]
    paths += sorted(ROOT.glob("src/*.py"))
    paths += sorted((ROOT / "data" / "raw").glob("*.pdf"))
    return [p for p in paths if p.exists()]


def main() -> None:
    files = archivos()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, f"{ARCROOT}/{p.relative_to(ROOT).as_posix()}")
    kb = OUT.stat().st_size // 1024
    print(f"OK: {OUT}  ({len(files)} archivos, {kb} KB)")
    for p in files:
        print("  ", p.relative_to(ROOT).as_posix())
    # check mínimo: los archivos nuevos clave tienen que estar
    nombres = {p.name for p in files}
    assert "db.py" in nombres, "falta src/db.py en el zip"
    assert "SistemasOrdenanza1877.pdf" in nombres, "falta la Ordenanza 1877 en el zip"


if __name__ == "__main__":
    main()
