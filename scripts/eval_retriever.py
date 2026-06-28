"""Evaluación del RETRIEVER (base vectorial): hit-rate@k y MRR.

Mide precisión de RECUPERACIÓN, no de generación: no usa el LLM ni GPU. Para cada consulta-tema
(redactada SIN nombrar la materia, para que el match sea semántico y no léxico), comprueba si la
ficha de la materia esperada aparece en el top-k de la base vectorial.

- hit-rate@k = fracción de consultas cuya materia esperada está entre los k primeros resultados.
- MRR       = media del inverso del rango del primer acierto (1/rank); 0 si no aparece en el top-k.

Uso:  python scripts/eval_retriever.py     (requiere la base vectorial construida: src/ingest.py)
Salida: data/eval_retriever.json + tabla por consola.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from documents import canon  # noqa: E402
from ingest import buscar  # noqa: E402

# Gold set: (consulta por TEMA — sin el nombre de la materia, evita el match léxico trivial; materia esperada)
GOLD: list[tuple[str, str]] = [
    ("control de versiones, pruebas de software, aseguramiento y modelos de calidad", "Ingeniería y Calidad de Software"),
    ("gestión de riesgos, auditoría, marco normativo y peritaje informático forense", "Seguridad en los Sistemas de Información"),
    ("modelo relacional, normalización de tablas, lenguaje de consultas y transacciones", "Bases de Datos"),
    ("límites, derivadas e integrales de funciones de una variable", "Análisis Matemático I"),
    ("protocolos de red, modelo de capas, ruteo y conmutación de paquetes", "Redes de Datos"),
    ("procesos, planificación de CPU, memoria virtual y sistemas de archivos", "Sistemas Operativos"),
    ("aprendizaje automático, modelos predictivos y análisis de grandes volúmenes de datos", "Ciencia de Datos"),
    ("agentes inteligentes, búsqueda heurística y representación del conocimiento", "Inteligencia Artificial"),
    ("vectores, matrices, espacios vectoriales y transformaciones lineales", "Álgebra y Geometría Analítica"),
    ("patrones de diseño, arquitectura de software y experiencia de usuario", "Diseño de Sistemas de Información"),
    ("variables aleatorias, distribuciones de probabilidad e inferencia estadística", "Probabilidad y Estadística"),
    ("paradigma funcional, orientado a objetos y lógico de la programación", "Paradigmas de Programación"),
]

K = 5


def rank_de_acierto(consulta: str, esperada: str, k: int = K) -> int:
    """Rango (1-based) de la primera ficha cuya materia es la esperada; 0 si no está en el top-k."""
    hits = buscar(consulta, k=k, where={"fuente": "contenidos_materia"})
    objetivo = canon(esperada)
    for i, h in enumerate(hits, start=1):
        if canon(h["metadata"].get("materia", "")) == objetivo:
            return i
    return 0


def main() -> None:
    detalle = []
    for consulta, esperada in GOLD:
        r = rank_de_acierto(consulta, esperada)
        detalle.append({"consulta": consulta, "esperada": esperada, "rank": r})

    n = len(detalle)
    hit1 = sum(1 for d in detalle if d["rank"] == 1) / n
    hit3 = sum(1 for d in detalle if 1 <= d["rank"] <= 3) / n
    hit5 = sum(1 for d in detalle if 1 <= d["rank"] <= 5) / n
    mrr = sum((1 / d["rank"]) if d["rank"] else 0 for d in detalle) / n

    resumen = {"n": n, "k": K, "hit_rate@1": round(hit1, 3), "hit_rate@3": round(hit3, 3),
               "hit_rate@5": round(hit5, 3), "mrr": round(mrr, 3)}

    out = ROOT / "data" / "eval_retriever.json"
    out.write_text(json.dumps({"resumen": resumen, "detalle": detalle}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"Retriever sobre {n} consultas-tema (top-{K}, fichas de materia):")
    print(f"  hit-rate@1 = {hit1:.2f}   hit-rate@3 = {hit3:.2f}   hit-rate@5 = {hit5:.2f}   MRR = {mrr:.3f}\n")
    for d in detalle:
        marca = "ok" if d["rank"] == 1 else (f"#{d['rank']}" if d["rank"] else "MISS")
        print(f"  [{marca:>4}] {d['esperada']}")
    print(f"\nGuardado: {out}")
    # self-check: un retriever razonable acierta la mayoría dentro del top-5
    assert hit5 >= 0.7, f"hit-rate@5 inesperadamente bajo ({hit5}); ¿está construida la base vectorial?"


if __name__ == "__main__":
    main()
