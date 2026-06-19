"""Capa de análisis: del corpus a un DataFrame de notas de los 6 integrantes del grupo.

Todos los datos son REALES: salen de los estados académicos + exámenes de los 6 integrantes,
parseados por documents.construir_corpus (auto-descubre Estado-Academico-*.pdf y Notas-*.pdf).

Áreas temáticas (para EDA, radar y clustering): mapeo de las 36 obligatorias del plan 2023.
"""
from __future__ import annotations

import pandas as pd

from documents import construir_corpus, normalizar, parsear_correlatividades
from extract import pdf_to_text
from documents import RAW_DIR

# --- Áreas temáticas por Nº de plan (36 obligatorias) ----------------------
AREAS: dict[str, list[int]] = {
    "Matemática y Física": [1, 2, 3, 5, 9, 10, 17, 22],
    "Programación": [6, 13, 14, 19, 20],
    "Sistemas y Gestión": [8, 11, 16, 18, 23, 24, 25, 30, 33, 34, 36],
    "Infraestructura y Redes": [7, 15, 21, 26, 29, 35],
    "Datos e Inteligencia": [27, 28, 31, 32],
    "Idiomas": [4, 12],
}
_NUM_AREA = {num: area for area, nums in AREAS.items() for num in nums}


def _mapa_nombre_area() -> dict[str, str]:
    """{materia_normalizada: area} usando los nombres del plan parseado."""
    plan = parsear_correlatividades(pdf_to_text(RAW_DIR / "Plan-Sistemas-2023.pdf"))
    return {normalizar(d["nombre"]): _NUM_AREA.get(num, "Otras") for num, d in plan.items()}


def area_de(materia: str, _cache: dict = {}) -> str:
    if not _cache:
        _cache.update(_mapa_nombre_area())
    return _cache.get(normalizar(materia), "Otras")


# --- Tabla de estudiantes reales (desde el corpus) -------------------------
def tabla_real() -> pd.DataFrame:
    """Un registro por (alumno, materia) con nota, año y área, solo materias con nota."""
    filas = []
    for d in construir_corpus():
        m = d["metadata"]
        if m["fuente"] != "estado_academico" or m["nota"] < 0:
            continue
        filas.append({
            "alumno": m["alumno"],
            "materia": m["materia"],
            "anio": m["anio"],
            "nota": m["nota"],
            "area": area_de(m["materia"]),
            "sintetico": False,
        })
    return pd.DataFrame(filas)


def tabla_grupo() -> pd.DataFrame:
    """Los 6 integrantes del Grupo 14, todos con datos reales (alias de tabla_real)."""
    return tabla_real()


# --- Agregaciones para EDA -------------------------------------------------
def promedio_por_alumno(df: pd.DataFrame) -> pd.Series:
    return df.groupby("alumno")["nota"].mean().round(2).sort_values(ascending=False)


def perfil_por_area(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz alumno x área con la nota media (features de clustering / radar)."""
    return df.pivot_table(index="alumno", columns="area", values="nota", aggfunc="mean").round(2)


if __name__ == "__main__":
    real = tabla_real()
    assert not real.empty, "tabla real vacía"
    assert real["alumno"].nunique() == 6, real["alumno"].unique()
    assert set(real["alumno"]) >= {"Simón Ocampo", "Santiago Natalichio"}, real["alumno"].unique()
    assert real["nota"].between(1, 10).all(), "notas fuera de rango"
    assert not real["sintetico"].any(), "ya no hay perfiles sintéticos"
    print(f"Reales: {len(real)} registros, {real['alumno'].nunique()} alumnos")
    print(promedio_por_alumno(real).to_string())
    print("\nPerfil por área (nota media):")
    print(perfil_por_area(tabla_grupo()).to_string())
