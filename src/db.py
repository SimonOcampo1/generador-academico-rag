"""Base de datos RELACIONAL (SQLite) para los datos TABULARES del sistema.

Por qué relacional y no vectorial: el historial académico (alumno, materia, nota, año, estado)
y las correlatividades (Anexo I del plan: Nº, materia, correlativas para cursar/rendir) son
registros estructurados. Una base relacional los modela y los consulta con exactitud (filtros,
joins, agregaciones); meterlos en una base vectorial sería usar similitud semántica para algo
que es un lookup determinístico. La DOCUMENTACIÓN de la carrera (prosa del plan), que sí es
texto largo, va en la base vectorial (ver ingest.py). El agente COMBINA ambas para su contexto.

sqlite3 es stdlib: cero dependencias nuevas, archivo único y persistente en data/academico.db.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from documents import construir_corpus

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "academico.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def reindexar() -> dict:
    """(Re)construye la base relacional desde el corpus parseado. Devuelve conteos por tabla.

    El corpus (documents.construir_corpus) trae tres fuentes; acá sólo se persisten las dos
    tabulares (estado académico y correlatividades). La prosa del plan va a Chroma (ingest.py).
    """
    docs = construir_corpus()
    historia = [d["metadata"] for d in docs if d["metadata"]["fuente"] == "estado_academico"]
    corr = [d["metadata"] for d in docs if d["metadata"]["fuente"] == "correlatividades"]
    with _conn() as c:
        c.executescript(
            """
            DROP TABLE IF EXISTS historia;
            DROP TABLE IF EXISTS correlatividades;
            CREATE TABLE historia (
                alumno TEXT, materia TEXT, codigo TEXT,
                anio INTEGER, nota INTEGER, estado TEXT
            );
            CREATE TABLE correlatividades (
                num_plan INTEGER PRIMARY KEY, materia TEXT, cursar TEXT, rendir TEXT
            );
            """
        )
        c.executemany(
            "INSERT INTO historia (alumno, materia, codigo, anio, nota, estado) VALUES (?,?,?,?,?,?)",
            [(m["alumno"], m["materia"], m["codigo"], m["anio"], m["nota"], m["estado"]) for m in historia],
        )
        c.executemany(
            "INSERT INTO correlatividades (num_plan, materia, cursar, rendir) VALUES (?,?,?,?)",
            [(m["num_plan"], m["materia"], m["cursar"], m["rendir"]) for m in corr],
        )
    return {"historia": len(historia), "correlatividades": len(corr)}


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


# --- Consultas (lo que antes resolvía ingest.obtener con filtros de metadata) ---
def historia(alumno: str) -> list[dict]:
    """Todas las filas del historial de un alumno (aprobadas, en curso, no iniciadas)."""
    return _rows("SELECT * FROM historia WHERE alumno = ?", (alumno,))


def aprobadas(alumno: str) -> list[dict]:
    """Materias con nota (nota >= 0): el rendimiento real del alumno."""
    return _rows("SELECT * FROM historia WHERE alumno = ? AND nota >= 0", (alumno,))


def correlativas() -> list[dict]:
    """Las 36 obligatorias del plan con sus correlativas (para cursar / para rendir)."""
    return _rows("SELECT * FROM correlatividades ORDER BY num_plan")


def alumnos() -> list[str]:
    return [r["alumno"] for r in _rows("SELECT DISTINCT alumno FROM historia ORDER BY alumno")]


def todas_aprobadas() -> list[dict]:
    """Todas las filas con nota de todos los alumnos (para el EDA en analisis.py)."""
    return _rows("SELECT * FROM historia WHERE nota >= 0")


def registro(alumno: str) -> dict[str, int]:
    """{canon(materia): nota} de las aprobadas del alumno (verdad de referencia para la eval)."""
    from documents import canon
    return {canon(r["materia"]): r["nota"] for r in aprobadas(alumno)}


def existe() -> bool:
    """True si la base relacional ya está construida y cargada."""
    if not DB_PATH.exists():
        return False
    try:
        return _rows("SELECT count(*) AS n FROM historia")[0]["n"] > 0
    except sqlite3.Error:
        return False


def asegurar() -> None:
    """Construye la base relacional si todavía no existe (idempotente)."""
    if not existe():
        reindexar()


if __name__ == "__main__":
    n = reindexar()
    print(f"Base relacional construida en {DB_PATH}")
    print(f"  historia: {n['historia']} filas · correlatividades: {n['correlatividades']} filas")
    assert n["correlatividades"] == 36, f"se esperaban 36 obligatorias, hay {n['correlatividades']}"

    al = alumnos()
    assert len(al) == 6, f"se esperaban 6 alumnos, hay {len(al)}: {al}"
    simon = next(a for a in al if a.startswith("Sim"))
    ap = aprobadas(simon)
    assert ap and all(r["nota"] >= 0 for r in ap), "aprobadas debería traer solo filas con nota"
    reg = registro(simon)
    assert reg and all(isinstance(v, int) for v in reg.values()), "registro mal formado"
    print(f"  {len(al)} alumnos · {len(ap)} aprobadas de {simon}")
    print("OK: la base relacional responde historial y correlatividades.")
