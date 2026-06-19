"""Generadores de artefactos académicos sobre el pipeline RAG.

Cada artefacto ancla la generación en contexto recuperado de la base vectorial (grounding):
- determinístico por metadata (las materias aprobadas del alumno, las 36 correlatividades),
- + semántico cuando hace falta (ingest.buscar).

El LLM (rag.chat) recibe SOLO ese contexto. Si Ollama está apagado, `salida` es None y queda
el contexto + prompt listos (la parte RAG funciona igual; la generación se completa con Ollama).

Tono controlable: mismo dato, distinto registro -> esto es lo que lo hace generativo y no un SQL.
"""
from __future__ import annotations

import analisis
from documents import normalizar
from ingest import buscar, obtener
from rag import chat

TONOS = {
    "tecnico": "Registro técnico, preciso y sobrio. Sin adornos.",
    "motivacional": "Registro cálido y motivador, en segunda persona, que destaque logros.",
    "honesto": "Registro directo y honesto: señalá fortalezas pero también debilidades sin rodeos.",
}

_SYSTEM = (
    "Sos un asistente académico de la UTN FRLP. Generás artefactos personalizados usando "
    "EXCLUSIVAMENTE los datos del CONTEXTO (notas, materias, años, correlatividades reales). "
    "No inventes notas ni materias; si un dato no está en el contexto, no lo afirmes. "
    "Citá los datos concretos que uses."
)


# --- Contexto (grounding) --------------------------------------------------
def _aprobadas(alumno: str) -> list[dict]:
    docs = obtener({"$and": [{"fuente": "estado_academico"}, {"alumno": alumno}]})
    return [d for d in docs if d["metadata"]["nota"] >= 0]


def _correlativas() -> list[dict]:
    return obtener({"fuente": "correlatividades"})


def _ctx(docs: list[dict]) -> str:
    return "\n".join(f"- {d['text']}" for d in docs)


def _ctx_aprobadas_compacto(docs: list[dict]) -> str:
    """Aprobadas en una línea por materia ('Materia (nota, año N)'), no la prosa verbosa.

    En CPU el costo dominante es procesar el contexto; compactarlo acelera la generación sin
    perder los datos que el grounding necesita (nombre de materia + nota).
    """
    filas = sorted(docs, key=lambda d: (d["metadata"]["anio"], d["metadata"]["materia"]))
    return "\n".join(
        f"- {d['metadata']['materia']} (nota {d['metadata']['nota']}, año {d['metadata']['anio']})"
        for d in filas
    )


def _contexto_plan(alumno: str) -> str:
    """Contexto mínimo para el plan: aprobadas compactas + correlativas SOLO de lo no aprobado."""
    aprob = _aprobadas(alumno)
    ya = {normalizar(d["metadata"]["materia"]) for d in aprob}
    pendientes = [d for d in _correlativas() if normalizar(d["metadata"]["materia"]) not in ya]
    return (
        f"MATERIAS APROBADAS por {alumno}:\n{_ctx_aprobadas_compacto(aprob)}\n\n"
        f"CORRELATIVIDADES de las materias que aún NO aprobó:\n{_ctx(pendientes)}"
    )


def _generar(system: str, contexto: str, instruccion: str, temperatura: float = 0.4) -> dict:
    prompt = f"CONTEXTO:\n{contexto}\n\nTAREA: {instruccion}"
    salida = chat(system, prompt, temperature=temperatura)
    return {
        "salida": salida,
        "contexto": contexto,
        "prompt": prompt,
        "fuente": "ollama" if salida is not None else "solo_retrieval",
    }


# --- Artefactos ------------------------------------------------------------
def plan_cursada(alumno: str, tono: str = "tecnico") -> dict:
    contexto = _contexto_plan(alumno)
    instr = (
        f"Armá el plan de cursada del próximo cuatrimestre para {alumno}. Listá qué materias "
        "PUEDE cursar (correlativas para cursar ya cumplidas según las aprobadas del contexto) y "
        "cuáles conviene PRIORIZAR, justificando en prosa con su rendimiento y las correlativas. "
        f"{TONOS.get(tono, '')}"
    )
    return {"artefacto": "plan_cursada", "alumno": alumno, "tono": tono, **_generar(_SYSTEM, contexto, instr)}


def informe_trayectoria(alumno: str, tono: str = "motivacional") -> dict:
    contexto = _ctx(_aprobadas(alumno))
    instr = (
        f"Escribí un informe narrativo de la trayectoria académica de {alumno}: evolución por año, "
        "fortalezas y debilidades por área, y un consejo final. Apoyate en notas y materias reales. "
        f"{TONOS.get(tono, '')}"
    )
    return {"artefacto": "informe_trayectoria", "alumno": alumno, "tono": tono, **_generar(_SYSTEM, contexto, instr, 0.6)}


def carta_pasantia(alumno: str, objetivo: str = "una pasantía en Ciencia de Datos", tono: str = "tecnico") -> dict:
    contexto = _ctx(_aprobadas(alumno))
    instr = (
        f"Redactá una carta de motivación de {alumno} para {objetivo}. Seleccioná y destacá las "
        "materias y notas del contexto más relevantes al objetivo. Formato carta, 2-3 párrafos. "
        f"{TONOS.get(tono, '')}"
    )
    return {"artefacto": "carta_pasantia", "alumno": alumno, "objetivo": objetivo, "tono": tono,
            **_generar(_SYSTEM, contexto, instr, 0.6)}


def recomendar_orientacion(alumno: str, tono: str = "tecnico") -> dict:
    """Matching semántico: cruza las materias donde mejor le fue con el espacio de materias."""
    mejores = sorted(_aprobadas(alumno), key=lambda d: d["metadata"]["nota"], reverse=True)[:6]
    consulta = " ".join(d["metadata"]["materia"] for d in mejores)
    vecinas = buscar(consulta, k=6, where={"fuente": "correlatividades"})
    contexto = _ctx(mejores) + "\n\nMaterias semánticamente afines a sus fortalezas:\n" + \
        "\n".join(f"- {h['text']}" for h in vecinas)
    instr = (
        f"Recomendá una orientación/perfil profesional para {alumno} a partir de las materias donde "
        "mejor rindió y sus afines. Justificá con los datos. No inventes electivas que no figuren. "
        f"{TONOS.get(tono, '')}"
    )
    return {"artefacto": "recomendar_orientacion", "alumno": alumno, "tono": tono,
            **_generar(_SYSTEM, contexto, instr)}


def diagnostico_grupal(tono: str = "tecnico") -> dict:
    """Reparte roles de un proyecto según las fortalezas reales de los 6 integrantes."""
    perfil = analisis.perfil_por_area(analisis.tabla_grupo())
    contexto = "Perfil por área (nota media de cada integrante):\n"
    for alumno, row in perfil.iterrows():
        areas = ", ".join(f"{a}: {v:.1f}" for a, v in row.dropna().items() if a != "Otras")
        contexto += f"- {alumno} -> {areas}\n"
    instr = (
        "Para un proyecto de software en equipo, asigná un rol a cada integrante según sus "
        "fortalezas por área, justificando con los números del contexto. "
        f"{TONOS.get(tono, '')}"
    )
    return {"artefacto": "diagnostico_grupal", "tono": tono, **_generar(_SYSTEM, contexto, instr)}


def simulacion_whatif(alumno: str, materias_nota: dict[str, int], tono: str = "tecnico") -> dict:
    """Proyecta el promedio si aprobara ciertas materias con ciertas notas."""
    df = analisis.tabla_real()
    actuales = df[df["alumno"] == alumno]["nota"].tolist()
    proyectado = actuales + list(materias_nota.values())
    prom_actual = sum(actuales) / len(actuales) if actuales else 0
    prom_nuevo = sum(proyectado) / len(proyectado) if proyectado else 0
    contexto = (
        f"{alumno} tiene promedio actual {prom_actual:.2f} sobre {len(actuales)} materias con nota.\n"
        f"Escenario: aprueba {', '.join(f'{m} con {n}' for m, n in materias_nota.items())}.\n"
        f"Promedio proyectado: {prom_nuevo:.2f} sobre {len(proyectado)} materias."
    )
    instr = (
        f"Explicá en prosa el impacto del escenario en el promedio de {alumno} y qué implica para su "
        f"avance. Usá los números del contexto. {TONOS.get(tono, '')}"
    )
    return {"artefacto": "simulacion_whatif", "alumno": alumno, "tono": tono,
            "prom_actual": round(prom_actual, 2), "prom_nuevo": round(prom_nuevo, 2),
            **_generar(_SYSTEM, contexto, instr)}


if __name__ == "__main__":
    from rag import ollama_disponible
    print("Ollama:", "UP" if ollama_disponible() else "DOWN (modo fallback: contexto sin generación)")

    r = plan_cursada("Simón Ocampo")
    assert "Bases de Datos" in r["contexto"], "el contexto no trae las correlativas"
    assert "Simón Ocampo" in r["contexto"], "el contexto no trae las aprobadas del alumno"
    print(f"\n[plan_cursada] fuente={r['fuente']} · contexto {len(r['contexto'])} chars")

    w = simulacion_whatif("Simón Ocampo", {"Ciencia de Datos": 9, "Inteligencia Artificial": 8})
    assert w["prom_nuevo"] != w["prom_actual"], "el what-if no cambió el promedio"
    print(f"[whatif] {w['prom_actual']} -> {w['prom_nuevo']}")

    g = diagnostico_grupal()
    assert "(sim.)" not in g["contexto"], "ya no hay perfiles simulados; todos son reales"
    assert g["contexto"].count("->") == 6, "deberían ser 6 integrantes"
    print(f"[diagnostico_grupal] 6 integrantes reales, fuente={g['fuente']}")

    if r["salida"]:
        print("\n--- ejemplo de salida generada (plan_cursada) ---\n", r["salida"][:600])
