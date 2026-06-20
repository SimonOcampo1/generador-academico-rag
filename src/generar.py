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
from documents import canon
from ingest import buscar, obtener
from rag import chat

TONOS = {
    "tecnico": "Registro técnico, preciso y sobrio. Sin adornos.",
    "motivacional": "Registro cálido y motivador, en segunda persona, que destaque logros.",
    "honesto": "Registro directo y honesto: señalá fortalezas pero también debilidades sin rodeos.",
}

_SYSTEM = (
    "Sos un asesor académico de la UTN FRLP (carrera Ingeniería en Sistemas de Información). "
    "Trabajás SOLO con los datos del CONTEXTO (notas, materias, años, correlatividades reales): "
    "no inventes ni supongas datos que no estén ahí. "
    "Pero no sos un loro de datos: INTERPRETÁ y SELECCIONÁ. Elegí los 3-5 datos más relevantes al "
    "objetivo y explicá QUÉ revelan (una fortaleza, un patrón, una afinidad); nunca enumeres todas "
    "las notas ni repitas el listado completo. Cuando cites un dato, que sea para sostener una idea. "
    "Escribí en español rioplatense, claro y profesional. Usá markdown con criterio: **negrita** "
    "para lo clave y listas solo cuando aporten."
)


# --- Contexto (grounding) --------------------------------------------------
def _aprobadas(alumno: str) -> list[dict]:
    docs = obtener({"$and": [{"fuente": "estado_academico"}, {"alumno": alumno}]})
    return [d for d in docs if d["metadata"]["nota"] >= 0]


def _correlativas() -> list[dict]:
    return obtener({"fuente": "correlatividades"})


def _ctx_aprobadas_compacto(docs: list[dict]) -> str:
    """Aprobadas en una línea por materia ('Materia (nota, año N)'), no la prosa verbosa.

    En CPU el costo dominante es procesar el contexto; compactarlo acelera la generación sin
    perder los datos que el grounding necesita (nombre de materia + nota).
    """
    filas = sorted(docs, key=lambda d: (d["metadata"]["anio"] or 99, d["metadata"]["materia"]))

    def fmt(d: dict) -> str:
        md = d["metadata"]
        ano = f", año {md['anio']}" if md["anio"] else ""  # las del acta de notas no traen año
        return f"- {md['materia']} (nota {md['nota']}{ano})"

    return "\n".join(fmt(d) for d in filas)


def _ctx_estado_rico(docs: list[dict]) -> str:
    """Como el compacto pero con el código de materia: más strings concretos para anclar.

    Saca el ruido (Tomo/Folio/horas del PDF) pero conserva código + nota + año, que es lo que el
    modelo copia para mantenerse anclado en datos reales (sube el grounding sin ensuciar la prosa).
    """
    filas = sorted(docs, key=lambda d: (d["metadata"]["anio"] or 99, d["metadata"]["materia"]))

    def fmt(d: dict) -> str:
        md = d["metadata"]
        cod = f"código {md['codigo']}, " if md["codigo"] else ""
        ano = f", año {md['anio']}" if md["anio"] else ""
        return f"- {md['materia']} ({cod}nota {md['nota']}{ano})"

    return "\n".join(fmt(d) for d in filas)


def _cursables(alumno: str) -> list[dict]:
    """Materias que el alumno YA puede cursar, derivadas del grafo de correlatividades.

    Retrieval determinístico (parte del método RAG, no un atajo): una materia es cursable si
    NO está aprobada y TODAS sus correlativas 'para cursar' (Anexo I del plan) ya están aprobadas.
    Devuelve, por materia, qué correlativas la habilitan (para que el modelo justifique sobre datos
    reales, no sobre suposiciones). La priorización y la prosa las genera el modelo.
    """
    aprob_canon = {canon(d["metadata"]["materia"]) for d in _aprobadas(alumno)}
    # Materias que el alumno está cursando AHORA (sin nota todavía, estado "Cursa en..."): no son
    # candidatas a "cursar el próximo cuatrimestre" -> sin esto se recomendaban materias en curso.
    en_curso_canon = {
        canon(d["metadata"]["materia"])
        for d in obtener({"$and": [{"fuente": "estado_academico"}, {"alumno": alumno}]})
        if str(d["metadata"].get("estado", "")).startswith("Cursa")
    }
    excluir = aprob_canon | en_curso_canon
    corr = _correlativas()
    num_a_nombre = {int(d["metadata"]["num_plan"]): d["metadata"]["materia"] for d in corr}
    # Para CURSAR una materia hace falta tener CURSADAS sus correlativas (no aprobadas). Una materia
    # en curso estará cursada el próximo cuatrimestre -> cuenta para habilitar la siguiente.
    cursadas_nums = {n for n, nom in num_a_nombre.items() if canon(nom) in excluir}
    cursables: list[dict] = []
    for d in corr:
        md = d["metadata"]
        if canon(md["materia"]) in excluir:
            continue  # ya aprobada o cursándose ahora: no es candidata a cursar
        req = [int(x) for x in md["cursar"].split(",") if x]
        if all(n in cursadas_nums for n in req):  # todas las correlativas para cursarla, cumplidas
            cursables.append({
                "materia": md["materia"], "num": int(md["num_plan"]),
                "habilitada_por": [num_a_nombre.get(n, f"materia {n}") for n in req],
            })
    return sorted(cursables, key=lambda c: c["num"])


def _electivas_disponibles(alumno: str) -> list[str]:
    """Electivas del plan que el alumno NO inició (ni aprobó ni cursa): opciones a futuro.

    Las electivas no están en la tabla de correlatividades (solo las 36 obligatorias), así que se
    toman del propio estado académico: materias no iniciadas que no son obligatorias.
    """
    obligatorias = {canon(d["metadata"]["materia"]) for d in _correlativas()}
    out: list[str] = []
    for d in obtener({"$and": [{"fuente": "estado_academico"}, {"alumno": alumno}]}):
        md = d["metadata"]
        est = str(md.get("estado", ""))
        if md["nota"] >= 0 or est.startswith("Cursa") or est.startswith("Aprobada"):
            continue  # aprobada, en curso, o aprobada por equivalencia: no es "disponible a futuro"
        if canon(md["materia"]) not in obligatorias:  # no es de las 36 -> electiva
            out.append(md["materia"])
    return sorted(set(out))


def _contexto_plan(alumno: str) -> str:
    """Contexto del plan: rendimiento real + obligatorias habilitadas + electivas disponibles."""
    aprob = _aprobadas(alumno)
    cursables = _cursables(alumno)
    if cursables:
        filas = "\n".join(
            f"- {c['materia']}"
            + (f" (habilitada por: {', '.join(c['habilitada_por'])})" if c["habilitada_por"]
               else " (sin correlativas para cursar)")
            for c in cursables
        )
    else:
        filas = "- (no quedan obligatorias nuevas habilitadas: están aprobadas o en curso)"

    electivas = _electivas_disponibles(alumno)
    bloque_elec = ""
    if electivas:
        bloque_elec = (
            "\n\nELECTIVAS del plan aún disponibles (no iniciadas; el alumno puede elegirlas):\n"
            + "\n".join(f"- {e}" for e in electivas)
        )
    return (
        f"RENDIMIENTO de {alumno} (materias aprobadas con su nota y año):\n"
        f"{_ctx_aprobadas_compacto(aprob)}\n\n"
        f"OBLIGATORIAS QUE {alumno} YA PUEDE CURSAR el próximo cuatrimestre "
        f"(verificado contra las correlatividades del plan):\n{filas}{bloque_elec}"
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
        f"Armá el plan de cursada del próximo cuatrimestre para {alumno}.\n"
        "Reglas duras: elegí SOLO materias listadas en 'OBLIGATORIAS QUE YA PUEDE CURSAR' y/o en "
        "'ELECTIVAS disponibles'; jamás incluyas materias aprobadas, en curso o fuera de esas listas.\n"
        "Tarea: recomendá un orden de prioridad (no hace falta usar todas). Priorizá las OBLIGATORIAS "
        "habilitadas (destraban la carrera) y, si hay cupo, sumá 1-2 ELECTIVAS afines a sus fortalezas. "
        "Para cada materia, una justificación corta que combine: (1) afinidad con las áreas donde mejor "
        "rinde, (2) qué destraba o qué perfil refuerza. Cerrá con una sugerencia de carga realista. "
        f"{TONOS.get(tono, '')}"
    )
    return {"artefacto": "plan_cursada", "alumno": alumno, "tono": tono, **_generar(_SYSTEM, contexto, instr)}


def informe_trayectoria(alumno: str, tono: str = "motivacional") -> dict:
    contexto = _ctx_estado_rico(_aprobadas(alumno))
    instr = (
        f"Escribí un informe de la trayectoria académica de {alumno}, en prosa narrativa (no un "
        "listado).\n"
        "Cubrí tres cosas: (1) la evolución a lo largo de los años (cómo arrancó y cómo progresó), "
        "(2) sus áreas más fuertes y las más flojas, INTERPRETANDO qué dicen las notas sobre su "
        "perfil —no las recites—, (3) un consejo final accionable y concreto. Usá pocas notas, las "
        "más ilustrativas, como evidencia de lo que afirmás. "
        f"{TONOS.get(tono, '')}"
    )
    return {"artefacto": "informe_trayectoria", "alumno": alumno, "tono": tono, **_generar(_SYSTEM, contexto, instr, 0.6)}


def carta_pasantia(alumno: str, objetivo: str = "una pasantía en Ciencia de Datos", tono: str = "tecnico") -> dict:
    contexto = _ctx_estado_rico(_aprobadas(alumno))
    instr = (
        f"Escribí, en PRIMERA PERSONA (la voz de {alumno}), una carta de presentación para postularse "
        f"a {objetivo}.\n"
        "Estructura: (1) saludo y una frase de interés genuino por el puesto; (2) el cuerpo: elegí 2 o "
        "3 fortalezas REALMENTE relevantes a ese objetivo y conectá cada una con la competencia que el "
        "puesto pide (ej.: una materia o área fuerte → una habilidad concreta que aporta). NO enumeres "
        "el historial ni listes notas año por año; mencioná a lo sumo una o dos notas como respaldo; "
        "(3) cierre breve con disponibilidad y agradecimiento.\n"
        "Que suene a una persona, no a un currículum. Máximo ~230 palabras. "
        f"{TONOS.get(tono, '')}"
    )
    return {"artefacto": "carta_pasantia", "alumno": alumno, "objetivo": objetivo, "tono": tono,
            **_generar(_SYSTEM, contexto, instr, 0.6)}


def recomendar_orientacion(alumno: str, tono: str = "tecnico") -> dict:
    """Recomienda una rama/especialización de la carrera anclada en el promedio por área real.

    Todos cursan la misma carrera: la pregunta no es 'qué carrera' sino en qué ÁREA temática
    (las de analisis.AREAS) conviene perfilarse según dónde rinde mejor. Se le da al modelo el
    promedio por área + ejemplos de materias, y se le pide elegir y comparar entre esas áreas.
    """
    df = analisis.tabla_real()
    sub = df[df["alumno"] == alumno]
    agg = (sub[sub["area"] != "Otras"].groupby("area")["nota"]
           .agg(prom="mean", n="count").sort_values("prom", ascending=False))
    lineas = []
    for area, r in agg.iterrows():
        top = sub[sub["area"] == area].sort_values("nota", ascending=False).head(3)
        ej = ", ".join(f"{t.materia} ({t.nota})" for t in top.itertuples())
        lineas.append(f"- {area}: promedio {r.prom:.1f} en {int(r.n)} materias. Mejores: {ej}")
    contexto = (
        "Desempeño de " + alumno + " por área temática de la carrera (de mejor a peor promedio). "
        "Es EVIDENCIA de sus fortalezas, no la lista de respuestas:\n" + "\n".join(lineas)
    )
    instr = (
        f"Recomendá a {alumno} hacia qué especialización o rama profesional de la Ingeniería en "
        "Sistemas le conviene orientarse.\n"
        "Cómo razonar: identificá en qué áreas es más fuerte según la evidencia, y traducí esas "
        "fortalezas a especializaciones REALES del mundo laboral del software (las que vos conocés: "
        "p. ej. desarrollo, datos, infraestructura, gestión, etc. — elegí las que de verdad encajen, "
        "no las nombres todas). Recomendá 1 o 2.\n"
        "Para cada una: nombrala, conectala con las materias/áreas concretas que la respaldan, y "
        "describí qué tipo de trabajo o perfil implica. Interpretá los datos, no los repitas; y sé "
        "honesto: no sugieras una rama que su desempeño no sostenga. "
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
        "Para un proyecto de software en equipo, asigná a CADA integrante un rol concreto (p. ej. "
        "líder técnico, backend, datos/ML, infra, QA, análisis funcional) según su área más fuerte. "
        "Justificá cada asignación con el número que la respalda. Evitá repetir el mismo rol salvo "
        "que los datos lo justifiquen, y buscá que el equipo quede balanceado. "
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
        f"Explicá en prosa qué impacto tiene el escenario en el promedio de {alumno}: no solo el número "
        "nuevo, sino qué SIGNIFICA (mejora/empeora, cuánto pesa, si conviene priorizar esas materias). "
        "Apoyate en los números del contexto y dale una lectura útil para decidir. "
        f"{TONOS.get(tono, '')}"
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

    # notas lejos de cualquier promedio para que el escenario siempre mueva el promedio
    w = simulacion_whatif("Simón Ocampo", {"Refuerzo I": 4, "Refuerzo II": 4})
    assert w["prom_nuevo"] != w["prom_actual"], "el what-if no cambió el promedio"
    print(f"[whatif] {w['prom_actual']} -> {w['prom_nuevo']}")

    g = diagnostico_grupal()
    assert "(sim.)" not in g["contexto"], "ya no hay perfiles simulados; todos son reales"
    assert g["contexto"].count("->") == 6, "deberían ser 6 integrantes"
    print(f"[diagnostico_grupal] 6 integrantes reales, fuente={g['fuente']}")

    if r["salida"]:
        print("\n--- ejemplo de salida generada (plan_cursada) ---\n", r["salida"][:600])
