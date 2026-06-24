"""Evaluación de la generación: ¿la salida se apoya en datos reales del RAG?

Métrica central: GROUNDING = fracción de los tokens informativos de la salida (nombres propios
de materias y notas numéricas) que efectivamente aparecen en el contexto recuperado. Una salida
bien anclada cita el contexto -> grounding alto; una salida que inventa -> grounding bajo.

Es un proxy léxico (no semántico), explicable y suficiente para el TP. La demo estrella compara
el mismo pedido CON RAG (contexto real) vs SIN RAG (sin contexto): se ve que sin el conocimiento
privado el modelo no puede anclar y baja el grounding (inventa o se generaliza).
"""
from __future__ import annotations

import re

from rag import chat

# tokens informativos: palabras Capitalizadas de >=4 letras (nombres de materias) y números 1-10
_PALABRA = re.compile(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{3,}\b")
_NOTA = re.compile(r"\b(?:10|[1-9])\b")
_STOP = {"Para", "Como", "Esto", "Esta", "Este", "Cada", "Pero", "Según"}  # conectores frecuentes


def tokens_informativos(texto: str) -> set[str]:
    pal = {p.lower() for p in _PALABRA.findall(texto) if p not in _STOP}
    return pal | set(_NOTA.findall(texto))


def grounding_score(salida: str, contexto: str) -> float:
    """Fracción de tokens informativos de la salida presentes en el contexto. [0, 1]."""
    s = tokens_informativos(salida)
    if not s:
        return 0.0
    c = tokens_informativos(contexto)
    return round(len(s & c) / len(s), 3)


def _oraciones(texto: str) -> list[str]:
    partes = re.split(r"[.\n;]+", texto or "")
    return [p.strip() for p in partes if len(p.strip()) >= 15]


_EMBED = None  # MiniLM por defecto de Chroma; se carga una sola vez (perezoso)


def grounding_semantico(salida: str, contexto: str) -> float:
    """Grounding por similitud de embeddings: promedio del máximo coseno de cada oración de la
    salida contra los fragmentos del contexto. Detecta paráfrasis legítimas que el léxico penaliza.
    """
    global _EMBED
    import numpy as np

    oraciones, fragmentos = _oraciones(salida), _oraciones(contexto)
    if not oraciones or not fragmentos:
        return 0.0
    if _EMBED is None:
        from chromadb.utils import embedding_functions
        _EMBED = embedding_functions.DefaultEmbeddingFunction()
    emb = np.array(_EMBED(oraciones + fragmentos))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    so, sf = emb[: len(oraciones)], emb[len(oraciones):]
    sim = so @ sf.T  # coseno (vectores normalizados)
    return round(float(sim.max(axis=1).mean()), 3)


# --- Precisión factual: ¿los pares (materia, nota) que afirma son CORRECTOS vs el registro? -------
# Mide corrección contra la verdad (no solapamiento): complementa al grounding. Se extraen pares
# "Materia (9)" / "Materia (nota 9)" / "Materia con un 10" de la salida y se verifican uno a uno.
_PAR_PARENT = re.compile(r"([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ ]{3,45}?)\s*\(\s*(?:nota\s*)?(10|[1-9])\s*\)")
_PAR_CON = re.compile(r"([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ ]{3,45}?)\s+con\s+(?:un[ao]?\s+)?(10|[1-9])\b")


def _pares_materia_nota(salida: str) -> list[tuple[str, int]]:
    pares = []
    for rx in (_PAR_PARENT, _PAR_CON):
        for m in rx.finditer(salida or ""):
            pares.append((m.group(1).strip(), int(m.group(2))))
    return pares


def registro_alumno(alumno: str) -> dict[str, int]:
    """{canon(materia): nota} de las materias aprobadas del alumno (la verdad de referencia).

    Sale de la base relacional: la verdad factual son datos tabulares, no recuperación semántica.
    """
    import db
    return db.registro(alumno)


def _match_claim(mat_canon: str, claves: list[str]) -> str | None:
    """Asocia la materia afirmada a una del registro. La frase capturada suele traer palabras de
    más al principio ('rindió bases de datos'), así que el nombre real es el SUFIJO."""
    from documents import _match_prefijo
    cand = _match_prefijo(mat_canon, claves)  # exacto o truncación del PDF
    if cand:
        return cand
    for k in sorted(claves, key=len, reverse=True):  # el más específico primero
        if len(k) >= 6 and mat_canon.endswith(k):
            return k
    return None


def precision_factual(salida: str, registro: dict[str, int]) -> dict | None:
    """Fracción de pares (materia, nota) afirmados que coinciden con el registro real. None si no
    se afirmó ningún par verificable. Penaliza inventar materias o equivocar la nota."""
    from documents import canon
    pares = _pares_materia_nota(salida)
    if not pares:
        return None
    claves = list(registro)
    ok, errores = 0, []
    for mat, nota in pares:
        k = _match_claim(canon(mat), claves)
        if k is not None and registro[k] == nota:
            ok += 1
        else:
            errores.append({"materia": mat, "nota_dicha": nota, "nota_real": registro.get(k)})
    return {"precision": round(ok / len(pares), 3), "pares": len(pares), "errores": errores}


# --- Faithfulness con LLM como juez: el propio modelo puntúa la fidelidad al contexto ------------
_JUEZ = (
    "Sos un evaluador de sistemas RAG. Puntuás de 0.0 a 1.0 la FIDELIDAD de la RESPUESTA al "
    "CONTEXTO mirando SOLO los DATOS CONCRETOS que afirma (materias, notas, años, correlativas):\n"
    "1.0 = todos los datos concretos que menciona aparecen en el CONTEXTO;\n"
    "0.5 = mayormente apoyada, pero con algún dato concreto que no está en el contexto;\n"
    "0.0 = inventa o contradice datos (materias/notas que no están en el contexto).\n"
    "IMPORTANTE: la interpretación, los consejos y las opiniones NO se penalizan; no exijas que "
    "cada oración sea verificable, penalizá únicamente DATOS inventados. Respondé SOLO el número."
)


def faithfulness_llm(salida: str, contexto: str) -> float | None:
    """Pide al LLM (juez) un puntaje 0.0-1.0 de fidelidad de la salida al contexto. None si no hay LLM."""
    if not (salida or "").strip():
        return None
    out = chat(_JUEZ, f"CONTEXTO:\n{contexto}\n\nRESPUESTA:\n{salida}\n\nPuntaje de fidelidad (0.0-1.0):",
               temperature=0.0, num_predict=8)
    if out is None:
        return None
    m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", out)
    return float(m.group(1)) if m else None


def comparar_con_sin_rag(generador, **kwargs) -> dict:
    """Corre un generador con su contexto RAG y repite el pedido SIN contexto. Compara grounding.

    Requiere Ollama para tener salidas que comparar; si está apagado, devuelve solo el contexto.
    """
    con = generador(**kwargs)
    instruccion = con["prompt"].split("TAREA:", 1)[-1].strip()
    sin_salida = chat(
        "Sos un asistente académico. Respondé el pedido lo mejor que puedas.",
        f"CONTEXTO:\n(sin datos disponibles)\n\nTAREA: {instruccion}",
    )
    ctx = con["contexto"]
    # Registro real del alumno (si el artefacto es por alumno) para la precisión factual.
    registro = registro_alumno(kwargs["alumno"]) if kwargs.get("alumno") else {}
    g_con = grounding_score(con["salida"] or "", ctx)
    g_sin = grounding_score(sin_salida or "", ctx)
    return {
        "artefacto": con.get("artefacto"),
        # Cuánto del contexto inyectado usó el modelo: grounding con RAG = fracción anclada en el
        # contexto; (1 - g_con) ≈ lo que puso de su conocimiento propio; el delta con−sin = aporte del RAG.
        "uso_contexto_con_rag": g_con,
        "conocimiento_propio_con_rag": round(1 - g_con, 3),
        "aporte_contexto": round(g_con - g_sin, 3),
        "con_rag": {"salida": con["salida"],
                    "grounding": grounding_score(con["salida"] or "", ctx),
                    "grounding_sem": grounding_semantico(con["salida"] or "", ctx),
                    "precision_factual": precision_factual(con["salida"] or "", registro) if registro else None,
                    "faithfulness": faithfulness_llm(con["salida"] or "", ctx)},
        "sin_rag": {"salida": sin_salida,
                    "grounding": grounding_score(sin_salida or "", ctx),
                    "grounding_sem": grounding_semantico(sin_salida or "", ctx),
                    "precision_factual": precision_factual(sin_salida or "", registro) if registro else None,
                    "faithfulness": faithfulness_llm(sin_salida or "", ctx)},
        "contexto": ctx,
    }


if __name__ == "__main__":
    contexto = (
        "- Simón Ocampo aprobó Bases de Datos con 9 en el año 3.\n"
        "- Simón Ocampo aprobó Análisis Matemático I con 9 en el año 1."
    )
    anclada = "Simón rindió Bases de Datos con 9 y Análisis Matemático con 9, muy buen desempeño."
    inventada = "Recomiendo cursar Astrofísica Cuántica y Derecho Romano el próximo cuatrimestre."
    g_ok = grounding_score(anclada, contexto)
    g_mal = grounding_score(inventada, contexto)
    print(f"grounding léxico   anclada={g_ok}  inventada={g_mal}")
    assert g_ok > 0.6, f"la salida anclada debería tener grounding alto, fue {g_ok}"
    assert g_mal < 0.3, f"la salida inventada debería tener grounding bajo, fue {g_mal}"

    s_ok = grounding_semantico(anclada, contexto)
    s_mal = grounding_semantico(inventada, contexto)
    print(f"grounding semántico anclada={s_ok}  inventada={s_mal}")
    assert s_ok > s_mal, f"el semántico debería separar anclada ({s_ok}) de inventada ({s_mal})"

    # precisión factual: pares (materia, nota) correctos vs el registro de referencia
    registro = {"bases de datos": 9, "analisis matematico i": 9}
    p_ok = precision_factual("Rindió Bases de Datos con 9 y Análisis Matemático I con 9.", registro)
    p_mal = precision_factual("Tiene Bases de Datos con 5 y Álgebra con 4.", registro)
    print(f"precisión factual  correcta={p_ok['precision']}  con errores={p_mal['precision']}")
    assert p_ok["precision"] == 1.0 and p_ok["pares"] == 2, p_ok
    assert p_mal["precision"] == 0.0, p_mal  # nota equivocada + materia inexistente
    assert precision_factual("No menciona ninguna nota.", registro) is None
    print("OK: groundings y precisión factual distinguen salida anclada de inventada.")
