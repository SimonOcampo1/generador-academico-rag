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
    return {
        "artefacto": con.get("artefacto"),
        "con_rag": {"salida": con["salida"],
                    "grounding": grounding_score(con["salida"] or "", ctx),
                    "grounding_sem": grounding_semantico(con["salida"] or "", ctx)},
        "sin_rag": {"salida": sin_salida,
                    "grounding": grounding_score(sin_salida or "", ctx),
                    "grounding_sem": grounding_semantico(sin_salida or "", ctx)},
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
    print("OK: ambos groundings distinguen salida anclada de inventada.")
