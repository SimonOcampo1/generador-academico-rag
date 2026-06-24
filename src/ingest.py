"""Documentación de la carrera -> base vectorial Chroma (persistente en data/chroma).

La base vectorial guarda SÓLO texto no estructurado: la prosa del plan de estudios (régimen,
descripciones, párrafos largos). Es donde el matching semántico le gana a un SELECT. Los datos
TABULARES (historial académico y correlatividades) viven en la base relacional (ver db.py); el
agente combina ambas. Antes acá entraba todo el corpus —incluidos los datos tabulares—, lo que
era usar la base vectorial para algo que es un lookup determinístico.

Chroma trae por defecto el embedding all-MiniLM-L6-v2 (el mismo MiniLM que usábamos antes),
así que no hace falta gestionar sentence-transformers aparte: corre local y offline.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

from documents import construir_corpus

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION = "academico"


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION)


def reindexar() -> int:
    """Reconstruye la colección desde cero con la documentación del plan (texto no estructurado).

    Sólo la documentación de la carrera (prosa: 'plan_estudios' = diseño curricular y
    'contenidos_materia' = contenidos mínimos por materia) se vectoriza; el historial académico y
    las correlatividades (tabulares) los carga db.reindexar() en la base relacional.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # no existía
    col = client.get_or_create_collection(COLLECTION)
    VECTOR = {"plan_estudios", "contenidos_materia"}
    docs = [d for d in construir_corpus() if d["metadata"]["fuente"] in VECTOR]
    col.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    return len(docs)


def buscar(consulta: str, k: int = 5, where: dict | None = None) -> list[dict]:
    col = get_collection()
    res = col.query(query_texts=[consulta], n_results=k, where=where)
    return [
        {"text": t, "metadata": m, "distancia": d}
        for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


if __name__ == "__main__":
    n = reindexar()
    print(f"Indexados {n} documentos (prosa del plan) en Chroma ({CHROMA_DIR}).\n")
    # Búsquedas semánticas sobre la DOCUMENTACIÓN del plan (lo no estructurado). Las consultas
    # tabulares (notas de un alumno, correlativas de una materia) las resuelve db.py por SQL.
    for q in [
        "régimen de cursada y modalidad del plan de estudios",
        "perfil profesional del ingeniero en sistemas de información",
        "alcances del título y campo de actuación",
    ]:
        print(f"### {q}")
        for r in buscar(q, k=3):
            print(f"  [{r['distancia']:.3f}] ({r['metadata']['fuente']}) {r['text'][:120]}")
        print()
