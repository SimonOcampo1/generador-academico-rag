"""Documentos -> base vectorial Chroma (persistente en data/chroma).

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
    """Reconstruye la colección desde cero con el corpus actual."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # no existía
    col = client.get_or_create_collection(COLLECTION)
    docs = construir_corpus()
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


def obtener(where: dict) -> list[dict]:
    """Trae documentos por filtro de metadata (grounding determinístico, sin semántica)."""
    res = get_collection().get(where=where)
    return [{"text": t, "metadata": m} for t, m in zip(res["documents"], res["metadatas"])]


if __name__ == "__main__":
    n = reindexar()
    print(f"Indexados {n} documentos en Chroma ({CHROMA_DIR}).\n")
    for q in [
        "¿Qué nota se sacó en Bases de Datos?",
        "¿Qué necesito para cursar Ciencia de Datos?",
        "materias de programación que aprobó",
    ]:
        print(f"### {q}")
        for r in buscar(q, k=3):
            print(f"  [{r['distancia']:.3f}] ({r['metadata']['fuente']}) {r['text'][:120]}")
        print()
