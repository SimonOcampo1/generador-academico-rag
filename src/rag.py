"""RAG: recupera contexto de Chroma y genera la respuesta con el LLM local.

Flujo: consulta -> retrieval (ingest.buscar) -> prompt con contexto -> qwen2.5 vía Ollama.
Si Ollama no está corriendo, devuelve el contexto recuperado (la parte RAG funciona igual).
Mismo modelo local que el proyecto de notas de cata: OLLAMA_MODEL, default qwen2.5:3b-instruct.
"""
from __future__ import annotations

import os

import requests

from ingest import buscar

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Default 1.5b: equilibrio velocidad/calidad en CPU con poca RAM (8 GB). En una máquina con GPU,
# exportá OLLAMA_MODEL=qwen2.5:7b-instruct para mejor redacción sin penalizar tiempo. Ver README.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
# Mantener el modelo cargado entre pedidos evita el costo de recarga en cada generación.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

SYSTEM = (
    "Sos un asistente académico de la UTN FRLP. Respondé SOLO con la información del "
    "CONTEXTO. Si el contexto no alcanza, decílo; no inventes notas, materias ni "
    "correlatividades. Sé concreto y citá los datos (materia, nota, año) que uses."
)


def ollama_disponible() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def chat(system: str, user: str, temperature: float = 0.4, num_predict: int = 400) -> str | None:
    """Una respuesta del LLM local. None si Ollama no está disponible (modo fallback).

    num_predict acota la longitud de salida: clave en CPU, donde una respuesta sin tope puede
    tardar minutos. 400 tokens alcanzan para los artefactos y mantienen la demo ágil.
    """
    if not ollama_disponible():
        return None
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE,
              "options": {"temperature": temperature, "num_predict": num_predict},
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def generar(consulta: str, k: int = 6) -> dict:
    """Devuelve {'respuesta', 'contexto', 'fuente'} para una consulta (caso QA simple)."""
    contexto = "\n".join(f"- {h['text']}" for h in buscar(consulta, k=k))
    respuesta = chat(SYSTEM, f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {consulta}")
    return {
        "respuesta": respuesta,
        "contexto": contexto,
        "fuente": "ollama" if respuesta is not None else "solo_retrieval",
    }


if __name__ == "__main__":
    out = generar("¿En qué materias de programación le fue mejor y qué le conviene cursar ahora?")
    print("FUENTE:", out["fuente"])
    if out["respuesta"]:
        print("\nRESPUESTA:\n", out["respuesta"])
    else:
        print("\n(Ollama apagado) CONTEXTO recuperado:\n", out["contexto"])
