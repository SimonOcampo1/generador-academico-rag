"""RAG: recupera contexto de Chroma y genera la respuesta con el LLM local.

Flujo: consulta -> retrieval (ingest.buscar) -> prompt con contexto -> Phi-4-mini vía Ollama.
Si Ollama no está corriendo, devuelve el contexto recuperado (la parte RAG funciona igual).
Modelo local configurable por OLLAMA_MODEL; default phi4-mini (3.8B).
"""
from __future__ import annotations

import os

import requests

from ingest import buscar

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Default phi4-mini (3.8B): mejor redacción y grounding léxico que modelos más chicos, sigue
# corriendo en CPU. En una máquina con GPU se puede exportar un modelo más grande. Ver README.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi4-mini")
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


# Opciones de generación compartidas por la web y por chat() para que las salidas sean idénticas.
# num_predict: tope de longitud. 400 cortaba a mitad de oración los artefactos narrativos; pero con
# modelos chicos un tope alto los hace DIVAGAR/loopear en vez de aportar. 512 deja terminar la prosa
# sin darle cuerda para repetirse.
# repeat_penalty 1.15 (>1.1 default) desalienta el reinicio literal del texto en modelos chicos.
NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "512"))
REPEAT_PENALTY = float(os.environ.get("OLLAMA_REPEAT_PENALTY", "1.15"))


def gen_options(temperature: float = 0.4, num_predict: int = NUM_PREDICT) -> dict:
    return {"temperature": temperature, "num_predict": num_predict, "repeat_penalty": REPEAT_PENALTY}


def chat(system: str, user: str, temperature: float = 0.4, num_predict: int = NUM_PREDICT) -> str | None:
    """Una respuesta del LLM local. None si Ollama no está disponible (modo fallback)."""
    if not ollama_disponible():
        return None
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE,
              "options": gen_options(temperature, num_predict),
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
