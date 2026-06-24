"""RAG: recupera contexto de las DOS bases de datos y genera la respuesta con el LLM local.

Flujo: consulta -> retrieval HÍBRIDO (base relacional db.py + base vectorial ingest.buscar)
-> prompt con contexto -> Phi-4-mini vía Ollama.
Si Ollama no está corriendo, devuelve el contexto recuperado (la parte RAG funciona igual).
Modelo local configurable por OLLAMA_MODEL; default phi4-mini (3.8B).
"""
from __future__ import annotations

import os

import requests

import db
from documents import normalizar
from ingest import buscar

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Default phi4-mini (3.8B): mejor redacción y grounding léxico que modelos más chicos, sigue
# corriendo en CPU. En una máquina con GPU se puede exportar un modelo más grande. Ver README.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi4-mini")
# Mantener el modelo cargado entre pedidos evita el costo de recarga en cada generación.
# Ollama acepta keep_alive como duración ("30m", "24h") o como NÚMERO de segundos (-1 = para
# siempre). Si viene un entero (p. ej. "-1"), hay que mandarlo como número: el string "-1" no es
# una duración válida y Ollama responde 400. Por eso lo coercionamos.
_KEEP_ALIVE_RAW = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
try:
    KEEP_ALIVE: int | str = int(_KEEP_ALIVE_RAW)  # "-1", "300" -> segundos
except ValueError:
    KEEP_ALIVE = _KEEP_ALIVE_RAW                   # "30m", "24h" -> duración
OLLAMA_KEEP_ALIVE = KEEP_ALIVE  # compat: el resto del código lee este nombre
# Cuando el LLM corre en Colab vía túnel ngrok-free, éste intercala una página de aviso salvo que
# se mande este header. Inofensivo contra un Ollama local. Ver colab/generar_en_colab.ipynb.
HEADERS = {"ngrok-skip-browser-warning": "true"}


def es_remoto() -> bool:
    """True si la generación corre en otra máquina (Colab) y no en un Ollama local."""
    return "localhost" not in OLLAMA_URL and "127.0.0.1" not in OLLAMA_URL

SYSTEM = (
    "Sos un asistente académico de la UTN FRLP. Respondé SOLO con la información del "
    "CONTEXTO. Si el contexto no alcanza, decílo; no inventes notas, materias ni "
    "correlatividades. Sé concreto y citá los datos (materia, nota, año) que uses."
)


def ollama_disponible() -> bool:
    # timeout holgado: vía túnel ngrok el primer /api/tags puede tardar más que en localhost.
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", headers=HEADERS, timeout=5).status_code == 200
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
        headers=HEADERS,
        json={"model": OLLAMA_MODEL, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE,
              "options": gen_options(temperature, num_predict),
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _contexto_relacional(consulta: str) -> str:
    """Datos TABULARES pertinentes a la consulta, traídos por SQL de la base relacional.

    Si la consulta nombra a un alumno -> su rendimiento; si nombra una materia -> sus
    correlativas. Es la mitad 'relacional' del contexto híbrido (exacta, no semántica).
    """
    q = normalizar(consulta)
    lineas: list[str] = []
    for al in db.alumnos():
        nombre, apellido = al.split()[0], al.split()[-1]
        if normalizar(nombre) in q or normalizar(apellido) in q:
            ap = sorted(db.aprobadas(al), key=lambda r: r["materia"])
            if ap:
                lineas.append(
                    f"{al} aprobó {len(ap)} materias: "
                    + ", ".join(f"{r['materia']} ({r['nota']})" for r in ap)
                )
    for c in db.correlativas():
        if normalizar(c["materia"]) in q:
            cursar = c["cursar"] or "ninguna"
            lineas.append(
                f"Correlativas de '{c['materia']}' (Nº {c['num_plan']} del plan): "
                f"para cursar requiere las materias Nº {cursar}; para rendir Nº {c['rendir'] or 'ninguna'}."
            )
    return "\n".join(f"- {l}" for l in lineas)


def generar(consulta: str, k: int = 6) -> dict:
    """Devuelve {'respuesta', 'contexto', 'fuente'} para una consulta (caso QA simple).

    Contexto HÍBRIDO: datos tabulares de la base relacional (db.py) + documentación del plan
    de la base vectorial (Chroma). El agente combina ambas para responder.
    """
    relacional = _contexto_relacional(consulta)
    vectorial = "\n".join(f"- {h['text']}" for h in buscar(consulta, k=k))
    partes = []
    if relacional:
        partes.append("DATOS DEL ALUMNO / CORRELATIVAS (base relacional):\n" + relacional)
    partes.append("DOCUMENTACIÓN DEL PLAN (base vectorial):\n" + vectorial)
    contexto = "\n\n".join(partes)
    respuesta = chat(SYSTEM, f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {consulta}")
    return {
        "respuesta": respuesta,
        "contexto": contexto,
        "fuente": "ollama" if respuesta is not None else "solo_retrieval",
    }


if __name__ == "__main__":
    db.asegurar()  # base relacional lista (la vectorial se construye con `python src/ingest.py`)
    out = generar("¿Qué necesito para cursar Ciencia de Datos y cómo le fue a Simón en programación?")
    print("FUENTE:", out["fuente"])
    if out["respuesta"]:
        print("\nRESPUESTA:\n", out["respuesta"])
    else:
        print("\n(Ollama apagado) CONTEXTO recuperado:\n", out["contexto"])
