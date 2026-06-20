"""Web app de testeo en vivo del pipeline RAG académico.

Reusa /src. Muestra el mecanismo: contexto recuperado -> tokens en vivo -> grounding.
Demo estrella: mismo pedido CON RAG (contexto real) vs SIN RAG.

Endpoints:
  GET  /                  -> frontend estático
  GET  /api/health        -> estado de Ollama + modelo
  GET  /api/opciones      -> artefactos, alumnos, tonos (para armar la UI)
  POST /api/start_model   -> enciende Ollama desde la UI
  POST /api/stop_model    -> apaga Ollama
  POST /api/generar_stream-> NDJSON en vivo: meta -> token* -> done (grounding)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import analisis  # noqa: E402
import generar  # noqa: E402
import rag  # noqa: E402
from evaluar import grounding_score  # noqa: E402
from ingest import get_collection, reindexar  # noqa: E402

app = FastAPI(title="Generador Académico RAG")
STATIC = Path(__file__).parent / "static"
ASSETS = ROOT / "assets"
FIGS = ROOT / "figs"

# Artefactos disponibles en la UI: (función, requiere_alumno, campos_extra)
ARTEFACTOS = {
    "plan_cursada": {"label": "Plan de cursada", "alumno": True, "tono": "tecnico"},
    "informe_trayectoria": {"label": "Informe de trayectoria", "alumno": True, "tono": "motivacional"},
    "carta_pasantia": {"label": "Carta de pasantía", "alumno": True, "tono": "tecnico", "objetivo": True},
    "recomendar_orientacion": {"label": "Recomendar orientación", "alumno": True, "tono": "tecnico"},
    "diagnostico_grupal": {"label": "Diagnóstico grupal", "alumno": False, "tono": "tecnico"},
    "simulacion_whatif": {"label": "Simulación what-if", "alumno": True, "tono": "tecnico", "whatif": True},
}

_SIN_RAG_SYSTEM = "Sos un asistente académico. Respondé el pedido lo mejor que puedas."


@app.on_event("startup")
def _ensure_index():
    try:
        if get_collection().count() == 0:
            print("Chroma vacío: indexando corpus...")
            reindexar()
    except Exception as e:  # noqa: BLE001
        print("Aviso: no se pudo verificar/indexar Chroma:", e)


class GenReq(BaseModel):
    artefacto: str = "plan_cursada"
    alumno: str | None = None
    tono: str = "tecnico"
    rag: bool = True
    objetivo: str = "una pasantía en Ciencia de Datos"
    materias_nota: dict[str, int] | None = None


@contextmanager
def _sin_llm():
    """Fuerza el modo fallback del generador para obtener contexto+prompt sin llamar al LLM."""
    orig = rag.ollama_disponible
    rag.ollama_disponible = lambda: False
    try:
        yield
    finally:
        rag.ollama_disponible = orig


def _kwargs(req: GenReq) -> dict:
    spec = ARTEFACTOS[req.artefacto]
    kw: dict = {"tono": req.tono}
    if spec.get("alumno"):
        kw["alumno"] = req.alumno
    if spec.get("objetivo"):
        kw["objetivo"] = req.objetivo
    if spec.get("whatif"):
        kw["materias_nota"] = req.materias_nota or {"Ciencia de Datos": 9, "Inteligencia Artificial": 8}
    return kw


@app.get("/api/health")
def health():
    remoto = rag.es_remoto()
    host = rag.OLLAMA_URL.split("://", 1)[-1]
    return {"ollama": rag.ollama_disponible(), "model": rag.OLLAMA_MODEL,
            "remoto": remoto, "backend": "Colab" if remoto else "local", "host": host,
            "tonos": list(generar.TONOS)}


@app.get("/api/opciones")
def opciones():
    alumnos = sorted(analisis.tabla_real()["alumno"].unique().tolist())
    return {
        "artefactos": [{"id": k, **v} for k, v in ARTEFACTOS.items()],
        "alumnos": alumnos,
        "tonos": [{"id": k, "desc": v} for k, v in generar.TONOS.items()],
    }


def _find_ollama() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    cands = [os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe"),
             r"C:\Program Files\Ollama\ollama.exe"]
    return next((c for c in cands if os.path.exists(c)), None)


@app.post("/api/start_model")
def start_model():
    if rag.ollama_disponible():
        return {"ok": True, "already": True}
    exe = _find_ollama()
    if not exe:
        return {"ok": False, "error": "No se encontró Ollama instalado en el equipo."}
    try:
        flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        subprocess.Popen([exe, "serve"], creationflags=flags,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
    for _ in range(15):
        time.sleep(1)
        if rag.ollama_disponible():
            return {"ok": True, "started": True}
    return {"ok": True, "starting": True}


@app.post("/api/stop_model")
def stop_model():
    if not rag.ollama_disponible():
        return {"ok": True, "already": True}
    try:
        if os.name == "nt":
            for img in ("ollama.exe", "ollama app.exe"):
                subprocess.run(["taskkill", "/F", "/IM", img], capture_output=True)
        else:
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
    return {"ok": True, "stopped": True}


def _nd(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _stream(req: GenReq):
    if req.artefacto not in ARTEFACTOS:
        yield _nd({"type": "error", "msg": f"artefacto inválido: {req.artefacto}"})
        return
    spec = ARTEFACTOS[req.artefacto]
    if spec.get("alumno") and not req.alumno:
        yield _nd({"type": "error", "msg": "Elegí un alumno."})
        return

    # 1) Reconstruir contexto + prompt SIN llamar al LLM (rápido).
    try:
        with _sin_llm():
            base = getattr(generar, req.artefacto)(**_kwargs(req))
    except Exception as e:  # noqa: BLE001
        yield _nd({"type": "error", "msg": f"No se pudo armar el contexto: {e}"})
        return

    contexto_real = base["contexto"]
    instruccion = base["prompt"].split("TAREA:", 1)[-1].strip()

    if req.rag:
        system = generar._SYSTEM
        user = base["prompt"]
        contexto_mostrado = contexto_real
    else:  # demo SIN RAG: mismo pedido, sin contexto
        system = _SIN_RAG_SYSTEM
        user = f"CONTEXTO:\n(sin datos disponibles)\n\nTAREA: {instruccion}"
        contexto_mostrado = "(sin datos: el modelo responde sin acceso al historial)"

    yield _nd({"type": "meta", "model": rag.OLLAMA_MODEL, "rag": req.rag,
               "artefacto": req.artefacto, "tono": req.tono,
               "system": system, "contexto": contexto_mostrado, "instruccion": instruccion})

    # 2) Streamear la generación.
    if not rag.ollama_disponible():
        msg = ("[Ollama apagado] La parte RAG funciona igual: arriba está el contexto recuperado "
               "de la base vectorial. Encendé el modelo para ver la generación en vivo.")
        for ch in msg:
            yield _nd({"type": "token", "text": ch}); time.sleep(0.003)
        yield _nd({"type": "done", "fuente": "solo_retrieval", "grounding": None, "stats": {}})
        return

    t0, full = time.time(), ""
    try:
        resp = requests.post(
            f"{rag.OLLAMA_URL}/api/chat",
            headers=rag.HEADERS,
            json={"model": rag.OLLAMA_MODEL, "stream": True, "keep_alive": rag.OLLAMA_KEEP_ALIVE,
                  "options": rag.gen_options(0.4),
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            stream=True, timeout=180)
        resp.raise_for_status()
        stats = {}
        for line in resp.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            tok = d.get("message", {}).get("content", "")
            if tok:
                full += tok
                yield _nd({"type": "token", "text": tok})
            if d.get("done"):
                ev, ed = d.get("eval_count", 0), d.get("eval_duration", 1) / 1e9
                stats = {"tokens": ev, "tok_s": round(ev / ed, 1) if ed else 0,
                         "segundos": round(time.time() - t0, 1)}
    except requests.RequestException as e:
        # Ollama está prendido pero la generación falló (p. ej. el modelo crasheó: 500).
        # Degradamos elegante: mostramos el contexto recuperado (la parte RAG funciona igual).
        if not full:
            msg = ("[El modelo no pudo generar: " + str(e)[:120] + "]\n\n"
                   "La recuperación RAG funcionó igual (ver el contexto). Suele ser un crash de "
                   "Ollama: probá re-descargar el modelo (ollama rm + ollama pull) o actualizar Ollama.")
            for ch in msg:
                yield _nd({"type": "token", "text": ch}); time.sleep(0.002)
            yield _nd({"type": "done", "fuente": "solo_retrieval", "grounding": None, "stats": {}})
            return

    # 3) Grounding SIEMPRE contra el contexto real del RAG (mide cuánto se ancló en datos).
    g = grounding_score(full, contexto_real)
    yield _nd({"type": "done", "fuente": "ollama", "grounding": g, "stats": stats})


@app.post("/api/generar_stream")
def generar_stream(req: GenReq):
    return StreamingResponse(_stream(req), media_type="application/x-ndjson")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
app.mount("/figs", StaticFiles(directory=FIGS), name="figs")
