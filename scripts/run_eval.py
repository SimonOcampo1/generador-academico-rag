"""Corre la demo con/sin RAG sobre varios artefactos y guarda los números para el informe.

Salida: data/eval_resultados.json (grounding léxico y semántico, con vs sin RAG por artefacto).
Requiere Ollama corriendo. Reproducible salvo el muestreo del LLM (temperature baja).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluar
import generar

CASOS = [
    ("plan_cursada", dict(alumno="Simón Ocampo", tono="tecnico")),
    ("recomendar_orientacion", dict(alumno="Mora Gentil", tono="tecnico")),
    ("informe_trayectoria", dict(alumno="Santiago Natalichio", tono="honesto")),
]

resultados = []
for nombre, kwargs in CASOS:
    gen = getattr(generar, nombre)
    print(f"... {nombre} {kwargs}", flush=True)
    r = evaluar.comparar_con_sin_rag(gen, **kwargs)
    resultados.append({
        "artefacto": nombre,
        "args": kwargs,
        "grounding_con_rag": r["con_rag"]["grounding"],
        "grounding_sin_rag": r["sin_rag"]["grounding"],
        "grounding_sem_con_rag": r["con_rag"].get("grounding_sem"),
        "grounding_sem_sin_rag": r["sin_rag"].get("grounding_sem"),
        "salida_con_rag": r["con_rag"]["salida"],
        "salida_sin_rag": r["sin_rag"]["salida"],
        "contexto": r["contexto"],
    })
    print(f"    lex con={r['con_rag']['grounding']} sin={r['sin_rag']['grounding']} "
          f"sem con={r['con_rag'].get('grounding_sem')} sin={r['sin_rag'].get('grounding_sem')}",
          flush=True)

out = Path(__file__).resolve().parent.parent / "data" / "eval_resultados.json"
out.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
prom_con = sum(x["grounding_con_rag"] for x in resultados) / len(resultados)
prom_sin = sum(x["grounding_sin_rag"] for x in resultados) / len(resultados)
print(f"\nPROMEDIO grounding léxico  con_rag={prom_con:.3f}  sin_rag={prom_sin:.3f}")
print("Guardado en", out)
