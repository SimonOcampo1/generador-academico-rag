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
    ("carta_pasantia", dict(alumno="Lautaro Aubert", tono="tecnico")),
]

resultados = []
for nombre, kwargs in CASOS:
    gen = getattr(generar, nombre)
    print(f"... {nombre} {kwargs}", flush=True)
    r = evaluar.comparar_con_sin_rag(gen, **kwargs)
    pf_con = (r["con_rag"].get("precision_factual") or {}).get("precision")
    pf_sin = (r["sin_rag"].get("precision_factual") or {}).get("precision")
    resultados.append({
        "artefacto": nombre,
        "args": kwargs,
        "grounding_con_rag": r["con_rag"]["grounding"],
        "grounding_sin_rag": r["sin_rag"]["grounding"],
        "grounding_sem_con_rag": r["con_rag"].get("grounding_sem"),
        "grounding_sem_sin_rag": r["sin_rag"].get("grounding_sem"),
        "precision_factual_con_rag": pf_con,
        "precision_factual_sin_rag": pf_sin,
        "faithfulness_con_rag": r["con_rag"].get("faithfulness"),
        "faithfulness_sin_rag": r["sin_rag"].get("faithfulness"),
        "salida_con_rag": r["con_rag"]["salida"],
        "salida_sin_rag": r["sin_rag"]["salida"],
        "contexto": r["contexto"],
    })
    print(f"    lex con={r['con_rag']['grounding']} sin={r['sin_rag']['grounding']} "
          f"| sem con={r['con_rag'].get('grounding_sem')} sin={r['sin_rag'].get('grounding_sem')} "
          f"| precision con={pf_con} sin={pf_sin} "
          f"| faith con={r['con_rag'].get('faithfulness')} sin={r['sin_rag'].get('faithfulness')}",
          flush=True)

out = Path(__file__).resolve().parent.parent / "data" / "eval_resultados.json"
out.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")


def _prom(clave):
    vals = [x[clave] for x in resultados if x.get(clave) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


print(f"\nPROMEDIOS  (n={len(resultados)})")
print(f"  grounding léxico    con={_prom('grounding_con_rag'):.3f}  sin={_prom('grounding_sin_rag'):.3f}")
print(f"  grounding semántico con={_prom('grounding_sem_con_rag'):.3f}  sin={_prom('grounding_sem_sin_rag'):.3f}")
print(f"  precisión factual   con={_prom('precision_factual_con_rag'):.3f}  sin={_prom('precision_factual_sin_rag'):.3f}")
print(f"  faithfulness (juez) con={_prom('faithfulness_con_rag'):.3f}  sin={_prom('faithfulness_sin_rag'):.3f}")
print("Guardado en", out)
