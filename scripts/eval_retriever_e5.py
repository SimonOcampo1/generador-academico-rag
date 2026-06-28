"""Evaluación del retriever con un embedder MULTILINGÜE (multilingual-e5-large).

Reproduce el resultado que el informe reporta como mejora validada y opcional: cambiando el
embedder por uno multilingüe, la recuperación sobre el mismo gold set de 12 consultas-tema sube
a hit-rate@1 = 1.00 y MRR = 1.00 (incluida «Bases de Datos», el único miss del MiniLM por defecto).

Es OPCIONAL: no toca el pipeline. El default sigue siendo MiniLM (liviano y offline); este script
solo demuestra el techo alcanzable con un modelo más fuerte para texto técnico en español.

Requiere fastembed (no está en requirements.txt para no imponer la descarga del modelo):
    pip install fastembed
La 1ª corrida descarga el modelo e5-large (~2.2 GB) a la caché de fastembed; después es instantáneo.
No usa el LLM ni GPU. Lee las fichas ya indexadas en Chroma (construí la base con src/ingest.py).

Uso:  python scripts/eval_retriever_e5.py
Salida: data/eval_retriever_e5.json + tabla por consola.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from documents import canon  # noqa: E402
from eval_retriever import GOLD  # mismo gold set que el retriever por defecto (única fuente)  # noqa: E402
from ingest import get_collection  # noqa: E402

MODEL = "intfloat/multilingual-e5-large"
K = 5


def main() -> None:
    try:
        import numpy as np
        from fastembed import TextEmbedding
    except ImportError:
        sys.exit("Falta fastembed. Instalá con:  pip install fastembed")

    # Fichas de materia ya indexadas (mismo texto que usa el retriever por defecto).
    raw = get_collection().get(where={"fuente": "contenidos_materia"})
    materias = [m["materia"] for m in raw["metadatas"]]
    docs = list(raw["documents"])

    emb = TextEmbedding(model_name=MODEL)
    # e5 espera prefijos "query:" / "passage:" para distinguir consulta de documento.
    D = np.array(list(emb.embed([f"passage: {d}" for d in docs])))
    Q = np.array(list(emb.embed([f"query: {q}" for q, _ in GOLD])))
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    sim = Q @ D.T

    detalle, ranks = [], []
    for i, (consulta, esperada) in enumerate(GOLD):
        objetivo = canon(esperada)
        orden = np.argsort(-sim[i])
        r = 0
        for pos, idx in enumerate(orden[:K], start=1):
            if canon(materias[idx]) == objetivo:
                r = pos
                break
        ranks.append(r)
        detalle.append({"consulta": consulta, "esperada": esperada, "rank": r})

    n = len(ranks)
    resumen = {
        "embedder": MODEL, "n": n, "k": K,
        "hit_rate@1": round(sum(r == 1 for r in ranks) / n, 3),
        "hit_rate@3": round(sum(1 <= r <= 3 for r in ranks) / n, 3),
        "hit_rate@5": round(sum(1 <= r <= 5 for r in ranks) / n, 3),
        "mrr": round(sum((1 / r) if r else 0 for r in ranks) / n, 3),
    }
    out = ROOT / "data" / "eval_retriever_e5.json"
    out.write_text(json.dumps({"resumen": resumen, "detalle": detalle}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"Retriever con {MODEL} sobre {n} consultas-tema (top-{K}):")
    print(f"  hit-rate@1 = {resumen['hit_rate@1']:.2f}   hit-rate@3 = {resumen['hit_rate@3']:.2f}   "
          f"hit-rate@5 = {resumen['hit_rate@5']:.2f}   MRR = {resumen['mrr']:.3f}\n")
    for d in detalle:
        marca = "ok" if d["rank"] == 1 else (f"#{d['rank']}" if d["rank"] else "MISS")
        print(f"  [{marca:>4}] {d['esperada']}")
    print(f"\nGuardado: {out}")
    # self-check: el modelo multilingüe debería recuperar casi todo dentro del top-5
    assert resumen["hit_rate@5"] >= 0.9, f"hit-rate@5 inesperadamente bajo ({resumen['hit_rate@5']})"


if __name__ == "__main__":
    main()
