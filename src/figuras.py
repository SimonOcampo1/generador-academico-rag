"""EDA + clustering -> PNGs con estilo 'Graphite & Cobalt' (homogéneo con los entregables).

Genera en figs/:
  notas_distribucion.png   distribución de notas por integrante
  avance_anual.png         materias aprobadas por año
  radar_areas.png          perfil por área temática (radar) de los 6 integrantes
  clustering_perfiles.png  KMeans sobre el perfil por área (2D vía PCA)
  materias_embeddings.png  las 36 obligatorias proyectadas (t-SNE de los embeddings) por área

Todo es reproducible y offline. Los embeddings se leen de la base vectorial Chroma ya construida.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import analisis
from ingest import get_collection

FIGS = Path(__file__).resolve().parent.parent / "figs"
FIGS.mkdir(exist_ok=True)

# --- Paleta Graphite & Cobalt ---------------------------------------------
BG = "#FAFAFB"
INK = "#0E0F12"
INK2 = "#5B6066"
LINE = "#E6E7EB"
COBALT = "#3A5BFF"
# categóricos derivados (cobalto + neutros fríos + un acento cálido puntual)
CATS = ["#3A5BFF", "#0E0F12", "#7C8AA5", "#00A6A6", "#9B6DFF", "#E0A100"]

_SANS = next((f for f in ("Inter", "Segoe UI", "Arial") if f in
              {ff.name for ff in font_manager.fontManager.ttflist}), "DejaVu Sans")

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": _SANS, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": LINE, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": LINE,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 130,
})


def _save(fig, nombre: str):
    fig.tight_layout()
    fig.savefig(FIGS / nombre, bbox_inches="tight")
    plt.close(fig)
    print("  [ok]", nombre)


def fig_notas_distribucion(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    alumnos = list(analisis.promedio_por_alumno(df).index)
    datos = [df[df["alumno"] == a]["nota"].values for a in alumnos]
    parts = ax.violinplot(datos, orientation="horizontal", showmeans=True, widths=0.85)
    for i, b in enumerate(parts["bodies"]):
        real = not df[df["alumno"] == alumnos[i]]["sintetico"].iloc[0]
        b.set_facecolor(COBALT if real else INK2)
        b.set_alpha(0.85 if real else 0.35)
    for k in ("cmeans", "cmaxes", "cmins", "cbars"):
        parts[k].set_color(INK)
    ax.set_yticks(range(1, len(alumnos) + 1))
    ax.set_yticklabels([f"{a}{'' if not df[df.alumno==a].sintetico.iloc[0] else '  (sim.)'}" for a in alumnos])
    ax.set_xlabel("Nota"); ax.set_xlim(3.5, 10.5)
    ax.set_title("Distribución de notas por integrante")
    _save(fig, "notas_distribucion.png")


def fig_avance_anual(df):
    real = df[~df["sintetico"]]
    piv = real.pivot_table(index="anio", columns="alumno", values="nota", aggfunc="count").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n = len(piv.columns); x = np.arange(len(piv.index)); w = 0.8 / n
    for i, a in enumerate(piv.columns):
        ax.bar(x + (i - (n - 1) / 2) * w, piv[a], w, label=a, color=CATS[i % len(CATS)], edgecolor=BG)
    ax.set_xticks(x); ax.set_xticklabels([f"Año {int(y)}" for y in piv.index])
    ax.set_ylabel("Materias aprobadas"); ax.set_title("Avance académico por año (datos reales)")
    ax.legend(frameon=False)
    _save(fig, "avance_anual.png")


def fig_radar(perfil):
    areas = [c for c in perfil.columns if c != "Otras"]
    ang = np.linspace(0, 2 * np.pi, len(areas), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, (alumno, row) in enumerate(perfil.iterrows()):
        vals = [row[a] for a in areas]; vals += vals[:1]
        ax.plot(ang, vals, color=CATS[i % len(CATS)], linewidth=2, label=alumno)
        ax.fill(ang, vals, color=CATS[i % len(CATS)], alpha=0.06)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(areas, fontsize=9)
    ax.set_ylim(5, 10); ax.set_facecolor(BG)
    ax.set_title("Perfil por área temática", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False, fontsize=9)
    _save(fig, "radar_areas.png")


def fig_clustering_perfiles(perfil, k=3):
    areas = [c for c in perfil.columns if c != "Otras"]
    X = StandardScaler().fit_transform(perfil[areas].values)
    labels = KMeans(n_clusters=k, random_state=14, n_init=10).fit_predict(X)
    xy = PCA(n_components=2, random_state=14).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(k):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=320, color=CATS[c], alpha=0.85,
                   edgecolor=BG, linewidth=2, label=f"Cluster {c + 1}")
    for (x, y), alumno in zip(xy, perfil.index):
        ax.annotate(alumno, (x, y), fontsize=9, ha="center", va="center", color=INK)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Clustering de perfiles académicos (KMeans, k={k})")
    ax.legend(frameon=False)
    _save(fig, "clustering_perfiles.png")


def fig_materias_embeddings():
    """t-SNE de los embeddings de las 36 obligatorias, coloreado por área."""
    col = get_collection()
    got = col.get(where={"fuente": "correlatividades"}, include=["embeddings", "metadatas"])
    emb = np.array(got["embeddings"]); metas = got["metadatas"]
    areas = [analisis.area_de(m["materia"]) for m in metas]
    xy = TSNE(n_components=2, random_state=14, perplexity=8, init="pca").fit_transform(emb)
    fig, ax = plt.subplots(figsize=(9, 7))
    cats = sorted(set(areas))
    for i, area in enumerate(cats):
        m = [a == area for a in areas]
        ax.scatter(xy[m, 0], xy[m, 1], s=120, color=CATS[i % len(CATS)],
                   alpha=0.85, edgecolor=BG, linewidth=1.5, label=area)
    for (x, y), meta in zip(xy, metas):
        ax.annotate(meta["materia"][:18], (x, y), fontsize=7, color=INK2, alpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Materias en el espacio semántico (t-SNE de embeddings)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _save(fig, "materias_embeddings.png")


def generar_todo():
    grupo = analisis.tabla_grupo()
    perfil = analisis.perfil_por_area(grupo)
    print("Generando figuras en", FIGS)
    fig_notas_distribucion(grupo)
    fig_avance_anual(grupo)
    fig_radar(perfil)
    fig_clustering_perfiles(perfil)
    fig_materias_embeddings()


if __name__ == "__main__":
    generar_todo()
    pngs = list(FIGS.glob("*.png"))
    assert len(pngs) >= 5, f"faltan figuras: {[p.name for p in pngs]}"
    print(f"\n{len(pngs)} figuras generadas.")
