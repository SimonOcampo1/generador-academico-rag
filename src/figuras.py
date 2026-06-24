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

import json
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
DATA = Path(__file__).resolve().parent.parent / "data"

# --- Paleta Graphite & Cobalt (misma fuente de verdad que assets/design.css) ---------
BG = "#FAFAFB"
INK = "#0E0F12"
INK2 = "#5B6066"      # ink-soft
INK3 = "#8B9099"      # muted, para anotaciones tenues
LINE = "#E6E7EB"
COBALT = "#3A5BFF"
# categóricos fríos reservados a data-viz: cobalto, grafito, acero, teal, violeta, ámbar
CATS = ["#3A5BFF", "#0E0F12", "#7C8AA5", "#00A6A6", "#9B6DFF", "#E0A100"]

_have = {ff.name for ff in font_manager.fontManager.ttflist}
# display = Space Grotesk → Segoe UI (mismo orden de fallback que el sistema de diseño)
_DISPLAY = next((f for f in ("Space Grotesk", "Segoe UI", "Arial") if f in _have), "DejaVu Sans")
_SANS = next((f for f in ("Inter", "Segoe UI", "Arial") if f in _have), "DejaVu Sans")
_MONO = next((f for f in ("JetBrains Mono", "Cascadia Mono", "Consolas") if f in _have), "monospace")

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": _SANS, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": LINE, "axes.linewidth": 1.0,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.major.size": 0, "ytick.major.size": 0,  # sin marcas de tick, solo etiquetas
    "axes.titlecolor": INK, "axes.grid": False,     # grilla explícita y sutil por figura
    "grid.color": LINE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,  # sin caja; se activa lo justo
    "font.size": 10.5, "figure.dpi": 150,
})


def _titulo(ax, titulo: str, subtitulo: str = ""):
    """Título editorial alineado a la izquierda (display) + subtítulo tenue (mono)."""
    ax.set_title(titulo, loc="left", fontfamily=_DISPLAY, fontsize=15, fontweight="600",
                 color=INK, pad=22 if subtitulo else 12)
    if subtitulo:
        ax.annotate(subtitulo, xy=(0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", ha="left", va="bottom",
                    fontfamily=_MONO, fontsize=8.5, color=INK3)


def _save(fig, nombre: str):
    fig.tight_layout()
    fig.savefig(FIGS / nombre, bbox_inches="tight")
    plt.close(fig)
    print("  [ok]", nombre)


def fig_notas_distribucion(df):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    # de mayor a menor promedio, dibujado de arriba hacia abajo
    alumnos = list(analisis.promedio_por_alumno(df).index)[::-1]
    datos = [df[df["alumno"] == a]["nota"].values for a in alumnos]
    parts = ax.violinplot(datos, orientation="horizontal", showmeans=False,
                          showextrema=False, widths=0.82)
    for b in parts["bodies"]:
        b.set_facecolor(COBALT); b.set_alpha(0.16)
        b.set_edgecolor(COBALT); b.set_linewidth(1.1)
    # promedio como marcador cobalto nítido sobre cada violín
    medias = [df[df["alumno"] == a]["nota"].mean() for a in alumnos]
    ax.scatter(medias, range(1, len(alumnos) + 1), color=COBALT, s=42, zorder=5,
               edgecolor=BG, linewidth=1.5)
    for i, m in enumerate(medias):
        ax.annotate(f"{m:.1f}", (m, i + 1), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontfamily=_MONO, fontsize=8.5, color=COBALT, fontweight="600")
    ax.set_yticks(range(1, len(alumnos) + 1)); ax.set_yticklabels(alumnos, color=INK)
    ax.set_xlim(5.5, 10.5); ax.set_xticks(range(6, 11))
    ax.set_xlabel("Nota", color=INK2)
    ax.grid(axis="x", linestyle=(0, (2, 4)), linewidth=0.7, color=LINE)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(LINE)
    _titulo(ax, "Distribución de notas", "Densidad de calificaciones y promedio por integrante · datos reales")
    _save(fig, "notas_distribucion.png")


def fig_avance_anual(df):
    piv = df.pivot_table(index="anio", columns="alumno", values="nota", aggfunc="count").fillna(0)
    # ordenar columnas por promedio (leyenda y barras coherentes con las otras figuras)
    orden = [a for a in analisis.promedio_por_alumno(df).index if a in piv.columns]
    piv = piv[orden]
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(piv.columns); x = np.arange(len(piv.index)); w = 0.82 / n
    for i, a in enumerate(piv.columns):
        ax.bar(x + (i - (n - 1) / 2) * w, piv[a], w, label=a,
               color=CATS[i % len(CATS)], edgecolor=BG, linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"Año {int(y)}" for y in piv.index], color=INK)
    ax.set_ylabel("Materias aprobadas", color=INK2)
    ax.grid(axis="y", linewidth=0.7, color=LINE); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=9, columnspacing=1.4, handletextpad=0.5)
    _titulo(ax, "Avance académico por año", "Materias aprobadas por integrante en cada nivel · datos reales")
    _save(fig, "avance_anual.png")


def fig_radar(perfil):
    areas = [c for c in perfil.columns if c != "Otras"]
    # ordenar integrantes por promedio para color/leyenda coherentes entre figuras
    orden = [a for a in analisis.promedio_por_alumno(analisis.tabla_grupo()).index if a in perfil.index]
    perfil = perfil.loc[orden]
    ang = np.linspace(0, 2 * np.pi, len(areas), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7.5, 8), subplot_kw=dict(polar=True))
    for i, (alumno, row) in enumerate(perfil.iterrows()):
        vals = [row[a] for a in areas]; vals += vals[:1]
        ax.plot(ang, vals, color=CATS[i % len(CATS)], linewidth=1.7, label=alumno, zorder=3)
        ax.fill(ang, vals, color=CATS[i % len(CATS)], alpha=0.04)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(areas, fontsize=9.5, color=INK)
    ax.tick_params(pad=14)
    ax.set_ylim(5, 10); ax.set_yticks([6, 7, 8, 9])
    ax.set_yticklabels(["6", "7", "8", "9"], fontsize=8, color=INK3)
    ax.set_facecolor(BG)
    ax.spines["polar"].set_color(LINE)
    ax.grid(color=LINE, linewidth=0.8)
    ax.set_title("Perfil por área temática", loc="center", fontfamily=_DISPLAY,
                 fontsize=15, fontweight="600", color=INK, pad=34)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False,
              fontsize=9, ncol=3, columnspacing=1.4, handletextpad=0.5)
    _save(fig, "radar_areas.png")


def fig_clustering_perfiles(perfil, k=3):
    areas = [c for c in perfil.columns if c != "Otras"]
    X = StandardScaler().fit_transform(perfil[areas].values)
    labels = KMeans(n_clusters=k, random_state=14, n_init=10).fit_predict(X)
    xy = PCA(n_components=2, random_state=14).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for c in range(k):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=240, color=CATS[c], alpha=0.85,
                   edgecolor=BG, linewidth=2, label=f"Cluster {c + 1}", zorder=3)
    for (x, y), alumno in zip(xy, perfil.index):
        ax.annotate(alumno, (x, y), xytext=(0, 14), textcoords="offset points",
                    fontsize=9, ha="center", va="bottom", color=INK)
    ax.set_xlabel("Componente principal 1", color=INK2)
    ax.set_ylabel("Componente principal 2", color=INK2)
    ax.margins(0.16)
    ax.grid(linewidth=0.7, color=LINE); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="best")
    _titulo(ax, "Clustering de perfiles académicos",
            f"KMeans (k={k}) sobre el perfil por área, proyectado con PCA")
    _save(fig, "clustering_perfiles.png")


def fig_materias_embeddings():
    """t-SNE de los embeddings de las 36 obligatorias (fichas de contenidos), coloreado por área."""
    col = get_collection()
    # Embeddings de las fichas de materia (fuente 'contenidos_materia'); las correlatividades viven
    # ahora en la base relacional, no en Chroma. Las fichas dan un espacio semántico más rico.
    got = col.get(where={"fuente": "contenidos_materia"}, include=["embeddings", "metadatas"])
    emb = np.array(got["embeddings"]); metas = got["metadatas"]
    if len(emb) < 3:
        print("  [skip] materias_embeddings.png (sin fichas de materia en Chroma; corré ingest.reindexar)")
        return
    areas = [analisis.area_de(m["materia"]) for m in metas]
    perplexity = min(8, len(emb) - 1)  # t-SNE exige perplexity < n_samples
    xy = TSNE(n_components=2, random_state=14, perplexity=perplexity, init="pca").fit_transform(emb)
    fig, ax = plt.subplots(figsize=(10.5, 8))
    cats = sorted(set(areas))
    for i, area in enumerate(cats):
        m = [a == area for a in areas]
        ax.scatter(xy[m, 0], xy[m, 1], s=110, color=CATS[i % len(CATS)],
                   alpha=0.9, edgecolor=BG, linewidth=1.5, label=area, zorder=3)
    for (x, y), meta in zip(xy, metas):
        # nombre completo (sin truncar a mitad de palabra), desplazado para no pisar el punto
        ax.annotate(meta["materia"], (x, y), xytext=(7, 4), textcoords="offset points",
                    fontsize=7, color=INK2, alpha=0.95, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.margins(0.10)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelspacing=0.7,
              borderpad=0.2, handletextpad=0.4)
    _titulo(ax, "Materias en el espacio semántico",
            "t-SNE de los embeddings de las 36 obligatorias, coloreadas por área")
    _save(fig, "materias_embeddings.png")


def fig_aporte_rag():
    """Cuánto del contexto inyectado usó el modelo: grounding CON vs SIN RAG por artefacto.

    Es la figura que pide la consigna de métricas: cuantifica qué parte de la respuesta se ancla en
    el contexto inyectado (con RAG) frente a lo que el modelo resuelve con su conocimiento propio
    (sin RAG). El salto con−sin es el aporte del RAG. Lee data/eval_resultados.json (scripts/run_eval.py).
    """
    p = DATA / "eval_resultados.json"
    if not p.exists():
        print("  [skip] aporte_rag.png (falta data/eval_resultados.json; corré scripts/run_eval.py)")
        return
    datos = json.loads(p.read_text(encoding="utf-8"))
    arts = [d["artefacto"].replace("_", " ") for d in datos]
    con = [d["grounding_con_rag"] for d in datos]
    sin = [d["grounding_sin_rag"] for d in datos]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(arts)); w = 0.38
    ax.bar(x - w / 2, con, w, label="con RAG (anclado en el contexto)", color=COBALT,
           edgecolor=BG, linewidth=0.8)
    ax.bar(x + w / 2, sin, w, label="sin RAG (conocimiento propio del modelo)", color=INK3,
           edgecolor=BG, linewidth=0.8)
    for i, (c, s) in enumerate(zip(con, sin)):
        ax.annotate(f"{c:.2f}", (i - w / 2, c), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=COBALT, fontfamily=_MONO, fontweight="600")
        ax.annotate(f"{s:.2f}", (i + w / 2, s), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK2, fontfamily=_MONO)
    ax.set_xticks(x); ax.set_xticklabels(arts, fontsize=9, color=INK)
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Grounding (fracción de la salida anclada en el contexto)", color=INK2)
    ax.grid(axis="y", linewidth=0.7, color=LINE); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _titulo(ax, "¿Cuánto del contexto inyectado usó el modelo?",
            "Grounding con vs sin RAG por artefacto · el salto cuantifica el aporte del conocimiento privado")
    _save(fig, "aporte_rag.png")


def generar_todo():
    grupo = analisis.tabla_grupo()
    perfil = analisis.perfil_por_area(grupo)
    print("Generando figuras en", FIGS)
    fig_notas_distribucion(grupo)
    fig_avance_anual(grupo)
    fig_radar(perfil)
    fig_clustering_perfiles(perfil)
    fig_materias_embeddings()
    fig_aporte_rag()


if __name__ == "__main__":
    generar_todo()
    pngs = list(FIGS.glob("*.png"))
    assert len(pngs) >= 5, f"faltan figuras: {[p.name for p in pngs]}"
    print(f"\n{len(pngs)} figuras generadas.")
