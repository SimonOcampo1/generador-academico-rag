"""Construye notebook.ipynb integrador (reproducible). Ejecutar desde la raíz del proyecto.

ponytail: builder one-shot del entregable; regenera el .ipynb desde acá si cambia el pipeline.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Generador Académico RAG — Notebook integrador
**TP Integrador de Ciencia de Datos · UTN FRLP 2026 · Grupo 14**

Sistema de IA generativa que, alimentado por una base vectorial con el **historial académico
real** del grupo + el **plan de estudios** de la carrera, genera *artefactos académicos
personalizados* (planes de cursada, informes, cartas de pasantía, recomendaciones,
diagnóstico grupal, simulaciones). No responde preguntas: **sintetiza documentos nuevos**
anclados en datos privados que el modelo no conoce.

### Por qué es generativo y no un SQL
Un SQL te dice *qué* materias podés cursar (lookup). Esto te dice *qué conviene y por qué*,
redactado, con tono controlable, cruzando notas + correlativas + rendimiento. La salida es
texto nuevo sintetizado, no un campo de una tabla.

### Pipeline (lo construimos nosotros, no es NotebookLM)
```
PDFs ──pdfplumber──▶ documentos (estructurado + no estructurado)
     ──MiniLM──▶ embeddings ──▶ Chroma (base vectorial)
     ──retrieval top-k──▶ contexto ──▶ qwen2.5 (Ollama) ──▶ artefacto
```
Local, offline, privado. Si Ollama está apagado, la parte RAG funciona igual (recupera el
contexto) y la generación se completa al encender el modelo.
""")

md("## 0 · Setup\nTodo el código vive en `src/`. Cada módulo tiene self-checks; acá los orquestamos.")
code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))  # importar los módulos del proyecto

import analisis, documents, extract, figuras, generar, ingest, rag, evaluar
from IPython.display import Image, display

print("Ollama:", "ENCENDIDO" if rag.ollama_disponible() else "apagado (modo solo-retrieval)")
""")

md(r"""
## 1 · Extracción de los PDFs
`extract.pdf_to_text` usa **pdfplumber**. La fuente son estados académicos reales, exports de
notas de exámenes y el plan de estudios 2023 (correlatividades + equivalencias).
""")
code(r"""
raw = documents.RAW_DIR
print("PDFs disponibles:", [p.name for p in sorted(raw.glob("*.pdf"))])
texto = extract.pdf_to_text(raw / "Plan-Sistemas-2023.pdf")
print(f"\nPlan 2023: {len(texto)} caracteres. Primeras líneas:\n")
print("\n".join(texto.splitlines()[:8]))
""")

md(r"""
## 2 · Construcción del corpus
`documents.construir_corpus` fusiona lo **estructurado** (una entrada por materia con nota,
año, alumno) con lo **no estructurado** (chunks del plan) y parsea las **36 correlatividades**.
La clave de join entre estado y plan es el *nombre normalizado* (`normalizar`), porque los
códigos institucionales no coinciden entre documentos.
""")
code(r"""
corpus = documents.construir_corpus()
print(f"Total documentos en el corpus: {len(corpus)}")
from collections import Counter
print("Por fuente:", dict(Counter(d["metadata"]["fuente"] for d in corpus)))

print("\nEjemplo (estado académico):")
ej = next(d for d in corpus if d["metadata"]["fuente"] == "estado_academico" and d["metadata"]["nota"] >= 0)
print(" texto:", ej["text"])
print(" metadata:", ej["metadata"])
""")

md(r"""
## 3 · Embeddings + base vectorial Chroma
`ingest.reindexar` vectoriza el corpus con **all-MiniLM-L6-v2** (el embedding por defecto de
Chroma, local y offline) y lo persiste. Esta es la base que el modelo consultará.
""")
code(r"""
n = ingest.reindexar()
print(f"Indexados {n} documentos en Chroma → {ingest.CHROMA_DIR}")
""")

md(r"""
## 4 · Retrieval
Dos modos de recuperación:
- **Semántico** (`ingest.buscar`): similitud de embeddings, para preguntas en lenguaje natural.
- **Determinístico** (`ingest.obtener`): por metadata, para grounding exacto (las materias
  aprobadas de un alumno, las correlatividades). Es lo que ancla a los generadores.
""")
code(r"""
print("### Búsqueda semántica: '¿qué necesito para cursar Ciencia de Datos?'")
for r in ingest.buscar("¿qué necesito para cursar Ciencia de Datos?", k=3):
    print(f"  [{r['distancia']:.3f}] ({r['metadata']['fuente']}) {r['text'][:90]}")

print("\n### Grounding determinístico: materias aprobadas de un alumno (por metadata)")
aprobadas = ingest.obtener({"$and": [{"fuente": "estado_academico"}, {"alumno": "Simón Ocampo"}]})
aprobadas = [d for d in aprobadas if d["metadata"]["nota"] >= 0]
print(f"  {len(aprobadas)} materias con nota. Ejemplos:", [d["metadata"]["materia"] for d in aprobadas[:5]])
""")

md(r"""
## 5 · EDA + visualización
`analisis` arma un DataFrame de notas **reales de los 6 integrantes del grupo**, fusionando el
estado académico de cada uno con su export de exámenes. Cada registro es `(alumno, materia, año,
nota, área)`; las áreas temáticas mapean las 36 obligatorias del plan en 6 grupos.
""")
code(r"""
real = analisis.tabla_real()
print(f"Registros reales: {len(real)} | alumnos: {real['alumno'].nunique()}")
display(analisis.promedio_por_alumno(real).to_frame("promedio"))

grupo = analisis.tabla_grupo()  # los 6 integrantes, todos con datos reales
perfil = analisis.perfil_por_area(grupo)
print("\nPerfil por área (nota media) — los 6 integrantes:")
display(perfil)
""")
code(r"""
figuras.generar_todo()  # regenera los PNG con estilo Graphite & Cobalt
for png in ["notas_distribucion.png", "avance_anual.png", "radar_areas.png"]:
    display(Image(filename=str(figuras.FIGS / png)))
""")

md(r"""
## 6 · Clustering de perfiles
KMeans sobre el perfil por área (estandarizado), proyectado a 2D con PCA. Y, aparte, las 36
obligatorias proyectadas con **t-SNE de sus embeddings**: las materias de la misma área caen
juntas, lo que valida que el espacio semántico capturó la estructura del plan.
""")
code(r"""
display(Image(filename=str(figuras.FIGS / "clustering_perfiles.png")))
display(Image(filename=str(figuras.FIGS / "materias_embeddings.png")))
""")

md(r"""
## 7 · Generación de artefactos con tono
Cada generador ancla la salida en el contexto recuperado (grounding) y pasa **solo ese
contexto** al LLM. El **tono** (técnico / motivacional / honesto) cambia el registro sobre los
mismos datos: eso es lo que lo hace generativo y no un reporte fijo.
""")
code(r"""
art = generar.plan_cursada("Simón Ocampo", tono="tecnico")
print("Artefacto:", art["artefacto"], "| fuente:", art["fuente"])
print(f"\nCONTEXTO recuperado ({len(art['contexto'])} chars), primeras líneas:")
print("\n".join(art["contexto"].splitlines()[:6]))

print("\n--- SALIDA ---")
if art["salida"]:
    print(art["salida"])
else:
    print("(Ollama apagado) El contexto de arriba ya es la parte RAG. Encendé el modelo para la generación.")
""")

md(r"""
## 8 · Evaluación — grounding con/sin RAG (demo estrella)
Medimos el anclaje de la salida con **dos métricas complementarias**:
- **Grounding léxico**: fracción de los tokens informativos de la salida (nombres de materias,
  notas) que aparecen literalmente en el contexto real. Explicable, pero penaliza paráfrasis.
- **Grounding semántico**: promedio del máximo coseno entre cada oración de la salida y los
  fragmentos del contexto (mismos embeddings MiniLM). Reconoce la paráfrasis legítima.

El mismo pedido CON RAG (contexto real) vs SIN RAG (sin contexto) hace medible el aporte del
conocimiento privado: sin él, el modelo no puede anclar y ambos groundings caen.
""")
code(r"""
contexto = ("- Simón Ocampo aprobó Bases de Datos con 9 en el año 3.\n"
            "- Simón Ocampo aprobó Análisis Matemático I con 9 en el año 1.")
anclada = "Simón rindió Bases de Datos con 9 y Análisis Matemático con 9, muy buen desempeño."
inventada = "Recomiendo cursar Astrofísica Cuántica y Derecho Romano el próximo cuatrimestre."
print("léxico   — anclada:", evaluar.grounding_score(anclada, contexto),
      "| inventada:", evaluar.grounding_score(inventada, contexto))
print("semántico — anclada:", evaluar.grounding_semantico(anclada, contexto),
      "| inventada:", evaluar.grounding_semantico(inventada, contexto))

if rag.ollama_disponible():
    cmp = evaluar.comparar_con_sin_rag(generar.plan_cursada, alumno="Simón Ocampo")
    print(f"\n[plan_cursada] grounding léxico   CON RAG = {cmp['con_rag']['grounding']}"
          f"  ·  SIN RAG = {cmp['sin_rag']['grounding']}")
    print(f"[plan_cursada] grounding semántico CON RAG = {cmp['con_rag']['grounding_sem']}"
          f"  ·  SIN RAG = {cmp['sin_rag']['grounding_sem']}")
else:
    print("\n(Ollama apagado) Encendé el modelo para la comparación con/sin RAG sobre un artefacto real.")
""")

md(r"""
## 9 · Conclusiones
- Construimos el **pipeline RAG completo** (extracción, corpus mixto, embeddings, Chroma,
  retrieval, generación) en vez de usar una caja negra.
- El sistema **genera** artefactos personalizados con tono, no responde preguntas: cada salida
  es contenido nuevo anclado en datos privados.
- La **demo con/sin RAG** y el **grounding** hacen medible y visible el aporte del conocimiento
  privado: sin él, el modelo no puede anclar y la calidad cae.
- Pipeline de Ciencia de Datos completo: EDA, visualización, clustering y evaluación.

La web app (`iniciar.bat`) permite probar todo esto en vivo, con la consola de contexto y el
medidor de grounding en tiempo real.
""")

nb["cells"] = cells
nb["metadata"] = {"language_info": {"name": "python"}, "kernelspec":
                  {"name": "python3", "display_name": "Python 3", "language": "python"}}
out = ROOT / "notebook.ipynb"
nbf.write(nb, out)
nbf.validate(nb)
print(f"OK: {out} ({len(cells)} celdas)")
