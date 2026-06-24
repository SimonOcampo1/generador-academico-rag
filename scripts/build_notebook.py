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

Sistema de IA generativa con **persistencia híbrida** —base **relacional** (SQLite: historial +
correlatividades) + base **vectorial** (Chroma: documentación y contenidos de materias)— que
**combina ambas** y *sintetiza* artefactos académicos personalizados (planes de cursada, informes,
cartas, recomendaciones). No responde preguntas: genera documentos nuevos anclados en datos
privados que el modelo no conoce.
""")
md(r"""
<div style="background:#FFF8E1;border-left:6px solid #E0A100;padding:14px 18px;border-radius:8px;font-size:15px;line-height:1.5">
<b>⚠️ Antes de ejecutar — leé esto</b>
<ol style="margin:8px 0 0;padding-left:20px">
<li>🖥️ <b>Activá la GPU T4</b>: <i>Entorno de ejecución → Cambiar tipo de entorno → <b>T4 GPU</b></i>. Sin esto la generación no corre en GPU.</li>
<li>⏳ <b>Tené paciencia</b>: la corrida completa tarda <b>unos minutos</b> (instala dependencias y carga el modelo en la GPU; la 1ª generación es la más lenta).</li>
<li>📦 <b>El Paso 1 te pide el proyecto</b>: cuando lo pida, subí el archivo <code>Entrega-Grupo14-GeneradorAcademico.zip</code> (trae <code>src/</code> y <code>data/raw/</code>). Lo descomprime solo.</li>
</ol>
<span style="color:#5B6066">Después: <i>Entorno de ejecución → Ejecutar todo</i> (Ctrl+F9).</span>
</div>
""")
md(r"""
**Por qué híbrido y no todo vectorial:** los datos tabulares (notas, correlativas) se consultan con
exactitud (lookup/join SQL); la documentación de la carrera (texto largo) es donde los embeddings
le ganan a un `SELECT`. Cada dato en la base que le corresponde.

```
                       ┌─ tabular ─────────▶ SQLite  (base relacional) ─┐
PDFs ─pdfplumber─▶ docs ┤                                               ├─▶ contexto combinado
                       └─ prosa + fichas ──▶ MiniLM ─▶ Chroma (vector) ─┘   ─▶ LLM ─▶ artefacto
```
Si no hay GPU/LLM, la parte RAG funciona igual (recupera el contexto) y solo falta la generación.
""")

md(r"""
## 0 · Setup

Dos celdas: **(1) Bootstrap** —en Colab, si el proyecto no está presente, te pide subir el `.zip`
de la entrega y lo descomprime solo— y **(2) Entorno** —instala dependencias y, en Colab, levanta
el modelo en la GPU—. **En local** (parado en la carpeta del proyecto) ninguna de las dos hace
falta tocar: el bootstrap no hace nada y el setup solo agrega `src/` al path.
""")
md(r"""
### Paso 1 · Cargar el proyecto (Colab)
Cuando esta celda lo pida, **subí `Entrega-Grupo14-GeneradorAcademico.zip`** (el zip de la entrega;
contiene `src/` y `data/raw/`). La celda lo descomprime sola — no hay que tocar nada más.

> *Alternativa con Drive:* subí esa **carpeta `generador-academico-rag/`** a tu Drive, y al inicio
> de esta celda agregá `from google.colab import drive; drive.mount('/content/drive')` y
> `os.chdir('/content/drive/MyDrive/generador-academico-rag')`. En **local** esta celda no hace nada.
""")
code(r"""
# Bootstrap (solo Colab): si el proyecto no está presente, pide el .zip de la entrega y lo descomprime.
import os, sys
from pathlib import Path

if "google.colab" in sys.modules and not Path("src").exists():
    import zipfile
    from google.colab import files
    print("Subí el .zip de la entrega: Entrega-Grupo14-GeneradorAcademico.zip")
    subido = files.upload()
    zips = [n for n in subido if n.endswith(".zip")]
    if not zips:
        raise RuntimeError(
            "No subiste ningún .zip. Subí 'Entrega-Grupo14-GeneradorAcademico.zip' (debe contener "
            "src/ y data/raw/) y volvé a correr esta celda.")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall()
        raiz = z.namelist()[0].split("/")[0]   # carpeta raíz dentro del zip
    if Path(raiz, "src").exists():
        os.chdir(raiz)                          # el zip trae una carpeta raíz: entrar en ella
    if not Path("src").exists():
        raise RuntimeError(
            "El .zip no contenía 'src/'. Subí el zip correcto (Entrega-Grupo14-GeneradorAcademico.zip, "
            "armado con scripts/build_zip.py).")
    print("✅ Proyecto cargado. Carpeta de trabajo:", os.getcwd())
""")
md(r"""
### Paso 2 · Entorno (dependencias + modelo)
Instala las dependencias y, **en Colab con GPU**, levanta el modelo para ver la generación en vivo.
""")
code(r"""
import os, sys, subprocess, time
from pathlib import Path

EN_COLAB = "google.colab" in sys.modules

if not Path("src").exists():
    raise RuntimeError(
        "No encuentro 'src/'. En Colab: corré primero la celda de Bootstrap (Paso 1) y subí el "
        ".zip de la entrega. En local: abrí el notebook DESDE la carpeta del proyecto (con src/ y data/raw/).")

if EN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    # LLM en la GPU de Colab para ver la generación. Si falla (sin GPU), seguimos en solo-retrieval.
    MODELO = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")  # 7B = rápido; 14B para mejor calidad
    try:
        subprocess.run("apt-get -qq install -y zstd pciutils", shell=True, check=True)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
        os.environ["OLLAMA_HOST"] = "0.0.0.0:11434"
        subprocess.Popen(["ollama", "serve"]); time.sleep(5)
        subprocess.run(["ollama", "pull", MODELO], check=True)
        os.environ["OLLAMA_MODEL"] = MODELO
    except Exception as e:
        print("No se pudo levantar Ollama en Colab (¿sin GPU?); sigo en modo solo-retrieval:", e)

sys.path.insert(0, str(Path.cwd() / "src"))  # importar los módulos del proyecto
import analisis, db, documents, extract, figuras, generar, ingest, rag, evaluar
from IPython.display import Image, display

print("Modelo:", rag.OLLAMA_MODEL, "|", "ENCENDIDO" if rag.ollama_disponible() else "apagado (solo-retrieval)")
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
`documents.construir_corpus` parsea los PDFs en fuentes que van a **bases distintas según
su naturaleza**:
- `estado_academico` (tabular) → base **relacional**: una fila por (alumno, materia) con nota, año.
- `correlatividades` (tabular) → base **relacional**: las 36 obligatorias con sus correlativas.
- `plan_estudios` (prosa) → base **vectorial**: chunks del diseño curricular (Ord. 1877).
- `contenidos_materia` (prosa) → base **vectorial**: una ficha por materia con sus contenidos mínimos.

La clave de join entre estado y plan es el *nombre normalizado* (`normalizar`), porque los
códigos institucionales no coinciden entre documentos.
""")
code(r"""
corpus = documents.construir_corpus()
print(f"Total documentos en el corpus: {len(corpus)}")
from collections import Counter
print("Por fuente (→ destino):", dict(Counter(d["metadata"]["fuente"] for d in corpus)))
print("  estado_academico, correlatividades → SQLite (relacional) · plan_estudios, contenidos_materia → Chroma (vectorial)")

print("\nEjemplo (estado académico, tabular → relacional):")
ej = next(d for d in corpus if d["metadata"]["fuente"] == "estado_academico" and d["metadata"]["nota"] >= 0)
print(" texto:", ej["text"])
print(" metadata:", ej["metadata"])
""")

md(r"""
## 3 · Persistencia: base relacional (SQLite) + base vectorial (Chroma)
Cada dato en la base que le corresponde:
- `db.reindexar` carga los **datos tabulares** (historial + correlatividades) en **SQLite**. Son
  registros estructurados: lookup y join determinístico, no semántica.
- `ingest.reindexar` vectoriza la **documentación de la carrera** (prosa del diseño curricular +
  fichas de materia con sus contenidos mínimos) con **all-MiniLM-L6-v2** (embedding por defecto de
  Chroma, local y offline) y la persiste para el matching semántico.
""")
code(r"""
rel = db.reindexar()
print(f"Base RELACIONAL (SQLite) → {db.DB_PATH}")
print(f"  historia: {rel['historia']} filas · correlatividades: {rel['correlatividades']} filas")

n = ingest.reindexar()
print(f"\nBase VECTORIAL (Chroma) → {ingest.CHROMA_DIR}")
print(f"  {n} documentos de la carrera (prosa del diseño + fichas de materia)")
""")

md(r"""
## 4 · Retrieval híbrido: combinar las dos bases
- **Relacional** (`db`, SQL): grounding exacto sobre datos tabulares (las aprobadas de un alumno,
  las correlativas de una materia). Es lo que ancla a los generadores.
- **Vectorial** (`ingest.buscar`): similitud de embeddings sobre la documentación de la carrera
  (diseño curricular + contenidos de cada materia), para texto libre.
- `rag.generar` los **combina**: una misma consulta trae datos del alumno (SQL) + documentación
  de la carrera (vector) en un solo contexto.
""")
code(r"""
print("### Relacional (SQL): materias aprobadas de un alumno")
aprobadas = db.aprobadas("Simón Ocampo")
print(f"  {len(aprobadas)} materias con nota. Ejemplos:", [r["materia"] for r in aprobadas[:5]])
print("  Correlativas de Ciencia de Datos:",
      next(c for c in db.correlativas() if c["materia"].startswith("Ciencia de Datos")))

print("\n### Vectorial (embeddings): contenidos de materias por similitud semántica")
for r in ingest.buscar("machine learning y modelos de datos", k=2):
    print(f"  [{r['distancia']:.3f}] ({r['metadata']['fuente']}) {r['metadata'].get('materia','')}: {r['text'][:70]}")

print("\n### Híbrido (rag.generar combina ambas en el contexto)")
ctx = rag.generar("¿Qué necesito para cursar Ciencia de Datos y cómo le fue a Simón?")["contexto"]
print(ctx[:400], "...")
""")

md(r"""
## 5 · EDA + visualización
`analisis` arma un DataFrame de notas **reales de los 6 integrantes del grupo** consultando la
**base relacional** (los datos tabulares se analizan donde corresponde, con SQL, no en la
vectorial). Cada registro es `(alumno, materia, año, nota, área)`; las áreas temáticas mapean las
36 obligatorias del plan en 6 grupos.
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
## 8 · Evaluación — ¿cuánto del contexto usó el modelo?
**Qué medimos:** cuánto de la respuesta se ancla en el **contexto inyectado** (lo que tomó del RAG)
y cuánto resuelve el modelo con su **conocimiento propio**. El *grounding* mide esa fracción
anclada; correr el mismo pedido **con** vs **sin** contexto aísla el aporte del conocimiento privado
(el salto con−sin). Lo cuantificamos con **cuatro métricas sobre tres ejes**:
- **Solapamiento** — *grounding léxico* (fracción de tokens informativos de la salida presentes
  literalmente en el contexto; penaliza paráfrasis) y *grounding semántico* (máximo coseno de
  embeddings salida↔contexto; reconoce la reformulación legítima).
- **Corrección factual** — *precisión factual*: extrae los pares (materia, nota) que la salida
  afirma y verifica cada uno contra el registro real del alumno. Mide acierto, no cercanía.
- **Juicio de un LLM** — *faithfulness*: el propio modelo puntúa 0–1 cuánto se apoya la respuesta
  solo en el contexto (LLM-as-judge).

El mismo pedido CON RAG vs SIN RAG hace medible el aporte del conocimiento privado.
""")
code(r"""
# Validación rápida de las métricas léxica/factual sobre un caso de juguete (no requiere LLM)
contexto = ("- Simón Ocampo aprobó Bases de Datos con 9 en el año 3.\n"
            "- Simón Ocampo aprobó Análisis Matemático I con 9 en el año 1.")
anclada = "Simón rindió Bases de Datos con 9 y Análisis Matemático I con 9, muy buen desempeño."
inventada = "Recomiendo cursar Astrofísica Cuántica y Derecho Romano el próximo cuatrimestre."
print("léxico    — anclada:", evaluar.grounding_score(anclada, contexto),
      "| inventada:", evaluar.grounding_score(inventada, contexto))
registro = {"bases de datos": 9, "analisis matematico i": 9}
print("precisión — anclada:", evaluar.precision_factual(anclada, registro)["precision"],
      "| inventada:", evaluar.precision_factual(inventada, registro))
""")
code(r"""
# Demo estrella CON vs SIN RAG sobre un artefacto real (requiere el LLM encendido)
if rag.ollama_disponible():
    cmp = evaluar.comparar_con_sin_rag(generar.recomendar_orientacion, alumno="Mora Gentil")
    con, sin = cmp["con_rag"], cmp["sin_rag"]
    pf = (con.get("precision_factual") or {}).get("precision")
    print(f"USO DEL CONTEXTO (con RAG): {cmp['uso_contexto_con_rag']:.0%} de la respuesta anclada "
          f"en el contexto · {cmp['conocimiento_propio_con_rag']:.0%} conocimiento propio")
    print(f"APORTE DEL RAG (con − sin): +{cmp['aporte_contexto']:.3f} de grounding\n")
    print(f"grounding léxico    CON={con['grounding']}  SIN={sin['grounding']}")
    print(f"grounding semántico CON={con['grounding_sem']}  SIN={sin['grounding_sem']}")
    print(f"precisión factual   CON={pf}  SIN={(sin.get('precision_factual') or {}).get('precision')}")
    print(f"faithfulness (juez) CON={con.get('faithfulness')}  SIN={sin.get('faithfulness')}")
    print("\nResultados agregados (4 artefactos) en data/eval_resultados.json; ver scripts/run_eval.py")
else:
    print("(LLM apagado) Encendé el modelo (celda 0 en Colab) para la comparación con/sin RAG.")
""")
md(r"""
La figura agrega los 4 artefactos: para cada uno, el grounding **con** vs **sin** RAG. El salto
entre las barras es **cuánto aportó el contexto inyectado** (lo que el modelo no resolvía solo).
""")
code(r"""
# Figura de la evaluación (usa data/eval_resultados.json; regenerable con scripts/run_eval.py)
from pathlib import Path
_p = figuras.FIGS / "aporte_rag.png"
display(Image(filename=str(_p))) if Path(_p).exists() else print("Corré scripts/run_eval.py para generar la figura.")
""")

md(r"""
## 9 · Conclusiones
- Construimos el **pipeline RAG completo** (extracción, corpus mixto, **persistencia híbrida**
  relacional + vectorial, retrieval combinado, generación) en vez de usar una caja negra.
- **Cada dato en su base:** lo tabular (historial, correlatividades) en SQLite; la documentación
  del plan en Chroma. El agente combina ambas para su contexto.
- El sistema **genera** artefactos personalizados con tono, no responde preguntas: cada salida
  es contenido nuevo anclado en datos privados.
- La **demo con/sin RAG** y las **cuatro métricas** (léxico, semántico, precisión factual y
  faithfulness) hacen medible el aporte del conocimiento privado: con RAG la precisión factual es
  1.0 (cero notas inventadas); sin RAG el modelo no puede anclar.
- Pipeline de Ciencia de Datos completo: EDA, visualización, clustering y evaluación.

La web app (`iniciar.bat` local, o `iniciar-colab.bat` apuntando a la GPU de Colab) permite probar
todo esto en vivo, con la consola de contexto y el medidor de grounding en tiempo real.
""")

nb["cells"] = cells
nb["metadata"] = {"language_info": {"name": "python"}, "kernelspec":
                  {"name": "python3", "display_name": "Python 3", "language": "python"}}
out = ROOT / "notebook.ipynb"
nbf.write(nb, out)
nbf.validate(nb)
print(f"OK: {out} ({len(cells)} celdas)")
