# Generador académico (RAG)

**Del PDF frío del campus a una respuesta que razona sobre tu carrera.**

Sistema de IA generativa que se alimenta de una **base de datos vectorial** con el historial
académico real del grupo (notas, materias, fechas) más la documentación de la carrera (plan
de estudios, correlatividades) y **genera** artefactos personalizados —planes de cursada,
informes de trayectoria, recomendaciones de electivas— que el modelo **no podría producir sin
ese conocimiento privado**. Proyecto integrador de Ciencia de Datos (UTN FRLP, 2026).

---

## 🎯 Por qué este proyecto

El modelo no conoce nuestras notas ni el régimen de correlatividades de la UTN FRLP: son
datos privados/de nicho. Sin la base vectorial **no puede responder**; con ella **genera**
respuestas concretas y citadas. El RAG no es un adorno: es el corazón del sistema.

- **No es un buscador SQL.** El corpus mezcla datos **estructurados** (notas, fechas) con
  **texto libre** (plan, correlatividades). El matching semántico de los embeddings hace lo
  que un `SELECT` no puede.
- **No es NotebookLM.** El entregable es **construir el pipeline RAG nosotros** (chunking →
  embeddings → base vectorial → retrieval → generación) y sumarle análisis y visualización.

---

## 🧠 Cómo funciona

```
PDFs (plan + estados académicos) → extracción → documentos → embeddings (MiniLM)
   → Chroma (vector DB) → retrieval top-k → qwen2.5 (Ollama) → respuesta citada
```

El estado académico se convierte en **un documento por materia** (hecho atómico); el plan se
parte en **chunks** de texto libre. La generación se ancla solo en lo recuperado.

---

## 📂 Estructura

```
data/raw/         PDFs fuente (plan de estudios, estados académicos, notas)
data/chroma/      base vectorial persistente (se genera; ignorada por git)
src/
  extract.py      PDF → texto (pdfplumber)
  documents.py    texto → documentos (estado: 1 por materia · plan: chunks)
  ingest.py       documentos → embeddings → Chroma + búsqueda semántica
  rag.py          retrieval → qwen2.5 (Ollama) → respuesta
  analisis.py     DataFrame de notas reales (6 integrantes) + agregaciones por área (EDA)
  figuras.py      EDA, radar, clustering (KMeans/PCA), t-SNE de embeddings → figs/
  generar.py      6 generadores de artefactos con control de tono
  evaluar.py      grounding léxico + semántico + comparar_con_sin_rag() (demo estrella)
app/              web app FastAPI: demo en vivo con/sin RAG (static/ = frontend)
assets/design.css sistema visual Graphite & Cobalt (compartido por todos los entregables)
figs/             PNGs del EDA/clustering (estilo homogéneo)
informe/          informe técnico (HTML + PDF)
slides/           slides de presentación (HTML con navegación + PDF)
guion/            guión de exposición por integrante (HTML + PDF)
scripts/          build_notebook.py · export_pdf.py
notebook.ipynb    notebook integrador (pipeline de punta a punta, ejecutado)
PRODUCT.md DESIGN.md  contexto de producto y sistema de diseño
docs/             propuesta, mail y notas de diseño del proyecto
```

---

## 💻 Puesta en marcha

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

python src/documents.py    # ver el corpus parseado (sin base vectorial)
python src/ingest.py       # construir la base vectorial + probar retrieval
python src/rag.py          # generar una respuesta (requiere Ollama; si no, muestra el contexto)
```

**Web app (demo en vivo):** doble clic en `iniciar.bat` (enciende Ollama + levanta la app +
abre el navegador), o manual:

```bash
python -m uvicorn app.main:app --port 8000   # http://localhost:8000
```

**Regenerar entregables:**

```bash
python scripts/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
python src/figuras.py        # regenerar figuras
python scripts/export_pdf.py # informe + slides + guión a PDF (Edge/Chrome headless)
```

Modelo local (recomendado para la generación):

```bash
ollama pull qwen2.5:1.5b-instruct
ollama serve
```

---

## ⚡ Rendimiento y elección de modelo

La generación corre sobre un LLM local; su velocidad depende del hardware. El default es
**`qwen2.5:1.5b-instruct`**, elegido como equilibrio velocidad/calidad para una notebook con
CPU y poca RAM (8 GB). Levers de velocidad ya aplicados: salida acotada (`num_predict=400`),
contexto del plan reducido a lo no aprobado, y modelo mantenido en memoria (`OLLAMA_KEEP_ALIVE`).

| Entorno | Modelo sugerido | Generación típica |
|---|---|---|
| CPU + 8 GB RAM (notebook básica) | `qwen2.5:1.5b-instruct` | ~60-70 s por artefacto |
| **GPU NVIDIA (≥6 GB VRAM)** | `qwen2.5:7b-instruct` | **pocos segundos** |

**Para la demo en vivo conviene una máquina con GPU NVIDIA.** Setup en esa máquina:

```bash
# 1) Clonar/copiar el repo (data/raw/ ya trae los PDFs → la base Chroma se reconstruye sola)
# 2) Instalar Ollama (detecta la GPU NVIDIA automáticamente, sin configuración)
ollama pull qwen2.5:7b-instruct
# 3) Entorno Python
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python src/ingest.py                       # reconstruye la base vectorial
set OLLAMA_MODEL=qwen2.5:7b-instruct        # usar el modelo grande
iniciar.bat                                 # o uvicorn (ver arriba)
```

> En CPU, cerrá apps pesadas y **pausá la sincronización de OneDrive** antes de la demo: con
> 8 GB de RAM el margen es chico y el swap a disco ralentiza la inferencia.

---

## 🛠️ Stack

- **Extracción:** `pdfplumber`.
- **Base vectorial + embeddings:** `chromadb` (embedding por defecto all-MiniLM-L6-v2, local y offline).
- **Generación:** LLM local vía **Ollama** (`qwen2.5`), configurable por `OLLAMA_MODEL`.

---

## 🚧 Estado

- [x] Extracción de PDFs (plan + estado académico) — verificada, texto digital limpio.
- [x] Documentos estructurados (1 por materia, con notas) + chunks del plan.
- [x] Base vectorial Chroma + retrieval semántico.
- [x] Generación RAG con Ollama (con fallback a solo-retrieval).
- [x] Parsear correlatividades a documentos por materia (36 obligatorias, join por nombre).
- [x] Datos reales de los 6 integrantes del grupo + EDA + visualización + clustering.
- [x] 6 generadores de artefactos con control de tono.
- [x] Evaluación: grounding léxico + semántico + demo con/sin RAG.
- [x] Sistema de diseño Graphite & Cobalt (PRODUCT.md / DESIGN.md / assets).
- [x] Web app (consola en vivo, demo con/sin RAG) + notebook integrador.
- [x] Informe técnico + slides + guión (HTML y PDF).
- [x] Web app: dark mode, vista única con pills, fallback elegante si Ollama crashea.
- [x] Equivalencias 2008↔2023 (Anexo II) para recuperar notas por equivalencia.
- [x] Ajustes de velocidad para CPU + guía de setup con GPU para la demo.

---

<sub>Grupo 14 · Ciencia de Datos · UTN Facultad Regional La Plata · 2026</sub>
