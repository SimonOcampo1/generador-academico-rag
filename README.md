# Generador académico (RAG)

**Del PDF frío del campus a una respuesta que razona sobre tu carrera.**

Sistema de IA generativa con **persistencia híbrida**: una **base relacional** (SQLite) con el
historial académico real del grupo (notas, materias, años) y las correlatividades del plan, y una
**base vectorial** (Chroma) con la documentación de la carrera. Combinando ambas, **genera**
artefactos personalizados —planes de cursada, informes de trayectoria, recomendaciones de
electivas— que el modelo **no podría producir sin ese conocimiento privado**. Proyecto integrador
de Ciencia de Datos (UTN FRLP, 2026).

---

## 🎯 Por qué este proyecto

El modelo no conoce nuestras notas ni el régimen de correlatividades de la UTN FRLP: son
datos privados/de nicho. Sin esas bases **no puede responder**; con ellas **genera**
respuestas concretas y citadas. El RAG no es un adorno: es el corazón del sistema.

- **Cada dato en la base que le corresponde.** Lo **tabular** (notas, años, correlatividades)
  vive en una base **relacional** (lookup/join exacto, no semántica); la **documentación** del
  plan —texto largo— vive en la base **vectorial**, donde los embeddings hacen lo que un `SELECT`
  no puede. El agente **combina ambas** para su contexto.
- **No es NotebookLM.** El entregable es **construir el pipeline RAG nosotros** (parseo →
  persistencia híbrida → retrieval combinado → generación) y sumarle análisis y visualización.

---

## 🧠 Cómo funciona

```
                        ┌─ tabular (notas, correlativas) → SQLite (relacional) ─┐
PDFs → extracción → docs ┤                                                      ├─ contexto
                        └─ prosa del plan → embeddings (MiniLM) → Chroma (vector)┘  combinado
                                                                                   → Phi-4-mini
                                                                                     (Ollama)
                                                                                   → respuesta
```

El estado académico y las correlatividades se cargan como **filas** en SQLite (datos tabulares);
el plan se parte en **chunks** de texto libre en Chroma. El retrieval combina lookup relacional +
búsqueda semántica, y la generación se ancla solo en lo recuperado.

---

## 📂 Estructura

```
data/raw/         PDFs fuente (plan, estados académicos, notas, diseño curricular Ord. 1877)
data/academico.db base RELACIONAL SQLite (historial + correlatividades; se genera, ignorada por git)
data/chroma/      base VECTORIAL persistente (documentación del plan; se genera, ignorada por git)
src/
  extract.py      PDF → texto (pdfplumber)
  documents.py    texto → documentos (estado: 1 por materia · correlativas · plan: chunks · fichas de materia)
  db.py           datos tabulares → SQLite (relacional) + consultas SQL (historial, correlativas)
  ingest.py       prosa del plan + contenidos de materias → embeddings → Chroma + búsqueda semántica
  rag.py          retrieval HÍBRIDO (SQL + vector) → Phi-4-mini (Ollama) → respuesta
  analisis.py     DataFrame de notas reales (6 integrantes, desde SQLite) + agregaciones (EDA)
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

python src/documents.py    # ver el corpus parseado (sin persistir)
python src/db.py           # construir la base RELACIONAL (SQLite) + probar consultas
python src/ingest.py       # construir la base VECTORIAL (Chroma) + probar retrieval
python src/rag.py          # respuesta híbrida (relacional + vector; requiere Ollama, si no muestra contexto)
```

**Web app (demo en vivo):** doble clic en `iniciar.bat` (enciende Ollama + levanta la app +
abre el navegador), o manual:

```bash
python -m uvicorn app.main:app --port 8000   # http://localhost:8000
```

> **Portabilidad entre máquinas (OneDrive).** Los lanzadores (`iniciar.bat` / `iniciar-colab.bat`)
> crean su propio entorno Python en `%LOCALAPPDATA%\generador-academico-rag-venv` —**fuera** de
> OneDrive— e instalan las dependencias la primera vez en cada máquina (`scripts/setup-venv.bat`).
> Así el proyecto funciona en notebook y escritorio sin recrear nada: un venv dentro de OneDrive se
> sincroniza pero queda atado al intérprete de la PC que lo creó y se rompe en la otra. Si tenés un
> `.venv` viejo en la carpeta del proyecto, podés borrarlo: ya no se usa.

**Regenerar entregables:**

```bash
python scripts/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
python src/figuras.py        # regenerar figuras
python scripts/export_pdf.py # informe + slides + guión a PDF (Edge/Chrome headless)
```

Modelo local (recomendado para la generación):

```bash
ollama pull phi4-mini
ollama serve
```

---

## ⚡ Rendimiento y elección de modelo

La generación corre sobre un LLM por Ollama; el modelo es configurable con `OLLAMA_MODEL` y la
velocidad depende del hardware. Para la **mejor calidad** usamos **`qwen2.5:14b-instruct`** en GPU
(la redacción interpreta y selecciona datos en vez de listar notas); en CPU sin GPU el default es
**`phi4-mini`** (3.8B), más liviano. Levers ya aplicados: tope de salida (`OLLAMA_NUM_PREDICT`),
contexto compacto y modelo mantenido en memoria (`OLLAMA_KEEP_ALIVE`).

| Entorno | Modelo sugerido | Generación típica |
|---|---|---|
| CPU + 8 GB RAM (notebook básica) | `phi4-mini` | ~60-130 s por artefacto |
| **GPU NVIDIA local (≥8 GB VRAM)** | `qwen2.5:14b-instruct` | **~30-45 s** |
| **GPU en Colab (T4 16 GB, gratis)** | `qwen2.5:14b-instruct` (o `7b` para más velocidad) | **~30-45 s** |

El pipeline RAG (extracción, embeddings, Chroma, retrieval) corre siempre local en CPU; lo
único que pesa es la generación del LLM. Hay dos formas de darle GPU para la demo, y en ambas
**no cambia una línea de código**: `rag.py` lee la URL del modelo de la variable `OLLAMA_URL`.

### Opción A — PC con GPU NVIDIA local

Todo offline y privado en una sola máquina. Setup en la máquina con GPU:

```bash
# 1) Clonar/copiar el repo (data/raw/ ya trae los PDFs → ambas bases se reconstruyen solas)
# 2) Instalar Ollama (detecta la GPU NVIDIA automáticamente, sin configuración)
ollama pull qwen2.5:7b-instruct
# 3) Entorno Python
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python src/db.py && python src/ingest.py    # reconstruye la base relacional + la vectorial
set OLLAMA_MODEL=qwen2.5:7b-instruct        # o editá iniciar.bat
iniciar.bat                                 # o uvicorn (ver arriba)
```

### Opción B — LLM en Google Colab (GPU T4 gratis, sin GPU propia)

Si nadie tiene GPU, Colab presta una T4 gratis. La web sigue en tu PC; solo la generación
viaja a Colab por un túnel con **URL fija** (ngrok), así no se copia ni pega nada.

**Setup por única vez:**

```text
1) Cuenta gratis en ngrok → copiá tu authtoken y reclamá tu dominio estático gratis
   (Dashboard → Domains, ej. grupo14-rag.ngrok-free.app).
2) En Colab: panel 🔑 (Secrets) → agregá NGROK_AUTHTOKEN con tu token.
3) Poné tu dominio en la variable DOMAIN del notebook Y en iniciar-colab.bat (OLLAMA_URL).
```

**Cada demo (2 pasos, cero copy-paste):**

```text
1) Abrí colab/generar_en_colab.ipynb → GPU (T4) → Entorno de ejecución → Ejecutar todo.
2) En tu PC: doble clic en iniciar-colab.bat. Listo.
```

La respuesta aparece donde siempre: en la web app (`localhost:8000`) y en la terminal si
corrés los scripts. Colab solo mueve el cómputo; el modelo 7b en GPU responde en pocos segundos.

> Privacidad: con la Opción B los prompts corren en **tu propio Ollama** sobre una VM efímera de
> Colab que se destruye al cerrar el runtime —ningún proveedor de inferencia ingiere los datos—,
> a diferencia de una API hosted. No es 100 % offline como la Opción A, pero sí "tu modelo, VM
> descartable". Para la demo conviene mostrar ambas: B para que se vea fluido, A como prod offline.

> En CPU, cerrá apps pesadas y **pausá la sincronización de OneDrive** antes de la demo: con
> 8 GB de RAM el margen es chico y el swap a disco ralentiza la inferencia.

### Notebook integrador en Colab (autocontenida)

`notebook.ipynb` recorre el pipeline de punta a punta y se reproduce con *Ejecutar todo*
(en Colab con GPU T4 se ven también las generaciones del LLM).

Para correrla en Google Colab: subí `notebook.ipynb`, elegí *Entorno de ejecución → GPU T4*
y *Ejecutar todo*. La 2ª celda detecta Colab y pide subir el zip de la entrega
(`Entrega-Grupo14-GeneradorAcademico.zip`); lo descomprime sola, sin editar nada. Después la celda
de setup instala dependencias, levanta `qwen2.5:14b-instruct` en la GPU y corre el pipeline
completo (corpus → SQLite + Chroma → retrieval híbrido → generación → evaluación con las 4 métricas).

Es **completamente funcional sin GPU**: si no hay modelo disponible, el pipeline RAG (extracción,
persistencia relacional + vectorial, retrieval, EDA, clustering y métricas) corre igual en modo
solo-retrieval; lo único que requiere GPU es la generación del texto del LLM.

---

## 🛠️ Stack

- **Extracción:** `pdfplumber`.
- **Base relacional:** `sqlite3` (stdlib): historial académico + correlatividades (datos tabulares).
- **Base vectorial + embeddings:** `chromadb` (embedding por defecto all-MiniLM-L6-v2, local y offline): documentación del plan.
- **Generación:** LLM vía **Ollama** (`qwen2.5:14b-instruct` en GPU; `phi4-mini` en CPU), configurable por `OLLAMA_MODEL`.

---

## 🚧 Estado

- [x] Extracción de PDFs (plan + estado académico) — verificada, texto digital limpio.
- [x] Documentos estructurados (1 por materia, con notas) + chunks del plan.
- [x] **Persistencia híbrida**: datos tabulares (historial + correlatividades) en SQLite (relacional); documentación del plan en Chroma (vectorial).
- [x] Retrieval combinado (lookup SQL + búsqueda semántica) + generación RAG con Ollama (fallback a solo-retrieval).
- [x] Fichas de las 36 materias (contenidos mínimos, Ord. 1877) en la vectorial → recomendador que cita contenidos reales, no solo nombres.
- [x] Parsear correlatividades a tabla relacional (36 obligatorias, join por nombre).
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
