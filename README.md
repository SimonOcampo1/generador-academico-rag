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
  vive en una base **relacional** (lookup/join exacto, no semántica); la **documentación** de la
  carrera —texto largo— vive en la base **vectorial**, donde los embeddings hacen lo que un
  `SELECT` no puede. El agente **combina ambas** para su contexto.
- **No es NotebookLM.** El entregable es **construir el pipeline RAG nosotros** (parseo →
  persistencia híbrida → retrieval combinado → generación) y sumarle análisis y visualización.

---

## 🧠 Cómo funciona

```
                        ┌─ tabular (notas, correlativas) → SQLite (relacional) ─┐
PDFs → extracción → docs ┤                                                      ├─ contexto
                        └─ prosa + contenidos → embeddings (MiniLM) → Chroma ───┘  combinado
                                                                                   → LLM (Ollama)
                                                                                   → respuesta
```

El historial y las correlatividades se cargan como **filas** en SQLite; el diseño curricular y los
contenidos de cada materia van como **texto** en Chroma. El retrieval combina lookup relacional +
búsqueda semántica, y la generación se ancla solo en lo recuperado.

---

## 🚀 Cómo usarlo

> ⚙️ La generación corre sobre un LLM por **Ollama**. Lo más simple es darle la **GPU T4 gratis de
> Colab**. Modelo configurable con `OLLAMA_MODEL` (`qwen2.5:14b-instruct` en GPU · `phi4-mini` en CPU).

### Opción 1 · Web app interactiva, con la GPU de Colab (recomendado)

La consola en vivo con la demo **con / sin RAG**. La web corre en tu PC; solo la generación viaja a
Colab por un túnel de URL fija (ngrok).

1. Abrí `colab/generar_en_colab.ipynb` en Colab → **GPU T4** → *Ejecutar todo* (levanta el LLM + túnel).
2. En tu PC: doble clic en **`iniciar-colab.bat`** → abre `http://localhost:8000`.

<sub>Setup ngrok (una sola vez): cuenta gratis → authtoken + dominio fijo · en Colab **Secrets** agregá
`NGROK_AUTHTOKEN` · poné tu dominio en `DOMAIN` (notebook) y en `iniciar-colab.bat`.</sub>

### Opción 2 · Correr el generador punta a punta en Colab (sin web app, sin instalar nada)

Si solo querés ver el pipeline completo de principio a fin:

1. Subí **`notebook.ipynb`** a Colab → **GPU T4** → *Ejecutar todo*.
2. La 1ª celda te pide el **`.zip` de la entrega** (`Entrega-Grupo14-GeneradorAcademico.zip`) y lo
   descomprime sola. Corre todo: parseo → SQLite + Chroma → retrieval → generación → evaluación.

> Tarda unos minutos (la 1ª generación carga el modelo en la GPU). **Tené paciencia.** Sin GPU el
> pipeline RAG corre igual en modo solo-retrieval; solo la generación del texto necesita la GPU.

### Opción 3 · 100% local (GPU NVIDIA o CPU, offline)

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct     # o phi4-mini en CPU
iniciar.bat                          # enciende Ollama + web app + navegador → localhost:8000
```

Los lanzadores crean su propio entorno Python por máquina (en `%LOCALAPPDATA%`, **fuera** de
OneDrive) la 1ª vez, así el proyecto funciona en cualquier PC sin recrear nada.

<details><summary>Módulos sueltos / regenerar entregables</summary>

```bash
python src/db.py && python src/ingest.py   # construir base relacional + vectorial
python src/rag.py                           # respuesta híbrida (requiere Ollama)
python src/figuras.py                       # regenerar figuras
python scripts/build_notebook.py            # regenerar notebook.ipynb
python scripts/export_pdf.py                # informe + slides + guión a PDF (Edge/Chrome headless)
python scripts/build_zip.py                 # .zip de la entrega para Colab
```
</details>

---

## 🛠️ Stack

- **Extracción:** `pdfplumber`.
- **Base relacional:** `sqlite3` (stdlib) — historial académico + correlatividades (datos tabulares).
- **Base vectorial:** `chromadb` (embedding all-MiniLM-L6-v2, local y offline) — documentación + contenidos de materias.
- **Análisis/EDA:** `pandas`, `scikit-learn`, `matplotlib`.
- **Generación:** LLM vía **Ollama** (`qwen2.5:14b-instruct` en GPU · `phi4-mini` en CPU), configurable por `OLLAMA_MODEL`.

---

## 🚧 Estado

- [x] Persistencia híbrida: tabular → SQLite (relacional) · documentación → Chroma (vectorial).
- [x] Retrieval combinado (SQL + semántico) + generación RAG con Ollama (fallback a solo-retrieval).
- [x] Fichas de las 36 materias (contenidos mínimos, Ord. 1877) → recomendador que cita contenidos reales.
- [x] Correlatividades parseadas (36 obligatorias) + equivalencias 2008↔2023 (Anexo II).
- [x] Datos reales de los 6 integrantes + EDA + visualización + clustering.
- [x] 6 generadores de artefactos con control de tono.
- [x] Evaluación: grounding léxico + semántico + precisión factual + faithfulness (demo con/sin RAG).
- [x] Web app (consola en vivo, dark mode) + notebook integrador + informe / slides / guión.

---

<sub>Grupo 14 · Ciencia de Datos · UTN Facultad Regional La Plata · 2026</sub>
