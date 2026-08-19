# Generador Académico (RAG)

**Del PDF frío del campus a una respuesta que razona sobre tu carrera.**

Sistema de IA generativa con persistencia híbrida: una base relacional con el historial académico real del grupo y una base vectorial con la documentación de la carrera. No responde preguntas — sintetiza documentos nuevos anclados en datos privados. Proyecto integrador de Ciencia de Datos, UTN FRLP 2026.

## 🚀 Características

- **Dos bases, cada dato donde corresponde** — lo tabular (notas, años, correlatividades) en SQLite, con lookup y join exactos; la documentación de la carrera en Chroma, con búsqueda semántica.
- **Artefactos, no respuestas** — planes de cursada, informes de trayectoria, cartas y recomendaciones de electivas, personalizados por alumno.
- **Demo con RAG vs sin RAG** — el mismo pedido, lado a lado, para ver qué aporta el contexto recuperado.
- **Streaming con grounding** — la API emite NDJSON token a token y devuelve de qué fragmentos salió cada afirmación.
- **Corre sin GPU y sin internet** — sin un LLM local disponible, el pipeline sigue funcionando en modo solo-retrieval: extracción, corpus, embeddings, retrieval, EDA, clustering y métricas.
- **Evaluación incluida** — métricas de retriever y de generación en `data/`.

## 📂 Estructura

```
app/                  Web app de testeo en vivo (API + frontend estático)
src/                  Pipeline RAG: extracción, corpus, embeddings, retrieval
data/raw/             Historial académico y plan de estudios (PDF)
data/                 Resultados de evaluación
colab/                Notebooks para correr y evaluar en Colab
informe/ guion/       Informe técnico y guion de presentación
estudio/              Guía de estudio generada
```

## 🛠️ Stack

Python · SQLite · ChromaDB · Ollama (LLM local, opcional) · Jupyter · HTML/CSS/JS para el frontend

## 💻 Puesta en marcha

```bash
git clone https://github.com/SimonOcampo1/generador-academico-rag.git
cd generador-academico-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

La notebook viene ejecutada de punta a punta: se puede leer sin correr nada. Para generación en vivo hace falta [Ollama](https://ollama.com); sin él, el resto del pipeline corre igual.
