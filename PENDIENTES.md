# Estado del proyecto

Actualizado: 2026-06-19. El trabajo está terminado de punta a punta y funcional.

## ✅ Hecho

- **Datos reales de los 6 integrantes** (estados académicos + exámenes). Eliminados los perfiles
  sintéticos: `analisis.tabla_grupo()` usa los 6 reales. Corpus: 418 docs, 199 notas reales.
- **Pipeline RAG completo** (pdfplumber → MiniLM → Chroma → qwen2.5/Ollama) con fallback a
  solo-retrieval si Ollama está apagado.
- **Parseo de correlatividades** arreglado (Nº35 "Seguridad en los Sistemas de Información"
  completa; Nº36 "Proyecto Final" sin arrastrar "(integradora)"). 36/36.
- **Evaluación**: grounding léxico **+ semántico** (embeddings) + demo con/sin RAG. Números
  reales en `data/eval_resultados.json` (léxico 0.74 con RAG vs 0.32 sin; semántico 0.61 vs 0.52).
- **EDA + figuras** regeneradas con los 6 reales (distribución, avance, radar, clustering, t-SNE).
- **Entregables** actualizados con datos reales y diseño homogéneo: `notebook.ipynb` (ejecutado
  end-to-end, 0 errores), informe / slides / guión (HTML + PDF con portada a página completa),
  web app FastAPI (demo en vivo con/sin RAG).
- **Velocidad en CPU**: default `qwen2.5:1.5b-instruct`, `num_predict=400`, contexto del plan
  reducido, `keep_alive`. ~60-70 s por artefacto en notebook con 8 GB.

## ⏳ Para la corrida final (en máquina con GPU)

- El integrante con GPU NVIDIA ejecuta con `qwen2.5:7b-instruct` para salidas más prolijas y demo
  ágil. Setup en el README (sección "Rendimiento y elección de modelo"). Es **un comando**:
  `set OLLAMA_MODEL=qwen2.5:7b-instruct` y re-correr `scripts/run_eval.py` + el notebook.
- Pasar de vuelta `data/eval_resultados.json` con los números del 7b y, el día de la demo, correr
  la web app en esa máquina.

## 💡 Trabajo futuro (documentado en el informe, no bloqueante)

- Incorporar descripciones reales de cátedras al corpus (recomendador por contenido, no solo nombre).
- Grounding con verificación de consistencia (detectar contradicciones, no solo cercanía).
