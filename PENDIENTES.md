# Estado del proyecto

Actualizado: 2026-06-19. El trabajo está terminado de punta a punta y funcional.

## ✅ Hecho

- **Datos reales de los 6 integrantes** (estados académicos + exámenes). Eliminados los perfiles
  sintéticos: `analisis.tabla_grupo()` usa los 6 reales. Corpus: 418 docs, 199 notas reales.
- **Pipeline RAG completo** (pdfplumber → MiniLM → Chroma → Phi-4-mini/Ollama) con fallback a
  solo-retrieval si Ollama está apagado.
- **Parseo de correlatividades** arreglado (Nº35 "Seguridad en los Sistemas de Información"
  completa; Nº36 "Proyecto Final" sin arrastrar "(integradora)"). 36/36.
- **Evaluación**: grounding léxico **+ semántico** (embeddings) + demo con/sin RAG. Números
  reales en `data/eval_resultados.json` (léxico 0.80 con RAG vs 0.29 sin; semántico 0.64 vs 0.52).
- **EDA + figuras** regeneradas con los 6 reales (distribución, avance, radar, clustering, t-SNE).
- **Entregables** actualizados con datos reales y diseño homogéneo: `notebook.ipynb` (ejecutado
  end-to-end, 0 errores), informe / slides / guión (HTML + PDF con portada a página completa),
  web app FastAPI (demo en vivo con/sin RAG).
- **Modelo**: se adoptó **`phi4-mini` (3.8B)** tras comparar contra `qwen2.5:1.5b` en el mismo
  pipeline: mejor grounding léxico (0.80 vs 0.72 con RAG) y redacción mucho más fluida. Se arregló
  el `plan_cursada` (ahora deriva las materias cursables del grafo de correlatividades, sin sugerir
  aprobadas) y la truncación (`num_predict=512`; un tope alto hacía divagar al modelo chico).
- **Velocidad en CPU**: default `phi4-mini`, `num_predict=512`, `keep_alive`. ~60-130 s por
  artefacto en notebook con 8 GB.

## ⏳ Para la corrida final (en máquina con GPU)

- El integrante con GPU NVIDIA puede ejecutar `phi4-mini` (o un modelo más grande) para una demo
  más ágil. Setup en el README ("Rendimiento y elección de modelo"). Es **un comando**: re-correr
  `scripts/run_eval.py` + el notebook, y correr la web app en esa máquina el día de la demo.

## 📌 Tareas pendientes (próxima sesión)

1. **Auditoría final vs. la consigna.** Comparar el trabajo contra la consigna del TP
   (`Lineamientos TP Grupal 2026.pdf`), los resultados obtenidos y los entregables requeridos;
   verificar que esté todo cubierto y **dar una puntuación del 1 al 10** del trabajo.
2. **Superar las limitaciones documentadas.** Intentar implementar/mejorar los puntos de la sección
   "Limitaciones" del informe para hacer todo más robusto:
   - descripciones reales de cátedras/electivas en el corpus (recomendador por contenido, no solo nombre);
   - grounding con verificación de consistencia lógica (detectar contradicciones, no solo cercanía);
   - acotar las imprecisiones puntuales del modelo chico (p. ej. validar notas citadas contra el contexto).

## 💡 Trabajo futuro (documentado en el informe, no bloqueante)

- Incorporar descripciones reales de cátedras al corpus (recomendador por contenido, no solo nombre).
- Grounding con verificación de consistencia (detectar contradicciones, no solo cercanía).
