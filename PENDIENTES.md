# Estado del proyecto

Actualizado: 2026-06-19. El trabajo está terminado de punta a punta y funcional.

## ✅ Hecho

- **Datos reales de los 6 integrantes** (estados académicos + exámenes). Eliminados los perfiles
  sintéticos: `analisis.tabla_grupo()` usa los 6 reales. Corpus: 413 filas relacional + 136 docs
  vectorial, 221 notas reales.
- **Pipeline RAG completo** (pdfplumber → SQLite + Chroma → Phi-4-mini/Ollama) con fallback a
  solo-retrieval si Ollama está apagado.
- **Persistencia híbrida (corrección de cátedra, Tomás).** Los datos tabulares (historial
  académico + correlatividades) se movieron a una base **relacional** (SQLite, `src/db.py`); la
  base **vectorial** (Chroma) queda solo con la documentación del plan. El retrieval ahora combina
  ambas (`rag.py` y `generar.py`). Cada dato en la base que le corresponde.
- **Diseño Curricular (Ord. 1877) en la vectorial.** Se ingieren las 36 fichas de materia (con
  objetivos, competencias, contenidos mínimos y cargas horarias) + la prosa del diseño curricular.
  El recomendador (`recomendar_orientacion`, `plan_cursada`) ahora **cita contenidos reales** de
  las materias, no solo nombres (resuelve el "trabajo futuro" del recomendador por contenido).
- **Matching de nombres de materias robustecido.** Alias para variantes (p. ej. "Desarrollo de
  Aplicaciones Móviles" ≡ "Aplicaciones Móviles"), limpieza de "(Elec" truncado, y el acta de
  notas manda (si tiene nota ahí, está aprobada). Sin duplicados ni electivas aprobadas ofrecidas
  como disponibles.
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
   - ~~descripciones de cátedras en el corpus (recomendador por contenido)~~ ✅ hecho (fichas Ord. 1877);
     falta solo sumar descripciones de las **electivas** (no detalladas en la ordenanza);
   - grounding con verificación de consistencia lógica (detectar contradicciones, no solo cercanía);
   - acotar las imprecisiones puntuales del modelo chico (p. ej. validar notas citadas contra el contexto).

## 💡 Trabajo futuro (documentado en el informe, no bloqueante)

- Sumar descripciones de las **electivas** (las obligatorias ya tienen contenidos en la vectorial).
- Grounding con verificación de consistencia (detectar contradicciones, no solo cercanía).
