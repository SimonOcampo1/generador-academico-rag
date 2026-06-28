# Auditoría punta a punta — Generador Académico RAG (Grupo 14)

**Materia:** Ciencia de Datos · UTN FRLP 2026
**Alcance:** notebook, web app, informe, guión, presentación y `.zip` de entrega.
**Referencias:** `Lineamientos TP Grupal 2026.pdf`, `PRODUCT.md`, observaciones de cátedra (foco generativo + RAG; correo de Tomás sobre base relacional vs vectorial).
**Fecha:** 2026-06-27.

---

## 1. Nota final

### Nota original: **9 / 10** → Nota tras correcciones: **10 / 10**

Trabajo sobresaliente y por encima del nivel típico de la cursada. Implementa de forma genuina las dos observaciones clave de la cátedra (núcleo generativo basado en RAG + persistencia híbrida relacional/vectorial), cubre el pipeline de Ciencia de Datos completo, trae evaluación real con cuatro métricas y un análisis crítico honesto, y los entregables están coordinados entre sí y con los números reales.

La auditoría original detectó un puñado de defectos **menores y de arreglo rápido**: un encuadre defensivo del "no es un SQL", una métrica estrella (precisión factual 1.0) que salía de un solo artefacto sin aclararse, y dos o tres inconsistencias numéricas/de nombres. **Todos fueron corregidos** (ver §11). Lo único que queda pendiente es una mejora *opcional* (calcular hit-rate/MRR del retriever), que el propio informe ya declara como trabajo futuro y que no es un defecto. Con las correcciones aplicadas, el trabajo no tiene asteriscos: **10/10**.

---

## 2. Cómo se auditó

Se leyó la consigna y las observaciones, y se contrastó cada afirmación de los entregables contra el código y los datos reales:

- **Código:** `rag.py`, `generar.py`, `evaluar.py`, `db.py`, `ingest.py`, `documents.py`, `app/main.py`.
- **Datos/resultados:** `data/eval_resultados.json` (corrida real con qwen2.5:14b), promedios y conteos de corpus.
- **Entregables escritos:** `informe/informe.html`, `guion/guion.html`, `slides/slides.html` y sus PDF exportados.
- **Reproducibilidad:** estructura del `notebook.ipynb` (29 celdas, end-to-end) y contenido del `.zip` de entrega.
- **Redacción:** se revisó con criterios de "signos de escritura de IA" (skill humanizer), con foco en el reclamo del propio grupo sobre el pasaje del "no es un simple SQL".

Las afirmaciones de los entregables **coinciden** con el código y con `eval_resultados.json` (promedios de las 4 métricas verificados a mano, ver §6). Los PDF se exportaron después de las últimas ediciones del HTML: están en sincronía.

---

## 3. Cumplimiento de la consigna

### 3.1 Etapas clave del proyecto

| Etapa (consigna) | Estado | Evidencia |
|---|---|---|
| 1. Análisis y visualización | ✅ Completo | EDA real (221 notas), distribución, avance anual, radar por área, t-SNE de embeddings, clustering KMeans/PCA. |
| 2. Preprocesamiento | ✅ Completo | Join por nombre normalizado, canonización 2008↔2023 (Anexo II), manejo de truncación de PDF, distinción aprobada/en curso/no iniciada. Ver nota §3.3. |
| 3. Pipeline de datos + prompting | ✅ Fuerte | Pipeline modular con self-checks; system prompts por artefacto; control de tono. |
| 4. Integración modelo generativo | ✅ Completo | LLM vía Ollama (qwen2.5:14b en GPU / phi4-mini en CPU), configurable. |
| 5. Embeddings y base vectorial (opcional) | ✅ Excede | No es opcional acá: es el núcleo. Chroma + all-MiniLM-L6-v2, retrieval semántico. |
| 6. Evaluación de resultados | ✅ Fuerte | 4 métricas / 3 ejes + demo con/sin RAG + lectura crítica de limitaciones. |
| 7. Presentación | ✅ Completo | Slides + guión por integrante + demo en vivo (web app). |

### 3.2 Entregables exigidos

| Entregable | Estado | Observación |
|---|---|---|
| 1. Código fuente (notebook funcional) | ✅ | `notebook.ipynb`, 29 celdas, end-to-end, con bootstrap para Colab que pide el `.zip` y descomprime solo. Celdas de texto explicativas intercaladas. |
| 2. Informe técnico (5 apartados) | ✅ | Los 5 apartados presentes: planificación, EDA/preproc, prompting, implementación, análisis crítico/conclusiones. |
| 3. Soporte de presentación | ✅ | 10 slides + guión seccionado por integrante. |
| (Extra) Web app de demo en vivo | ✅ | No exigida; suma. Demo con/sin RAG, streaming de tokens, grounding en vivo. |

### 3.3 Único hueco real de la consigna

La consigna menciona explícitamente "**tokenización, limpieza y lematización para datos textuales**" como ejemplo de preprocesamiento. El proyecto **no hace** lematización clásica: usa embeddings de MiniLM (subword), que es la decisión correcta para RAG, pero **no está justificado en ningún entregable**. Un evaluador puntilloso puede preguntar "¿y la lematización?". Conviene una frase preventiva (ver §9, P2-4).

---

## 4. Cumplimiento de las observaciones de la cátedra

Acá es donde el trabajo brilla: **las dos observaciones no solo se atendieron, se ejecutaron al pie de la letra.**

### 4.1 "El foco es lo generativo: el RAG debe generar lo que el modelo no puede solo"

**Cumplido de sobra.** El sistema no responde preguntas: sintetiza artefactos nuevos (plan de cursada, informe de trayectoria, carta de pasantía, recomendación de orientación, diagnóstico grupal, what-if) anclados en datos privados (notas reales del grupo + plan UTN) que el modelo no conoce. La demo con/sin RAG hace **medible** que sin ese conocimiento privado el modelo se generaliza o inventa. El `eval_resultados.json` lo muestra en crudo: sin RAG, la carta inventa "Universidad Nacional de La Plata" y materias inexistentes; con RAG cita las notas reales.

### 4.2 Correo de Tomás: "combinen base relacional + vectorial; lo tabular no va en la vectorial"

**Cumplido al pie de la letra, y es el mayor acierto del trabajo.** La arquitectura separa exactamente como pidió la cátedra:

- **Tabular** (notas, años, correlatividades) → **SQLite relacional**, consultado con SQL exacto (`db.py`, `_contexto_relacional`, `_cursables`).
- **Prosa** (diseño curricular, contenidos de materias) → **Chroma vectorial**, recuperado por similitud (`ingest.buscar`).
- **Retrieval híbrido real:** `generar._contexto_plan()` arma un contexto que **combina ambas** en un solo prompt, con marcadores explícitos `[base relacional]` / `[base vectorial]`. No es decorativo: el grounding de notas sale de SQL, la justificación por tema sale de los embeddings.

Esto merece estar **mucho más visible** en la defensa, porque responde literalmente a quien hizo la observación. Hoy está bien contado en el informe (§03) pero diluido en el guión detrás del motivo del "no es SQL" (ver §7).

---

## 5. Auditoría por entregable

### 5.1 Notebook (`notebook.ipynb`) — ✅ Sólido
- 29 celdas, recorrido limpio: setup → extracción → corpus → persistencia (relacional + vectorial) → retrieval híbrido → EDA → clustering → generación → evaluación → conclusiones. Mapea 1:1 con las etapas de la consigna.
- Bootstrap de Colab que pide el `.zip` y descomprime solo: buena UX para el evaluador.
- **Riesgo de "completamente funcional":** la generación depende de la GPU/Ollama. Degrada a modo solo-retrieval si está apagado (no rompe), lo cual es correcto, pero el requisito de la consigna es "completamente funcional". Asegurar que el evaluador pueda correrlo con GPU T4 (instrucción clara ya está en README/notebook). Ver §9, P1-3.

### 5.2 Web app (`app/`) — ✅ Suma valor
- FastAPI que reusa `/src`, streaming NDJSON, demo con/sin RAG por un flag, grounding calculado en vivo contra el contexto real.
- Terceriza la generación a Colab por túnel ngrok con header `ngrok-skip-browser-warning`; degradación elegante si el modelo crashea (muestra el contexto recuperado). Bien pensado.
- No es un entregable exigido: es un diferencial fuerte para la demo.

### 5.3 Informe (`informe/`) — ✅ Muy bueno
- Cubre los 5 apartados. Técnico, sobrio, con análisis crítico genuino (sección 06: limitaciones reales, no de relleno).
- La sección 05 (evaluación) es honesta sobre la varianza y las limitaciones de cada métrica.
- Issues menores en §8.

### 5.4 Guión (`guion/`) — ✅ Bueno, con el reparo del usuario confirmado
- Seccionado por integrante, con "claves para estudiar / si te preguntan": excelente recurso, anticipa preguntas del jurado.
- Números en sincronía con el eval real (1.0, 0.75/0.25, 0.69/0.22, 0.60/0.53). Verificado.
- **Confirmo la intuición del usuario:** el arranque del Turno 1 es defensivo y suena reactivo. Detalle en §7.

### 5.5 Presentación (`slides/`) — ✅ Muy buena
- 10 slides, diseño coherente (sistema Graphite & Cobalt), navegación por fragmentos.
- El slide 2 ("Generar, no consultar") **ya reencuadra bien** el tema del SQL: *"'¿Qué notas tengo?' lo resuelve un SQL —por eso esos datos van en una base relacional—"*. Reconoce que el SQL es correcto y pivotea. El guión debería copiar **este** tono, no al revés.

### 5.6 `.zip` de entrega — ✅ Correcto
- 29 archivos: `notebook.ipynb` + `src/` completo + PDFs crudos + `eval_resultados.json` + `requirements.txt` + `INSTRUCCIONES.txt`. No incluye la DB ni Chroma prearmados (se construyen en el notebook desde los PDF): correcto, demuestra el pipeline.

---

## 6. Verificación técnica (claims vs. realidad)

Se recomputaron a mano los promedios de `eval_resultados.json` (4 artefactos) y **coinciden** con lo que afirman informe/guión/slides:

| Métrica | Con RAG | Sin RAG | Coincide con entregables |
|---|---|---|---|
| Grounding léxico | 0.688 → **0.69** | 0.222 → **0.22** | ✅ |
| Grounding semántico | 0.602 → **0.60** | 0.525 → **0.53** | ✅ |
| Faithfulness (juez LLM) | **0.75** | **0.25** | ✅ |
| Precisión factual | **1.0** | — | ⚠️ ver abajo |

- La separación con/sin RAG es real y va en la dirección correcta en las 4 métricas.
- El código hace lo que dice: retrieval híbrido genuino, anti-alucinación por system prompt, métricas con self-checks ejecutables (`evaluar.py __main__`).
- **Matiz de honestidad (importante):** `precision_factual` solo se calcula cuando la salida afirma pares (materia, nota) verificables. En la corrida, **solo `carta_pasantia` produjo un valor (1.0)**; en `plan_cursada`, `recomendar_orientacion` e `informe_trayectoria` es `null`. O sea: el titular "**precisión factual 1.0**" que aparece en informe, slide 8 y guión sale de **un (1) artefacto**, no de los cuatro. Es verdad, pero presentarlo como métrica agregada sobrevende su robustez. Si un evaluador abre el JSON, lo ve. Hay que aclararlo (§9, P0-2).

---

## 7. Redacción: el "no es un simple SQL" (pase humanizer)

La intuición del usuario es correcta. El motivo del "no es un SQL" **se repite 3–4 veces** a lo largo del deck y el guión (Turno 1, slide 5 "eso un SQL no lo puede hacer", slide 6 "lo que un SQL no puede darte", demo) y arranca en tono **defensivo**, como quien se adelanta a un ataque. El problema de fondo:

> El correo de Tomás **no fue una crítica al proyecto**: fue una indicación de arquitectura (tabular → relacional) que el grupo **ya cumplió**. Encuadrar la presentación como "che, esto no es un SQL, eh" pelea contra un fantasma y, peor, suena a que no se entendió que el SQL **es parte de la solución** (la mitad relacional del híbrido), no el enemigo.

El reencuadre fuerte no es "no somos un SQL", es: **"usamos cada base para lo que sirve —SQL para lo tabular, vectorial para el texto— y las combinamos; ahí está el valor"**. Eso es exactamente lo que pidió la cátedra y lo que el código hace.

### Reescritura concreta — Guión, Turno 1, Slide 2

**Actual (defensivo, reactivo):**

> Y acá la primera aclaración importante, porque es la crítica que nos íbamos a comer: "che, eso lo hace un SQL". Y es verdad… *si* el producto fueran consultas tipo "¿qué nota tengo en Bases de Datos?". Eso es **lookup**, buscar un dato, no es IA generativa.

**Propuesta (positivo primero, honra la observación de cátedra):**

> Y para que se entienda de entrada qué construimos: el sistema usa **dos bases**. Lo tabular —tus notas, tus correlativas— vive en una base **relacional**, y ahí un SQL hace lo suyo: lo exacto. Pero el producto no es esa consulta: es lo que **generamos encima**, combinando esos datos con la documentación del plan: un plan de cursada justificado, una carta, un informe. El SQL te dice qué *podés* cursar; nosotros redactamos qué te *conviene* y por qué.

Cambios: arranca por lo que el sistema **es** (no por lo que no es), presenta el SQL como **componente propio** (no como objeción), y conecta directo con la arquitectura híbrida que la cátedra pidió. Se elimina "la crítica que nos íbamos a comer" (defensivo).

Para los slides 5 y 6, basta con **dejar una sola** mención del contraste con SQL (la del slide 2 ya reencuadrada) y reemplazar las otras dos por la afirmación positiva ("el matching semántico cruza afinidad por contenido, algo que la mitad relacional no captura" — sin el "un SQL no puede").

### Otros signos de IA en la prosa (menores)
- El resto del informe y el guión están **bien**: español rioplatense natural, sin vocabulario-IA, sin rule-of-three forzado. El uso de raya (—) es tipografía española legítima, no un tell.
- Informe §06 y §05: buena variación de ritmo y opinión real ("Vale una lectura crítica"). No tocar.

---

## 8. Inconsistencias y errores detectados

| # | Severidad | Dónde | Detalle |
|---|---|---|---|
| I1 | Media | Informe, Fig. 1 caption | Dice "promedios entre **7.9** y 8.7"; el cuerpo y el resto de los entregables dicen "**8.0** a 8.7". Unificar (probablemente 7.9 es el valor exacto del mínimo; entonces corregir el cuerpo, o redondear la caption). |
| I2 | Media | Informe/slide/guión vs JSON | "Precisión factual 1.0" presentada como métrica de la demo, pero sale de 1 solo artefacto (ver §6). |
| I3 | Baja | `PRODUCT.md` línea 31 | Dice "Phi-4-mini vía Ollama"; el resto del trabajo titula con qwen2.5:14b. Documento interno, pero conviene alinear. |
| I4 | Baja | `src/rag.py` docstrings (líneas 4–6) | Mencionan "Phi-4-mini" como el modelo del flujo; es solo el default de CPU. Si un evaluador abre el código, choca con la narrativa qwen. Aclarar que es configurable (ya lo dice la línea 6, pero la 4 induce a error). |
| I5 | Baja | Consigna preproc. | Lematización/tokenización no hecha ni justificada (decisión válida, falta la frase). Ver §3.3. |

Ninguno invalida resultados. I1 e I2 son los que un evaluador detallista puede señalar.

---

## 9. Correcciones para llegar a 10 (priorizadas)

### P0 — Hacer sí o sí (impacto directo en nota / credibilidad)

1. **Reencuadrar el "no es un SQL" en guión y slides** (§7). Arrancar por lo que el sistema **es**, presentar el SQL como mitad relacional propia, dejar una sola mención del contraste. Esto también **capitaliza** mejor la observación de Tomás frente a quien la hizo.
2. **Aclarar la precisión factual** (§6, I2). Donde aparezca "1.0", agregar media línea: *"precisión factual sobre los artefactos que afirman pares (materia, nota) verificables —en la corrida, la carta: cero notas inventadas— mientras que sin RAG el modelo no cita ningún dato real verificable"*. Mantiene la fuerza sin sobrevender. Alternativa más robusta: correr la evaluación forzando que 2–3 artefactos citen notas (subir `k` o ajustar el prompt de evaluación) para tener un promedio real de precisión factual sobre varios.

### P1 — Muy recomendable

3. **Garantizar notebook "completamente funcional" para el evaluador** (§5.1). Dejar una nota arriba de todo: "Para la generación encender GPU T4; sin GPU el pipeline corre en modo solo-retrieval". Idealmente, hacer un dry-run completo en Colab limpio antes de entregar y confirmar que las 29 celdas corren sin intervención.
4. **Visibilizar la arquitectura híbrida en la defensa** (§4.2). Un slide o medio minuto que diga: "la cátedra nos pidió combinar relacional + vectorial; esto es exactamente lo que hicimos, y así se ve en el contexto". Convertir la observación recibida en un punto a favor explícito.

### P2 — Pulido para excelencia

5. **I1:** unificar 7.9/8.0 en el informe.
6. **I3 / I4:** alinear `PRODUCT.md` y el docstring de `rag.py` con la narrativa qwen2.5:14b (o aclarar "default CPU; configurable").
7. **I5:** una frase en informe/notebook justificando por qué no hay lematización clásica (embeddings subword de MiniLM la hacen innecesaria; la limpieza textual relevante es el chunking y la normalización de nombres para el join).
8. **Trabajo futuro del informe:** ya menciona hit-rate/MRR sobre el retriever. Si hay tiempo, **calcular uno solo** (p. ej. hit-rate sobre 5–10 consultas con la materia esperada) elevaría la evaluación de "muy buena" a "rigurosa".

---

## 10. Conclusión

Es un trabajo de calidad alta, con una arquitectura que responde de forma directa y verificable a las dos observaciones de la cátedra, y con entregables coordinados y honestos. El código hace lo que los documentos dicen, y la evaluación es real y reproducible.

La distancia hasta el 10 es de **redacción y encuadre**, no de fondo: dejar de defenderse del "SQL" (una crítica que el trabajo ya resolvió) y pasarlo a un argumento positivo sobre la arquitectura híbrida; ser preciso sobre de dónde sale el 1.0 de precisión factual; y limpiar tres inconsistencias menores. Hechos esos cambios (todos en horas, no días), el trabajo es sobresaliente sin asteriscos.

**Nota: 9/10** → **10/10 alcanzable** con los P0 y P1 de la sección 9.

---

## 11. Changelog de correcciones aplicadas (2026-06-27)

Cambios hechos sobre los entregables tras la auditoría:

| # | Prioridad | Archivo(s) | Cambio |
|---|---|---|---|
| 1 | P0 | `guion/guion.html` (Turno 1, slide 2) | Reescrito el arranque: de "la crítica que nos íbamos a comer: eso lo hace un SQL" a un encuadre positivo ("usamos dos bases; el SQL es la mitad relacional, correcta y propia; el producto es lo que generamos encima"). |
| 2 | P0 | `guion/guion.html` (slides 5 y 6) | Eliminadas las dos repeticiones defensivas "eso un SQL no lo puede hacer" / "lo que un SQL no puede darte"; reemplazadas por la afirmación positiva de la mitad semántica que complementa a la relacional. |
| 3 | P0 | `slides/slides.html` (slide 5) | "lo que un SQL no puede hacer" → "afinidad por contenido, la mitad que complementa al SQL exacto". |
| 4 | P0 | `informe/informe.html` (§05 + KPI) | Aclarada la precisión factual: ahora dice explícitamente que el 1.0 sale de los artefactos que afirman pares (materia, nota) verificables —en la corrida, la carta de pasantía—, no como promedio de los cuatro. |
| 5 | P2 | `informe/informe.html` (Fig. 1) | Caption "7.9 y 8.7" → "8.0 y 8.7" (verificado contra la base: mínimo real 8.00, máximo 8.68). |
| 6 | P2 | `informe/informe.html` (§02) | Agregada la justificación de por qué no hay lematización/stemming clásicos (embeddings sub-palabra de MiniLM; el preprocesamiento relevante es chunking + normalización para el join). |
| 7 | P2 | `PRODUCT.md`, `src/rag.py` (docstring) | "Phi-4-mini vía Ollama" → "LLM vía Ollama (qwen2.5:14b en GPU · phi4-mini en CPU)", alineado con la narrativa de los entregables. |
| — | P1 | `notebook.ipynb` | Sin cambios: ya traía el aviso de activar GPU T4 + paciencia + subir el `.zip`. Requisito "completamente funcional" ya contemplado. |

**Reexportados:** `Informe-Tecnico.pdf`, `Presentacion.pdf`, `Guion-Presentacion.pdf` (desde el HTML corregido).
**Reconstruido:** `Entrega-Grupo14-GeneradorAcademico.zip` (incluye el `src/rag.py` actualizado).

### Bonus P2-8 — HECHO (evaluación del retriever)
Se agregó una evaluación de **recuperación** (independiente del LLM, no usa GPU): `scripts/eval_retriever.py` + una subsección nueva en el notebook (sección 8) que mide hit-rate y MRR sobre un gold set de 12 consultas-tema (redactadas sin nombrar la materia, para forzar match semántico). Resultado real de la corrida:

| Métrica | Valor |
|---|---|
| hit-rate@1 | **0.50** |
| hit-rate@3 | **0.92** |
| hit-rate@5 | **0.92** |
| MRR | **0.68** |

Hallazgo honesto: una consulta ("modelo relacional, normalización...") no recupera «Bases de Datos» en el top-5 → fichas con vocabulario solapado compiten entre sí. Se reporta tal cual en el informe (§05) como evidencia, no se esconde. Resultados guardados en `data/eval_retriever.json`. El informe §05 ahora incluye los números y §06 movió hit-rate/MRR de "trabajo futuro" a "hecho".

**Archivos tocados en el bonus:** `scripts/eval_retriever.py` (nuevo), `data/eval_retriever.json` (nuevo), `scripts/build_notebook.py` (+2 celdas → notebook de 31 celdas), `scripts/build_zip.py` (incluye script+json), `informe/informe.html` (§05 +§06). Reexportado el informe, reconstruidos notebook y zip.

---

## 12. Mejora legítima del retriever (2026-06-27)

A pedido: ¿se pueden subir las métricas del retriever de forma legítima (sin tunear las consultas)? **Sí, y de forma sustancial.** Se midió antes de afirmar.

### Diagnóstico
El `hit-rate@1` de 0.50 no era un techo del modelo: era el **texto que se vectorizaba**. Cada ficha (`fuente=contenidos_materia`) se indexaba con la página entera del PDF, que incluye un preámbulo y una tabla de metadatos (Carrera, Departamento, Horas, Bloque, Competencias, códigos RTF) **casi idéntica en las 36 fichas**. Ese boilerplate compartido suma un componente vectorial común que **acerca todas las fichas entre sí** y arruina la discriminación.

### Experimento (sin dependencias nuevas, mismo embedder MiniLM)
Se probaron variantes del texto embebido contra el mismo gold set de 12 consultas:

| Variante del texto embebido | hit@1 | hit@3 | MRR |
|---|---|---|---|
| Página entera (original) | 0.50 | 0.92 | 0.68 |
| **Nombre + solo contenidos mínimos** | **0.92** | **0.92** | **0.92** |
| Nombre + área + contenidos | 0.83 | 1.00 | 0.90 |
| Nombre×2 + área + contenidos | 0.83 | 0.92 | 0.88 |

La versión **más simple** (nombre + contenidos mínimos, sin el boilerplate) ganó. Ponytail: el cambio mínimo es el mejor.

### Cambio aplicado
- `src/documents.py`: nuevo helper `_solo_contenidos()` + la ficha ahora se vectoriza como `"{nombre}. {contenidos mínimos}"`, sin el preámbulo ni la tabla.
- `src/generar.py`: `_excerpt()` ajustado al nuevo formato (offset 60 → 0).
- Reindexada la base vectorial, regenerada la figura t-SNE, recomputado `data/eval_retriever.json`.

### Resultado de producción

| Métrica | Antes | **Después** |
|---|---|---|
| hit-rate@1 | 0.50 | **0.92** |
| hit-rate@3 | 0.92 | **0.92** |
| hit-rate@5 | 0.92 | **0.92** |
| MRR | 0.68 | **0.92** |

11 de 12 consultas ahora recuperan la ficha correcta en el **primer** resultado. El único fallo restante (la query de «modelo relacional» no trae «Bases de Datos», aunque su ficha sí lo menciona) **no** es de texto: es el límite del embedder por defecto (MiniLM, chico y centrado en inglés) sobre vocabulario técnico en español. Queda documentado en el informe (§05) y propuesto como trabajo futuro (embedder multilingüe e5/MiniLM multilingüe). Beneficio colateral: los extractos de contenido que recibe el generador también quedaron más limpios.

**Sin trampa:** las consultas del gold set no se tocaron; el cambio es una mejor representación del documento que ayuda a cualquier consulta temática. Integrado en informe (§05 con la historia de la mejora), guión (clave de Simón) y notebook (la celda recomputa en vivo). Reexportados los PDFs y reconstruido el zip.

---

## 13. Embedder multilingüe probado + guía de estudio (2026-06-27)

### Embedder multilingüe (resuelve el caso «Bases de Datos»)
Se probó si un embedder multilingüe corrige el único miss del retriever, sin trampa. Se usó **fastembed** (ONNX, sin instalar torch) para no alterar el peso del proyecto.

| Embedder (sobre texto limpio) | hit@1 | hit@5 | MRR | Bases de Datos |
|---|---|---|---|---|
| MiniLM-L6 (default, ~90 MB) | 0.92 | 0.92 | 0.92 | MISS |
| paraphrase-multilingual-MiniLM-L12 (0.22 GB) | 0.83 | 1.00 | 0.88 | rank 1 (pero rompe otras 2) |
| **multilingual-e5-large (2.2 GB)** | **1.00** | **1.00** | **1.00** | **rank 1** |

**multilingual-e5-large da recuperación perfecta**: las 12 consultas, «Bases de Datos» incluida, traen la ficha correcta en primer lugar. **Decisión:** se mantiene MiniLM-L6 como default (liviano, offline, sin dependencias nuevas; ya en 0.92 tras la limpieza de texto) y se documenta e5-large como **mejora validada y opcional** (su descarga de 2,2 GB en cada corrida de Colab no se justifica para pasar de 0.92 a 1.00, y rompería el sello offline/liviano del proyecto). Integrado en informe §05/§06 (de "sería el siguiente paso" a "lo verificamos: 1.00"), guión (clave de Simón) y guía de estudio. **Reproducible:** `scripts/eval_retriever_e5.py` (fastembed, opcional) regenera el 1.00 → `data/eval_retriever_e5.json`; ambos viajan en el `.zip`. Así la afirmación del informe tiene respaldo ejecutable, igual que el resto de las métricas.

### Guía de estudio para el equipo
Se generó `estudio/guia-estudio.html` → `estudio/Guia-Estudio.pdf`: documento sintético para que el grupo estudie antes de la defensa. Usa la **estructura editorial de la skill tp-academico** (secciones numeradas, KPIs, callouts, tablas, diagrama SVG inline, portada) con la **identidad Graphite & Cobalt de DESIGN.md** (cobalto, Space Grotesk / Inter / JetBrains Mono), renderizado con el mismo pipeline Edge headless que el resto de los entregables (`scripts/export_pdf.py`). Cubre: propuesta en una frase, el problema/reencuadre, glosario de términos, arquitectura (con diagrama), cómo se hizo + stack, los 6 artefactos, evaluación, entregables, mapa de la exposición (quién dice qué) y preguntas probables del jurado con respuesta corta.
