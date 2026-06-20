# Propuestas TP Grupal — Sistema RAG (Grupo 14)

> **Por qué cambiamos de rumbo.** El dataset de vinos es algo que los modelos modernos
> ya dominan: el modelo responde bien *con o sin* el dataset, así que la base vectorial
> no es fundamental. El profe pide que el **RAG sea el corazón** del trabajo: una base de
> datos vectorial con información que el modelo **NO conoce**, y que el modelo use ese
> conocimiento para responder. Las 4 propuestas cumplen eso.

---

## Lo que se reutiliza (sirve para las 4)

- **Modelo local**: Ollama + `phi4-mini` (3.8B). Corre en la PC, sin GPU.
- **Frontend**: FastAPI + la web app con la "consola en vivo" (se ve el prompt real + respuesta token a token). Ideal para la demo.
- **Embeddings**: `sentence-transformers` (MiniLM) que ya usábamos para el recomendador.
- **Evaluación**: módulo de métricas (grounding, coherencia) ya existente.
- **Notebook + informe + slides**: misma estructura de entregables.

## Lo que se agrega (el núcleo nuevo, igual para las 4)

```
documentos → chunking → embeddings (MiniLM) → Chroma (vector DB) → retrieval top-k → Phi-4-mini → respuesta citada
```

- **Chroma** como base vectorial (liviana, local, `pip install`).
- **Demo estrella**: botón "con RAG / sin RAG". Sin RAG el modelo alucina o dice "no tengo esa info"; con RAG responde **correcto y citando la fuente**. Eso prueba ante el profe, en vivo, que el dataset es indispensable.
- Pega con la teoría: **Chunking**, **Embeddings y Bases Vectoriales** (etapa 5), **Prompt Engineering**, **IA Agéntica**.

---

## Propuesta 1 — Asistente académico del Grupo 14 ⭐ (recomendada)

**Idea.** RAG sobre los datos académicos reales de cada integrante: materias cursadas,
notas, fechas de aprobación, finales pendientes, correlatividades, promedio. Sacados del
campus virtual / certificados analíticos. Le preguntás en lenguaje natural y responde.

**Ejemplos de preguntas.** "¿Qué materias aprobó Simón en 2025?" · "¿Quién tiene mejor
promedio del grupo?" · "¿Qué le falta a Cata para recibirse?" · "Generá un resumen del
avance académico de cada uno." · "¿Qué finales podemos rendir juntos el próximo turno?"

**Por qué resuelve la consigna.** Imposible que el modelo conozca esto: son datos
privados nuestros. El contraste con/sin RAG es brutal y obvio para cualquiera.

**Wow factor.** Altísimo: el profe ve al instante que el modelo aprendió de *nuestros*
datos. Personal, divertido, demostrable.

**Esfuerzo.** Bajo. Pocos datos, fáciles de estructurar (un CSV/JSON por integrante).
**Riesgo.** Bajo. Único cuidado: son notas reales (ok, son nuestras; podemos anonimizar nombres si queremos).

---

## Propuesta 2 — Asistente de Reglamento y Trámites UTN FRLP

**Idea.** RAG sobre la normativa de la facultad: reglamento de cursada, régimen de
correlatividades, plan de estudios, calendario académico, instructivos de trámites
(equivalencias, regularidad, exámenes). PDFs públicos pero "oscuros".

**Ejemplos.** "¿Cuántas materias puedo rendir por turno?" · "¿Qué necesito para mantener
la regularidad?" · "¿Qué correlativas pide Ciencia de Datos?" · "¿Cómo tramito una equivalencia?"

**Por qué resuelve la consigna.** El modelo NO conoce los artículos ni fechas específicas
de la UTN FRLP → sin RAG **inventa** (alucinación clarísima). Con RAG responde citando el
artículo exacto. Demo de contraste perfecta.

**Wow factor.** Alto y útil de verdad (lo usaría cualquier alumno).
**Esfuerzo.** Medio (juntar y limpiar los PDFs de reglamento).
**Riesgo.** Bajo. Datos públicos, sin tema de privacidad.

---

## Propuesta 3 — Memoria del Grupo (RAG sobre el chat de WhatsApp)

**Idea.** Exportamos el historial del grupo de WhatsApp y lo vectorizamos. El asistente
responde sobre lo que pasó: decisiones, acuerdos, links compartidos, chistes internos.

**Ejemplos.** "¿Qué decidimos sobre la fecha de entrega?" · "¿Quién dijo que iba a hacer
el informe?" · "Resumí lo que hablamos del proyecto la semana pasada." · "¿Qué links pasaron sobre RAG?"

**Por qué resuelve la consigna.** Datos 100% privados y post-cutoff: el modelo no tiene
forma de conocerlos. Imposible de responder sin la base vectorial.

**Wow factor.** Muy alto, gracioso y personal. Engancha en la presentación.
**Esfuerzo.** Bajo (exportar chat es un toque; parsear el .txt es trivial).
**Riesgo.** Bajo. Cuidar privacidad → es nuestro propio chat, ok.

---

## Propuesta 4 — Base de conocimiento de un dominio de nicho

**Idea.** RAG sobre un corpus que el modelo demostrablemente NO domina: wiki/patch-notes
de un videojuego, reglas de un juego de mesa, lore de una saga, o documentación técnica
de nicho. Elegimos algo donde el modelo *se equivoca* sin contexto.

**Ejemplos** (según el dominio): "¿Qué cambió en el último parche del personaje X?" ·
"¿Cómo se resuelve este caso según las reglas oficiales?" · "Explicá la mecánica Y."

**Por qué resuelve la consigna.** Mostramos primero al modelo fallando solo, después
acertando con RAG. El más "blindado" contra el reclamo "esto ya lo sabía el modelo".

**Wow factor.** Medio-alto (depende de qué tan copado sea el dominio elegido).
**Esfuerzo.** Medio (conseguir y limpiar el corpus).
**Riesgo.** Bajo, pero hay que elegir bien el dominio para que el contraste sea fuerte.

---

## Comparación rápida

| # | Propuesta | Modelo no lo conoce | Wow | Esfuerzo | Riesgo |
|---|-----------|:---:|:---:|:---:|:---:|
| 1 | Asistente académico del grupo ⭐ | ✅✅ | ✅✅✅ | Bajo | Bajo |
| 2 | Reglamento/Trámites UTN | ✅✅ | ✅✅ | Medio | Bajo |
| 3 | Memoria del chat del grupo | ✅✅✅ | ✅✅✅ | Bajo | Bajo |
| 4 | Dominio de nicho | ✅✅✅ | ✅✅ | Medio | Bajo |

**Recomendación.** La **1** (o combinarla con la 3) da el contraste con/sin RAG más claro,
el menor esfuerzo de datos, y la demo más vistosa. Las cuatro se terminan en horas
reutilizando casi todo el proyecto de ayer.

> Pasen la encuesta, elijan tópico, y arrancamos.
