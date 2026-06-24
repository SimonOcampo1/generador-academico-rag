# Propuesta 1 (votada) — reencuadre: de "asistente de preguntas" a GENERADOR académico

## Las dos dudas a resolver

1. **"Un SQL lo resuelve."** Verdad si el producto son consultas tipo "¿qué notas tiene
   X?". Eso es *lookup*, no IA generativa. → Reencuadre: el producto son **artefactos
   nuevos generados** (planes, informes, CV, simulaciones), no respuestas.
2. **"Esto ya es NotebookLM."** → El entregable del TP es **construir el pipeline RAG
   nosotros** y mostrarlo; NotebookLM es caja negra. Además nosotros *generamos* (no solo
   respondemos) y hacemos el pipeline DS completo (EDA, visualización, clustering).

---

## Objetivo y alcance (concreto)

**Objetivo.** Construir un sistema de IA generativa que, alimentado por **dos bases de datos**
—una relacional con el historial académico del grupo + correlatividades, y una vectorial con la
documentación de la carrera— **genere artefactos académicos personalizados** (planes de cursada,
informes de trayectoria, perfiles para pasantías, simulaciones) imposibles de producir sin ese
conocimiento privado.

**Alcance (lo que SÍ hace).**
- Ingesta de un corpus mixto con **persistencia híbrida**: lo estructurado (notas, fechas,
  correlativas, promedio) en una base **relacional** (SQLite); lo **no estructurado** (prosa del
  plan, descripciones de cátedras) en una base **vectorial** (Chroma).
- Pipeline RAG propio: parseo → (SQLite + embeddings MiniLM/Chroma) → retrieval híbrido → Phi-4-mini.
- Generación de los artefactos de abajo, con **tono controlable**.
- EDA + visualización del avance académico + clustering de perfiles.
- Evaluación: grounding (¿la salida usa datos reales del RAG?) + contraste con/sin RAG.

**Fuera de alcance.** No es un sistema de gestión, no edita datos, no se conecta al campus
en vivo. Datos cargados como CSV/JSON + PDFs una vez.

---

## Por qué es generativo y NO un SQL (los casos que sorprenden)

> La regla: cada salida es **texto/contenido nuevo sintetizado**, con razonamiento o
> estilo, no un campo de una tabla.

1. **Plan de cursada personalizado y justificado.** "Armá mi próximo cuatrimestre."
   → Genera un plan ordenado con *justificación en prosa*: cruza correlativas aprobadas +
   calendario + tu patrón de rendimiento. SQL te da qué *podés* cursar; esto te dice qué
   *conviene* y por qué.

2. **Recomendador de electivas/orientación (acá los embeddings le ganan al SQL).**
   "¿Qué electivas me convienen?" → Hace *matching semántico* entre las materias donde te
   fue bien/te gustaron (estructurado) y las **descripciones de las electivas** (texto
   libre). Esto un SQL no lo puede hacer: requiere similitud semántica sobre texto.

3. **Informe de trayectoria narrativa, con tono.** "Contá mi historia académica."
   → Genera una narrativa coherente + fortalezas/debilidades + consejo, en registro
   *motivacional*, *técnico* o *brutalmente honesto*. El control de tono = puro generativo.

4. **CV académico / carta de motivación para una pasantía.** "Generá mi carta para una
   pasantía en Data Science." → El modelo *selecciona y redacta* desde los datos privados
   apuntando a un objetivo. Esto NotebookLM no lo hace; es generación de documento con formato.

5. **Diagnóstico grupal generativo.** "Repartí roles para un proyecto de X según las
   fortalezas académicas de cada integrante." → Sintetiza entre todos los perfiles y
   asigna con justificación.

6. **Simulación what-if.** "¿Cómo queda mi promedio y mi fecha estimada de recibida si
   apruebo estas 3 con 8?" → Proyecta el escenario y lo explica en prosa.

**Demo estrella:** mismo input, botón **con RAG / sin RAG**. Sin RAG el modelo inventa o
dice "no sé"; con RAG genera el plan/CV/informe correcto citando los datos reales.

---

## Defensa vs NotebookLM (para la exposición)

| | NotebookLM | Nuestro TP |
|---|---|---|
| Pipeline RAG | caja negra | **lo construimos y mostramos** (chunking, embeddings, vector DB, retrieval) |
| Salida | responde preguntas | **genera artefactos** (plan, CV, informe, simulación) con tono |
| Datos | solo texto subido | **estructurado + no estructurado fusionados** + lógica de dominio (correlativas, promedio) |
| Análisis | no | **EDA + visualización + clustering** (etapas que pide el lineamiento) |
| Entorno | nube/Google | local, privado, offline |

El punto de fondo: el TP no se evalúa por "tener un chatbot", se evalúa por **demostrar que
sabemos armar el pipeline RAG** y usarlo para generar. NotebookLM no demuestra eso, lo oculta.

---

## Reutilización y esfuerzo

- **Reutilizamos:** Ollama+Phi-4-mini, web app con consola en vivo, MiniLM, evaluación, notebook/informe/slides.
- **Agregamos:** SQLite (relacional) + Chroma (vectorial) + chunking + retrieval híbrido + los generadores de artefactos.
- **Esfuerzo:** bajo-medio. El dato es poco; lo nuevo es el pipeline RAG (igual para cualquier propuesta).
- **Riesgo:** notas reales → podemos anonimizar nombres si alguien prefiere.

---

## Alternativas específicas (por si el grupo quiere re-votar)

### Alt A — Generador de material de estudio desde los apuntes de la cátedra
RAG sobre los PDFs de **teoría de esta materia**. Genera **parciales de práctica, flashcards
y guías de estudio** con la *terminología y el enfoque exactos del profe* (no la versión
genérica de internet). Generativo (crea preguntas nuevas), no SQL.
*Pro:* re-usa casi 100% del proyecto, corpus ya lo tenés. *Contra:* el modelo sabe DS
genérico → el valor es que genera en el marco específico de la cátedra (defendible, no blindado).

### Alt B — Generador de fichas/perfiles desde un corpus privado del grupo
RAG sobre un corpus que armamos nosotros (ej. relevamiento propio: encuestas, datos
recolectados de un dominio que elijamos). Genera fichas, reportes o descripciones.
*Pro:* dato 100% desconocido por el modelo. *Contra:* hay que recolectar el corpus.

---

## Recomendación

**Mantener la 1 reencuadrada como generador.** Es la que mejor combina: dato que el modelo
no conoce (blindado), generación real (no SQL), demo con/sin RAG vistosa, y bajo esfuerzo de
datos. Si la quieren más "blindada-académica", la **Alt A** es el plan B natural y reusa todo.
