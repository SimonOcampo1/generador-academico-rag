# PRODUCT.md — Generador Académico RAG

## Product Purpose
Sistema de IA generativa con **persistencia híbrida** —una base **relacional** (SQLite)
con el historial académico real del grupo + las correlatividades, y una base **vectorial**
(Chroma) con la documentación del plan— que combinando ambas **genera artefactos
académicos personalizados** (planes de cursada justificados, informes de trayectoria,
cartas de pasantía, recomendación de orientación, diagnóstico grupal, simulación
what-if). No responde preguntas: **sintetiza documentos nuevos** anclados en datos
privados que el modelo no conoce.

TP Integrador de Ciencia de Datos, UTN FRLP 2026, Grupo 14. Local, offline, privado.

## Register
product

El núcleo es una herramienta (web app de testeo en vivo de un pipeline RAG). Los
entregables de comunicación (informe, slides, guión) comparten el mismo sistema visual.

## Users
- **Jurado/cátedra:** evalúa que el grupo sepa construir el pipeline RAG (parseo, persistencia
  híbrida relacional+vectorial, retrieval combinado, generación) y lo use para generar, eligiendo
  la base correcta para cada tipo de dato (tabular → relacional, texto → vectorial).
- **El grupo (6 integrantes):** expone el trabajo y opera la demo en vivo.
- **Estudiante final (caso de uso):** pide un artefacto sobre su propia trayectoria.

## What it does
- Ingesta corpus mixto y lo persiste según su naturaleza: tabular (notas, años, correlativas)
  → base relacional SQLite; no estructurado (prosa del plan) → base vectorial Chroma.
- Pipeline RAG propio: PDFs → pdfplumber → documentos → (SQLite + embeddings MiniLM/Chroma) →
  retrieval híbrido (SQL + semántico) → Phi-4-mini vía Ollama (fallback a solo-retrieval si apagado).
- 6 generadores de artefactos con **tono controlable** (técnico / motivacional / honesto).
- EDA + visualización + clustering de perfiles (KMeans, t-SNE de embeddings).
- Evaluación de **grounding** y **demo estrella con RAG / sin RAG**.

## Tone & Voice
Técnico, sobrio, honesto. Riguroso sin solemnidad. Muestra el mecanismo (no caja negra):
el contexto recuperado se ve, el grounding se mide, la diferencia con/sin RAG se exhibe.

## Strategic principles
1. **Generar, no consultar.** Cada salida es contenido nuevo sintetizado, no un campo.
2. **Mostrar el pipeline.** El valor del TP es el RAG visible, no un chatbot pulido.
3. **Anclar todo.** Sin grounding no hay producto: se cita el dato real o no se afirma.
4. **Honestidad de datos.** Los perfiles sintéticos se marcan siempre como simulados.

## Anti-references
- NotebookLM (caja negra; nosotros mostramos el pipeline).
- Un asistente de preguntas / chatbot Q&A (eso sería lookup, un SQL lo resuelve).
- Dashboards SaaS genéricos (hero-metric template, tarjetas idénticas, glassmorphism).
- La paleta del TP anterior (editorial enológico, granate sobre crema serif).
