# DESIGN.md — Sistema visual "Graphite & Cobalt"

Línea HOMOGÉNEA y coherente para todos los entregables (web app, informe, slides, guión,
figuras). Distinta del TP anterior (que era editorial enológico, granate/crema, serif).
Acá: claro, frío, técnico, premium-minimalista. Estrategia de color **restrained**: neutros
grafito tintados al azul + un único acento cobalto (≤10% de superficie) + categóricos fríos
reservados para data-viz.

Fuente de verdad del color: las figuras (`src/figuras.py`) ya usan estos hex. La UI los
replica exactos para que gráficos y pantallas se vean del mismo sistema.

## Theme
Dual, con toggle (persistido en localStorage; respeta `prefers-color-scheme`).
- **Light** (default para informe/slides/lámina proyectada): aula iluminada, de día, proyector.
  Fondo claro frío = lectura larga de prosa + lámina legible con luz ambiente.
- **Dark** (la web app puede arrancar en dark): demo en vivo sobre pantalla, foco en la consola
  y la generación. Grafito profundo, mismo acento cobalto (aclarado para contraste AA).

Dark tokens: `--bg #0E0F12`, `--surface #16181D`, `--ink #ECEEF2`, `--ink-soft #9AA0AA`,
`--line #262A31`, `--cobalt #6E86FF` (aclarado), consola `#08090B`.

## Color (hex, tintados al azul grafito)
- `--bg` `#FAFAFB` — fondo, blanco frío (no #fff).
- `--surface` `#F2F3F6` — paneles/secciones sutiles.
- `--ink` `#0E0F12` — texto principal (no #000).
- `--ink-soft` `#5B6066` — texto secundario.
- `--line` `#E6E7EB` — bordes finos 1px.
- `--cobalt` `#3A5BFF` — acento único. Estados activos, links, foco, dato anclado.
- `--cobalt-soft` `#E7EBFF` — fondo de chip/realce cobalto muy tenue.
- **Categóricos (solo data-viz / tonos / fuentes):** cobalto `#3A5BFF`, grafito `#0E0F12`,
  acero `#7C8AA5`, teal `#00A6A6`, violeta `#9B6DFF`, ámbar `#E0A100`.
- **Semánticos de la demo:** con-RAG = cobalto; sin-RAG = acero apagado; grounding alto =
  teal, bajo = ámbar.

## Typography
- **Display / títulos:** `"Space Grotesk", "Segoe UI", sans-serif` — grotesca moderna con
  carácter geométrico, sobria y premium.
- **UI / cuerpo / datos / código de materias:** `"Inter", system-ui, sans-serif`.
- **Mono (prompt, contexto, métricas):** `"JetBrains Mono", ui-monospace, monospace`.
- Escala: 13 / 15 / 18 / 24 / 34 / 48. Contraste de peso ≥1.25 entre niveles.
- Cuerpo de lectura 64–72ch.

## Layout
- Márgenes generosos, ritmo de espaciado variable (no padding uniforme).
- Bordes completos 1px o nada. **Prohibido side-stripe** (border-left de color como acento).
- Sin tarjetas idénticas repetidas, sin contenedor envolviendo todo, sin nested cards.

## Motion
- Solo opacidad/transform, ease-out-expo, <300ms, sin bounce. No animar layout.

## Componentes clave
- **Consola RAG en vivo:** prompt + contexto recuperado (mono) → tokens streaming → métricas
  (grounding, tok/s, fuente). El mecanismo a la vista.
- **Switch con RAG / sin RAG:** segmented control; con-RAG cobalto, sin-RAG acero.
- **Selector de artefacto** (6) + **selector de tono** (3, color propio por tono).
- **Medidor de grounding:** barra teal→ámbar con el valor [0,1].
- **Chip de dato anclado:** materia/nota citada del contexto, fondo cobalt-soft.

## Export PDF (informe/slides)
- `@media print`: fondo blanco forzado, sin sombras, acento conservado,
  `break-inside: avoid` en bloques, saltos de página controlados.
