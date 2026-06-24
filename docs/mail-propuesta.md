Buenas,

Somos el grupo 14: Martin Tomas, Gentil Mora, Natalichio Santiago, Cuenca Juan Bautista, Ocampo Simon y Aubert Lautaro.

Replanteamos la propuesta para que el RAG sea el núcleo, con una persistencia híbrida que combina dos bases de datos. Tomando tu observación, separamos los datos según su naturaleza: el historial académico y las correlatividades —datos tabulares— van en una base relacional (SQLite), donde un lookup o un join exacto es lo correcto; la documentación de la carrera —texto largo no estructurado— va en una base vectorial (Chroma) para búsqueda semántica. El "Generador Académico RAG" combina ambas para obtener el contexto y, a partir de él, genera artefactos personalizados. No busca respuestas: sintetiza contenido nuevo a partir de datos privados que ningún modelo conoce. Para la demo lo cargamos con los datos del grupo, pero es escalable a cualquier alumno.

Ejemplos de pedidos:

- "Armame el plan de cursada del próximo cuatrimestre y justificá cada materia."
- "Redactá mi carta de motivación para una pasantía en Ciencia de Datos, destacando las materias más relevantes."
- "Escribí un informe narrativo de mi trayectoria, con tono motivacional."
- "Recomendá una orientación profesional según las materias donde mejor rindo."
- "Simulá cómo cambia mi perfil si me saco un 9 en Inteligencia Artificial."

El foco es construir nosotros el pipeline RAG completo (parseo, persistencia híbrida relacional + vectorial, embeddings y recuperación combinada), más el análisis exploratorio y el clustering de los perfiles.

Desde ya, gracias.
