"""Texto crudo de los PDFs -> documentos listos para la base vectorial.

Dos fuentes, dos tratamientos:
- Estado académico (estructurado): una línea por materia -> un documento atómico por materia,
  con metadatos (alumno, materia, codigo, anio, nota, estado). Hechos puntuales, sin chunking.
- Plan de estudios / correlatividades (texto libre): chunking por ventana de caracteres.
  Es la parte donde el RAG semántico le gana a un SQL.

Cada documento es {"id", "text", "metadata"}.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from extract import pdf_to_text

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def normalizar(nombre: str) -> str:
    """Clave de join entre estado (códigos institucionales) y plan (Nº 1-36).

    Los dos PDFs usan códigos distintos, así que el único nexo confiable es el
    nombre de la materia. Bajamos a minúsculas, sacamos acentos, el marcador
    '(elec.)'/'(integradora)' —incluso truncado por el PDF, p. ej. '(Elec'— y
    espacios redundantes.
    """
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    s = s.lower()
    # El PDF a veces corta el marcador: '(Elec.)', '(Elec', '(integradora'. Sacar desde el '('.
    s = re.sub(r"\(\s*(elec|integradora)[^)]*\)?", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Anexo II del Plan 2023 (Ord. 1877): equivalencias de NOMBRE entre Plan 2008 (Ord. 1150) y
# Plan 2023. El export de notas (Notas-*.pdf) usa nombres del Plan 2008; el estado académico
# usa los del 2023. Canonizamos ambos al nombre 2023 para que el join recupere notas de
# materias aprobadas por equivalencia. Solo van las materias que CAMBIARON de nombre (las
# idénticas no necesitan mapeo; Química y Sistemas de Representación no tienen equivalente 2023).
# ponytail: tabla fija de una ordenanza oficial; si cambia, re-derivar del Anexo II del PDF.
EQUIV_2008_2023 = {
    "matematica discreta": "logica y estructuras discretas",
    "sistemas y organizaciones": "sistemas y procesos de negocio",
    "analisis de sistemas": "analisis de sistemas de informacion",
    "probabilidades y estadisticas": "probabilidad y estadistica",
    "diseno de sistemas": "diseno de sistemas de informacion",
    "matematica superior": "analisis numerico",
    "gestion de datos": "bases de datos",
    "redes de informacion": "redes de datos",
    "administracion de recursos": "administracion de sistemas de informacion",
    "ingenieria de software": "ingenieria y calidad de software",
    "teoria de control": "tecnologias para la automatizacion",
    "administracion gerencial": "gestion gerencial",
    "practica supervisada": "practica profesional supervisada",
}


# Variantes del MISMO nombre de materia entre el estado académico y el acta de notas (no son
# equivalencias de plan, sino el mismo electivo nombrado distinto). El estado usa una etiqueta
# corta y el acta el nombre completo. Un match difuso automático NO sirve: uniría electivas
# distintas que comparten palabras (p. ej. 'Seguridad en Redes' vs 'Gestión Operativa y Seguridad
# en Redes'), así que listamos los casos confirmados, como con EQUIV_2008_2023.
# ponytail: tabla explícita; si aparece otra electiva nombrada distinto, sumar la línea acá.
ALIAS_MATERIAS = {
    "desarrollo de aplicaciones moviles": "aplicaciones moviles",  # acta -> etiqueta del estado
}


def canon(nombre: str) -> str:
    """Nombre normalizado y llevado a su forma canónica (equivalencias 2008↔2023 + alias)."""
    n = normalizar(nombre)
    n = EQUIV_2008_2023.get(n, n)
    return ALIAS_MATERIAS.get(n, n)


def _buscar_nota(nombre: str, notas: dict[str, int]) -> int | None:
    """Busca la nota de una materia tolerando equivalencias y la truncación de nombres del PDF.

    El match tolerante usa _match_prefijo (gap<=4): cubre la truncación del PDF ('...informacio'
    vs '...informacion') SIN confundir materias que solo comparten prefijo, p. ej. 'Administración
    de Sistemas de Información' con 'Administración de Bases de Datos'. El viejo prefijo de 18
    chars ('administracion de ') las colisionaba y devolvía la nota equivocada.
    """
    c = canon(nombre)
    if c in notas:
        return notas[c]
    k = _match_prefijo(c, list(notas))
    return notas[k] if k else None


def _match_prefijo(clave: str, candidatos) -> str | None:
    """Match tolerante SOLO a la truncación del PDF (~40 chars), no a prefijos de otra materia.

    El PDF corta los nombres largos, así que el truncado es prefijo del completo con muy pocos
    chars de diferencia. Un umbral chico (<=4) evita confundir 'Desarrollo de Software' con
    'Desarrollo de Software Cloud' (materias distintas) o cualquier 'Administración de ...'.
    """
    if clave in candidatos:
        return clave
    mejor, mejor_gap = None, 99
    for c in candidatos:
        corto, largo = (clave, c) if len(clave) <= len(c) else (c, clave)
        gap = len(largo) - len(corto)
        if largo.startswith(corto) and gap <= 4 and gap < mejor_gap:
            mejor, mejor_gap = c, gap
    return mejor

# "0 Física 6 Aprobada en 2022 2016"
# "1 Algoritmos y Estructuras de Datos 106 Aprobada con 8 (5 hs.) Tomo: 386 Folio: 2 2023"
# "3 Responsabilidad Social e Institucional ( 331 2023"   (sin estado)
_FILA = re.compile(r"^(\d+)\s+(.+?)\s+(\d+)\s+(?:(.*?)\s+)?(\d{4})$")
_NOTA = re.compile(r"con\s+(\d+)")


_NOMBRE_HEADER = re.compile(r"Estado acad[eé]mico de\s+(.+?)\s+\(Legajo", re.IGNORECASE)
# Notas-*.pdf (export de exámenes): "27/4/2022 Inglés I 9 (nueve) Ing. Sist. Inf. 2008 111"
_FILA_NOTA = re.compile(r"^\d+/\d+/\d+\s+(.+?)\s+(\d{1,2})\s+\(\w+\)\s+Ing\. Sist\. Inf\.")


def nombre_alumno(texto_estado: str) -> str:
    """Header 'OCAMPO, SIMÓN TADEO (Legajo ...)' -> 'Simón Ocampo'."""
    m = _NOMBRE_HEADER.search(texto_estado)
    if not m:
        return "Alumno"
    apellidos, _, nombres = m.group(1).partition(",")
    return f"{nombres.split()[0].title()} {apellidos.split()[0].title()}"


def notas_de_examenes(texto: str) -> dict[str, int]:
    """Notas-*.pdf -> {materia_normalizada: nota}. Fuente autoritativa de notas.

    Recupera notas que el estado a veces no muestra (p. ej. equivalencias del plan 2008,
    que figuran como 'Aprobada en 2022' sin número).
    """
    notas: dict[str, int] = {}
    for ln in texto.splitlines():
        m = _FILA_NOTA.match(ln.strip())
        if m:
            notas[canon(m.group(1))] = int(m.group(2))  # canon: nombre 2008 -> 2023 (Anexo II)
    return notas


def documentos_estado(texto: str, alumno: str, notas_examenes: dict[str, int] | None = None,
                      nombres_plan: dict[str, str] | None = None) -> list[dict]:
    """Una línea de materia -> un documento por materia."""
    docs: list[dict] = []
    for linea in texto.splitlines():
        linea = linea.strip()
        m = _FILA.match(linea)
        if not m:
            continue
        anio, nombre, codigo, estado, _plan = m.groups()
        nombre = nombre.strip(" (")
        estado = (estado or "").strip() or "sin información de cursada"
        nota_m = _NOTA.search(estado)
        nota = int(nota_m.group(1)) if nota_m else None
        if nota is None and notas_examenes:  # recuperar nota desde Notas-*.pdf (vía equivalencias)
            nota = _buscar_nota(nombre, notas_examenes)
            if nota is not None:
                # El acta de exámenes manda: si la materia tiene nota ahí, está APROBADA, aunque el
                # estado diga 'en curso' o 'sin información de cursada' (suele ser un snapshot viejo).
                estado = f"Aprobada con {nota} (registrada en el acta de exámenes)"

        texto_doc = (
            f"{alumno}, en la carrera Ingeniería en Sistemas de Información (Plan 2023), "
            f"con la materia '{nombre}' (código {codigo}, año {anio} de la carrera): {estado}."
        )
        docs.append({
            "id": f"estado-{alumno.lower().split()[0]}-{codigo}",
            "text": texto_doc,
            "metadata": {
                "fuente": "estado_academico",
                "alumno": alumno,
                "materia": nombre,
                "codigo": codigo,
                "anio": int(anio),
                "nota": nota if nota is not None else -1,
                "estado": estado,
            },
        })

    # Materias aprobadas que figuran en Notas-*.pdf pero NO tienen fila en el estado académico
    # (el export de notas es más completo): se perdían por completo. Se agregan como aprobadas
    # con su nota numérica. anio=0 = año de carrera no disponible en el export de notas.
    if notas_examenes:
        nombres_plan = nombres_plan or {}
        presentes = {canon(d["metadata"]["materia"]) for d in docs}
        for clave, nota in notas_examenes.items():
            if _match_prefijo(clave, presentes):
                continue  # ya está en el estado (su nota se enriqueció arriba)
            kp = _match_prefijo(clave, nombres_plan)  # nombre legible del plan 2023 si es obligatoria
            nombre = nombres_plan[kp] if kp else clave.title()
            docs.append({
                "id": f"estado-{alumno.lower().split()[0]}-nota-{canon(nombre)[:24].replace(' ', '_')}",
                "text": (f"{alumno}, en la carrera Ingeniería en Sistemas de Información (Plan 2023), "
                         f"aprobó la materia '{nombre}' con nota {nota}."),
                "metadata": {
                    "fuente": "estado_academico", "alumno": alumno, "materia": nombre,
                    "codigo": "", "anio": 0, "nota": nota,
                    "estado": f"Aprobada con {nota} (registrada en el acta de exámenes)",
                },
            })
    return docs


# membrete/pie que se repite en cada página del PDF del plan: puro ruido para el RAG
_BOILERPLATE = re.compile(
    r"malvinas son argentinas|ministerio de educaci|universidad tecnol|rectorado|"
    r"pablo a\. huel|jefe de departamento|apoyo al consejo superior|r\s*e\s*g\s*i\s*s\s*t",
    re.IGNORECASE,
)


def _limpiar_plan(texto: str) -> str:
    return "\n".join(
        ln for ln in texto.splitlines()
        if ln.strip() and not _BOILERPLATE.search(ln)
    )


def chunk_text(texto: str, size: int = 700, overlap: int = 120) -> list[str]:
    """Ventana deslizante por caracteres.

    ponytail: chunking por caracteres, simple y suficiente para textos cortos como el plan.
    Si la calidad de retrieval lo pide, pasar a chunking por tokens o por secciones.
    """
    texto = re.sub(r"\n{2,}", "\n", texto).strip()
    chunks, i = [], 0
    while i < len(texto):
        chunks.append(texto[i:i + size].strip())
        i += size - overlap
    return [c for c in chunks if c]


def documentos_plan(texto: str) -> list[dict]:
    """Plan / correlatividades -> chunks de texto libre."""
    return [
        {
            "id": f"plan-{n:03d}",
            "text": c,
            "metadata": {"fuente": "plan_estudios", "chunk": n},
        }
        for n, c in enumerate(chunk_text(_limpiar_plan(texto)))
    ]


# Diseño curricular completo (Ord. 1877): una materia por página (págs ~39-74, con "Nº de orden:
# N"), con objetivos, competencias y contenidos mínimos; el resto es prosa del diseño (perfil,
# alcances, fundamentación). Todo es texto no estructurado -> base vectorial.
ORDENANZA = "SistemasOrdenanza1877.pdf"
_ORDEN = re.compile(r"orden:\s*(\d+)", re.IGNORECASE)


def documentos_ordenanza(plan_corr: dict[int, dict]) -> list[dict]:
    """Ordenanza 1877 -> documentos para la base vectorial.

    - Página con 'Nº de orden: N' (N en 1..36): un documento por materia con sus contenidos
      mínimos (fuente 'contenidos_materia'). El nombre limpio sale del plan (el Nº de orden
      coincide con el Nº de plan), no del texto desprolijo de la página.
    - Resto de páginas: prosa del diseño curricular -> chunks (fuente 'plan_estudios').
    """
    from extract import pdf_pages
    path = RAW_DIR / ORDENANZA
    if not path.exists():
        return []
    nombre_de = {n: d["nombre"] for n, d in plan_corr.items()}
    docs: list[dict] = []
    prosa: list[str] = []
    for pag in pdf_pages(path):
        limpio = _limpiar_plan(pag)
        if not limpio:
            continue
        m = _ORDEN.search(pag)
        if m and int(m.group(1)) in nombre_de:
            n = int(m.group(1))
            nombre = nombre_de[n]
            docs.append({
                "id": f"contenidos-{n:02d}",
                "text": (f"Descripción y contenidos mínimos de la materia '{nombre}' "
                         f"(asignatura Nº {n} del Plan 2023 de Ingeniería en Sistemas de "
                         f"Información):\n{limpio}"),
                "metadata": {"fuente": "contenidos_materia", "materia": nombre, "num_plan": n},
            })
        else:
            prosa.append(limpio)
    docs += [
        {"id": f"diseno-{i:03d}", "text": c, "metadata": {"fuente": "plan_estudios", "chunk": i}}
        for i, c in enumerate(chunk_text("\n".join(prosa)))
    ]
    return docs


# --- Correlatividades (Anexo I del plan) -----------------------------------
# Tabla "NIVEL Nº ASIGNATURA | Para cursar (Cursadas) | Para rendir (Aprobadas)".
# Cada fila referencia otras materias por su Nº de plan (1-36), separados por '-'.
NIVEL_ROMANO = re.compile(r"^[IVX]{1,4}$")
_ROW_NAMED = re.compile(r"^(\d{1,2})\s+(\D.*?)\s+(-|[\d\-]+)\s+(-|[\d\-]+)$")
_ROW_NONAME = re.compile(r"^(\d{1,2})\s+(-|[\d\-]+)\s+(-|[\d\-]+)$")  # integradora: nombre en línea previa
_RUIDO_CORR = re.compile(
    r"malvinas|ministerio|universidad tecnol|pablo|rectorado|apoyo al consejo|"
    r"para cursar|y rendir|nivel|cursadas aprobadas|^\d+$",
    re.IGNORECASE,
)
_INTEGRADORA = re.compile(r"^(.*?)\s*\(integradora\)\s*$", re.IGNORECASE)


def _codigos(campo: str) -> list[int]:
    return [] if campo == "-" else [int(x) for x in campo.split("-") if x]


def parsear_correlatividades(texto: str) -> dict[int, dict]:
    """Texto del plan -> {Nº: {nombre, cursar:[Nº], rendir:[Nº]}} para las 36 obligatorias."""
    lineas = [ln.strip() for ln in texto.splitlines()]
    plan: dict[int, dict] = {}
    pending: list[str] = []
    started = False  # ignoramos el preámbulo (menciona "Anexo II" antes de la tabla)
    i = 0
    while i < len(lineas):
        ln = lineas[i]
        if not started:
            if re.search(r"asignatura", ln, re.IGNORECASE):
                started = True  # header "NIVEL Nº ASIGNATURA": empieza la tabla
            i += 1
            continue
        if re.search(r"anexo ii", ln, re.IGNORECASE):
            break  # arranca el régimen de equivalencias, no nos interesa
        ln = re.sub(r"^[IVX]{1,4}\s+(?=\d)", "", ln)  # nivel romano inline: "III 20 Desarrollo..." -> "20 Desarrollo..."
        if not ln or _RUIDO_CORR.search(ln) or NIVEL_ROMANO.match(ln) or _INTEGRADORA.match(ln):
            # línea de ruido/encabezado/nivel/"(integradora)": no es nombre pendiente útil
            if _INTEGRADORA.match(ln) is None:
                pending = []
            i += 1
            continue
        m = _ROW_NAMED.match(ln)
        if m:
            num, nombre, cursar, rendir = m.groups()
            nombre = _INTEGRADORA.sub(r"\1", nombre).strip()  # Nº36: nombre trae "(integradora)" inline
            plan[int(num)] = {"nombre": nombre, "cursar": _codigos(cursar), "rendir": _codigos(rendir)}
            pending = []
            i += 1
            continue
        m = _ROW_NONAME.match(ln)
        if m:  # integradora/nombre partido: nombre = líneas previas (+ continuación en la siguiente)
            num, cursar, rendir = m.groups()
            nombre = " ".join(pending).strip()
            nxt = lineas[i + 1] if i + 1 < len(lineas) else ""
            cont = _INTEGRADORA.match(nxt)
            if cont and cont.group(1).strip():
                nombre = f"{nombre} {cont.group(1).strip()}".strip()
            elif nxt and not (_RUIDO_CORR.search(nxt) or NIVEL_ROMANO.match(nxt)
                              or _ROW_NAMED.match(nxt) or _ROW_NONAME.match(nxt)):
                # continuación simple del nombre partido por el salto de página (Nº35: "Información")
                nombre = f"{nombre} {nxt.strip()}".strip()
                i += 1  # consumir la línea de continuación
            nombre = _INTEGRADORA.sub(r"\1", nombre).strip()
            plan[int(num)] = {"nombre": nombre, "cursar": _codigos(cursar), "rendir": _codigos(rendir)}
            pending = []
            i += 1
            continue
        pending.append(ln)  # candidato a nombre de integradora
        i += 1
    return plan


def documentos_correlatividades(plan: dict[int, dict]) -> list[dict]:
    """Cada materia obligatoria -> un documento en prosa con sus correlativas (nombres resueltos)."""
    nombre_de = {n: d["nombre"] for n, d in plan.items()}
    docs: list[dict] = []
    for num, d in sorted(plan.items()):
        cursar = [nombre_de.get(c, f"materia {c}") for c in d["cursar"]]
        rendir = [nombre_de.get(c, f"materia {c}") for c in d["rendir"]]
        partes = [f"En el plan 2023 de Ingeniería en Sistemas de Información, '{d['nombre']}' es la materia Nº {num}."]
        partes.append(
            f"Para cursarla hay que tener cursadas: {', '.join(cursar)}." if cursar
            else "No tiene correlativas para cursarla."
        )
        if rendir:
            partes.append(f"Para rendirla (final) hay que tener aprobadas: {', '.join(rendir)}.")
        docs.append({
            "id": f"correlativa-{num:02d}",
            "text": " ".join(partes),
            "metadata": {
                "fuente": "correlatividades",
                "materia": d["nombre"],
                "num_plan": num,
                "cursar": ",".join(map(str, d["cursar"])),
                "rendir": ",".join(map(str, d["rendir"])),
            },
        })
    return docs


def construir_corpus() -> list[dict]:
    """Arma todos los documentos desde los PDFs en data/raw.

    Auto-descubre todos los Estado-Academico-*.pdf (un alumno por archivo) y, si existe el
    Notas-<alumno>.pdf correspondiente, lo usa para enriquecer notas faltantes.
    """
    docs: list[dict] = []
    # Plan primero: sus nombres canónicos (2023) sirven para nombrar prolijo las materias que
    # vienen sólo del export de notas (con nombre 2008 y a veces truncado).
    plan_path = RAW_DIR / "Plan-Sistemas-2023.pdf"
    texto_plan = pdf_to_text(plan_path) if plan_path.exists() else None
    plan_corr = parsear_correlatividades(texto_plan) if texto_plan else {}
    nombres_plan = {canon(d["nombre"]): d["nombre"] for d in plan_corr.values()}

    for estado_pdf in sorted(RAW_DIR.glob("Estado-Academico-*.pdf")):
        texto = pdf_to_text(estado_pdf)
        alumno = nombre_alumno(texto)
        sufijo = estado_pdf.stem.split("-")[-1]  # 'Simon', 'Santi'
        notas_pdf = RAW_DIR / f"Notas-{sufijo}.pdf"
        notas = notas_de_examenes(pdf_to_text(notas_pdf)) if notas_pdf.exists() else None
        docs += documentos_estado(texto, alumno=alumno, notas_examenes=notas, nombres_plan=nombres_plan)
    if texto_plan:
        docs += documentos_correlatividades(plan_corr)  # -> base relacional
    # Prosa de la carrera (-> base vectorial): el diseño curricular completo (Ord. 1877) con los
    # contenidos mínimos por materia. Si no está ese PDF, se usa la prosa del Plan-Sistemas-2023.
    ord_docs = documentos_ordenanza(plan_corr)
    if ord_docs:
        docs += ord_docs
    elif texto_plan:
        docs += documentos_plan(texto_plan)
    return docs


if __name__ == "__main__":
    docs = construir_corpus()
    estado = [d for d in docs if d["metadata"]["fuente"] == "estado_academico"]
    plan = [d for d in docs if d["metadata"]["fuente"] == "plan_estudios"]
    print(f"Total documentos: {len(docs)}  (estado: {len(estado)}, plan: {len(plan)})")
    print("\n--- ejemplos estado académico ---")
    for d in estado[:3]:
        print(" *", d["text"])
    print("\n--- ejemplo chunk plan ---")
    if plan:
        print(" *", plan[3]["text"][:300], "...")
    # check mínimo: parseo de notas
    con_nota = [d for d in estado if d["metadata"]["nota"] >= 0]
    assert con_nota, "no se parseó ninguna nota del estado académico"
    print(f"\nMaterias con nota parseada: {len(con_nota)}")

    # check: la canonización de equivalencias (Anexo II) lleva nombres 2008 -> 2023
    assert canon("Matemática Discreta") == "logica y estructuras discretas"
    assert canon("Gestión de Datos") == "bases de datos"
    assert canon("Análisis Matemático I") == "analisis matematico i"  # sin equivalencia: queda igual

    # check: correlatividades (36 obligatorias, con valores conocidos del Anexo I)
    plan_pdf = RAW_DIR / "Plan-Sistemas-2023.pdf"
    if plan_pdf.exists():
        corr = parsear_correlatividades(pdf_to_text(plan_pdf))
        assert len(corr) == 36, f"se esperaban 36 obligatorias, hay {len(corr)}: {sorted(corr)}"
        assert corr[19]["nombre"].startswith("Bases de Datos"), corr[19]
        assert corr[19]["cursar"] == [13, 16] and corr[19]["rendir"] == [5, 6], corr[19]
        assert corr[32]["cursar"] == [28] and corr[32]["rendir"] == [17, 19], corr[32]  # Ciencia de Datos
        assert corr[16]["nombre"] == "Análisis de Sistemas de Información", corr[16]   # integradora
        assert corr[30]["nombre"] == "Administración de Sistemas de Información", corr[30]
        print(f"Correlatividades parseadas: {len(corr)} obligatorias")
        print(" * ", documentos_correlatividades(corr)[18]["text"])
