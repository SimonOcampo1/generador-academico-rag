"use strict";
const $ = (s) => document.querySelector(s);
const state = { tono: "tecnico", artefactos: {}, busy: false, genRaw: "", rag: true };

/* ---- markdown mínimo (lo que generan los artefactos: títulos, **negrita**, *cursiva*, listas) ---- */
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function inlineMd(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}
function mdToHtml(src) {
  const lines = escapeHtml(src).split(/\r?\n/);
  let html = "", list = null;
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    if (/^\s*$/.test(line)) { closeList(); continue; }
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) { closeList(); const n = m[1].length; html += `<h${n}>${inlineMd(m[2])}</h${n}>`; continue; }
    if ((m = line.match(/^\s*[-*]\s+(.*)$/))) { if (list !== "ul") { closeList(); list = "ul"; html += "<ul>"; } html += `<li>${inlineMd(m[1])}</li>`; continue; }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { if (list !== "ol") { closeList(); list = "ol"; html += "<ol>"; } html += `<li>${inlineMd(m[1])}</li>`; continue; }
    closeList(); html += `<p>${inlineMd(line)}</p>`;
  }
  closeList();
  return html;
}

/* ---- tema ---- */
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem("theme", t);
  $("#toggleTheme").textContent = t === "dark" ? "☀" : "◐";
}
(function initTheme() {
  const saved = localStorage.getItem("theme")
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", saved);
})();

/* ---- combobox custom (mantiene el <select> nativo como fuente de verdad) ---- */
const ARROW = '<svg width="12" height="8" viewBox="0 0 12 8" fill="none"><path d="M1 1l5 5 5-5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
function enhanceSelect(sel) {
  sel.classList.add("enhanced");
  const wrap = document.createElement("div"); wrap.className = "cselect";
  sel.parentNode.insertBefore(wrap, sel); wrap.appendChild(sel);
  const trig = document.createElement("button");
  trig.type = "button"; trig.className = "cs-trigger";
  trig.setAttribute("aria-haspopup", "listbox"); trig.setAttribute("aria-expanded", "false");
  trig.innerHTML = `<span class="cs-label"></span>${ARROW}`;
  const menu = document.createElement("ul"); menu.className = "cs-menu"; menu.hidden = true; menu.setAttribute("role", "listbox");
  wrap.append(trig, menu);

  const sync = () => {
    trig.querySelector(".cs-label").textContent = sel.options[sel.selectedIndex]?.text || "";
    [...menu.children].forEach((li) => li.setAttribute("aria-selected", li.dataset.value === sel.value));
  };
  const render = () => {
    menu.innerHTML = "";
    [...sel.options].forEach((o) => {
      const li = document.createElement("li");
      li.className = "cs-opt"; li.dataset.value = o.value; li.textContent = o.text;
      li.setAttribute("role", "option");
      li.onclick = () => { sel.value = o.value; sel.dispatchEvent(new Event("change")); sync(); close(); };
      menu.appendChild(li);
    });
    sync();
  };
  const open = () => { render(); menu.hidden = false; wrap.classList.add("open"); trig.setAttribute("aria-expanded", "true"); };
  const close = () => { menu.hidden = true; wrap.classList.remove("open"); trig.setAttribute("aria-expanded", "false"); };
  trig.onclick = () => (menu.hidden ? open() : close());
  document.addEventListener("click", (e) => { if (!wrap.contains(e.target)) close(); });
  sync();
}

async function init() {
  applyTheme(document.documentElement.getAttribute("data-theme"));
  $("#toggleTheme").onclick = () =>
    applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");

  const o = await (await fetch("/api/opciones")).json();
  const aSel = $("#artefacto");
  o.artefactos.forEach((a) => { state.artefactos[a.id] = a; aSel.add(new Option(a.label, a.id)); });
  o.alumnos.forEach((a) => $("#alumno").add(new Option(a, a)));

  const tSeg = $("#tonoSeg");
  o.tonos.forEach((t, i) => {
    const b = document.createElement("button");
    b.textContent = t.id; b.title = t.desc; b.dataset.tono = t.id;
    b.setAttribute("aria-selected", i === 0);
    if (i === 0) state.tono = t.id;
    b.onclick = () => { state.tono = t.id; selectGroup(tSeg, b); };
    tSeg.appendChild(b);
  });

  const rSeg = $("#ragSeg");
  rSeg.querySelectorAll("button").forEach((b) => {
    b.onclick = () => { state.rag = b.dataset.rag === "1"; selectGroup(rSeg, b); };
  });

  aSel.onchange = syncFields; syncFields();
  enhanceSelect(aSel); enhanceSelect($("#alumno"));
  bindViewPills();
  $("#go").onclick = generar;
  $("#toggleModel").onclick = toggleModel;
  health(); setInterval(health, 8000);
}

function selectGroup(group, active) {
  group.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", b === active));
}
function bindViewPills() {
  $("#viewPills").querySelectorAll("button").forEach((b) => (b.onclick = () => showView(b.dataset.view)));
}
function showView(v) {
  $("#viewPills").querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", b.dataset.view === v));
  $("#viewArtifact").hidden = v !== "artifact";
  $("#viewCtx").hidden = v !== "ctx";
}

function syncFields() {
  const spec = state.artefactos[$("#artefacto").value] || {};
  $("#alumnoField").classList.toggle("hidden", !spec.alumno);
  $("#objetivoField").classList.toggle("hidden", !spec.objetivo);
  $("#whatifField").classList.toggle("hidden", !spec.whatif);
}

let _downStreak = 0;  // health: evita que un blip aislado tire el banner a "desconectado"
async function health() {
  // Durante la generación la GPU está ocupada con el request largo: el health /api/tags puede
  // tardar/timeout y reportar un falso "desconectado". No chequeamos mientras se genera.
  if (state.busy) return;
  try {
    const h = await (await fetch("/api/health")).json();
    _downStreak = h.ollama ? 0 : _downStreak + 1;
    const conectado = h.ollama || _downStreak < 2;  // hace falta 2 fallos seguidos para marcar caído

    $("#dot").classList.toggle("up", conectado);
    $("#ollamaTxt").textContent = conectado ? `${h.model} listo` : "modelo apagado · solo-retrieval";

    // Banner de backend: deja claro si el modelo corre local o en Colab, y si está conectado.
    const bar = $("#backendBar");
    bar.classList.toggle("up", conectado);
    bar.classList.toggle("down", h.remoto && !conectado);
    if (h.remoto) {
      $("#backendTxt").textContent = conectado
        ? `Generación en Colab (GPU) · ${h.model}`
        : "Colab no conectado · corré el notebook (Ejecutar todo) · solo-retrieval";
      $("#backendHost").textContent = h.host ? `— ${h.host}` : "";
    } else {
      $("#backendTxt").textContent = conectado
        ? `Modelo local · ${h.model}`
        : "Modelo local apagado · solo-retrieval";
      $("#backendHost").textContent = "";
    }

    // El botón de encender/apagar maneja un Ollama local: no aplica cuando el modelo está en Colab.
    $("#toggleModel").hidden = h.remoto;
    $("#toggleModel").dataset.up = conectado ? "1" : "0";
    $("#toggleModel").title = conectado ? "Apagar modelo" : "Encender modelo";
  } catch { /* server caído */ }
}

async function toggleModel() {
  const btn = $("#toggleModel"), up = btn.dataset.up === "1";
  btn.disabled = true; $("#ollamaTxt").textContent = up ? "apagando…" : "encendiendo…";
  try { await fetch(up ? "/api/stop_model" : "/api/start_model", { method: "POST" }); }
  finally { btn.disabled = false; await health(); }
}

function body() {
  const spec = state.artefactos[$("#artefacto").value];
  if (!spec) return null;  // opciones aún cargando (la 1ra vez se arma el corpus): evitar romper
  const r = { artefacto: $("#artefacto").value, tono: state.tono, rag: state.rag };
  if (spec.alumno) r.alumno = $("#alumno").value;
  if (spec.objetivo) r.objetivo = $("#objetivo").value;
  if (spec.whatif) r.materias_nota = { [$("#wifMateria").value]: Number($("#wifNota").value) };
  return r;
}

async function generar() {
  if (state.busy) return;
  const req = body();
  if (!req) { $("#hint").textContent = "Esperá a que terminen de cargar las opciones…"; return; }
  state.busy = true;
  const go = $("#go"); go.disabled = true; go.textContent = "Generando…";
  $("#hint").textContent = "";
  state.genRaw = ""; $("#gen").classList.remove("placeholder"); $("#gen").classList.add("rendered"); $("#gen").innerHTML = "";
  $("#ctx").textContent = "recuperando…";
  $("#groundBox").classList.add("hidden"); $("#stats").textContent = "generando…"; $("#srcTag").textContent = "";
  showView("artifact");

  try {
    const resp = await fetch("/api/generar_stream", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req),
    });
    const reader = resp.body.getReader(), dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (line) handle(JSON.parse(line));
      }
    }
  } catch (e) {
    // distinguir un fallo real de red de un error del cliente (no confundir el diagnóstico)
    const red = e instanceof TypeError && /fetch/i.test(e.message);
    $("#hint").textContent = (red ? "Error de conexión: " : "Error: ") + e.message;
  } finally {
    state.busy = false; go.disabled = false; go.textContent = "Generar";
  }
}

function handle(ev) {
  if (ev.type === "meta") {
    $("#ctx").textContent = ev.contexto;
    $("#srcTag").textContent = `${ev.model} · ${ev.tono} · ${ev.rag ? "CON RAG" : "SIN RAG"}`;
  } else if (ev.type === "token") {
    state.genRaw += ev.text;
    // ponytail: re-render del buffer completo por token; los artefactos son cortos, sobra
    $("#gen").innerHTML = mdToHtml(state.genRaw);
    $("#viewArtifact").scrollTop = $("#viewArtifact").scrollHeight;
  } else if (ev.type === "done") {
    $("#gen").innerHTML = mdToHtml(state.genRaw);  // render final (cierra markdown incompleto)
    if (ev.grounding != null) showGround(ev.grounding);
    $("#stats").textContent = (ev.stats && ev.stats.tokens)
      ? `${ev.stats.tokens} tokens · ${ev.stats.tok_s} tok/s · ${ev.stats.segundos}s · ${ev.fuente}`
      : `fuente: ${ev.fuente}`;
  } else if (ev.type === "error") {
    $("#hint").textContent = ev.msg;
    $("#stats").textContent = "error";
  }
}

function showGround(g) {
  const box = $("#groundBox"), bar = $("#gbar");
  box.classList.remove("hidden");
  $("#gval").textContent = g.toFixed(3);
  bar.classList.toggle("is-low", g < 0.5);
  bar.querySelector("span").style.width = Math.round(g * 100) + "%";
}

init();
