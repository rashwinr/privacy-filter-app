// Privacy Filter frontend — vanilla JS, no build step.
const $ = (sel) => document.querySelector(sel);

const dz = $("#dropzone");
const fileInput = $("#file-input");
const statusEl = $("#status");
const resultsEl = $("#results");
const healthEl = $("#health");
const supportedEl = $("#supported");

async function pingHealth() {
  const badge = healthEl;
  const textEl = document.getElementById("health-text");
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    const ready = !!j.model_loaded;
    textEl.textContent = `${j.model} · ${j.device} · ${ready ? "READY" : "LOADING…"}`;
    badge.classList.toggle("ai-badge-off", !ready);
    if (!ready) setTimeout(pingHealth, 2000);
  } catch (e) {
    textEl.textContent = "MODEL SERVICE UNREACHABLE";
    badge.classList.add("ai-badge-off");
  }
}

async function loadSupported() {
  try {
    const r = await fetch("/api/supported-types");
    const j = await r.json();
    supportedEl.textContent = `Accepted: ${j.extensions.join(", ")}`;
  } catch {}
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.remove("hidden");
  statusEl.classList.toggle("error", isError);
}

function clearStatus() {
  statusEl.classList.add("hidden");
  statusEl.textContent = "";
}

function renderResult(res) {
  resultsEl.classList.remove("hidden");
  $("#dl-original").href = res.original_url;
  $("#dl-original").textContent = `Download original (${res.filename})`;
  $("#dl-redacted").href = res.redacted_url;

  const counts = $("#counts-list");
  counts.innerHTML = "";
  const entries = Object.entries(res.entity_counts || {}).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    counts.innerHTML = "<li>No personal information detected.</li>";
  } else {
    for (const [k, v] of entries) {
      const li = document.createElement("li");
      li.innerHTML = `<b>${v}</b>${k}`;
      counts.appendChild(li);
    }
  }

  $("#prev-original").textContent = res.text_preview_original || "(no text preview available)";
  $("#prev-redacted").textContent = res.text_preview_redacted || "(no text preview available)";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

// Build a fresh FormData each attempt so the body stream isn't reused.
async function postRedact(file) {
  const fd = new FormData();
  fd.append("file", file);
  return fetch("/api/redact", { method: "POST", body: fd });
}

async function uploadFile(file) {
  clearStatus();
  resultsEl.classList.add("hidden");
  // Reset previous result UI so a stale view never lingers.
  $("#counts-list").innerHTML = "";
  $("#prev-original").textContent = "";
  $("#prev-redacted").textContent = "";

  setStatus(`Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)…`);

  // Cloud Run can return 503 briefly while a fresh instance spins up
  // (cold start, autoscale, or container restart after the previous job).
  // Retry up to 3 times with backoff before giving up.
  const MAX_ATTEMPTS = 3;
  let lastErr = null;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const r = await postRedact(file);
      if (r.ok) {
        const j = await r.json();
        setStatus(`Done. Job ${j.job_id} · ${j.entities.length} entities detected.`);
        renderResult(j);
        return;
      }
      // Retry on transient server errors (503/504/502); fail fast on 4xx.
      if ([502, 503, 504].includes(r.status) && attempt < MAX_ATTEMPTS) {
        const wait = 2000 * attempt;
        setStatus(`Service warming up (HTTP ${r.status}) — retrying in ${wait / 1000}s…`);
        await new Promise((res) => setTimeout(res, wait));
        continue;
      }
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || `HTTP ${r.status}`);
    } catch (e) {
      lastErr = e;
      if (attempt < MAX_ATTEMPTS && /Failed to fetch|NetworkError/i.test(e.message)) {
        await new Promise((res) => setTimeout(res, 2000 * attempt));
        continue;
      }
      break;
    }
  }
  setStatus(`Failed: ${lastErr ? lastErr.message : "unknown error"}`, true);
}

dz.addEventListener("click", () => fileInput.click());
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("drag");
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) uploadFile(e.target.files[0]);
});

pingHealth();
loadSupported();

// ---------- Header tab switcher ----------
const filterPanel = document.getElementById("filter-panel");
const apiPanel = document.getElementById("api-panel");
const apiBaseUrlEl = document.getElementById("api-base-url");
const tabButtons = document.querySelectorAll(".header-nav .tablinks[data-tab]");

// Reflect the current origin in the API panel so the user sees the URL
// they'd actually hit (handy on localhost / staging).
if (apiBaseUrlEl) {
  apiBaseUrlEl.textContent = window.location.origin;
}

function activateTab(name) {
  tabButtons.forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  if (name === "api") {
    filterPanel?.classList.add("hidden");
    apiPanel?.classList.remove("hidden");
  } else {
    apiPanel?.classList.add("hidden");
    filterPanel?.classList.remove("hidden");
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

tabButtons.forEach((b) => {
  b.addEventListener("click", () => activateTab(b.dataset.tab));
});
