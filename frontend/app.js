// Privacy Filter frontend — vanilla JS, no build step.
const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------------------
// Auth helpers — demo JWT stored in sessionStorage
// ---------------------------------------------------------------------------
const TOKEN_KEY = "pf_demo_token";

function getStoredToken() {
  return sessionStorage.getItem(TOKEN_KEY) || "";
}

function storeToken(token) {
  sessionStorage.setItem(TOKEN_KEY, token);
}

/**
 * Drop-in fetch() replacement that injects Authorization: Bearer <token>
 * when a demo token is present. Falls back to a plain fetch if no token.
 */
async function authFetch(url, opts = {}) {
  const token = getStoredToken();
  if (token) {
    opts.headers = { ...(opts.headers || {}), Authorization: `Bearer ${token}` };
  }
  return fetch(url, opts);
}

/**
 * Decode the JWT payload without verifying the signature (client-side display only).
 * Returns null if the token is malformed.
 */
function decodeJwtPayload(token) {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const b64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(atob(b64));
  } catch {
    return null;
  }
}


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
  return authFetch("/api/redact", { method: "POST", body: fd });
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
const dashboardPanel = document.getElementById("dashboard-panel");
const apiBaseUrlEl = document.getElementById("api-base-url");
const tabButtons = document.querySelectorAll(".header-nav .tablinks[data-tab]");

// Reflect the current origin in the API panel so the user sees the URL
// they'd actually hit (handy on localhost / staging).
if (apiBaseUrlEl) {
  apiBaseUrlEl.textContent = window.location.origin;
}
// Also fill the curl example spans.
const curlSpan1 = document.getElementById("api-base-url-curl");
const curlSpan2 = document.getElementById("api-base-url-curl2");
if (curlSpan1) curlSpan1.textContent = window.location.origin;
if (curlSpan2) curlSpan2.textContent = window.location.origin;


function activateTab(name) {
  tabButtons.forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  filterPanel?.classList.add("hidden");
  apiPanel?.classList.add("hidden");
  dashboardPanel?.classList.add("hidden");

  if (name === "api") {
    apiPanel?.classList.remove("hidden");
  } else if (name === "dashboard") {
    dashboardPanel?.classList.remove("hidden");
    loadStats(); // Refresh immediately when switching to dashboard tab
  } else {
    filterPanel?.classList.remove("hidden");
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

tabButtons.forEach((b) => {
  b.addEventListener("click", () => activateTab(b.dataset.tab));
});

// ---------- Dashboard stats ----------
function animateCount(el, targetVal) {
  if (!el) return;
  const start = parseInt(el.textContent.replace(/[^0-9]/g, "")) || 0;
  const end = parseInt(targetVal) || 0;
  if (start === end) return;
  const duration = 600;
  const step = Math.ceil(Math.abs(end - start) / (duration / 16));
  let current = start;
  const timer = setInterval(() => {
    current = start < end
      ? Math.min(current + step, end)
      : Math.max(current - step, end);
    el.textContent = current.toLocaleString();
    if (current === end) clearInterval(timer);
  }, 16);
}

async function loadStats() {
  try {
    const r = await fetch("/api/stats");
    if (!r.ok) return;
    const s = await r.json();

    // Dashboard cards
    animateCount(document.getElementById("statPageVisits"), s.page_visits ?? 0);
    animateCount(document.getElementById("statUniqueVisitors"), s.unique_visitors ?? 0);
    animateCount(document.getElementById("statDocsRedacted"), s.docs_redacted ?? 0);

    // Footer bar
    const fv = document.getElementById("footerVisits");
    const fu = document.getElementById("footerVisitors");
    const fd = document.getElementById("footerDocs");
    if (fv) fv.textContent = (s.page_visits ?? 0).toLocaleString();
    if (fu) fu.textContent = (s.unique_visitors ?? 0).toLocaleString();
    if (fd) fd.textContent = (s.docs_redacted ?? 0).toLocaleString();
  } catch (e) {
    // Stats fetch failure is non-critical; ignore silently.
  }
}

// Load stats on page init and refresh every 30 seconds.
loadStats();
setInterval(loadStats, 30_000);

// ---------------------------------------------------------------------------
// Demo token form
// ---------------------------------------------------------------------------
(function initDemoTokenForm() {
  const form   = document.getElementById("demo-token-form");
  const submit = document.getElementById("demo-token-submit");
  const result = document.getElementById("demo-token-result");
  const errEl  = document.getElementById("demo-token-error");
  const output = document.getElementById("demo-token-output");
  const copyBtn = document.getElementById("demo-token-copy");
  const greeting = document.getElementById("demo-token-greeting");
  const expiryEl = document.getElementById("demo-token-expiry");
  const appliedEl = document.getElementById("demo-token-applied");

  if (!form) return;

  // If we already have a stored token from a previous session, pre-fill the
  // result box so the user can see/copy it without re-requesting.
  const existingToken = getStoredToken();
  if (existingToken) {
    _showTokenResult(existingToken, null, null, true);
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name  = document.getElementById("demo-token-name")?.value.trim();
    const email = document.getElementById("demo-token-email")?.value.trim();

    if (!name || !email) {
      _showError("Please fill in both your name and email address.");
      return;
    }

    submit.disabled = true;
    submit.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Requesting…';
    errEl.classList.add("hidden");
    result.classList.add("hidden");

    try {
      const r = await fetch("/api/demo-token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email }),
      });

      if (!r.ok) {
        const errJson = await r.json().catch(() => ({}));
        throw new Error(errJson.detail || `HTTP ${r.status}`);
      }

      const data = await r.json();
      storeToken(data.access_token);
      _showTokenResult(data.access_token, data.name, data.expires_in_days, false);
    } catch (err) {
      _showError(`Failed to request token: ${err.message}`);
    } finally {
      submit.disabled = false;
      submit.innerHTML = '<i class="fas fa-bolt"></i> Request Token';
    }
  });

  copyBtn?.addEventListener("click", async () => {
    const token = output?.textContent?.trim();
    if (!token) return;
    try {
      await navigator.clipboard.writeText(token);
      copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
      copyBtn.classList.add("copied");
      setTimeout(() => {
        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
        copyBtn.classList.remove("copied");
      }, 2000);
    } catch {
      // Clipboard API not available (non-HTTPS or blocked)
      copyBtn.textContent = "Use Ctrl+C";
    }
  });

  function _showTokenResult(token, name, expiresInDays, restored) {
    output.textContent = token;
    result.classList.remove("hidden");
    errEl.classList.add("hidden");

    // Decode payload for display
    const payload = decodeJwtPayload(token);
    const displayName = name || payload?.name || "";
    const expTs = payload?.exp;
    if (displayName) {
      greeting.textContent = restored
        ? `Welcome back, ${displayName}!`
        : `🎉 Token issued for ${displayName}`;
    }
    if (expTs) {
      const expDate = new Date(expTs * 1000);
      expiryEl.textContent = `Expires ${expDate.toLocaleDateString(undefined, { dateStyle: "medium" })}`;
    } else if (expiresInDays) {
      expiryEl.textContent = `Valid for ${expiresInDays} days`;
    }

    // Show the "applied" confirmation strip
    if (appliedEl) appliedEl.style.display = "flex";
  }

  function _showError(msg) {
    errEl.textContent = msg;
    errEl.classList.remove("hidden");
  }
})();
