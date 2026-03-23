const state = {
  runs: [],
  selectedRun: null,
  map: null,
  mapLayers: [],
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function runFormData(form) {
  const data = new FormData(form);
  return data;
}

function renderRuns() {
  const container = document.getElementById("run-list");
  container.innerHTML = "";
  state.runs.forEach((run) => {
    const card = document.createElement("div");
    card.className = `run-card status-${run.status}`;
    const button = document.createElement("button");
    const auc = run.metrics?.auc_mean ? `AUC ${run.metrics.auc_mean.toFixed(4)}` : run.status;
    button.innerHTML = `
      <strong>${run.model_name}</strong>
      <span>${run.dataset_id}</span>
      <span>${auc}</span>
    `;
    button.addEventListener("click", () => loadRun(run.run_id));
    card.appendChild(button);
    if (["queued", "running", "cancelling"].includes(run.status)) {
      const cancel = document.createElement("button");
      cancel.textContent = run.status === "cancelling" ? "Cancelling..." : "Stop";
      cancel.disabled = run.status === "cancelling";
      cancel.addEventListener("click", async () => {
        await fetchJson(`/api/runs/${run.run_id}/cancel`, { method: "POST" });
        await refreshRuns();
      });
      card.appendChild(cancel);
    }
    container.appendChild(card);
  });
}

function ensureMap() {
  if (state.map || typeof L === "undefined") {
    return;
  }
  state.map = L.map("result-map", {
    zoomControl: true,
    preferCanvas: true,
  }).setView([-26, 122], 5);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);
}

function clearMapLayers() {
  state.mapLayers.forEach((layer) => state.map.removeLayer(layer));
  state.mapLayers = [];
}

function rgbaFromRatio(ratio, alpha = 1) {
  const r = Math.round(255 * ratio);
  const g = Math.round(170 - 90 * ratio);
  const b = 40;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function anomalyOverlay(mapPayload) {
  if (!mapPayload?.grid?.length || !mapPayload.grid_x?.length || !mapPayload.grid_y?.length) {
    return null;
  }
  const rows = mapPayload.grid.length;
  const cols = mapPayload.grid[0].length;
  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  const values = mapPayload.grid.flat();
  const min = Math.min(...values);
  const max = Math.max(...values);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const ratio = (mapPayload.grid[r][c] - min) / Math.max(max - min, 1e-6);
      ctx.fillStyle = rgbaFromRatio(ratio, 0.42);
      ctx.fillRect(c, rows - 1 - r, 1, 1);
    }
  }
  const yMin = Math.min(...mapPayload.grid_y);
  const yMax = Math.max(...mapPayload.grid_y);
  const xMin = Math.min(...mapPayload.grid_x);
  const xMax = Math.max(...mapPayload.grid_x);
  return L.imageOverlay(canvas.toDataURL("image/png"), [[yMin, xMin], [yMax, xMax]], {
    opacity: 0.7,
    interactive: false,
  });
}

function drawMap(run) {
  ensureMap();
  if (!state.map) {
    return;
  }
  clearMapLayers();
  if (!run.payload) {
    return;
  }
  const coords = run.payload.coords || [];
  const scores = run.payload.scores || [];
  const sites = run.payload.sites || [];
  if (!coords.length) {
    return;
  }
  const scoreMin = Math.min(...scores);
  const scoreMax = Math.max(...scores);
  const bounds = [];
  const overlay = anomalyOverlay(run.payload.anomaly_map);
  if (overlay) {
    overlay.addTo(state.map);
    state.mapLayers.push(overlay);
  }
  coords.forEach((point, index) => {
    const lat = point[1];
    const lon = point[0];
    const ratio = (scores[index] - scoreMin) / Math.max(scoreMax - scoreMin, 1e-6);
    const marker = L.circleMarker([lat, lon], {
      radius: 3,
      weight: 0,
      fillOpacity: 0.8,
      fillColor: rgbaFromRatio(ratio, 0.95),
    }).bindPopup(
      `<div class="sample-popup"><strong>Sample</strong><br/>Score: ${scores[index].toFixed(4)}<br/>Lon: ${lon.toFixed(5)}<br/>Lat: ${lat.toFixed(5)}</div>`
    );
    marker.addTo(state.map);
    state.mapLayers.push(marker);
    bounds.push([lat, lon]);
  });
  sites.forEach((site) => {
    const marker = L.circleMarker([site[1], site[0]], {
      radius: 6,
      color: "#f5f7fa",
      weight: 2,
      fillOpacity: 0.1,
    }).bindPopup(
      `<div class="sample-popup"><strong>Known site</strong><br/>Lon: ${site[0].toFixed(5)}<br/>Lat: ${site[1].toFixed(5)}</div>`
    );
    marker.addTo(state.map);
    state.mapLayers.push(marker);
    bounds.push([site[1], site[0]]);
  });
  if (bounds.length) {
    state.map.fitBounds(bounds, { padding: [24, 24] });
  }
  setTimeout(() => state.map.invalidateSize(), 50);
}

function renderRunDetail(run) {
  const summary = document.getElementById("run-summary");
  if (!run) {
    summary.textContent = "No run selected.";
    return;
  }
  const metrics = run.metrics || {};
  const actions = document.getElementById("run-actions");
  summary.innerHTML = `
    <div class="metric-grid">
      <div><span>Status</span><strong>${run.status}</strong></div>
      <div><span>Dataset</span><strong>${run.dataset_id}</strong></div>
      <div><span>Model</span><strong>${run.model_name}</strong></div>
      <div><span>AUC</span><strong>${metrics.auc_mean ? metrics.auc_mean.toFixed(4) : "-"}</strong></div>
      <div><span>AP</span><strong>${metrics.ap_mean ? metrics.ap_mean.toFixed(4) : "-"}</strong></div>
      <div><span>Positives</span><strong>${metrics.positive_count ?? "-"}</strong></div>
      <div><span>DTD Top</span><strong>${metrics.dtd_top_mean ? metrics.dtd_top_mean.toFixed(4) : "-"}</strong></div>
      <div><span>Hit Rate</span><strong>${metrics.hit_rate_at_q1 ? metrics.hit_rate_at_q1.toFixed(4) : "-"}</strong></div>
      <div><span>Interpolation</span><strong>${run.payload?.anomaly_map?.method ?? "-"}</strong></div>
    </div>
  `;
  actions.innerHTML = "";
  if (["queued", "running", "cancelling"].includes(run.status)) {
    const cancel = document.createElement("button");
    cancel.textContent = run.status === "cancelling" ? "Cancelling..." : "Stop This Run";
    cancel.disabled = run.status === "cancelling";
    cancel.addEventListener("click", async () => {
      await fetchJson(`/api/runs/${run.run_id}/cancel`, { method: "POST" });
      await loadRun(run.run_id);
      await refreshRuns();
    });
    actions.appendChild(cancel);
  }
  drawMap(run);
}

async function loadRun(runId) {
  const run = await fetchJson(`/api/runs/${runId}`);
  state.selectedRun = run;
  renderRunDetail(run);
}

async function refreshRuns() {
  state.runs = await fetchJson("/api/runs");
  renderRuns();
  if (state.selectedRun) {
    const match = state.runs.find((run) => run.run_id === state.selectedRun.run_id);
    if (match) {
      await loadRun(match.run_id);
    }
  }
}

async function init() {
  await refreshRuns();
  document.getElementById("run-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = runFormData(event.currentTarget);
    if (!payload.get("dataset_name") || !payload.get("target_element")) {
      throw new Error("Dataset name and target element are required.");
    }
    if (!payload.get("sample_file") || !payload.get("site_file")) {
      throw new Error("Sample CSV and Site CSV are required.");
    }
    const response = await fetch("/api/runs/upload", {
      method: "POST",
      body: payload,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Run creation failed: ${response.status}`);
    }
    event.currentTarget.reset();
    setTimeout(refreshRuns, 500);
  });
  setInterval(refreshRuns, 5000);
}

init().catch((error) => {
  const summary = document.getElementById("run-summary");
  summary.textContent = error.message;
});
