const state = {
  runs: [],
  selectedRun: null,
  map: null,
  mapLayers: [],
  infoMarker: null,
  lastDrawnRunId: null,
  lastDrawnFingerprint: null,
};

function currentRunIdFromUrl() {
  return new URLSearchParams(window.location.search).get("run");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function runFormData(form) {
  return new FormData(form);
}

function renderRuns() {
  const container = document.getElementById("run-list");
  if (!container) {
    return;
  }
  container.innerHTML = "";
  state.runs.forEach((run) => {
    const card = document.createElement("div");
    card.className = `run-card status-${run.status}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "run-card-main";
    const auc = run.metrics?.auc_mean ? `AUC ${run.metrics.auc_mean.toFixed(4)}` : run.status;
    button.innerHTML = `
      <strong>${run.model_name}</strong>
      <span class="run-meta">${run.dataset_id}</span>
      <span class="run-meta">${auc}</span>
    `;
    button.addEventListener("click", () => {
      window.location.href = `/web/run.html?run=${encodeURIComponent(run.run_id)}`;
    });
    if (["queued", "running", "cancelling"].includes(run.status)) {
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "stop-button";
      cancel.textContent = run.status === "cancelling" ? "Cancelling..." : "Stop";
      cancel.disabled = run.status === "cancelling";
      cancel.addEventListener("click", async (event) => {
        event.stopPropagation();
        event.preventDefault();
        showFormStatus(`Stopping run ${run.dataset_id}...`);
        await fetchJson(`/api/runs/${run.run_id}/cancel`, { method: "POST" });
        await refreshRuns();
      });
      button.appendChild(cancel);
    }
    card.appendChild(button);
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
  if (state.infoMarker) {
    state.map.removeLayer(state.infoMarker);
    state.infoMarker = null;
  }
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
    interactive: true,
  });
}

function nearestSampleInfo(latlng, coords, scores) {
  let bestIndex = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  coords.forEach((point, index) => {
    const dx = point[0] - latlng.lng;
    const dy = point[1] - latlng.lat;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  if (bestIndex < 0) {
    return null;
  }
  return {
    index: bestIndex,
    lon: coords[bestIndex][0],
    lat: coords[bestIndex][1],
    score: scores[bestIndex],
    distance: Math.sqrt(bestDistance),
  };
}

function interpolatedValueAt(latlng, anomalyMap) {
  if (!anomalyMap?.grid?.length || !anomalyMap.grid_x?.length || !anomalyMap.grid_y?.length) {
    return null;
  }
  const { grid, grid_x: gridX, grid_y: gridY } = anomalyMap;
  const xMin = gridX[0];
  const xMax = gridX[gridX.length - 1];
  const yMin = gridY[0];
  const yMax = gridY[gridY.length - 1];
  if (latlng.lng < xMin || latlng.lng > xMax || latlng.lat < yMin || latlng.lat > yMax) {
    return null;
  }
  const xRatio = (latlng.lng - xMin) / Math.max(xMax - xMin, 1e-9);
  const yRatio = (latlng.lat - yMin) / Math.max(yMax - yMin, 1e-9);
  const col = Math.min(gridX.length - 1, Math.max(0, Math.round(xRatio * (gridX.length - 1))));
  const row = Math.min(gridY.length - 1, Math.max(0, Math.round(yRatio * (gridY.length - 1))));
  return grid[row]?.[col] ?? null;
}

function attachMapInspector(run, coords, scores) {
  if (!state.map) {
    return;
  }
  state.map.off("click");
  state.map.on("click", (event) => {
    const nearest = nearestSampleInfo(event.latlng, coords, scores);
    const interpolated = interpolatedValueAt(event.latlng, run.payload?.anomaly_map);
    const lines = [
      `<strong>Map point</strong>`,
      `Lon: ${event.latlng.lng.toFixed(5)}`,
      `Lat: ${event.latlng.lat.toFixed(5)}`,
    ];
    if (interpolated !== null) {
      lines.push(`Heat value: ${Number(interpolated).toFixed(4)}`);
    }
    if (nearest) {
      lines.push(`Nearest sample score: ${nearest.score.toFixed(4)}`);
      lines.push(`Nearest sample: (${nearest.lon.toFixed(5)}, ${nearest.lat.toFixed(5)})`);
    }
    if (state.infoMarker) {
      state.map.removeLayer(state.infoMarker);
    }
    state.infoMarker = L.popup({ maxWidth: 320 })
      .setLatLng(event.latlng)
      .setContent(`<div class="sample-popup">${lines.join("<br/>")}</div>`)
      .openOn(state.map);
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
  const drawFingerprint = JSON.stringify({
    runId: run.run_id,
    status: run.status,
    metricAuc: run.metrics?.auc_mean ?? null,
    scoreCount: scores.length,
  });
  const preserveView = state.lastDrawnRunId === run.run_id;
  const scoreMin = Math.min(...scores);
  const scoreMax = Math.max(...scores);
  const bounds = [];
  const overlay = anomalyOverlay(run.payload.anomaly_map);
  if (overlay) {
    overlay.addTo(state.map);
    state.mapLayers.push(overlay);
  }
  if (typeof L.heatLayer === "function") {
    const heatPoints = coords.map((point, index) => {
      const lat = point[1];
      const lon = point[0];
      const ratio = (scores[index] - scoreMin) / Math.max(scoreMax - scoreMin, 1e-6);
      bounds.push([lat, lon]);
      return [lat, lon, 0.15 + 0.85 * ratio];
    });
    const heatLayer = L.heatLayer(heatPoints, {
      radius: 22,
      blur: 18,
      minOpacity: 0.28,
      maxZoom: 12,
      gradient: {
        0.15: "#1a7f37",
        0.4: "#8ebf3f",
        0.65: "#f6c945",
        0.85: "#f17c36",
        1.0: "#c92a2a",
      },
    });
    heatLayer.addTo(state.map);
    state.mapLayers.push(heatLayer);
  }
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
  attachMapInspector(run, coords, scores);
  if (bounds.length && !preserveView) {
    state.map.fitBounds(bounds, { padding: [24, 24] });
  }
  state.lastDrawnRunId = run.run_id;
  state.lastDrawnFingerprint = drawFingerprint;
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
    actions.className = "summary run-actions-inline";
    const cancel = document.createElement("button");
    cancel.className = "stop-button";
    cancel.textContent = run.status === "cancelling" ? "Cancelling..." : "Stop";
    cancel.disabled = run.status === "cancelling";
    cancel.addEventListener("click", async () => {
      await fetchJson(`/api/runs/${run.run_id}/cancel`, { method: "POST" });
      await loadRun(run.run_id);
      await refreshRuns();
    });
    actions.appendChild(cancel);
  } else {
    actions.className = "summary";
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
  const runId = currentRunIdFromUrl();
  if (runId && document.getElementById("run-summary")) {
    const match = state.runs.find((run) => run.run_id === runId);
    if (match) {
      await loadRun(match.run_id);
    } else {
      showMessage("Run not found.");
    }
  }
}

function showMessage(message) {
  const summary = document.getElementById("run-summary");
  if (summary) {
    summary.textContent = message;
  }
}

function showFormStatus(message) {
  const status = document.getElementById("form-status");
  if (status) {
    status.textContent = message;
  }
}

async function init() {
  const backButton = document.getElementById("back-button");
  if (backButton) {
    backButton.addEventListener("click", () => {
      window.location.href = "/";
    });
  }
  await refreshRuns();
  const form = document.getElementById("run-form");
  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const submitButton = event.currentTarget.querySelector('button[type="submit"]');
      const payload = runFormData(event.currentTarget);
      if (!payload.get("dataset_name") || !payload.get("target_element")) {
        showFormStatus("Dataset name and target element are required.");
        return;
      }
      const sampleFile = payload.get("sample_file");
      const siteFile = payload.get("site_file");
      if (!(sampleFile instanceof File) || sampleFile.size === 0 || !(siteFile instanceof File) || siteFile.size === 0) {
        showFormStatus("Sample CSV and Site CSV are required.");
        return;
      }
      submitButton.disabled = true;
      showFormStatus("Submitting run to backend...");
      try {
        const response = await fetch("/api/runs/upload", {
          method: "POST",
          body: payload,
        });
        if (!response.ok) {
          const text = await response.text();
          showFormStatus(text || `Run creation failed: ${response.status}`);
          return;
        }
        const run = await response.json();
        state.runs = [run, ...state.runs];
        renderRuns();
        event.currentTarget.reset();
        showFormStatus(`Run created: ${run.dataset_id} (${run.run_id}). You can submit another dataset immediately.`);
        setTimeout(refreshRuns, 300);
      } catch (error) {
        showFormStatus(error instanceof Error ? error.message : "Run creation failed.");
      } finally {
        submitButton.disabled = false;
      }
    });
  }
  setInterval(refreshRuns, 5000);
}

init().catch((error) => {
  showMessage(error.message);
});
