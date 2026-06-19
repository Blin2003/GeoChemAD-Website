const SAMPLE_COLORS = {
  background: "#64748b",
  buffer: "#fbbf24",
  near_deposit: "#ef4444",
  survey: "#2563eb",
};

const LEVEL_COLORS = {
  LOW: "rgba(100,116,139,0.72)",
  MODERATE: "rgba(251,191,36,0.88)",
  HIGH: "rgba(255,208,96,0.92)",
};

const LEVEL_HEX = {
  LOW: "#94a3b8",
  MODERATE: "#fbbf24",
  HIGH: "#ffd060",
};

const state = {
  map: null,
  sampleLayer: null,
  siteLayer: null,
  coverageLayer: null,
  pickedLayer: null,
  feedbackLayer: null,
  sampleRenderer: null,
  sessionId: null,
  sessionKind: "research",
  target: "Cu",
  targetLabel: "Copper",
  targetOptions: [],
  samplePreview: [],
  sitePoints: [],
  coverageRegions: [],
  availableElements: [],
  knowledgeEntry: null,
  evaluation: null,
  referenceExample: null,
  manualTemplate: null,
  modelMetadata: null,
  dataProfile: null,
  layerInventory: null,
  gaugeChart: null,
  lastScorePct: 0,
  drawLayer: null,
  drawControl: null,
  placeTimer: null,
  isDrawingRegion: false,
  regionClickBlockUntil: 0,
  layerVisibility: {
    samples: true,
    sites: true,
  },
};

function $(id) {
  return document.getElementById(id);
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "n/a";
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toFixed(4);
}

function elementLabel(key) {
  return key.replace(/_(ppm|ppb|pct)$/i, "");
}

function escapeHtml(value) {
  return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.text();
    try {
      const parsed = JSON.parse(body);
      throw new Error(parsed.detail || body);
    } catch (error) {
      if (error instanceof SyntaxError) throw new Error(body);
      throw error;
    }
  }
  return response.json();
}

function ensureMap() {
  if (state.map || typeof L === "undefined") return;
  state.map = L.map("analysis-map", {
    zoomControl: true,
    preferCanvas: true,
  }).setView([-26, 122], 5);
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles &copy; Esri",
    maxZoom: 18,
  }).addTo(state.map);
  state.sampleRenderer = L.canvas({ padding: 0.4 });
  if (typeof L.Control.Draw !== "undefined") {
    state.drawLayer = new L.FeatureGroup().addTo(state.map);
    state.drawControl = new L.Control.Draw({
      position: "topright",
      draw: {
        polyline: false,
        marker: false,
        circlemarker: false,
        polygon: { allowIntersection: false, showArea: true },
        rectangle: true,
        circle: true,
      },
      edit: { featureGroup: state.drawLayer, remove: true },
    });
    state.map.addControl(state.drawControl);
    state.map.on(L.Draw.Event.DRAWSTART, () => {
      state.isDrawingRegion = true;
      state.regionClickBlockUntil = Date.now() + 1000;
    });
    state.map.on(L.Draw.Event.DRAWSTOP, () => {
      state.regionClickBlockUntil = Date.now() + 1000;
      window.setTimeout(() => {
        state.isDrawingRegion = false;
      }, 80);
    });
    state.map.on(L.Draw.Event.CREATED, (event) => handleRegionDraw(event.layer, event.layerType));
  }
}

function initGauge() {
  const canvas = $("gauge-canvas");
  if (!canvas || state.gaugeChart || typeof Chart === "undefined") return;
  state.gaugeChart = new Chart(canvas.getContext("2d"), {
    type: "doughnut",
    data: {
      datasets: [{
        data: [0, 100],
        backgroundColor: [LEVEL_COLORS.LOW, "rgba(255,255,255,0.055)"],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
        borderRadius: 5,
      }],
    },
    options: {
      responsive: false,
      cutout: "70%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 1100, easing: "easeInOutCubic" },
    },
  });
}

function animateCount(el, from, to) {
  const start = performance.now();
  const duration = 900;
  const step = (time) => {
    const t = Math.min((time - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = `${(from + (to - from) * eased).toFixed(1)}%`;
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function updateGauge(scorePct, level) {
  initGauge();
  const color = LEVEL_COLORS[level] || LEVEL_COLORS.LOW;
  if (state.gaugeChart) {
    state.gaugeChart.data.datasets[0].data = [scorePct, Math.max(0, 100 - scorePct)];
    state.gaugeChart.data.datasets[0].backgroundColor = [color, "rgba(255,255,255,0.055)"];
    state.gaugeChart.update();
  }
  const pct = $("score-pct");
  pct.style.color = LEVEL_HEX[level] || LEVEL_HEX.LOW;
  animateCount(pct, state.lastScorePct, scorePct);
  state.lastScorePct = scorePct;
}

function clearLayer(layer) {
  if (layer && state.map) state.map.removeLayer(layer);
}

function setMapHint(message, tone = "info") {
  const hint = $("map-hint");
  hint.textContent = message;
  hint.dataset.tone = tone;
}

function showMapFeedback(latlng, message, tone = "loading") {
  clearLayer(state.feedbackLayer);
  const color = tone === "error" ? "#dc2626" : "#2563eb";
  state.feedbackLayer = L.circleMarker([latlng.lat, latlng.lng], {
    radius: tone === "error" ? 9 : 7,
    color,
    weight: 2,
    fillColor: tone === "error" ? "#fee2e2" : "#dbeafe",
    fillOpacity: 0.95,
  }).addTo(state.map);
  state.feedbackLayer.bindPopup(
    `<div class="map-popup feedback-popup ${tone}">
      <div class="map-popup-loc">${latlng.lat.toFixed(4)}°, ${latlng.lng.toFixed(4)}°</div>
      <strong>${tone === "error" ? "No model result" : "Scoring point"}</strong>
      <div class="map-popup-exp">${escapeHtml(message)}</div>
    </div>`,
    { maxWidth: 320, closeButton: true, autoPan: true }
  ).openPopup();
}

function fitBoundsFromData() {
  if (state.coverageRegions.length) {
    const bounds = [];
    state.coverageRegions.forEach((region) => {
      const box = region.bounds;
      if (!box) return;
      bounds.push([box.south, box.west], [box.north, box.east]);
    });
    if (bounds.length) {
      state.map.fitBounds(bounds, { padding: [34, 34] });
      return;
    }
  }
  state.map.fitBounds([[-33.5, 114.5], [-15.0, 128.5]], { padding: [24, 24] });
}

function renderCoverageLayer() {
  clearLayer(state.coverageLayer);
  state.coverageLayer = L.layerGroup(
    state.coverageRegions.map((region) => {
      const box = region.bounds;
      const polygon = L.rectangle([[box.south, box.west], [box.north, box.east]], {
        color: "#16a34a",
        weight: 1.4,
        dashArray: region.type === "model-extent" ? "6 6" : null,
        fillColor: "#bbf7d0",
        fillOpacity: region.type === "model-extent" ? 0.10 : 0.22,
        interactive: false,
      });
      polygon.bindTooltip(`${region.label} data coverage`, {
        sticky: true,
        direction: "top",
        opacity: 0.88,
      });
      return polygon;
    })
  );
  state.coverageLayer.addTo(state.map);
}

function renderMapLayers() {
  ensureMap();
  renderCoverageLayer();
  clearLayer(state.sampleLayer);
  clearLayer(state.siteLayer);

  state.sampleLayer = L.layerGroup(
    state.samplePreview.map((row) =>
      L.circleMarker([row.lat, row.lon], {
        renderer: state.sampleRenderer,
        radius: row.label === "near_deposit" ? 3.2 : 2.3,
        stroke: false,
        fillColor: SAMPLE_COLORS[row.label] || SAMPLE_COLORS.buffer,
        fillOpacity: row.label === "background" ? 0.38 : 0.68,
        interactive: false,
      })
    )
  );
  if (state.layerVisibility.samples) state.sampleLayer.addTo(state.map);

  state.siteLayer = L.layerGroup(
    state.sitePoints.map((row) =>
      L.circleMarker([row.lat, row.lon], {
        radius: 5.8,
        color: "#0f172a",
        weight: 1.4,
        fillColor: "#ffd060",
        fillOpacity: 0.95,
      }).bindPopup(
        `<strong>${row.name}</strong><br>${row.type}<br>${row.commodity}<br>` +
        `<em>Known site marker; scoring still depends on nearby model evidence.</em>`
      )
    )
  );
  if (state.layerVisibility.sites) state.siteLayer.addTo(state.map);

  state.siteLayer.eachLayer((layer) => layer.on("click", (event) => {
    L.DomEvent.stop(event);
    analyzePoint(event.latlng, { source: "site" });
  }));
  state.map.off("click");
  state.map.on("click", (event) => {
    if (state.isDrawingRegion || Date.now() < state.regionClickBlockUntil) return;
    analyzePoint(event.latlng);
  });
  renderMapLayerControl();
  fitBoundsFromData();
}

function setOverlayVisible(key, visible) {
  state.layerVisibility[key] = visible;
  const layer = key === "samples" ? state.sampleLayer : state.siteLayer;
  if (!layer || !state.map) return;
  if (visible && !state.map.hasLayer(layer)) layer.addTo(state.map);
  if (!visible && state.map.hasLayer(layer)) state.map.removeLayer(layer);
}

function renderMapLayerControl() {
  const container = $("map-layer-list");
  if (!container) return;
  const inventory = state.layerInventory;
  const geophys = inventory?.groups?.find((group) => group.key === "geophys");
  const structure = inventory?.groups?.find((group) => group.key === "structure");
  container.innerHTML = [
    `<label class="layer-check"><input type="checkbox" data-layer-toggle="samples" ${state.layerVisibility.samples ? "checked" : ""} /> <span>GSWA samples</span></label>`,
    `<label class="layer-check"><input type="checkbox" data-layer-toggle="sites" ${state.layerVisibility.sites ? "checked" : ""} /> <span>Known ${escapeHtml(state.target)} sites</span></label>`,
    `<div class="layer-row"><span>Light green coverage</span><em>areas with data loaded for this profile</em></div>`,
    `<div class="layer-row disabled"><span>Magnetic / gravity rasters</span><em>${geophys ? "teacher data present; not displayed or used in this run" : "not loaded"}</em></div>`,
    `<div class="layer-row disabled"><span>Faults / worms / geology</span><em>${structure ? "teacher data present; not displayed or used in this run" : "not loaded"}</em></div>`,
  ].join("");
  container.querySelectorAll("[data-layer-toggle]").forEach((input) => {
    input.addEventListener("change", () => setOverlayVisible(input.dataset.layerToggle, input.checked));
  });
}

function updateSessionSummary(payload) {
  $("session-badge").textContent = `${payload.target} | supervisor GAD`;
  $("summary-samples").textContent = payload.sampleCount.toLocaleString();
  $("summary-sites").textContent = (payload.targetSiteCount ?? payload.siteCount).toLocaleString();
  $("summary-auc").textContent = payload.evaluation ? payload.evaluation.auc.toFixed(3) : "n/a";
  $("header-target-label").textContent = `${payload.targetLabel} prospectivity workspace`;
  $("gauge-target-label").textContent = `${payload.target} prospectivity`;
  $("legend-target-site").textContent = `${payload.target} site`;
  $("welcome-copy").textContent =
    `${payload.sampleCount.toLocaleString()} GSWA stream-sediment samples and ` +
    `${payload.targetSiteCount.toLocaleString()} known ${payload.target} sites are connected to the supervisor model.`;
  if (payload.modelWarning) {
    $("welcome-copy").textContent += ` ${payload.modelWarning}`;
  }
  const badges = $("research-badges");
  badges.hidden = false;
  badges.innerHTML = [
    `<span>Supervisor backend</span>`,
    `<span>${escapeHtml(payload.modelMetadata.dataProfile?.name || "Current data")}</span>`,
    `<span>Geochem scoring layer</span>`,
    `<span class="limited">Geophysics not active</span>`,
    `<span class="limited">Structure not active</span>`,
    `<span>WA coverage</span>`,
    `<span>Spatial AUC ${payload.modelMetadata.spatialAuc.toFixed(3)}</span>`,
  ].join("");
}

function renderTargetSelector() {
  const select = $("target-select");
  if (!select) return;
  select.innerHTML = state.targetOptions
    .map((target) => `<option value="${target.key}">${target.label} (${target.key})</option>`)
    .join("");
  select.value = state.target;
}

function updateModelPanel(payload) {
  const metadata = payload.modelMetadata;
  $("model-description").textContent = payload.knowledgeEntry.text_description;
  $("model-provenance").innerHTML = `
    <span>Source: ${metadata.source}</span>
    <span>Engine: ${metadata.engine}</span>
    <span>Owner: ${metadata.ownership}</span>
  `;
  $("model-detail").innerHTML = [
    `<div class="detail-row"><span>Target</span><strong>${payload.targetLabel} (${payload.target})</strong></div>`,
    `<div class="detail-row"><span>Data profile</span><strong>${escapeHtml(metadata.dataProfile?.name || "Current data")}</strong></div>`,
    `<div class="detail-row"><span>Data root</span><strong>${escapeHtml(metadata.dataRoot)}</strong></div>`,
    `<div class="detail-row"><span>Spatial AUC</span><strong>${metadata.spatialAuc.toFixed(3)} | ${metadata.validation}</strong></div>`,
    `<div class="detail-row"><span>Data layer</span><strong>${metadata.layers.join(", ")}</strong></div>`,
    `<div class="detail-row"><span>Active sources</span><strong>${metadata.activeSources.join(", ")}</strong></div>`,
    `<div class="detail-row"><span>Experts</span><strong>${metadata.experts.join(", ")}</strong></div>`,
    ...(metadata.layerStatus || [
      { name: "Geochemistry", active: true, role: "Active supervisor evidence layer" },
      { name: "Magnetics and gravity", active: false, role: "Not active in this local run" },
      { name: "Structure and geology", active: false, role: "Not active in this local run" },
    ]).map((layer) =>
      `<div class="detail-row"><span>${escapeHtml(layer.name)}</span>` +
      `<strong class="${layer.active ? "source-active" : "source-inactive"}">` +
      `${layer.active ? "ACTIVE" : "NOT ACTIVE"} · ${escapeHtml(layer.role)}</strong></div>`
    ),
    `<div class="detail-note"><span>Gravity and magnetics</span>` +
      `<p>Supervisor-model evidence, not user selection filters. They are not included ` +
      `in the current score. Raster overlays are listed as available teacher data and need tile conversion for browser display.</p></div>`,
    `<div class="detail-row"><span>Coverage</span><strong>Western Australia only</strong></div>`,
  ].join("");
}

function renderEvidenceLayers(result) {
  const layers = result.evidenceLayers || [];
  $("evidence-layer-list").innerHTML = layers.map((layer) => {
    const signals = (layer.topSignals || []).map((signal) =>
      `<span>${escapeHtml(signal.label)} ${signal.zScore >= 0 ? "+" : ""}${signal.zScore.toFixed(2)}σ</span>`
    ).join("");
    return `
      <div class="evidence-layer ${layer.active ? "active" : "available"}">
        <div class="evidence-layer-head">
          <strong>${escapeHtml(layer.label)}</strong>
          <em>${layer.active ? "USED" : "AVAILABLE"}</em>
        </div>
        <p>${escapeHtml(layer.note)}</p>
        <div class="detail-row compact-row">
          <span>Signals</span>
          <strong>${layer.signalCount.toLocaleString()} | contribution ${formatNumber(layer.contribution)}</strong>
        </div>
        ${signals ? `<div class="layer-signal-list">${signals}</div>` : ""}
      </div>
    `;
  }).join("");
}

function updatePickedMarker(latlng, result) {
  clearLayer(state.feedbackLayer);
  state.feedbackLayer = null;
  clearLayer(state.pickedLayer);
  const color = LEVEL_HEX[result.prospectivity.level] || LEVEL_HEX.LOW;
  state.pickedLayer = L.circleMarker([latlng.lat, latlng.lng], {
    radius: 11,
    color: "#111827",
    weight: 2.5,
    fillColor: color,
    fillOpacity: 0.95,
  }).addTo(state.map);
  state.pickedLayer
    .bindPopup(
      `<div class="map-popup">
        <div class="map-popup-loc">${latlng.lat.toFixed(4)}°, ${latlng.lng.toFixed(4)}°</div>
        <div class="map-popup-prob" style="color:${color}">${result.prospectivity.scorePct.toFixed(1)}%</div>
        <div class="map-popup-verdict" style="color:${color}">${result.prospectivity.level}</div>
        <div class="map-popup-exp">${escapeHtml(result.interpretation)}</div>
      </div>`,
      { maxWidth: 280 }
    )
    .openPopup();
}

function updateResult(result) {
  const methodText = result.mode === "region"
    ? "Supervisor regional grid"
    : "Supervisor ProspectivityModel";
  $("welcome-panel").hidden = true;
  $("result-panel").hidden = false;
  $("selection-card-label").textContent = result.mode === "region" ? "Selected Region" : "Selected Point";
  $("point-title").textContent = result.mode === "region"
    ? `${result.target} region summary`
    : result.point ? `${result.point.lat.toFixed(4)}, ${result.point.lon.toFixed(4)}` : "Manual values";
  $("point-copy").textContent = result.mode === "region" && result.region
    ? `${result.region.validGridCount.toLocaleString()} of ${result.region.candidateGridCount.toLocaleString()} candidate grid points scored`
    : methodText;
  updateGauge(result.prospectivity.scorePct, result.prospectivity.level);
  $("level-badge").textContent = result.prospectivity.level;
  $("level-badge").className = `level-badge ${result.prospectivity.level.toLowerCase()}`;
  $("zscore-val").textContent = result.researchModel
    ? `g-score ${result.anomaly.score.toFixed(3)}`
    : `Score ${result.anomaly.score >= 0 ? "+" : ""}${result.anomaly.score.toFixed(4)}`;
  $("signal-summary").textContent = result.explanation?.summary || result.interpretation;
  $("assessment-body").textContent = result.interpretation;
  $("expert-summary").textContent = result.explanation?.summary || "No expert narrative yet.";
  renderEvidenceLayers(result);
  renderExpertBreakdown(result);
  $("signal-detail").innerHTML = [
    `<div class="detail-row"><span>Weighted score</span><strong>${result.prospectivity.scorePct.toFixed(1)}%</strong></div>`,
    `<div class="detail-row"><span>${result.researchModel ? "G-score" : "Anomaly score"}</span><strong>${result.anomaly.score.toFixed(4)}</strong></div>`,
    `<div class="detail-row"><span>Method</span><strong>${methodText}</strong></div>`,
    `<div class="detail-note"><span>Evidence</span><p>${escapeHtml(result.evidence)}</p></div>`,
  ].join("");

  const contextRows = [];
  if (result.mode === "region" && result.region) {
    const tierText = (result.region.tierDistribution || [])
      .filter((item) => item.count)
      .map((item) => `${item.tier} ${item.pct.toFixed(0)}%`)
      .join(", ");
    contextRows.push(
      `<div class="detail-row"><span>Valid grid points</span><strong>${result.region.validGridCount.toLocaleString()} / ${result.region.candidateGridCount.toLocaleString()}</strong></div>`,
      `<div class="detail-row"><span>Model coverage</span><strong>${result.region.coveragePct.toFixed(0)}%</strong></div>`,
      `<div class="detail-row"><span>Tier distribution</span><strong>${escapeHtml(tierText)}</strong></div>`,
      `<div class="detail-row"><span>Centroid</span><strong>${result.region.centroid.lat.toFixed(5)}, ${result.region.centroid.lon.toFixed(5)}</strong></div>`,
      `<div class="detail-row"><span>Bounds</span><strong>${result.region.bounds.south.toFixed(3)} to ${result.region.bounds.north.toFixed(3)} lat</strong></div>`,
      `<div class="detail-row"><span>Longitude</span><strong>${result.region.bounds.west.toFixed(3)} to ${result.region.bounds.east.toFixed(3)}</strong></div>`
    );
  } else if (result.point) {
    contextRows.push(`<div class="detail-row"><span>Requested</span><strong>${result.point.lat.toFixed(5)}, ${result.point.lon.toFixed(5)}</strong></div>`);
  }
  if (Number.isFinite(result.nearestSampleKm)) {
    contextRows.push(`<div class="detail-row"><span>Nearest sample</span><strong>${result.nearestSampleKm.toFixed(2)} km</strong></div>`);
  }
  if (result.sampleRecord) {
    contextRows.push(`<div class="detail-row"><span>Matched sample</span><strong>${result.sampleRecord.lat.toFixed(5)}, ${result.sampleRecord.lon.toFixed(5)}</strong></div>`);
  }
  if (result.nearestSite) {
    contextRows.push(
      `<div class="detail-row"><span>Nearest ${result.target} site</span><strong>${result.nearestSite.name}</strong></div>`,
      `<div class="detail-row"><span>Site distance</span><strong>${result.nearestSite.distanceKm.toFixed(2)} km</strong></div>`,
      `<div class="detail-row"><span>Zone</span><strong>${result.zoneLabel}</strong></div>`,
      `<div class="detail-row"><span>Commodity</span><strong>${result.nearestSite.commodity}</strong></div>`
    );
  } else {
    contextRows.push(`<div class="detail-row"><span>Zone</span><strong>${result.zoneLabel}</strong></div>`);
  }
  $("point-context").innerHTML = contextRows.join("");

  renderPills(result);
  renderElementProfiles(result);
  renderPathfinderRatios(result);
  renderElementTable(result);
  document.querySelectorAll("#result-panel .glass-card").forEach((card, index) => {
    card.style.animation = "none";
    void card.offsetWidth;
    card.style.animation = `rise .42s cubic-bezier(.16,1,.3,1) ${index * 0.045}s both`;
  });
}

function renderPills(result) {
  const details = result.anomaly.details || {};
  $("anom-pills").innerHTML = Object.entries(details)
    .slice(0, 10)
    .map(([element, detail]) => {
      const hot = detail.anomalous || detail.suppressed;
      const mark = detail.anomalous ? "▲" : detail.suppressed ? "▼" : "✓";
      const regional = result.mode === "region" && Number.isFinite(detail.supportPct)
        ? ` · ${detail.supportPct.toFixed(0)}% support`
        : "";
      return `<span class="pill ${hot ? "pill-anom" : "pill-ok"}">${mark} ${elementLabel(element)}${regional}</span>`;
    })
    .join("");
}

function renderExpertBreakdown(result) {
  const experts = result.explanation?.expertBreakdown || [];
  const signals = result.explanation?.topSignals || [];
  const rows = experts.map((expert) => {
    const coverage = result.mode === "region" && Number.isFinite(expert.coveragePct)
      ? ` | ${expert.coveragePct.toFixed(0)}% coverage`
      : "";
    return `<div class="detail-row"><span>${escapeHtml(expert.name)}</span><strong>${(expert.score * 100).toFixed(0)}% median | w ${expert.weight.toFixed(2)}${coverage}</strong></div>`;
  });
  signals.slice(0, 4).forEach((signal, index) => {
    const signalText = result.mode === "region" && Number.isFinite(signal.supportPct)
      ? `${signal.supportPct.toFixed(0)}% support`
      : signal.direction;
    rows.push(
      `<div class="detail-row"><span>Signal ${index + 1}</span><strong>${escapeHtml(signal.label)} · ${escapeHtml(signalText)}</strong></div>`
    );
  });
  $("expert-breakdown").innerHTML = rows.join("") || `<div class="detail-row"><span>Status</span><strong>No expert signals</strong></div>`;
}

function renderElementProfiles(result) {
  const signals = result.featureSignals || [];
  $("elem-profile-list").innerHTML = signals.map((signal) => {
    const magnitude = Math.min(Math.abs(signal.zScore) * 18 + 12, 100);
    const cls = signal.zScore > 0 ? "anom" : "low";
    return `
      <div class="epr-row">
        <div class="epr-sym">${escapeHtml(signal.label.split("_")[0])}</div>
        <div class="epr-center">
          <div class="epr-val">${escapeHtml(signal.label)} <span>${escapeHtml(signal.source)}${result.mode === "region" ? ` · ${signal.coveragePct.toFixed(0)}% coverage` : ""}</span></div>
          <div class="epr-bar-track"><div class="epr-bar-fill ${cls}" data-width="${magnitude}"></div></div>
        </div>
        <div class="epr-right">
          <div class="epr-fold ${cls}">${signal.zScore >= 0 ? "+" : ""}${signal.zScore.toFixed(2)}σ</div>
          <div class="epr-tag ${cls}">${result.mode === "region" ? `${signal.supportPct.toFixed(0)}% support` : signal.direction}</div>
        </div>
      </div>
    `;
  }).join("");
  requestAnimationFrame(() => {
    document.querySelectorAll(".epr-bar-fill").forEach((bar) => {
      bar.style.width = `${bar.dataset.width}%`;
    });
  });
}

function renderPathfinderRatios(result) {
  const scope = result.mode === "region"
    ? "Signals shown here are median feature evidence aggregated across all valid grid points."
    : "Only signals returned in the selected point's NodeScore tree are displayed.";
  $("ratio-list").innerHTML =
    `<div class="detail-note"><span>Supervisor features</span>` +
    `<p>Configured ratios are evaluated by the supervisor pathfinder expert. ` +
    `${scope}</p></div>`;
}

function renderElementTable(result) {
  $("element-table").innerHTML = (result.featureSignals || []).map((signal) => `
    <div class="element-row">
      <span>${escapeHtml(signal.label)}</span>
      <strong>${signal.zScore >= 0 ? "+" : ""}${signal.zScore.toFixed(2)}σ</strong>
      <em>${escapeHtml(signal.source)} | w ${signal.weight.toFixed(2)}${result.mode === "region" ? ` | ${signal.supportPct.toFixed(0)}% support | ${signal.coveragePct.toFixed(0)}% coverage` : ""}</em>
    </div>
  `).join("");
}

async function analyzePoint(latlng, options = {}) {
  if (!state.sessionId) return;
  $("raw-lat").value = latlng.lat.toFixed(6);
  $("raw-lon").value = latlng.lng.toFixed(6);
  const params = new URLSearchParams({ lat: String(latlng.lat), lon: String(latlng.lng), target: state.target });
  const sourceText = options.source === "site"
    ? "known site marker; scoring nearby model evidence"
    : "coordinate";
  showMapFeedback(latlng, `Running the supervisor ${state.target} model at this ${sourceText}...`);
  setMapHint(`Scoring ${latlng.lat.toFixed(4)}, ${latlng.lng.toFixed(4)} with the supervisor model...`, "loading");
  try {
    const result = await fetchJson(`/api/research/${state.sessionId}/point?${params.toString()}`);
    updateResult(result);
    updatePickedMarker(latlng, result);
    setMapHint(`Supervisor model returned a ${result.prospectivity.level.toLowerCase()} result for this point.`, "success");
    return result;
  } catch (error) {
    console.error(error);
    let message = error.message || String(error);
    if (options.source === "site") {
      message = `This yellow marker is a known ${state.target} site, but the current data profile has no local scoring evidence here. ${message}`;
    }
    showMapFeedback(latlng, message, "error");
    setMapHint(message, "error");
    $("raw-check-status").textContent = message;
    return null;
  }
}

async function coordinateScore() {
  const lat = Number($("raw-lat").value);
  const lon = Number($("raw-lon").value);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    $("raw-check-status").textContent = "Enter latitude and longitude.";
    return;
  }
  $("raw-check-status").textContent = "Analyzing coordinate...";
  const result = await analyzePoint({ lat, lng: lon });
  if (result) $("raw-check-status").textContent = "Coordinate analysis complete.";
}

async function switchTarget() {
  if (!state.sessionId) return;
  state.target = $("target-select").value || "Cu";
  clearLayer(state.pickedLayer);
  state.pickedLayer = null;
  state.drawLayer?.clearLayers();
  $("region-detail").innerHTML = "";
  $("region-status").textContent = "Use the rectangle, polygon, or circle tools on the map to summarize a custom area.";
  $("welcome-panel").hidden = false;
  $("welcome-copy").textContent = `Loading the supervisor ${state.target} model. This may take 10–25 seconds...`;
  try {
    const payload = await fetchJson(`/api/research/${state.sessionId}?${new URLSearchParams({ target: state.target }).toString()}`);
    sessionStorage.setItem(
      `geochemad:research:${state.sessionId}`,
      JSON.stringify(payload)
    );
    hydrate(payload);
    $("welcome-panel").hidden = false;
    $("result-panel").hidden = true;
  } catch (error) {
    console.error(error);
    $("welcome-copy").textContent = `Target switch failed: ${error.message || error}`;
  }
}

function regionPayload(layer, layerType) {
  if (layerType === "circle") {
    const center = layer.getLatLng();
    return {
      type: "circle",
      center: { lat: center.lat, lon: center.lng },
      radiusKm: layer.getRadius() / 1000,
    };
  }
  const latlngs = layer.getLatLngs();
  const ring = Array.isArray(latlngs[0]) ? latlngs[0] : latlngs;
  return {
    type: layerType === "rectangle" ? "bbox" : "polygon",
    points: ring.map((point) => ({ lat: point.lat, lon: point.lng })),
  };
}

function regionPopupAnchor(layer, result) {
  if (typeof layer.getLatLng === "function") return layer.getLatLng();
  if (typeof layer.getBounds === "function") return layer.getBounds().getCenter();
  const centroid = result.region?.centroid || result.point;
  return L.latLng(centroid.lat, centroid.lon);
}

function regionPopupHtml(result) {
  const color = LEVEL_HEX[result.prospectivity.level] || LEVEL_HEX.LOW;
  const centroid = result.region?.centroid || result.point;
  return `
    <div class="map-popup region-popup">
      <div class="map-popup-loc">${centroid.lat.toFixed(4)}°, ${centroid.lon.toFixed(4)}° | ${result.region.validGridCount.toLocaleString()}/${result.region.candidateGridCount.toLocaleString()} grid points</div>
      <div class="map-popup-prob" style="color:${color}">${result.prospectivity.scorePct.toFixed(1)}%</div>
      <div class="map-popup-verdict" style="color:${color}">${result.prospectivity.level}</div>
      <div class="map-popup-exp">${escapeHtml(result.explanation?.summary || result.interpretation)}</div>
    </div>
  `;
}

function showRegionPopup(layer, result) {
  layer
    .bindPopup(regionPopupHtml(result), {
      maxWidth: 340,
      closeButton: true,
      className: "region-summary-popup",
      autoPan: true,
    })
    .openPopup(regionPopupAnchor(layer, result));
}

function showRegionError(layer, message) {
  layer
    .bindPopup(
      `<div class="map-popup feedback-popup error">
        <strong>Region not scored</strong>
        <div class="map-popup-exp">${escapeHtml(message)}</div>
      </div>`,
      {
        maxWidth: 340,
        closeButton: true,
        className: "region-summary-popup",
        autoPan: true,
      }
    )
    .openPopup(typeof layer.getBounds === "function" ? layer.getBounds().getCenter() : layer.getLatLng());
}

async function handleRegionDraw(layer, layerType) {
  if (!state.sessionId) return;
  state.regionClickBlockUntil = Date.now() + 1200;
  clearLayer(state.pickedLayer);
  state.pickedLayer = null;
  if (state.drawLayer) {
    state.drawLayer.clearLayers();
    state.drawLayer.addLayer(layer);
  }
  $("region-status").textContent = "Scoring selected region...";
  setMapHint("Scoring grid points inside the selected shape with the supervisor model...", "loading");
  try {
    const result = await fetchJson(`/api/research/${state.sessionId}/region?${new URLSearchParams({ target: state.target }).toString()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(regionPayload(layer, layerType)),
    });
    if (result.sampleCount === 0) {
      $("region-status").textContent = result.message;
      $("region-detail").innerHTML = "";
      return;
    }
    $("region-status").textContent = `Region aggregated from ${result.region.validGridCount.toLocaleString()} of ${result.region.candidateGridCount.toLocaleString()} candidate grid points.`;
    const tierText = (result.region.tierDistribution || [])
      .filter((item) => item.count)
      .map((item) => `${item.tier} ${item.pct.toFixed(0)}%`)
      .join(", ");
    $("region-detail").innerHTML = [
      `<div class="detail-row"><span>Target</span><strong>${result.targetLabel} (${result.target})</strong></div>`,
      `<div class="detail-row"><span>Valid grid points</span><strong>${result.region.validGridCount.toLocaleString()} / ${result.region.candidateGridCount.toLocaleString()}</strong></div>`,
      `<div class="detail-row"><span>Coverage</span><strong>${result.region.coveragePct.toFixed(0)}%</strong></div>`,
      `<div class="detail-row"><span>Regional median</span><strong>${result.prospectivity.scorePct.toFixed(1)}% | ${result.prospectivity.level}</strong></div>`,
      `<div class="detail-row"><span>Tier distribution</span><strong>${escapeHtml(tierText)}</strong></div>`,
      `<div class="detail-note"><span>Interpretation</span><p>${escapeHtml(result.explanation?.summary || result.interpretation)}</p></div>`,
    ].join("");
    showRegionPopup(layer, result);
    updateResult(result);
    setMapHint(`Region aggregated from ${result.region.validGridCount.toLocaleString()} valid supervisor-model grid points.`, "success");
  } catch (error) {
    console.error(error);
    const message = error.message || String(error);
    $("region-status").textContent = `Region scoring failed: ${message}`;
    $("region-detail").innerHTML = "";
    showRegionError(layer, message);
    setMapHint(message, "error");
  }
}

async function searchPlaces() {
  const query = $("place-search").value.trim();
  clearTimeout(state.placeTimer);
  if (query.length < 2) {
    $("place-suggestions").innerHTML = "";
    return;
  }
  state.placeTimer = setTimeout(async () => {
    try {
      const results = await fetchJson(`/api/places/search?${new URLSearchParams({ q: query }).toString()}`);
      $("place-suggestions").innerHTML = results
        .map((place, index) => `
          <button type="button" class="place-option" data-index="${index}">
            <strong>${escapeHtml(place.name.split(",")[0])}</strong>
            <span>${escapeHtml(place.name)}</span>
          </button>
        `)
        .join("");
      document.querySelectorAll(".place-option").forEach((button) => {
        button.addEventListener("click", () => choosePlace(results[Number(button.dataset.index)]));
      });
    } catch (error) {
      console.error(error);
      $("place-suggestions").innerHTML = "";
    }
  }, 240);
}

function choosePlace(place) {
  $("place-search").value = place.name.split(",")[0];
  $("place-suggestions").innerHTML = "";
  const latlng = { lat: place.lat, lng: place.lon };
  if (Array.isArray(place.bbox) && place.bbox.length === 4) {
    const bounds = [[place.bbox[0], place.bbox[2]], [place.bbox[1], place.bbox[3]]];
    state.map.fitBounds(bounds, { padding: [38, 38] });
  } else {
    state.map.setView([place.lat, place.lon], 13);
  }
  analyzePoint(latlng);
}

function hydrate(payload) {
  state.sessionKind = "research";
  state.target = payload.target || "Cu";
  state.targetLabel = payload.targetLabel || "Copper";
  state.targetOptions = payload.targetOptions || state.targetOptions || [];
  state.samplePreview = payload.samplePreview;
  state.sitePoints = payload.sitePoints;
  state.coverageRegions = payload.coverageRegions || [];
  state.availableElements = payload.availableElements;
  state.knowledgeEntry = payload.knowledgeEntry;
  state.evaluation = payload.evaluation;
  state.referenceExample = payload.referenceExample;
  state.manualTemplate = payload.manualTemplate;
  state.modelMetadata = payload.modelMetadata || null;
  state.dataProfile = payload.modelMetadata?.dataProfile || null;
  state.layerInventory = payload.layerInventory || null;
  renderTargetSelector();
  updateSessionSummary(payload);
  updateModelPanel(payload);
  renderMapLayers();
}

async function init() {
  ensureMap();
  initGauge();
  const query = new URLSearchParams(window.location.search);
  const researchId = query.get("research_id");
  state.sessionId = researchId;
  state.sessionKind = "research";
  if (!state.sessionId) {
    $("welcome-copy").textContent = "No supervisor-model session was provided. Return to the target selector.";
    return;
  }
  try {
    const cacheKey = `geochemad:research:${state.sessionId}`;
    const cached = sessionStorage.getItem(cacheKey);
    const cachedPayload = cached ? JSON.parse(cached) : null;
    const payload = cachedPayload?.targetOptions && cachedPayload?.layerInventory
      ? cachedPayload
      : await fetchJson(`/api/research/${state.sessionId}`);
    hydrate(payload);
  } catch (error) {
    console.error(error);
    $("welcome-copy").textContent = "This analysis session could not be loaded. It may have expired after a server restart.";
  }
  $("target-select").addEventListener("change", switchTarget);
  $("place-search").addEventListener("input", searchPlaces);
  $("coordinate-score").addEventListener("click", coordinateScore);
  $("coordinate-form").addEventListener("submit", (event) => {
    event.preventDefault();
    coordinateScore();
  });
}

window.addEventListener("DOMContentLoaded", init);
