const map = L.map('map', { preferCanvas: true }).setView([-31.95, 115.86], 5);
const canvasRenderer = L.canvas({ padding: 0.5 });

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors',
}).addTo(map);

const state = {
  sampleRows: [],
  siteRows: [],
  samplePoints: [],
  sitePoints: [],
  analyteFields: [],
  currentStats: null,
  sampleLayer: null,
  siteLayer: null,
};

const els = {
  sampleFile: document.getElementById('sampleFile'),
  siteFile: document.getElementById('siteFile'),
  clearBtn: document.getElementById('clearBtn'),
  fitBtn: document.getElementById('fitBtn'),
  applyBtn: document.getElementById('applyBtn'),
  analysisMode: document.getElementById('analysisMode'),
  scoreMethod: document.getElementById('scoreMethod'),
  elementSelect: document.getElementById('elementSelect'),
  multiElementSelect: document.getElementById('multiElementSelect'),
  logTransform: document.getElementById('logTransform'),
  clipNegativeScores: document.getElementById('clipNegativeScores'),
  invalidThreshold: document.getElementById('invalidThreshold'),
  anomalyThreshold: document.getElementById('anomalyThreshold'),
  showOnlyAnomalies: document.getElementById('showOnlyAnomalies'),
  statusText: document.getElementById('statusText'),
  rowsText: document.getElementById('rowsText'),
  pointsText: document.getElementById('pointsText'),
  sitePointsText: document.getElementById('sitePointsText'),
  elementCountText: document.getElementById('elementCountText'),
  currentElementText: document.getElementById('currentElementText'),
  thresholdText: document.getElementById('thresholdText'),
  anomalyCountText: document.getElementById('anomalyCountText'),
  minScoreText: document.getElementById('minScoreText'),
  medianScoreText: document.getElementById('medianScoreText'),
  maxScoreText: document.getElementById('maxScoreText'),
  nearestSiteText: document.getElementById('nearestSiteText'),
  previewBox: document.getElementById('previewBox'),
  previewType: document.getElementById('previewType'),
};

const samplePopupFields = ['SAMPLEID', 'SAMPLETYPE', 'COMPSAMPID', 'WAMEX_A_NO'];
const sitePopupFields = ['SITE_CODE', 'SITE_TITLE', 'SHORT_NAME', 'SITE_COMMO', 'SITE_TYPE_', 'SITE_STAGE', 'TARGET_COM'];

els.sampleFile.addEventListener('change', (event) => handleCsvUpload(event, 'sample'));
els.siteFile.addEventListener('change', (event) => handleCsvUpload(event, 'site'));
els.applyBtn.addEventListener('click', applyAnalysis);
els.clearBtn.addEventListener('click', clearAll);
els.fitBtn.addEventListener('click', fitToLoadedData);
els.analysisMode.addEventListener('change', syncModeOptions);
els.scoreMethod.addEventListener('change', syncModeOptions);

syncModeOptions();
refreshElementSelectors([]);

function handleCsvUpload(event, fileType) {
  const file = event.target.files?.[0];
  if (!file) return;

  setStatus(`Parsing ${fileType} CSV...`);

  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,
    complete: (results) => {
      if (results.errors?.length) console.warn('Papa Parse warnings:', results.errors);
      if (fileType === 'sample') {
        loadSampleRows(results.data, file.name);
      } else {
        loadSiteRows(results.data, file.name);
      }
    },
    error: (error) => {
      setStatus(`Failed to parse ${fileType} CSV`);
      els.previewBox.textContent = error.message;
    },
  });
}

function loadSampleRows(rows, fileName) {
  state.sampleRows = rows;
  state.samplePoints = rows.map(normalizeSampleRow).filter(Boolean);
  state.analyteFields = detectAnalyteFields(rows[0] || {});

  els.rowsText.textContent = String(rows.length);
  els.pointsText.textContent = String(state.samplePoints.length);
  els.elementCountText.textContent = String(state.analyteFields.length);

  refreshElementSelectors(state.analyteFields, guessDefaultElement(fileName, state.analyteFields));
  setStatus(`Loaded ${state.samplePoints.length} valid sample points from ${fileName}`);

  if (!state.samplePoints.length) {
    els.previewType.textContent = 'No valid sample points';
    els.previewBox.textContent = 'No valid sample rows with numeric X and Y were found.';
    clearLayer('sample');
    return;
  }

  renderPreview(state.samplePoints[0]);
  applyAnalysis();
}

function loadSiteRows(rows, fileName) {
  state.siteRows = rows;
  state.sitePoints = rows.map(normalizeSiteRow).filter(Boolean);
  els.sitePointsText.textContent = String(state.sitePoints.length);
  setStatus(`Loaded ${state.sitePoints.length} site points from ${fileName}`);
  renderSites();
  updateNearestSiteMetrics();
  fitToLoadedData();
}

function normalizeSampleRow(row) {
  const lng = parseCoordinate(row.X);
  const lat = parseCoordinate(row.Y);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) return null;
  return { type: 'sample', lat, lng, raw: row };
}

function normalizeSiteRow(row) {
  const lng = parseCoordinate(row.X);
  const lat = parseCoordinate(row.Y);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) return null;
  return { type: 'site', lat, lng, raw: row };
}

function parseCoordinate(value) {
  if (value === null || value === undefined || value === '') return NaN;
  const num = Number(value);
  return Number.isFinite(num) ? num : NaN;
}

function detectAnalyteFields(firstRow) {
  return Object.keys(firstRow)
    .filter((key) => key !== 'X' && key !== 'Y')
    .filter((key) => /(_ppm|_pct|_ppb|_ppt|_mgkg)$/i.test(key));
}

function guessDefaultElement(fileName, fields) {
  const lower = String(fileName || '').toLowerCase();
  const priorities = ['au', 'cu', 'w', 'li', 'ni', 'fe', 'zn', 'as'];
  for (const token of priorities) {
    const match = fields.find((field) => field.toLowerCase().startsWith(`${token}_`));
    if (lower.includes(`_${token}`) && match) return match;
  }
  return fields[0] || '';
}

function refreshElementSelectors(fields, selectedValue = '') {
  const options = fields.map((field) => `<option value="${escapeHtml(field)}">${escapeHtml(field)}</option>`).join('');
  els.elementSelect.innerHTML = options || '<option value="">No analyte fields found</option>';
  els.multiElementSelect.innerHTML = options;

  if (selectedValue && fields.includes(selectedValue)) {
    els.elementSelect.value = selectedValue;
  }

  const defaults = selectedValue ? [selectedValue] : fields.slice(0, Math.min(3, fields.length));
  for (const option of els.multiElementSelect.options) {
    option.selected = defaults.includes(option.value);
  }
}

function syncModeOptions() {
  const mode = els.analysisMode.value;
  const score = els.scoreMethod.value;
  const compositeNeeded = mode === 'composite' || score === 'compositeMean' || score === 'compositeMax';

  els.multiElementSelect.disabled = !compositeNeeded;
  if (compositeNeeded) {
    els.scoreMethod.value = score === 'zscore' || score === 'robustZ' || score === 'percentile' ? 'compositeMean' : score;
  }
}

function applyAnalysis() {
  if (!state.samplePoints.length) {
    setStatus('Upload a sample CSV first');
    return;
  }

  const config = getConfig();
  const selectedElements = config.analysisMode === 'composite'
    ? getSelectedMultiElements()
    : [config.primaryElement].filter(Boolean);

  if (!selectedElements.length) {
    setStatus('No analyte element selected');
    return;
  }

  const scoreMethod = resolveScoreMethod(config.analysisMode, config.scoreMethod);
  const scoredPoints = scorePoints(state.samplePoints, selectedElements, config, scoreMethod);

  state.currentStats = buildScoreSummary(scoredPoints, config.anomalyThreshold);
  updateSummaryUi(selectedElements, scoreMethod, state.currentStats, config);
  renderSamples(scoredPoints, config);
  renderSites();
  renderPreview(scoredPoints[0] || state.samplePoints[0]);
  updateNearestSiteMetrics(scoredPoints, config.anomalyThreshold);
  fitToLoadedData();

  const anomalyCount = state.currentStats ? state.currentStats.aboveThresholdCount : 0;
  setStatus(`Analysis complete: ${anomalyCount} anomaly points above threshold`);
}

function getConfig() {
  return {
    analysisMode: els.analysisMode.value,
    scoreMethod: els.scoreMethod.value,
    primaryElement: els.elementSelect.value,
    logTransform: els.logTransform.checked,
    clipNegativeScores: els.clipNegativeScores.checked,
    invalidThreshold: Number(els.invalidThreshold.value),
    anomalyThreshold: Number(els.anomalyThreshold.value),
    showOnlyAnomalies: els.showOnlyAnomalies.checked,
  };
}

function getSelectedMultiElements() {
  return Array.from(els.multiElementSelect.selectedOptions).map((opt) => opt.value);
}

function resolveScoreMethod(mode, scoreMethod) {
  if (mode === 'composite') {
    return scoreMethod === 'compositeMax' ? 'compositeMax' : 'compositeMean';
  }
  return scoreMethod;
}

function scorePoints(points, elements, config, scoreMethod) {
  const processedByElement = new Map();
  for (const element of elements) {
    const values = points.map((point) => preprocessValue(point.raw[element], points, element, config));
    processedByElement.set(element, values);
  }

  const elementStats = new Map();
  for (const [element, values] of processedByElement.entries()) {
    elementStats.set(element, computeStats(values));
  }

  return points.map((point, index) => {
    const elementDetails = elements.map((element) => {
      const processed = processedByElement.get(element)[index];
      const stats = elementStats.get(element);
      const rawValue = numericOrNull(point.raw[element]);
      const score = scoreSingleValue(processed, stats, scoreMethod === 'compositeMax' || scoreMethod === 'compositeMean' ? 'robustZ' : scoreMethod, config.clipNegativeScores);
      return { element, rawValue, processedValue: processed, score };
    });

    const aggregateScore = aggregateElementScores(elementDetails.map((item) => item.score), scoreMethod, config.clipNegativeScores);
    const nearestSiteKm = state.sitePoints.length ? computeNearestSiteKm(point, state.sitePoints) : null;

    return {
      ...point,
      score: aggregateScore,
      isAnomaly: Number.isFinite(aggregateScore) && aggregateScore >= config.anomalyThreshold,
      analyzedElements: elements,
      elementDetails,
      nearestSiteKm,
    };
  });
}

function preprocessValue(rawValue, points, element, config) {
  const num = numericOrNull(rawValue);
  if (!Number.isFinite(num)) return null;
  if (num <= config.invalidThreshold) return null;

  let cleaned = num;
  if (cleaned <= 0) {
    const substitute = estimatePositiveSubstitute(points, element, config.invalidThreshold);
    if (!Number.isFinite(substitute)) return null;
    cleaned = substitute;
  }

  if (config.logTransform) {
    return Math.log1p(cleaned);
  }
  return cleaned;
}

function estimatePositiveSubstitute(points, element, invalidThreshold) {
  const positives = [];
  for (const point of points) {
    const value = numericOrNull(point.raw[element]);
    if (Number.isFinite(value) && value > 0 && value > invalidThreshold) positives.push(value);
  }
  if (!positives.length) return null;
  positives.sort((a, b) => a - b);
  return positives[0] / 2;
}

function numericOrNull(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function computeStats(values) {
  const clean = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!clean.length) {
    return { clean, mean: null, std: null, median: null, mad: null };
  }
  const mean = clean.reduce((sum, value) => sum + value, 0) / clean.length;
  const variance = clean.reduce((sum, value) => sum + (value - mean) ** 2, 0) / clean.length;
  const std = Math.sqrt(variance) || 1e-9;
  const median = percentileFromSorted(clean, 0.5);
  const absDeviations = clean.map((value) => Math.abs(value - median)).sort((a, b) => a - b);
  const mad = percentileFromSorted(absDeviations, 0.5) || 1e-9;
  return { clean, mean, std, median, mad };
}

function scoreSingleValue(value, stats, method, clipNegative) {
  if (!Number.isFinite(value) || !stats.clean.length) return null;

  let score = null;
  if (method === 'zscore') {
    score = (value - stats.mean) / stats.std;
  } else if (method === 'robustZ') {
    score = (value - stats.median) / (1.4826 * stats.mad);
  } else if (method === 'percentile') {
    score = percentileRank(stats.clean, value) * 100;
  }

  if (!Number.isFinite(score)) return null;
  return clipNegative && method !== 'percentile' ? Math.max(0, score) : score;
}

function aggregateElementScores(scores, method, clipNegative) {
  const clean = scores.filter((value) => Number.isFinite(value));
  if (!clean.length) return null;
  if (method === 'compositeMax') return Math.max(...clean);
  if (method === 'compositeMean') {
    const mean = clean.reduce((sum, value) => sum + value, 0) / clean.length;
    return clipNegative ? Math.max(0, mean) : mean;
  }
  return clean[0];
}

function buildScoreSummary(scoredPoints, threshold) {
  const scores = scoredPoints.map((point) => point.score).filter((score) => Number.isFinite(score)).sort((a, b) => a - b);
  if (!scores.length) {
    return { min: null, median: null, max: null, aboveThresholdCount: 0 };
  }
  return {
    min: scores[0],
    median: percentileFromSorted(scores, 0.5),
    max: scores[scores.length - 1],
    aboveThresholdCount: scores.filter((score) => score >= threshold).length,
  };
}

function updateSummaryUi(elements, scoreMethod, summary, config) {
  els.currentElementText.textContent = elements.join(', ');
  els.thresholdText.textContent = `${config.anomalyThreshold} (${scoreMethod})`;
  els.anomalyCountText.textContent = String(summary?.aboveThresholdCount ?? 0);
  els.minScoreText.textContent = formatNumber(summary?.min);
  els.medianScoreText.textContent = formatNumber(summary?.median);
  els.maxScoreText.textContent = formatNumber(summary?.max);
}

function renderSamples(scoredPoints, config) {
  clearLayer('sample');
  const visiblePoints = config.showOnlyAnomalies
    ? scoredPoints.filter((point) => point.isAnomaly)
    : scoredPoints;

  const markers = visiblePoints.map((point) => {
    const color = colorForScore(point.score, state.currentStats);
    const radius = point.isAnomaly ? 6 : 4;
    return L.circleMarker([point.lat, point.lng], {
      renderer: canvasRenderer,
      radius,
      color: point.isAnomaly ? '#ffffff' : color,
      weight: point.isAnomaly ? 1.8 : 0.6,
      fillColor: color,
      fillOpacity: point.isAnomaly ? 0.95 : 0.8,
    }).bindPopup(buildSamplePopupHtml(point));
  });

  state.sampleLayer = L.layerGroup(markers).addTo(map);
}

function renderSites() {
  clearLayer('site');
  if (!state.sitePoints.length) return;

  const markers = state.sitePoints.map((point) =>
    L.circleMarker([point.lat, point.lng], {
      renderer: canvasRenderer,
      radius: 7,
      color: '#7c2d12',
      weight: 1.4,
      fillColor: '#f59e0b',
      fillOpacity: 0.95,
    }).bindPopup(buildSitePopupHtml(point))
  );

  state.siteLayer = L.layerGroup(markers).addTo(map);
}

function clearLayer(type) {
  const key = type === 'sample' ? 'sampleLayer' : 'siteLayer';
  if (state[key]) {
    state[key].remove();
    state[key] = null;
  }
}

function buildSamplePopupHtml(point) {
  const baseRows = samplePopupFields
    .map((field) => [field, point.raw[field]])
    .filter(([, value]) => value !== undefined && value !== null && value !== '');

  const scoreRows = [
    ['Anomaly score', formatNumber(point.score)],
    ['Above threshold', point.isAnomaly ? 'Yes' : 'No'],
    ['Nearest site (km)', formatNumber(point.nearestSiteKm)],
  ];

  const analyteRows = point.elementDetails.flatMap((item) => ([
    [`${item.element} raw`, formatNumber(item.rawValue)],
    [`${item.element} processed`, formatNumber(item.processedValue)],
    [`${item.element} score`, formatNumber(item.score)],
  ]));

  return buildPopupHtml('Sample point', [...baseRows, ...scoreRows, ...analyteRows, ['Longitude', point.lng], ['Latitude', point.lat]]);
}

function buildSitePopupHtml(point) {
  const rows = sitePopupFields
    .map((field) => [field, point.raw[field]])
    .filter(([, value]) => value !== undefined && value !== null && value !== '');

  rows.push(['Longitude', point.lng], ['Latitude', point.lat]);
  return buildPopupHtml('Known site', rows);
}

function buildPopupHtml(title, rows) {
  const htmlRows = rows
    .map(([key, value]) => `<tr><td>${escapeHtml(String(key))}</td><td>${escapeHtml(String(value))}</td></tr>`)
    .join('');

  return `
    <div>
      <div class="popup-title"><strong>${escapeHtml(title)}</strong></div>
      <table class="popup-table">${htmlRows}</table>
    </div>
  `;
}

function renderPreview(point) {
  if (!point) {
    els.previewType.textContent = 'No data';
    els.previewBox.textContent = 'Upload a sample CSV to inspect the first valid processed row.';
    return;
  }

  els.previewType.textContent = point.type || 'sample';
  const lines = [
    `X: ${point.lng}`,
    `Y: ${point.lat}`,
    ...samplePopupFields.map((field) => `${field}: ${point.raw?.[field] ?? '—'}`),
  ];

  if (point.elementDetails?.length) {
    for (const detail of point.elementDetails) {
      lines.push(`${detail.element} raw: ${formatNumber(detail.rawValue)}`);
      lines.push(`${detail.element} processed: ${formatNumber(detail.processedValue)}`);
      lines.push(`${detail.element} score: ${formatNumber(detail.score)}`);
    }
    lines.push(`Anomaly score: ${formatNumber(point.score)}`);
    lines.push(`Nearest site (km): ${formatNumber(point.nearestSiteKm)}`);
  }

  els.previewBox.textContent = lines.join('\n');
}

function fitToLoadedData() {
  const coords = [
    ...state.samplePoints.map((point) => [point.lat, point.lng]),
    ...state.sitePoints.map((point) => [point.lat, point.lng]),
  ];
  if (!coords.length) return;
  map.fitBounds(L.latLngBounds(coords).pad(0.08));
}

function clearAll() {
  state.sampleRows = [];
  state.siteRows = [];
  state.samplePoints = [];
  state.sitePoints = [];
  state.analyteFields = [];
  state.currentStats = null;

  clearLayer('sample');
  clearLayer('site');

  els.sampleFile.value = '';
  els.siteFile.value = '';
  els.rowsText.textContent = '0';
  els.pointsText.textContent = '0';
  els.sitePointsText.textContent = '0';
  els.elementCountText.textContent = '0';
  els.currentElementText.textContent = '—';
  els.thresholdText.textContent = '—';
  els.anomalyCountText.textContent = '0';
  els.minScoreText.textContent = '—';
  els.medianScoreText.textContent = '—';
  els.maxScoreText.textContent = '—';
  els.nearestSiteText.textContent = '—';
  refreshElementSelectors([]);
  renderPreview(null);
  setStatus('Cleared');
  map.setView([-31.95, 115.86], 5);
}

function updateNearestSiteMetrics(scoredPoints = null, threshold = null) {
  if (!state.sitePoints.length) {
    els.nearestSiteText.textContent = 'No site CSV';
    return;
  }
  const source = (scoredPoints || []).filter((point) => point.isAnomaly) ;
  if (!source.length || threshold === null) {
    els.nearestSiteText.textContent = `${state.sitePoints.length} sites loaded`;
    return;
  }
  const distances = source.map((point) => point.nearestSiteKm).filter((value) => Number.isFinite(value));
  if (!distances.length) {
    els.nearestSiteText.textContent = 'No anomaly-site distances';
    return;
  }
  const average = distances.reduce((sum, value) => sum + value, 0) / distances.length;
  els.nearestSiteText.textContent = `${formatNumber(average)} km avg`;
}

function computeNearestSiteKm(point, sites) {
  let min = Infinity;
  for (const site of sites) {
    const distance = haversineKm(point.lat, point.lng, site.lat, site.lng);
    if (distance < min) min = distance;
  }
  return Number.isFinite(min) ? min : null;
}

function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function toRad(deg) {
  return deg * (Math.PI / 180);
}

function percentileFromSorted(sorted, q) {
  if (!sorted.length) return null;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] !== undefined
    ? sorted[base] + rest * (sorted[base + 1] - sorted[base])
    : sorted[base];
}

function percentileRank(sorted, value) {
  if (!sorted.length) return null;
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (sorted[mid] <= value) low = mid + 1;
    else high = mid;
  }
  return low / sorted.length;
}

function colorForScore(score, summary) {
  if (!Number.isFinite(score) || !summary || !Number.isFinite(summary.max)) return '#64748b';
  const min = Number.isFinite(summary.min) ? summary.min : 0;
  const max = summary.max === min ? min + 1 : summary.max;
  const t = Math.max(0, Math.min(1, (score - min) / (max - min)));
  if (t < 0.33) return '#38bdf8';
  if (t < 0.66) return '#22c55e';
  return '#ef4444';
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return '—';
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(3);
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
