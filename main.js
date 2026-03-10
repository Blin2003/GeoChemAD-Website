const map = L.map('map').setView([-31.95, 115.86], 5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors',
}).addTo(map);

let currentLayer = null;

const els = {
  fileInput: document.getElementById('csvFile'),
  statusText: document.getElementById('statusText'),
  typeText: document.getElementById('typeText'),
  fileText: document.getElementById('fileText'),
  rowsText: document.getElementById('rowsText'),
  pointsText: document.getElementById('pointsText'),
  previewBox: document.getElementById('previewBox'),
  previewType: document.getElementById('previewType'),
};

const preferredFields = {
  sample: ['SAMPLEID', 'SAMPLETYPE', 'COMPSAMPID', 'Au_ppm', 'Cu_ppm', 'Fe_ppm', 'Zn_ppm', 'As_ppm'],
  site: ['SITE_CODE', 'SITE_TITLE', 'SHORT_NAME', 'SITE_TYPE_', 'SITE_SUB_T', 'COMMODITY_', 'Au_SITES'],
  common: ['X', 'Y'],
};

els.fileInput.addEventListener('change', handleFileUpload);

function handleFileUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  resetState();
  els.fileText.textContent = file.name;
  els.statusText.textContent = 'Parsing CSV...';

  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,
    complete: (results) => {
      if (results.errors?.length) {
        console.warn('Papa Parse warnings:', results.errors);
      }
      processRows(results.data, file.name);
    },
    error: (error) => {
      els.statusText.textContent = 'Failed to parse CSV';
      els.previewBox.textContent = error.message;
    },
  });
}

function processRows(rows, fileName) {
  els.rowsText.textContent = String(rows.length);

  if (!rows.length) {
    els.statusText.textContent = 'No rows found';
    els.previewBox.textContent = 'The file did not contain any readable rows.';
    return;
  }

  const fileType = detectFileType(rows[0]);
  els.typeText.textContent = fileType;
  els.previewType.textContent = fileType;

  const validPoints = rows
    .map((row) => normalizeRow(row, fileType))
    .filter(Boolean);

  els.pointsText.textContent = String(validPoints.length);

  if (!validPoints.length) {
    els.statusText.textContent = 'No valid coordinates found';
    els.previewBox.textContent = 'Rows were read, but no valid X/Y coordinates were found.';
    return;
  }

  els.statusText.textContent = `Loaded ${validPoints.length} points from ${fileName}`;
  renderPreview(validPoints[0]);
  renderPoints(validPoints, fileType);
}

function detectFileType(firstRow) {
  const keys = Object.keys(firstRow || {});
  if (keys.includes('SITE_CODE') || keys.includes('SITE_TITLE')) return 'site';
  if (keys.includes('SAMPLEID') || keys.includes('SAMPLETYPE')) return 'sample';
  return 'unknown';
}

function normalizeRow(row, fileType) {
  const lng = parseCoordinate(row.X);
  const lat = parseCoordinate(row.Y);

  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) return null;

  return {
    type: fileType,
    lat,
    lng,
    raw: row,
    popupFields: buildPopupFields(row, fileType),
  };
}

function parseCoordinate(value) {
  if (value === null || value === undefined || value === '') return NaN;
  const num = Number(value);
  return Number.isFinite(num) ? num : NaN;
}

function buildPopupFields(row, fileType) {
  const base = [...preferredFields.common];
  const fields = fileType === 'site'
    ? [...preferredFields.site, ...base]
    : fileType === 'sample'
      ? [...preferredFields.sample, ...base]
      : Object.keys(row).slice(0, 10);

  const seen = new Set();
  const output = [];

  for (const key of fields) {
    if (seen.has(key) || !(key in row)) continue;
    seen.add(key);
    const value = row[key];
    if (shouldSkipValue(value)) continue;
    output.push([key, value]);
  }

  return output;
}

function shouldSkipValue(value) {
  if (value === null || value === undefined) return true;
  const text = String(value).trim();
  return text === '' || text === '-9999' || text === '-9999.0' || text === '-9999.000000000000000';
}

function renderPreview(point) {
  const lines = point.popupFields.map(([k, v]) => `${k}: ${v}`);
  els.previewBox.textContent = lines.length ? lines.join('\n') : 'No preview fields available.';
}

function renderPoints(points, fileType) {
  if (currentLayer) {
    currentLayer.remove();
  }

  const color = fileType === 'site' ? '#f59e0b' : '#38bdf8';

  currentLayer = L.layerGroup(
    points.map((point) => {
      const marker = L.circleMarker([point.lat, point.lng], {
        radius: fileType === 'site' ? 7 : 5,
        color,
        weight: 1,
        fillColor: color,
        fillOpacity: 0.8,
      });
      marker.bindPopup(buildPopupHtml(point));
      return marker;
    })
  ).addTo(map);

  const bounds = L.latLngBounds(points.map((p) => [p.lat, p.lng]));
  map.fitBounds(bounds.pad(0.12));
}

function buildPopupHtml(point) {
  const title = point.type === 'site' ? 'Site point' : point.type === 'sample' ? 'Sample point' : 'Data point';
  const rows = point.popupFields
    .map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(String(value))}</td></tr>`)
    .join('');

  return `
    <div>
      <strong>${title}</strong>
      <table class="popup-table">${rows}</table>
    </div>
  `;
}

function escapeHtml(text) {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function resetState() {
  els.statusText.textContent = 'Reading file...';
  els.typeText.textContent = '—';
  els.rowsText.textContent = '0';
  els.pointsText.textContent = '0';
  els.previewType.textContent = 'No data';
  els.previewBox.textContent = 'Upload a CSV to preview the first valid row.';

  if (currentLayer) {
    currentLayer.remove();
    currentLayer = null;
  }
}
