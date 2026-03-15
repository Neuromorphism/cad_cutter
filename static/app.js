import * as THREE from '/static/vendor/three.module.js';
import { OrbitControls } from '/static/vendor/OrbitControls.js';
import { STLLoader } from '/static/vendor/STLLoader.js';

/* ── DOM refs ── */
const fileList       = document.getElementById('file-list');
const statusEl       = document.getElementById('status');
const statusDot      = document.getElementById('status-dot');
const gradientInput  = document.getElementById('gradient-input');
const gradientMode   = document.getElementById('gradient-mode');
const gradientOutput = document.getElementById('gradient-output');
const gradientRender = document.getElementById('gradient-render');
const mainCanvas     = document.getElementById('main-canvas');
const thumbs         = document.getElementById('thumbnails');
const workflowSelect = document.getElementById('workflow-select');
const fineOrientToggle = document.getElementById('fine-orient-toggle');
const axisSelect     = document.getElementById('axis-select');
const gapInput       = document.getElementById('gap-input');
const sectionInput   = document.getElementById('section-input');
const partsDirLabel  = document.getElementById('parts-dir-label');
const browsePartInput = document.getElementById('browse-part-input');
const partCountEl    = document.getElementById('part-count');
const viewportHint   = document.getElementById('viewport-hint');
const toastContainer = document.getElementById('toast-container');
const combinedBtn    = document.getElementById('combined-view');
const tileBtn        = document.getElementById('tile-view');
const dirOverlay     = document.getElementById('dir-overlay');
const dirList        = document.getElementById('dir-list');
const dirCurrent     = document.getElementById('dir-current');
const dirSelectBtn   = document.getElementById('dir-select');
const dirUpBtn       = document.getElementById('dir-up');
const dirRootBtn     = document.getElementById('dir-root');
const dirCloseBtn    = document.getElementById('dir-close');
const dirCancelBtn   = document.getElementById('dir-cancel');
const tooltipEl      = document.getElementById('ui-tooltip');
const debugToggle    = document.getElementById('debug-toggle');
const debugCopyBtn   = document.getElementById('debug-copy');
const debugLogEl     = document.getElementById('debug-log');
const debugCountEl   = document.getElementById('debug-count');
const midlayerConfigForms = document.getElementById('midlayer-config-forms');
const midlayerSolverDataEl = document.getElementById('midlayer-solver-data');

const progressContainer = document.getElementById('progress-container');
const progressFill      = document.getElementById('progress-fill');
const progressText      = document.getElementById('progress-text');

let sceneData = null;
let tileView  = false;
let mainCtx   = null;
let progressSSE = null;
let mainAnimation = null;
let dirBrowseState = { root: '.', current: '.', parent: null, directories: [], selected: '.' };
let refreshFilesRequestId = 0;
let progressHistory = [];
let progressLogTimer = null;
let progressLogCollapsed = false;
let debugEntries = [];
let debugPanelOpen = false;
let activeLoaderContext = null;
let stallWatchdog = null;
let lastProgressEventAt = 0;
let stallThresholdMs = 5000;
let pendingDebugFlush = [];
let debugFlushTimer = null;
let debugFlushInFlight = false;
let geometryCache = new WeakMap();
let remoteGeometryCache = new Map();
let thumbRenderGeneration = 0;
const MAX_WEBGL_THUMBNAILS = 8;
const stlLoader = new STLLoader();
const STAGE_TIMEOUT_MS = {
  auto_orient: 15000,
  auto_stack: 15000,
  auto_scale: 30000,
  auto_drop: 30000,
  design_midlayer_dl4to: 60000,
  design_midlayer_pymoto: 60000,
  cut_inner_from_mid: 30000,
  export_parts: 30000,
  render_whole: 45000,
  export_whole: 30000,
};

const MATERIAL_OPTIONS = [
  '', 'steel', 'aluminum', 'copper', 'brass', 'bronze', 'gold', 'titanium',
  'chrome', 'plastic', 'rubber', 'ceramic', 'glass', 'wood', 'oak', 'pine',
  'stone', 'concrete', 'red', 'green', 'blue', 'black', 'white'
];

const midlayerSolverData = (() => {
  if (!midlayerSolverDataEl?.textContent) return { solvers: {}, configs: {} };
  try {
    return JSON.parse(midlayerSolverDataEl.textContent);
  } catch (_) {
    return { solvers: {}, configs: {} };
  }
})();

/* ── Toast notifications ── */
function toast(message, type = 'info') {
  addDebugLog(`toast:${type}`, message);
  const icons = { success: '\u2713', error: '\u2717', info: '\u2139', warning: '\u26A0' };
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type] || ''}</span><span>${message}</span>`;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.classList.add('toast-out');
    el.addEventListener('animationend', () => el.remove());
  }, 3500);
}

/* ── Status helpers ── */
function setStatus(msg, busy = false) {
  statusEl.textContent = msg;
  statusDot.className = busy ? 'status-dot busy' : 'status-dot';
}

function setBusy(msg) { setStatus(msg, true); }
function setIdle(msg) { setStatus(msg || 'Ready'); }

function addDebugLog(kind, message, meta = null) {
  const entry = {
    id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
    ts: Date.now(),
    kind,
    message,
    meta,
  };
  debugEntries = [...debugEntries, entry].slice(-80);
  pendingDebugFlush.push(entry);
  renderDebugLog();
  scheduleDebugFlush();
  const metaText = meta ? ` ${JSON.stringify(meta)}` : '';
  console.info(`[cad-debug] ${kind}: ${message}${metaText}`);
}

function renderDebugLog() {
  if (!debugLogEl) return;
  debugLogEl.innerHTML = '';
  [...debugEntries].sort((a, b) => b.ts - a.ts).forEach((entry) => {
    const row = document.createElement('div');
    row.className = 'debug-entry';
    row.innerHTML = `
      <span class="debug-time">${new Date(entry.ts).toLocaleTimeString()}</span>
      <span class="debug-kind">${entry.kind}</span>
      <span class="debug-message">${entry.message}</span>`;
    if (entry.meta) {
      const meta = document.createElement('pre');
      meta.className = 'debug-meta';
      meta.textContent = JSON.stringify(entry.meta);
      row.appendChild(meta);
    }
    debugLogEl.appendChild(row);
  });
  if (debugCountEl) debugCountEl.textContent = String(debugEntries.length);
}

function formatDebugLogText() {
  return [...debugEntries]
    .sort((a, b) => a.ts - b.ts)
    .map((entry) => {
      const time = new Date(entry.ts).toISOString();
      const meta = entry.meta ? ` ${JSON.stringify(entry.meta)}` : '';
      return `${time} [${entry.kind}] ${entry.message}${meta}`;
    })
    .join('\n');
}

function scheduleDebugFlush() {
  if (debugFlushTimer) return;
  debugFlushTimer = window.setTimeout(() => {
    debugFlushTimer = null;
    flushDebugLog();
  }, 400);
}

async function flushDebugLog(force = false) {
  if (debugFlushInFlight) return;
  if (!pendingDebugFlush.length) return;
  if (!force && document.visibilityState === 'hidden' && pendingDebugFlush.length < 5) return;
  debugFlushInFlight = true;
  const batch = pendingDebugFlush.splice(0, 25);
  try {
    await fetch('/api/debug-log', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ entries: batch }),
      keepalive: true,
    });
  } catch (_) {
    pendingDebugFlush = [...batch, ...pendingDebugFlush].slice(-100);
  } finally {
    debugFlushInFlight = false;
    if (pendingDebugFlush.length) scheduleDebugFlush();
  }
}

function updateDebugPanelState() {
  if (!debugToggle || !debugLogEl) return;
  debugToggle.setAttribute('aria-expanded', debugPanelOpen ? 'true' : 'false');
  debugToggle.classList.toggle('open', debugPanelOpen);
  debugLogEl.classList.toggle('hidden', !debugPanelOpen);
}

function startStallWatchdog(context, thresholdMs = 5000) {
  stopStallWatchdog();
  activeLoaderContext = context;
  stallThresholdMs = thresholdMs;
  lastProgressEventAt = Date.now();
  stallWatchdog = window.setInterval(() => {
    const loader = document.getElementById('viewport-loader');
    if (!loader || !activeLoaderContext) return;
    const idleMs = Date.now() - lastProgressEventAt;
    if (idleMs >= stallThresholdMs) {
      const msg = document.querySelector('#viewport-loader .loader-msg')?.textContent || 'Loading parts...';
      addDebugLog('stall-warning', `${activeLoaderContext} has no new progress for ${Math.floor(idleMs / 1000)}s`, {
        loaderMessage: msg,
      });
      lastProgressEventAt = Date.now();
    }
  }, 1000);
}

function stopStallWatchdog() {
  if (stallWatchdog) {
    window.clearInterval(stallWatchdog);
    stallWatchdog = null;
  }
  activeLoaderContext = null;
  stallThresholdMs = 5000;
}

function stopMainAnimation() {
  if (mainAnimation) {
    mainAnimation();
    mainAnimation = null;
  }
}

function stopThumbAnimations() {
  // Thumbnails are rendered as static snapshots so they do not keep live WebGL
  // contexts around and starve the main viewport on large assemblies.
}

function showTooltip(target) {
  if (!tooltipEl || !target?.dataset?.tooltip) return;
  tooltipEl.textContent = target.dataset.tooltip;
  tooltipEl.classList.remove('hidden');
  tooltipEl.classList.add('visible');
  positionTooltip(target);
}

function positionTooltip(target) {
  if (!tooltipEl || tooltipEl.classList.contains('hidden')) return;
  const rect = target.getBoundingClientRect();
  const tooltipRect = tooltipEl.getBoundingClientRect();
  const margin = 10;
  let left = rect.left + (rect.width - tooltipRect.width) / 2;
  left = Math.min(window.innerWidth - tooltipRect.width - margin, Math.max(margin, left));
  let top = rect.bottom + 8;
  if (top + tooltipRect.height > window.innerHeight - margin) {
    top = rect.top - tooltipRect.height - 8;
  }
  top = Math.max(margin, top);
  tooltipEl.style.left = `${left}px`;
  tooltipEl.style.top = `${top}px`;
}

function hideTooltip() {
  if (!tooltipEl) return;
  tooltipEl.classList.remove('visible');
  tooltipEl.classList.add('hidden');
}

/* ── API wrapper with error handling ── */
async function api(path, opts = {}) {
  const { timeoutMs = 0, ...fetchOpts } = opts;
  const method = fetchOpts.method || 'GET';
  const startedAt = performance.now();
  addDebugLog('api-start', `${method} ${path}`);
  const controller = timeoutMs > 0 ? new AbortController() : null;
  const timeoutId = controller ? window.setTimeout(() => controller.abort(), timeoutMs) : null;
  let res;
  try {
    res = await fetch(path, {
      headers: { 'content-type': 'application/json' },
      ...fetchOpts,
      signal: controller?.signal,
    });
  } catch (error) {
    if (timeoutId) window.clearTimeout(timeoutId);
    const message = error?.name === 'AbortError'
      ? `${method} ${path} timed out after ${timeoutMs}ms`
      : `${method} ${path} failed: ${error?.message || error}`;
    addDebugLog('api-error', message);
    throw new Error(message);
  }
  if (timeoutId) window.clearTimeout(timeoutId);
  addDebugLog('api-done', `${method} ${path} -> ${res.status}`, {
    elapsedMs: Math.round(performance.now() - startedAt),
  });
  if (!res.ok) {
    const errText = await res.text();
    addDebugLog('api-error', `${method} ${path} failed`, { status: res.status, body: errText.slice(0, 300) });
    throw new Error(errText);
  }
  return res.json();
}

/* ── Loading overlay for viewport ── */
function showViewportLoader(msg = 'Loading...') {
  addDebugLog('loader-show', msg);
  removeViewportLoader();
  const overlay = document.createElement('div');
  overlay.className = 'loading-overlay';
  overlay.id = 'viewport-loader';
  overlay.innerHTML = `
    <div class="loader-shell">
      <div class="spinner"></div>
      <div class="loader-headline">
        <p class="loader-msg">${msg}</p>
        <span class="loader-elapsed" id="viewport-progress-elapsed">0s</span>
      </div>
      <div class="viewport-progress">
        <div class="progress-bar">
          <div class="progress-fill" id="viewport-progress-fill"></div>
        </div>
        <span class="progress-text" id="viewport-progress-text"></span>
      </div>
    </div>
    <div class="progress-log-shell">
      <button type="button" class="progress-log-toggle" id="viewport-progress-toggle" aria-expanded="${progressLogCollapsed ? 'false' : 'true'}">
        <span>Recent activity</span>
        <span class="progress-log-toggle-meta" id="viewport-progress-toggle-meta">0 messages</span>
      </button>
      <div class="progress-log ${progressLogCollapsed ? 'collapsed' : ''}" id="viewport-progress-log" aria-live="polite"></div>
    </div>`;
  mainCanvas.appendChild(overlay);
  const toggle = document.getElementById('viewport-progress-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      progressLogCollapsed = !progressLogCollapsed;
      updateProgressLogCollapsedState();
    });
  }
  startProgressLogTimer();
  updateProgressLogCollapsedState();
  renderProgressHistory();
}

function updateViewportLoader(msg) {
  const overlay = document.getElementById('viewport-loader');
  const msgEl = overlay?.querySelector('.loader-msg');
  if (!overlay || !msgEl) return;
  msgEl.textContent = msg;
  lastProgressEventAt = Date.now();
  addDebugLog('loader-update', msg);
}

function removeViewportLoader() {
  const existing = document.getElementById('viewport-loader');
  if (existing) existing.remove();
  stopProgressLogTimer();
  progressHistory = [];
  stopStallWatchdog();
  addDebugLog('loader-hide', 'Viewport loader removed');
}

function startProgressLogTimer() {
  if (progressLogTimer) return;
  progressLogTimer = window.setInterval(renderProgressHistory, 500);
}

function stopProgressLogTimer() {
  if (progressLogTimer) {
    window.clearInterval(progressLogTimer);
    progressLogTimer = null;
  }
}

function renderProgressHistory() {
  const list = document.getElementById('viewport-progress-log');
  const toggleMeta = document.getElementById('viewport-progress-toggle-meta');
  if (!list) return;
  const now = Date.now();
  const entries = [...progressHistory]
    .sort((a, b) => b.ts - a.ts)
    .slice(0, 8);
  if (toggleMeta) {
    toggleMeta.textContent = `${entries.length} message${entries.length !== 1 ? 's' : ''}`;
  }

  list.innerHTML = '';
  entries.forEach((entry, index) => {
    const ageSec = (now - entry.ts) / 1000;
    const item = document.createElement('div');
    item.className = 'progress-log-item';
    if (index >= 5 || ageSec > 5) item.classList.add('is-fading');
    const opacity = Math.max(0.15, 1 - Math.max(0, ageSec - 2) * 0.14 - Math.max(0, index - 2) * 0.12);
    item.style.opacity = opacity.toFixed(2);
    item.innerHTML = `
      <span class="progress-log-time">${Math.max(0, Math.floor(ageSec))}s</span>
      <span class="progress-log-message">${entry.message}</span>`;
    list.appendChild(item);
  });
}

function updateProgressLogCollapsedState() {
  const list = document.getElementById('viewport-progress-log');
  const toggle = document.getElementById('viewport-progress-toggle');
  if (!list || !toggle) return;
  list.classList.toggle('collapsed', progressLogCollapsed);
  toggle.setAttribute('aria-expanded', progressLogCollapsed ? 'false' : 'true');
  toggle.classList.toggle('collapsed', progressLogCollapsed);
}

/* ── Three.js helpers ── */
function buildRenderer(container, options = {}) {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: !!options.preserveDrawingBuffer,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const existingCanvas = container.querySelector('canvas');
  if (existingCanvas) existingCanvas.remove();
  container.insertBefore(renderer.domElement, container.firstChild);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111821);
  scene.fog = new THREE.FogExp2(0x111821, 0.00008);

  const camera = new THREE.PerspectiveCamera(
    45, container.clientWidth / container.clientHeight, 0.1, 10000
  );
  camera.position.set(160, 140, 160);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.8;

  const ambient = new THREE.AmbientLight(0xf6f8ff, 0.75);
  scene.add(ambient);

  const hemi = new THREE.HemisphereLight(0xe4efff, 0x26313f, 1.2);
  scene.add(hemi);

  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(120, 160, 90);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0xb7cae8, 1.0);
  fill.position.set(-80, 70, -50);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0xa5b9de, 0.7);
  rim.position.set(20, -10, -100);
  scene.add(rim);

  const grid = new THREE.GridHelper(600, 40, 0x1a2030, 0x141a22);
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  scene.add(grid);

  return { renderer, scene, camera, controls, grid };
}

function disposeRenderer(ctx) {
  if (!ctx) return;
  try {
    ctx.controls?.dispose?.();
  } catch (_) {
    // noop
  }
  try {
    ctx.renderer?.dispose?.();
  } catch (_) {
    // noop
  }
  try {
    ctx.renderer?.forceContextLoss?.();
  } catch (_) {
    // noop
  }
  try {
    ctx.renderer?.domElement?.remove?.();
  } catch (_) {
    // noop
  }
}

function geometryFromPayload(payload) {
  if (!payload || !Array.isArray(payload.vertices) || !Array.isArray(payload.indices)) {
    const summary = payload && typeof payload === 'object' ? Object.keys(payload) : payload;
    throw new Error(`Invalid mesh payload: ${JSON.stringify(summary)}`);
  }
  const cached = geometryCache.get(payload);
  if (cached) return cached;

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(payload.vertices), 3));
  const vertexCount = payload.vertices.length / 3;
  const IndexArray = vertexCount < 65536 ? Uint16Array : Uint32Array;
  geo.setIndex(new THREE.BufferAttribute(new IndexArray(payload.indices), 1));
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  geo.computeVertexNormals();
  geometryCache.set(payload, geo);
  return geo;
}

function meshFromPayload(payload, color = [0.6, 0.7, 0.8]) {
  const geo = geometryFromPayload(payload);
  const mat = materialForColor(color);
  return new THREE.Mesh(geo, mat);
}

function materialForColor(color = [0.6, 0.7, 0.8]) {
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color(...color),
    metalness: 0.05,
    roughness: 0.75,
    side: THREE.DoubleSide,
  });
}

function buildVisualFromGeometry(geometry, color = [0.6, 0.7, 0.8]) {
  const mesh = new THREE.Mesh(geometry, materialForColor(color));
  mesh.frustumCulled = false;
  return mesh;
}

function applyPartTransforms(mesh, part, offset = null) {
  if (!mesh || !part) return mesh;
  mesh.rotation.set(0, 0, 0);
  mesh.quaternion.identity();
  mesh.scale.setScalar(1);
  mesh.position.set(0, 0, 0);

  for (const step of part.orientationSteps || []) {
    const axis = Array.isArray(step?.axis) ? step.axis : [0, 0, 1];
    const angle = Number(step?.angle || 0);
    const vec = new THREE.Vector3(axis[0] || 0, axis[1] || 0, axis[2] || 0);
    if (vec.lengthSq() > 1e-12 && Math.abs(angle) > 1e-9) {
      vec.normalize();
      mesh.applyQuaternion(new THREE.Quaternion().setFromAxisAngle(vec, angle));
    }
  }

  const scale = Number(part.scale || 1);
  mesh.scale.setScalar(Number.isFinite(scale) && scale > 0 ? scale : 1);

  const rot = Array.isArray(part.rot) ? part.rot : [0, 0, 0];
  mesh.rotateX(THREE.MathUtils.degToRad(Number(rot[0] || 0)));
  mesh.rotateY(THREE.MathUtils.degToRad(Number(rot[1] || 0)));
  mesh.rotateZ(THREE.MathUtils.degToRad(Number(rot[2] || 0)));

  const translate = Array.isArray(part.translate) ? part.translate : [0, 0, 0];
  const tx = Number(translate[0] || 0);
  const ty = Number(translate[1] || 0);
  const tz = Number(translate[2] || 0);
  if (Array.isArray(offset) && offset.length === 3) {
    mesh.position.set(
      tx + Number(offset[0] || 0),
      ty + Number(offset[1] || 0),
      tz + Number(offset[2] || 0),
    );
  } else {
    mesh.position.set(tx, ty, tz);
  }
  return mesh;
}

function isLargeThumbPayload(payload) {
  const vertexCount = Array.isArray(payload?.vertices) ? payload.vertices.length / 3 : 0;
  return vertexCount > 250000;
}

function thumbPayloadForPart(part) {
  return part?.thumbMesh || null;
}

async function loadRemoteGeometry(part) {
  if (!part?.meshUrl) return null;
  const cached = remoteGeometryCache.get(part.meshUrl);
  if (cached) return cached;
  let geometry = null;
  if (part.meshFormat === 'stl') {
    geometry = await stlLoader.loadAsync(part.meshUrl);
    geometry.computeVertexNormals();
  } else if (part.meshFormat === 'payload') {
    const payload = await api(part.meshUrl, { timeoutMs: 30000 });
    geometry = geometryFromPayload(payload);
  }
  if (!geometry) return null;
  remoteGeometryCache.set(part.meshUrl, geometry);
  return geometry;
}

async function ensureSceneMeshes(scene, context = 'scene') {
  const pending = (scene?.parts || []).filter((part) => !part.mesh && part.meshUrl);
  if (!pending.length) return;

  stopProgressStream();
  startStallWatchdog(context);
  for (let idx = 0; idx < pending.length; idx += 1) {
    const part = pending[idx];
    const fetchMsg = `Fetching ${part.meshFormat?.toUpperCase() || 'mesh'} ${idx + 1} of ${pending.length}: ${part.name}`;
    updateViewportLoader(fetchMsg);
    addDebugLog('mesh-fetch', fetchMsg);
    const geometry = await loadRemoteGeometry(part);
    part.meshGeometry = geometry;
    lastProgressEventAt = Date.now();
    addDebugLog('mesh-fetch', `Fetched ${part.meshFormat?.toUpperCase() || 'mesh'} ${idx + 1} of ${pending.length}: ${part.name}`);
  }
}

function meshFromSceneEntry(entry, fallbackColor = [0.6, 0.7, 0.8]) {
  let mesh;
  const sourcePart = typeof entry.partIndex === 'number' ? sceneData?.parts?.[entry.partIndex] : null;
  if (entry.mesh) {
    mesh = buildVisualFromGeometry(geometryFromPayload(entry.mesh), entry.color || fallbackColor);
  } else if (typeof entry.partIndex === 'number' && sceneData?.parts?.[entry.partIndex]?.meshGeometry) {
    mesh = buildVisualFromGeometry(sceneData.parts[entry.partIndex].meshGeometry, entry.color || fallbackColor);
  } else if (typeof entry.partIndex === 'number' && sceneData?.parts?.[entry.partIndex]?.mesh) {
    mesh = buildVisualFromGeometry(geometryFromPayload(sceneData.parts[entry.partIndex].mesh), entry.color || fallbackColor);
  } else {
    return null;
  }
  if (sourcePart) {
    applyPartTransforms(mesh, sourcePart, entry.offset || [0, 0, 0]);
  } else if (Array.isArray(entry.offset) && entry.offset.length === 3) {
    mesh.position.set(entry.offset[0], entry.offset[1], entry.offset[2]);
  }
  return mesh;
}

function geometryForPart(part, context) {
  if (part?.meshGeometry) return part.meshGeometry;
  if (part?.mesh) return geometryFromPayload(part.mesh);
  addDebugLog('render-skip', `No mesh data available for ${context}`, {
    part: part?.name || null,
    keys: part && typeof part === 'object' ? Object.keys(part) : [],
  });
  return null;
}

function fitCamera(camera, controls, group) {
  const fallbackCenter = new THREE.Vector3(0, 0, 0);
  const fallbackDistance = 240;
  group.updateMatrixWorld(true);

  const box = new THREE.Box3().setFromObject(group);
  const center = box.isEmpty() ? fallbackCenter.clone() : box.getCenter(new THREE.Vector3());
  const sizeVec = box.isEmpty() ? new THREE.Vector3(1, 1, 1) : box.getSize(new THREE.Vector3());
  const size = sizeVec.length() || 1;
  const sphere = box.isEmpty() ? { radius: 1 } : box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius || 1, 1);
  const fovRad = THREE.MathUtils.degToRad(camera.fov || 45);
  const aspect = Math.max(camera.aspect || 1, 1e-3);
  const fitHeight = radius / Math.sin(Math.max(1e-3, fovRad / 2));
  const horizontalFov = 2 * Math.atan(Math.tan(fovRad / 2) * aspect);
  const fitWidth = radius / Math.sin(Math.max(1e-3, horizontalFov / 2));
  const distance = Math.max(
    fallbackDistance,
    fitHeight * 1.15,
    fitWidth * 1.15,
  );
  const direction = new THREE.Vector3(1, 0.72, 1).normalize();
  const nextPosition = center.clone().addScaledVector(direction, distance);

  if (!Number.isFinite(center.x) || !Number.isFinite(center.y) || !Number.isFinite(center.z)) {
    controls.target.copy(fallbackCenter);
    camera.position.set(fallbackDistance, fallbackDistance * 0.72, fallbackDistance);
  } else if (!Number.isFinite(nextPosition.x) || !Number.isFinite(nextPosition.y) || !Number.isFinite(nextPosition.z)) {
    controls.target.copy(center);
    camera.position.set(center.x + fallbackDistance, center.y + (fallbackDistance * 0.72), center.z + fallbackDistance);
  } else {
    controls.target.copy(center);
    camera.position.copy(nextPosition);
  }

  camera.near = Math.max(0.1, distance / 500);
  camera.far = Math.max(5000, distance * 12, size * 8);
  camera.updateProjectionMatrix();
  controls.update();
}

function updateSceneAtmosphere(ctx, group) {
  if (!ctx?.scene?.fog || !group) return;
  group.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(group);
  const sceneSize = box.isEmpty() ? 1 : (box.getSize(new THREE.Vector3()).length() || 1);
  ctx.scene.fog.density = Math.min(0.00018, Math.max(0.00001, 1 / (sceneSize * 18)));
}

function animate(ctx, frameHook = null) {
  let animId = null;
  let last = performance.now();
  const tick = () => {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (frameHook) frameHook(dt);
    ctx.controls.update();
    ctx.renderer.render(ctx.scene, ctx.camera);
    animId = requestAnimationFrame(tick);
  };
  tick();
  return () => {
    if (animId !== null) {
      cancelAnimationFrame(animId);
      animId = null;
    }
  };
}

/* ── Main 3D viewport ── */
function renderMain() {
  if (!sceneData) return;

  if (viewportHint) viewportHint.style.display = 'none';

  if (mainCtx) {
    mainCtx.scene.children
      .filter(c => c.type === 'Group')
      .forEach(c => mainCtx.scene.remove(c));
  } else {
    mainCtx = buildRenderer(mainCanvas);
    mainAnimation = animate(mainCtx);

    const ro = new ResizeObserver(() => {
      if (!mainCtx) return;
      const w = mainCanvas.clientWidth;
      const h = mainCanvas.clientHeight;
      mainCtx.renderer.setSize(w, h);
      mainCtx.camera.aspect = w / h;
      mainCtx.camera.updateProjectionMatrix();
    });
    ro.observe(mainCanvas);
  }

  const group = new THREE.Group();

  if (!tileView) {
    sceneData.combined.forEach((p) => {
      const mesh = meshFromSceneEntry(p, p.color);
      if (mesh) group.add(mesh);
    });
  } else {
    sceneData.parts.forEach((p, i) => {
      const geometry = geometryForPart(p, `tile:${p.name || i}`);
      if (!geometry) return;
      const mesh = buildVisualFromGeometry(geometry);
      applyPartTransforms(mesh, p, [i * 120, 0, 0]);
      group.add(mesh);
    });
  }

  mainCtx.scene.add(group);
  fitCamera(mainCtx.camera, mainCtx.controls, group);
  updateSceneAtmosphere(mainCtx, group);
  mainCtx.renderer.render(mainCtx.scene, mainCtx.camera);
  removeViewportLoader();
  addDebugLog('render-ready', `Rendered ${tileView ? 'tile' : 'combined'} view`, {
    parts: sceneData.parts?.length || 0,
    combined: sceneData.combined?.length || 0,
    drawCalls: mainCtx.renderer.info.render.calls,
    triangles: mainCtx.renderer.info.render.triangles,
    lines: mainCtx.renderer.info.render.lines,
    points: mainCtx.renderer.info.render.points,
    sceneChildren: group.children.length,
  });
}

/* ── Physics simulation ── */
function startPhysicsSim() {
  if (!sceneData || tileView) {
    toast('Switch to Combined View to run the physics sim.', 'warning');
    return;
  }

  const ctx = buildRenderer(mainCanvas);
  const group = new THREE.Group();
  const bodies = [];

  sceneData.combined.forEach((p, i) => {
    const mesh = meshFromSceneEntry(p, p.color);
    if (!mesh) return;
    const dropHeight = 100 + (i * 20);
    mesh.position.y += dropHeight;
    group.add(mesh);
    bodies.push({ mesh, targetY: mesh.position.y - dropHeight, vy: 0, settled: false });
  });

  ctx.scene.add(group);
  fitCamera(ctx.camera, ctx.controls, group);
  updateSceneAtmosphere(ctx, group);
  setBusy('Physics sim running...');

  const gravity = -260;
  let simDone = false;
  stopMainAnimation();
  mainAnimation = animate(ctx, (dt) => {
    let settledCount = 0;
    for (const body of bodies) {
      if (body.settled) {
        settledCount += 1;
        continue;
      }
      body.vy += gravity * dt;
      body.mesh.position.y += body.vy * dt;

      if (body.mesh.position.y <= body.targetY) {
        body.mesh.position.y = body.targetY;
        body.vy *= -0.28;
        if (Math.abs(body.vy) < 6) {
          body.vy = 0;
          body.settled = true;
          settledCount += 1;
        }
      }
    }
    if (!simDone && settledCount === bodies.length) {
      simDone = true;
      setIdle('Physics sim complete. Parts dropped into place.');
    }
  });
}

/* ── Thumbnails ── */
function buildThumb(part, idx, deferPreview = false, skipPreview = false) {
  const wrap = document.createElement('div');
  wrap.className = 'thumb';

  wrap.innerHTML = `
    <div class="thumb-header">
      <div class="thumb-title">
        <span class="part-name">${part.name}</span>
        ${part.previewLabel ? `<span class="preview-badge" title="${part.previewLabel}">${part.previewLabel}</span>` : ''}
      </div>
      <button data-act="focus" class="btn-sm" data-tooltip="Focus in viewport">&#x1F50D;</button>
    </div>
    <div class="thumb-canvas"></div>
    <div class="thumb-controls">
      <label><span class="ctrl-label">Tx</span> <input type="number" step="1" value="${(part.translate || [0, 0, 0])[0] || 0}" data-k="tx"/></label>
      <label><span class="ctrl-label">Ty</span> <input type="number" step="1" value="${(part.translate || [0, 0, 0])[1] || 0}" data-k="ty"/></label>
      <label><span class="ctrl-label">Tz</span> <input type="number" step="1" value="${(part.translate || [0, 0, 0])[2] || 0}" data-k="tz"/></label>
      <label><span class="ctrl-label">Scale</span> <input type="number" step="0.1" value="${part.scale}" data-k="scale"/></label>
      <label class="material-field"><span class="ctrl-label">Mat</span> <select data-k="material"></select></label>
    </div>`;

  thumbs.appendChild(wrap);

  const canvas = wrap.querySelector('.thumb-canvas');
  const renderPreview = () => {
    if (skipPreview) {
      canvas.innerHTML = `
        <div class="thumb-placeholder">
          <span class="thumb-placeholder-title">Assembly mode</span>
          <span class="thumb-placeholder-copy">3D thumbnails are skipped on large assemblies to protect the main viewport.</span>
        </div>`;
      return;
    }
    const thumbPayload = thumbPayloadForPart(part);
    if (isLargeThumbPayload(thumbPayload)) {
      canvas.innerHTML = `
        <div class="thumb-placeholder">
          <span class="thumb-placeholder-title">Large mesh</span>
          <span class="thumb-placeholder-copy">3D thumbnail skipped to keep the viewport responsive.</span>
        </div>`;
      return;
    }
    const ctx = buildRenderer(canvas, { preserveDrawingBuffer: true });
    if (ctx.grid) ctx.scene.remove(ctx.grid);
    const showMesh = (mesh) => {
      ctx.scene.add(mesh);
      fitCamera(ctx.camera, ctx.controls, mesh);
      ctx.renderer.render(ctx.scene, ctx.camera);
      const image = document.createElement('img');
      image.className = 'thumb-image';
      image.alt = `${part.name} preview`;
      image.src = ctx.renderer.domElement.toDataURL('image/png');
      canvas.innerHTML = '';
      canvas.appendChild(image);
      disposeRenderer(ctx);
    };
    if (part.meshGeometry && thumbPayload) {
      showMesh(applyPartTransforms(buildVisualFromGeometry(part.meshGeometry), part));
    } else if (thumbPayload) {
      showMesh(applyPartTransforms(buildVisualFromGeometry(geometryFromPayload(thumbPayload)), part));
    } else {
      disposeRenderer(ctx);
      canvas.innerHTML = `
        <div class="thumb-placeholder">
          <span class="thumb-placeholder-title">Viewport Proxy</span>
          <span class="thumb-placeholder-copy">3D thumbnail skipped to keep the main viewport responsive.</span>
        </div>`;
    }
  };

  if (deferPreview) {
    window.setTimeout(renderPreview, 0);
  } else {
    renderPreview();
  }

  const materialSelect = wrap.querySelector('select[data-k="material"]');
  MATERIAL_OPTIONS.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m || 'auto';
    if ((part.material || '') === m) opt.selected = true;
    materialSelect.appendChild(opt);
  });

  const sendUpdate = async () => {
    const translation = {
      x: +wrap.querySelector('input[data-k="tx"]').value,
      y: +wrap.querySelector('input[data-k="ty"]').value,
      z: +wrap.querySelector('input[data-k="tz"]').value,
    };
    const scale = +wrap.querySelector('input[data-k="scale"]').value;
    const material = wrap.querySelector('select[data-k="material"]').value;
    try {
      await api(`/api/part/${idx}`, { method: 'PATCH', body: JSON.stringify({ translation, scale, material }) });
      await refreshScene();
    } catch (e) {
      toast('Failed to update part: ' + e.message, 'error');
    }
  };

  wrap.querySelectorAll('input').forEach(inp => inp.addEventListener('change', sendUpdate));
  materialSelect.addEventListener('change', sendUpdate);

  wrap.querySelector('[data-act="focus"]').addEventListener('click', () => {
    tileView = true;
    updateViewButtons();
    renderMain();
  });
}

function renderThumbs() {
  stopThumbAnimations();
  thumbs.innerHTML = '';
  if (!sceneData || !sceneData.parts.length) {
    partCountEl.textContent = '0 parts';
    return;
  }
  partCountEl.textContent = `${sceneData.parts.length} part${sceneData.parts.length !== 1 ? 's' : ''}`;
  const generation = ++thumbRenderGeneration;
  const skipPreview = sceneData.parts.length > MAX_WEBGL_THUMBNAILS;
  const queue = sceneData.parts.map((part, idx) => ({ part, idx }));
  const renderNext = () => {
    if (generation !== thumbRenderGeneration) return;
    const next = queue.shift();
    if (!next) return;
    buildThumb(next.part, next.idx, true, skipPreview);
    if (queue.length) {
      window.setTimeout(renderNext, 0);
    }
  };
  window.setTimeout(renderNext, 0);
}

/* ── View button state ── */
function updateViewButtons() {
  combinedBtn.classList.toggle('active', !tileView);
  tileBtn.classList.toggle('active', tileView);
}

function updateWorkflowUI() {
  const workflow = workflowSelect?.value || 'cylinder';
  document.querySelectorAll('[data-workflows]').forEach((el) => {
    const allowed = (el.dataset.workflows || '')
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean);
    const visible = !allowed.length || allowed.includes(workflow);
    el.classList.toggle('hidden', !visible);
    if ('disabled' in el) el.disabled = !visible;
  });
}

function renderMidlayerConfigForms() {
  if (!midlayerConfigForms) return;
  const solvers = midlayerSolverData.solvers || {};
  const configs = midlayerSolverData.configs || {};
  midlayerConfigForms.innerHTML = '';

  Object.entries(solvers).forEach(([solverId, solver]) => {
    const current = configs[solverId] || solver.defaults || {};
    const wrapper = document.createElement('section');
    wrapper.className = 'solver-config';
    wrapper.innerHTML = `
      <div class="solver-config-header">
        <div>
          <h4>${solver.label}</h4>
          <p>${solver.availability?.notes || ''}</p>
        </div>
        <span class="solver-chip ${solver.availability?.installed ? 'ok' : 'fallback'}">
          ${solver.availability?.installed ? 'package detected' : 'scaffold mode'}
        </span>
      </div>
      <div class="solver-config-grid"></div>`;
    const grid = wrapper.querySelector('.solver-config-grid');
    (solver.schema || []).forEach((field) => {
      const label = document.createElement('label');
      label.className = 'solver-config-field';
      label.innerHTML = `
        <span>${field.label}</span>
        <input
          type="${field.type || 'number'}"
          step="${field.step ?? 1}"
          min="${field.min ?? ''}"
          max="${field.max ?? ''}"
          value="${current[field.key] ?? field.default ?? ''}"
          data-midlayer-solver="${solverId}"
          data-midlayer-key="${field.key}"
        />`;
      grid.appendChild(label);
    });
    midlayerConfigForms.appendChild(wrapper);
  });

  midlayerConfigForms.querySelectorAll('input[data-midlayer-solver]').forEach((input) => {
    input.addEventListener('change', async () => {
      const solverId = input.dataset.midlayerSolver;
      const solver = solvers[solverId];
      if (!solver) return;
      const values = {};
      (solver.schema || []).forEach((field) => {
        const fieldInput = midlayerConfigForms.querySelector(
          `input[data-midlayer-solver="${solverId}"][data-midlayer-key="${field.key}"]`
        );
        if (!fieldInput) return;
        values[field.key] = Number(fieldInput.value);
      });
      midlayerSolverData.configs[solverId] = values;
      try {
        await api('/api/config', {
          method: 'PATCH',
          body: JSON.stringify({ midlayer_configs: { [solverId]: values } }),
        });
        toast(`${solver.label} midlayer config updated`, 'success');
      } catch (e) {
        toast('Midlayer config update failed: ' + e.message, 'error');
      }
    });
  });
}

/* ── Data refresh ── */
async function refreshScene() {
  try {
    if (progressSSE) {
      addDebugLog('progress-stream', 'Closing progress stream before scene fetch');
      progressSSE.close();
      progressSSE = null;
    }
    addDebugLog('scene-refresh', 'Requesting scene payload');
    sceneData = await api('/api/scene', { timeoutMs: 15000 });
    await ensureSceneMeshes(sceneData, 'scene-fetch');
    addDebugLog('scene-refresh', 'Scene payload received', {
      parts: sceneData.parts?.length || 0,
      combined: sceneData.combined?.length || 0,
    });
    renderMain();
    renderThumbs();
  } catch (e) {
    addDebugLog('scene-error', e.message);
    throw e;
  }
}

async function refreshFiles() {
  const requestId = ++refreshFilesRequestId;
  setBusy('Scanning for files...');
  try {
    const data = await api('/api/files');
    if (requestId !== refreshFilesRequestId) return;
    if (partsDirLabel) {
      partsDirLabel.textContent = `Dir: ${data.parts_dir}`;
    }
    fileList.innerHTML = '';
    if (data.files.length === 0) {
      fileList.innerHTML = '<div class="empty-state"><span class="empty-icon">&#x1F4ED;</span><p>No supported CAD files found</p></div>';
    } else {
      data.files.forEach(f => {
        const ext = f.split('.').pop().toUpperCase();
        const row = document.createElement('div');
        row.className = 'file-row';
        row.innerHTML = `<input type="checkbox" value="${f}"/><span class="file-name">${f}</span><span class="file-ext">${ext}</span>`;
        row.addEventListener('click', e => {
          if (e.target.tagName !== 'INPUT') {
            const cb = row.querySelector('input');
            cb.checked = !cb.checked;
          }
        });
        fileList.appendChild(row);
      });
    }
    setIdle(`Found ${data.files.length} file${data.files.length !== 1 ? 's' : ''}`);
    toast(`Found ${data.files.length} CAD file${data.files.length !== 1 ? 's' : ''}`, 'info');
  } catch (e) {
    if (requestId !== refreshFilesRequestId) return;
    setIdle();
    toast('Failed to scan files: ' + e.message, 'error');
  }
}

async function changePartsDir() {
  openDirOverlay();
}

async function loadSelectedParts(explicitFiles = null) {
  const files = Array.isArray(explicitFiles) ? explicitFiles : [...fileList.querySelectorAll('input:checked')].map(x => x.value);
  if (files.length === 0) {
    toast('Select at least one file to load', 'warning');
    return;
  }
  addDebugLog('load-click', 'Load selected triggered', { files });
  setBusy('Loading parts...');
  showViewportLoader('Loading parts...');
  startStallWatchdog('load-selected');
  try {
    const loadResult = await api('/api/load', {
      method: 'POST',
      body: JSON.stringify({ files, include_scene: true }),
      timeoutMs: 30000,
    });
    addDebugLog('load-click', 'Load API complete', { files });
    toast(`Loaded ${files.length} part${files.length !== 1 ? 's' : ''}`, 'success');
    setIdle(`${files.length} part${files.length !== 1 ? 's' : ''} loaded`);
    if (loadResult?.scene) {
      addDebugLog('scene-refresh', 'Using scene payload returned from load', {
        parts: loadResult.scene.parts?.length || 0,
        combined: loadResult.scene.combined?.length || 0,
      });
      sceneData = loadResult.scene;
      await ensureSceneMeshes(sceneData, 'load-inline');
      renderMain();
      renderThumbs();
    } else {
      await refreshScene();
    }
  } catch (e) {
    addDebugLog('load-error', e.message, { files });
    removeViewportLoader();
    stopProgressStream();
    setIdle();
    toast('Failed to load parts: ' + e.message, 'error');
  }
}

async function fetchDirectoryListing(path = '.') {
  return api(`/api/directories?path=${encodeURIComponent(path)}`);
}

function renderDirectoryList() {
  if (!dirList || !dirCurrent || !dirSelectBtn) return;
  dirCurrent.textContent = dirBrowseState.current;
  dirList.innerHTML = '';

  if (!dirBrowseState.directories.length) {
    dirList.innerHTML = '<div class="empty-state"><span class="empty-icon">&#x1F4C1;</span><p>No subdirectories here</p></div>';
  } else {
    dirBrowseState.directories.forEach((entry) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'dir-entry';
      if (entry.path === dirBrowseState.selected) row.classList.add('active');
      row.innerHTML = `
        <span class="dir-icon">&#x1F4C1;</span>
        <span class="dir-name">${entry.name}</span>
        <span class="dir-meta">${entry.file_count} file${entry.file_count !== 1 ? 's' : ''}</span>`;
      row.addEventListener('click', () => {
        dirBrowseState.selected = entry.path;
        renderDirectoryList();
      });
      row.addEventListener('dblclick', async () => {
        await browseDirectory(entry.path);
      });
      dirList.appendChild(row);
    });
  }

  dirUpBtn.disabled = !dirBrowseState.parent;
  dirSelectBtn.disabled = !dirBrowseState.selected;
}

async function browseDirectory(path = '.') {
  setBusy('Updating parts directory...');
  try {
    dirBrowseState = await fetchDirectoryListing(path);
    dirBrowseState.selected = dirBrowseState.current;
    renderDirectoryList();
    setIdle('Browse to a directory and confirm');
  } catch (e) {
    setIdle();
    toast('Failed to browse directories: ' + e.message, 'error');
  }
}

function openDirOverlay() {
  if (!dirOverlay) return;
  dirOverlay.classList.remove('hidden');
  dirOverlay.setAttribute('aria-hidden', 'false');
  browseDirectory(partsDirLabel?.textContent.replace(/^Dir:\s*/, '') || '.');
}

function closeDirOverlay() {
  if (!dirOverlay) return;
  dirOverlay.classList.add('hidden');
  dirOverlay.setAttribute('aria-hidden', 'true');
}

async function applyPartsDir(path) {
  setBusy('Updating parts directory...');
  try {
    const data = await api('/api/parts-dir', {
      method: 'PATCH',
      body: JSON.stringify({ path }),
    });
    if (partsDirLabel) {
      partsDirLabel.textContent = `Dir: ${data.parts_dir}`;
    }
    closeDirOverlay();
    toast(`Parts directory set to ${data.parts_dir}`, 'success');
    await refreshFiles();
  } catch (e) {
    setIdle();
    toast('Failed to update parts directory: ' + e.message, 'error');
  }
}

async function uploadSelectedPartFile(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch('/api/upload-part', { method: 'POST', body: form });
  if (!res.ok) {
    let data = {};
    try {
      data = await res.json();
    } catch (_) {
      // noop
    }
    throw new Error(data.error || `HTTP ${res.status}`);
  }
}

async function refreshGradientFiles() {
  try {
    const data = await api('/api/gradient/files');
    gradientInput.innerHTML = '';
    if (data.files.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No files found';
      opt.disabled = true;
      gradientInput.appendChild(opt);
    } else {
      data.files.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f;
        opt.textContent = f;
        gradientInput.appendChild(opt);
      });
    }
  } catch (e) {
    toast('Failed to list gradient files', 'error');
  }
}

/* ── Progress bar helpers ── */
function showProgress(current, total, message, history = []) {
  lastProgressEventAt = Date.now();
  progressContainer.style.display = 'flex';
  if (total > 0) {
    const pct = Math.round((current / total) * 100);
    progressFill.style.width = pct + '%';
    progressFill.classList.remove('indeterminate');
    progressText.textContent = `${current}/${total} (${pct}%)`;
  } else {
    progressFill.classList.add('indeterminate');
    progressText.textContent = message || '';
  }

  if (Array.isArray(history) && history.length) {
    progressHistory = history.map((entry) => ({
      ...entry,
      ts: Math.round((entry.ts || Date.now() / 1000) * 1000),
    }));
  }
  const vpFill = document.getElementById('viewport-progress-fill');
  const vpText = document.getElementById('viewport-progress-text');
  const vpMsg = document.querySelector('#viewport-loader .loader-msg');
  const vpElapsed = document.getElementById('viewport-progress-elapsed');
  const latestEvent = progressHistory.length ? [...progressHistory].sort((a, b) => b.ts - a.ts)[0] : null;
  if (vpFill) {
    if (total > 0) {
      vpFill.style.width = Math.round((current / total) * 100) + '%';
      vpFill.classList.remove('indeterminate');
    } else {
      vpFill.classList.add('indeterminate');
      vpFill.style.width = '30%';
    }
  }
  if (vpText) {
    vpText.textContent = total > 0 ? `${current}/${total}` : 'Working';
  }
  if (vpMsg && message) {
    vpMsg.textContent = message;
  }
  if (vpElapsed && latestEvent) {
    const ageSec = Math.max(0, Math.floor((Date.now() - latestEvent.ts) / 1000));
    vpElapsed.textContent = `${ageSec}s`;
  }
  renderProgressHistory();
}

function hideProgress() {
  progressContainer.style.display = 'none';
  progressFill.style.width = '0%';
  progressFill.classList.remove('indeterminate');
  progressText.textContent = '';
  progressHistory = [];
  renderProgressHistory();
}

function startProgressStream() {
  if (progressSSE) progressSSE.close();
  addDebugLog('progress-stream', 'Opening progress stream');
  progressSSE = new EventSource('/api/progress/stream');
  progressSSE.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      addDebugLog('progress-event', data.message || 'Working', {
        current: data.current,
        total: data.total,
      });
      showProgress(data.current, data.total, data.message, data.history || []);
      setStatus(data.message || 'Working...', true);
    } catch (_) {}
  };
  progressSSE.onerror = () => {
    addDebugLog('progress-stream', 'Progress stream error/closed');
    if (progressSSE) progressSSE.close();
    progressSSE = null;
  };
}

function stopProgressStream() {
  if (progressSSE) {
    progressSSE.close();
    progressSSE = null;
  }
  addDebugLog('progress-stream', 'Closing progress stream');
  setTimeout(hideProgress, 600);
}

/* ── Button loading state helper ── */
function withLoading(btn, fn) {
  return async (...args) => {
    btn.classList.add('loading');
    btn.disabled = true;
    startProgressStream();
    try {
      await fn(...args);
    } finally {
      btn.classList.remove('loading');
      btn.disabled = false;
      stopProgressStream();
    }
  };
}

/* ── Event listeners ── */

// File management
document.getElementById('refresh-files').addEventListener('click', refreshFiles);
document.getElementById('change-dir').addEventListener('click', withLoading(
  document.getElementById('change-dir'),
  async () => {
    await changePartsDir();
  }
));
document.getElementById('browse-part').addEventListener('click', () => browsePartInput.click());
browsePartInput.addEventListener('change', withLoading(
  document.getElementById('browse-part'),
  async () => {
    const [file] = browsePartInput.files || [];
    if (!file) return;
    setBusy(`Uploading ${file.name}...`);
    try {
      await uploadSelectedPartFile(file);
      toast(`Uploaded ${file.name}`, 'success');
      await refreshFiles();
      const checkbox = [...fileList.querySelectorAll('input')].find((el) => el.value === file.name);
      if (checkbox) checkbox.checked = true;
      await loadSelectedParts([file.name]);
    } catch (e) {
      setIdle();
      toast('Upload failed: ' + e.message, 'error');
    } finally {
      browsePartInput.value = '';
    }
  }
));
document.getElementById('load-selected').addEventListener('click', withLoading(
  document.getElementById('load-selected'),
  async () => {
    await loadSelectedParts();
  }
));

// Config changes
workflowSelect?.addEventListener('change', async () => {
  try {
    await api('/api/config', { method: 'PATCH', body: JSON.stringify({ workflow: workflowSelect.value }) });
    updateWorkflowUI();
    toast(`Workflow set to ${workflowSelect.options[workflowSelect.selectedIndex]?.textContent || workflowSelect.value}`, 'success');
  } catch (e) {
    toast('Workflow update failed', 'error');
  }
});

fineOrientToggle?.addEventListener('change', async () => {
  try {
    await api('/api/config', { method: 'PATCH', body: JSON.stringify({ fine_orient: fineOrientToggle.checked }) });
    toast(fineOrientToggle.checked ? 'Fine-grained orient enabled' : 'Fast axis-aligned orient enabled', 'success');
  } catch (e) {
    toast('Orient mode update failed', 'error');
  }
});

axisSelect.addEventListener('change', async () => {
  try {
    await api('/api/config', { method: 'PATCH', body: JSON.stringify({ axis: axisSelect.value }) });
    await refreshScene();
  } catch (e) { toast('Config update failed', 'error'); }
});

gapInput.addEventListener('change', async () => {
  try {
    await api('/api/config', { method: 'PATCH', body: JSON.stringify({ gap: +gapInput.value }) });
    await refreshScene();
  } catch (e) { toast('Config update failed', 'error'); }
});

sectionInput.addEventListener('change', async () => {
  const raw = sectionInput.value.trim();
  const section_number = raw === '' ? null : Math.max(1, Math.floor(+raw));
  try {
    await api('/api/config', { method: 'PATCH', body: JSON.stringify({ section_number }) });
  } catch (e) { toast('Config update failed', 'error'); }
});

// Pipeline stages
document.querySelectorAll('[data-stage]').forEach(btn => {
  btn.addEventListener('click', withLoading(btn, async () => {
    const stageName = btn.textContent.trim().replace(/^\d+\s*/, '');
    addDebugLog('stage-click', `Requested stage ${btn.dataset.stage}`, { stage: btn.dataset.stage });
    await flushDebugLog(true);
    setBusy(`Running: ${stageName}...`);
    showViewportLoader(`Running: ${stageName}...`);
    const stallThreshold = btn.dataset.stage.startsWith('design_midlayer_') ? 20000 : 5000;
    startStallWatchdog(`stage:${btn.dataset.stage}`, stallThreshold);
    try {
      const out = await api(`/api/stage/${btn.dataset.stage}`, {
        method: 'POST',
        body: '{}',
        timeoutMs: STAGE_TIMEOUT_MS[btn.dataset.stage] || 30000,
      });
      toast(out.message || 'Stage complete', 'success');
      setIdle(out.message || 'Done');
      await refreshScene();
    } catch (e) {
      removeViewportLoader();
      stopProgressStream();
      setIdle();
      toast('Stage failed: ' + e.message, 'error');
    }
  }));
});

document.querySelectorAll('[data-client-stage]').forEach((btn) => {
  btn.addEventListener('click', () => {
    if (btn.dataset.clientStage === 'auto_drop') {
      startPhysicsSim();
    }
  });
});

// View mode toggle
combinedBtn.addEventListener('click', () => { tileView = false; updateViewButtons(); renderMain(); });
tileBtn.addEventListener('click', () => { tileView = true; updateViewButtons(); renderMain(); });

dirSelectBtn?.addEventListener('click', withLoading(dirSelectBtn, async () => {
  await applyPartsDir(dirBrowseState.selected || dirBrowseState.current);
}));
dirUpBtn?.addEventListener('click', async () => {
  if (dirBrowseState.parent) await browseDirectory(dirBrowseState.parent);
});
dirRootBtn?.addEventListener('click', async () => {
  await browseDirectory(dirBrowseState.root || '.');
});
dirCloseBtn?.addEventListener('click', closeDirOverlay);
dirCancelBtn?.addEventListener('click', closeDirOverlay);
dirOverlay?.addEventListener('click', (event) => {
  if (event.target === dirOverlay) closeDirOverlay();
});

document.querySelectorAll('[data-tooltip]').forEach((el) => {
  el.addEventListener('mouseenter', () => showTooltip(el));
  el.addEventListener('focus', () => showTooltip(el));
  el.addEventListener('mouseleave', hideTooltip);
  el.addEventListener('blur', hideTooltip);
  el.addEventListener('mousemove', () => positionTooltip(el));
});

debugToggle?.addEventListener('click', () => {
  debugPanelOpen = !debugPanelOpen;
  updateDebugPanelState();
});

debugCopyBtn?.addEventListener('click', async () => {
  const text = formatDebugLogText();
  if (!text) {
    toast('Debug log is empty', 'warning');
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    addDebugLog('debug-copy', 'Copied debug log to clipboard', { lines: debugEntries.length });
    toast('Copied debug log', 'success');
  } catch (e) {
    addDebugLog('debug-copy-error', e?.message || 'Clipboard write failed');
    toast('Failed to copy debug log', 'error');
  }
});

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    flushDebugLog(true);
  }
});

window.addEventListener('beforeunload', () => {
  flushDebugLog(true);
});

// Gradient capability
document.getElementById('refresh-gradient-files').addEventListener('click', refreshGradientFiles);
document.getElementById('run-gradient-capability').addEventListener('click', withLoading(
  document.getElementById('run-gradient-capability'),
  async () => {
    if (!gradientInput.value) {
      toast('No input file selected for gradient', 'warning');
      return;
    }
    setBusy('Running thermal gradient...');
    try {
      const out = await api('/api/capability/wrl_gradient', {
        method: 'POST',
        body: JSON.stringify({
          input: gradientInput.value,
          mode: gradientMode.value,
          output: gradientOutput.value || 'web_colored_output.ply',
          render: gradientRender.value || null,
        }),
      });
      toast(out.message || 'Gradient complete', 'success');
      setIdle(out.message || 'Gradient complete');
      await refreshFiles();
      await refreshGradientFiles();
    } catch (e) {
      setIdle();
      toast('Gradient failed: ' + e.message, 'error');
    }
  }
));

/* ── Keyboard shortcut: Ctrl+L = load ── */
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && dirOverlay && !dirOverlay.classList.contains('hidden')) {
    closeDirOverlay();
    hideTooltip();
    return;
  }
  if (e.ctrlKey && e.key === 'l') {
    e.preventDefault();
    document.getElementById('load-selected').click();
  }
});

/* ── Init ── */
refreshFiles();
refreshGradientFiles();
renderDebugLog();
updateDebugPanelState();
updateWorkflowUI();
renderMidlayerConfigForms();
addDebugLog('app-init', 'Web UI initialized');
