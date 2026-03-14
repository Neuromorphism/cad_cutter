import * as THREE from 'https://unpkg.com/three@0.161.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.161.0/examples/jsm/controls/OrbitControls.js';

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
const startSimBtn    = document.getElementById('start-sim');

const progressContainer = document.getElementById('progress-container');
const progressFill      = document.getElementById('progress-fill');
const progressText      = document.getElementById('progress-text');

let sceneData = null;
let tileView  = false;
let mainCtx   = null;
let progressSSE = null;
let mainAnimation = null;
let thumbAnimations = [];

const MATERIAL_OPTIONS = [
  '', 'steel', 'aluminum', 'copper', 'brass', 'bronze', 'gold', 'titanium',
  'chrome', 'plastic', 'rubber', 'ceramic', 'glass', 'wood', 'oak', 'pine',
  'stone', 'concrete', 'red', 'green', 'blue', 'black', 'white'
];

/* ── Toast notifications ── */
function toast(message, type = 'info') {
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

function stopMainAnimation() {
  if (mainAnimation) {
    mainAnimation();
    mainAnimation = null;
  }
}

function stopThumbAnimations() {
  thumbAnimations.forEach((stop) => stop());
  thumbAnimations = [];
}

/* ── API wrapper with error handling ── */
async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { 'content-type': 'application/json' }, ...opts });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText);
  }
  return res.json();
}

/* ── Loading overlay for viewport ── */
function showViewportLoader(msg = 'Loading...') {
  removeViewportLoader();
  const overlay = document.createElement('div');
  overlay.className = 'loading-overlay';
  overlay.id = 'viewport-loader';
  overlay.innerHTML = `
    <div class="spinner"></div>
    <p class="loader-msg">${msg}</p>
    <div class="viewport-progress">
      <div class="progress-bar">
        <div class="progress-fill" id="viewport-progress-fill"></div>
      </div>
      <span class="progress-text" id="viewport-progress-text"></span>
    </div>`;
  mainCanvas.appendChild(overlay);
}

function removeViewportLoader() {
  const existing = document.getElementById('viewport-loader');
  if (existing) existing.remove();
}

/* ── Three.js helpers ── */
function buildRenderer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const existingCanvas = container.querySelector('canvas');
  if (existingCanvas) existingCanvas.remove();
  container.insertBefore(renderer.domElement, container.firstChild);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  scene.fog = new THREE.FogExp2(0x0d1117, 0.0008);

  const camera = new THREE.PerspectiveCamera(
    45, container.clientWidth / container.clientHeight, 0.1, 10000
  );
  camera.position.set(160, 140, 160);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.8;

  const hemi = new THREE.HemisphereLight(0xc8d8f0, 0x1a2030, 0.6);
  scene.add(hemi);

  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(80, 120, 60);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0x8899bb, 0.4);
  fill.position.set(-60, 40, -30);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0x6688cc, 0.3);
  rim.position.set(0, -20, -80);
  scene.add(rim);

  const grid = new THREE.GridHelper(600, 40, 0x1a2030, 0x141a22);
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  scene.add(grid);

  return { renderer, scene, camera, controls, grid };
}

function meshFromPayload(payload, color = [0.6, 0.7, 0.8]) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(payload.vertices, 3));
  geo.setIndex(payload.indices);
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(...color),
    metalness: 0.35,
    roughness: 0.45,
    envMapIntensity: 0.5,
  });
  return new THREE.Mesh(geo, mat);
}

function fitCamera(camera, controls, group) {
  const box = new THREE.Box3().setFromObject(group);
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  camera.position.copy(center.clone().add(new THREE.Vector3(size * 0.9, size * 0.7, size * 0.9)));
  controls.update();
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
    sceneData.combined.forEach(p => group.add(meshFromPayload(p.mesh, p.color)));
  } else {
    sceneData.parts.forEach((p, i) => {
      const mesh = meshFromPayload(p.mesh);
      mesh.position.x = i * 120;
      group.add(mesh);
    });
  }

  mainCtx.scene.add(group);
  fitCamera(mainCtx.camera, mainCtx.controls, group);
  removeViewportLoader();
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
    const mesh = meshFromPayload(p.mesh, p.color);
    const dropHeight = 100 + (i * 20);
    mesh.position.y += dropHeight;
    group.add(mesh);
    bodies.push({ mesh, targetY: mesh.position.y - dropHeight, vy: 0, settled: false });
  });

  ctx.scene.add(group);
  fitCamera(ctx.camera, ctx.controls, group);
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
function buildThumb(part, idx) {
  const wrap = document.createElement('div');
  wrap.className = 'thumb';

  wrap.innerHTML = `
    <div class="thumb-header">
      <span class="part-name">${part.name}</span>
      <button data-act="focus" class="btn-sm" data-tooltip="Focus in viewport">&#x1F50D;</button>
    </div>
    <div class="thumb-canvas"></div>
    <div class="thumb-controls">
      <label><span class="ctrl-label">Rx</span> <input type="number" value="${part.rot[0]}" data-k="x"/></label>
      <label><span class="ctrl-label">Ry</span> <input type="number" value="${part.rot[1]}" data-k="y"/></label>
      <label><span class="ctrl-label">Rz</span> <input type="number" value="${part.rot[2]}" data-k="z"/></label>
      <label><span class="ctrl-label">Scale</span> <input type="number" step="0.1" value="${part.scale}" data-k="scale"/></label>
      <label class="material-field"><span class="ctrl-label">Mat</span> <select data-k="material"></select></label>
    </div>`;

  const canvas = wrap.querySelector('.thumb-canvas');
  const ctx = buildRenderer(canvas);
  if (ctx.grid) ctx.scene.remove(ctx.grid);
  const mesh = meshFromPayload(part.mesh);
  ctx.scene.add(mesh);
  fitCamera(ctx.camera, ctx.controls, mesh);
  const stopThumbAnimation = animate(ctx);
  thumbAnimations.push(stopThumbAnimation);

  const materialSelect = wrap.querySelector('select[data-k="material"]');
  MATERIAL_OPTIONS.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m || 'auto';
    if ((part.material || '') === m) opt.selected = true;
    materialSelect.appendChild(opt);
  });

  const sendUpdate = async () => {
    const rotation = {
      x: +wrap.querySelector('input[data-k="x"]').value,
      y: +wrap.querySelector('input[data-k="y"]').value,
      z: +wrap.querySelector('input[data-k="z"]').value,
    };
    const scale = +wrap.querySelector('input[data-k="scale"]').value;
    const material = wrap.querySelector('select[data-k="material"]').value;
    try {
      await api(`/api/part/${idx}`, { method: 'PATCH', body: JSON.stringify({ rotation, scale, material }) });
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

  thumbs.appendChild(wrap);
}

function renderThumbs() {
  stopThumbAnimations();
  thumbs.innerHTML = '';
  if (!sceneData || !sceneData.parts.length) {
    partCountEl.textContent = '0 parts';
    return;
  }
  partCountEl.textContent = `${sceneData.parts.length} part${sceneData.parts.length !== 1 ? 's' : ''}`;
  sceneData.parts.forEach(buildThumb);
}

/* ── View button state ── */
function updateViewButtons() {
  combinedBtn.classList.toggle('active', !tileView);
  tileBtn.classList.toggle('active', tileView);
}

/* ── Data refresh ── */
async function refreshScene() {
  try {
    sceneData = await api('/api/scene');
    renderMain();
    renderThumbs();
  } catch (e) {
    toast('Failed to load scene: ' + e.message, 'error');
  }
}

async function refreshFiles() {
  setBusy('Scanning for files...');
  try {
    const data = await api('/api/files');
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
    setIdle();
    toast('Failed to scan files: ' + e.message, 'error');
  }
}

async function changePartsDir() {
  const raw = prompt('Enter parts directory (relative to current working directory):', '.');
  if (raw === null) return;
  const path = raw.trim();
  if (!path) return;

  setBusy('Updating parts directory...');
  try {
    const data = await api('/api/parts-dir', {
      method: 'PATCH',
      body: JSON.stringify({ path }),
    });
    if (partsDirLabel) {
      partsDirLabel.textContent = `Dir: ${data.parts_dir}`;
    }
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
function showProgress(current, total, message) {
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
}

function hideProgress() {
  progressContainer.style.display = 'none';
  progressFill.style.width = '0%';
  progressFill.classList.remove('indeterminate');
  progressText.textContent = '';
}

function startProgressStream() {
  if (progressSSE) progressSSE.close();
  progressSSE = new EventSource('/api/progress/stream');
  progressSSE.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.total > 0) {
        showProgress(data.current, data.total, data.message);
        setStatus(data.message, true);
        const vpFill = document.getElementById('viewport-progress-fill');
        const vpText = document.getElementById('viewport-progress-text');
        const vpMsg = document.querySelector('#viewport-loader .loader-msg');
        if (vpFill) vpFill.style.width = Math.round((data.current / data.total) * 100) + '%';
        if (vpText) vpText.textContent = `${data.current}/${data.total}`;
        if (vpMsg) vpMsg.textContent = data.message;
      }
    } catch (_) {}
  };
  progressSSE.onerror = () => {
    if (progressSSE) progressSSE.close();
    progressSSE = null;
  };
}

function stopProgressStream() {
  if (progressSSE) {
    progressSSE.close();
    progressSSE = null;
  }
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
      document.getElementById('load-selected').click();
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
    const files = [...fileList.querySelectorAll('input:checked')].map(x => x.value);
    if (files.length === 0) {
      toast('Select at least one file to load', 'warning');
      return;
    }
    setBusy('Loading parts...');
    showViewportLoader('Loading parts...');
    try {
      await api('/api/load', { method: 'POST', body: JSON.stringify({ files }) });
      toast(`Loaded ${files.length} part${files.length !== 1 ? 's' : ''}`, 'success');
      setIdle(`${files.length} part${files.length !== 1 ? 's' : ''} loaded`);
      await refreshScene();
    } catch (e) {
      removeViewportLoader();
      setIdle();
      toast('Failed to load parts: ' + e.message, 'error');
    }
  }
));

// Config changes
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
    setBusy(`Running: ${stageName}...`);
    showViewportLoader(`Running: ${stageName}...`);
    try {
      const out = await api(`/api/stage/${btn.dataset.stage}`, { method: 'POST', body: '{}' });
      toast(out.message || 'Stage complete', 'success');
      setIdle(out.message || 'Done');
      await refreshScene();
    } catch (e) {
      removeViewportLoader();
      setIdle();
      toast('Stage failed: ' + e.message, 'error');
    }
  }));
});

// View mode toggle
combinedBtn.addEventListener('click', () => { tileView = false; updateViewButtons(); renderMain(); });
tileBtn.addEventListener('click', () => { tileView = true; updateViewButtons(); renderMain(); });
if (startSimBtn) {
  startSimBtn.addEventListener('click', startPhysicsSim);
}

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
  if (e.ctrlKey && e.key === 'l') {
    e.preventDefault();
    document.getElementById('load-selected').click();
  }
});

/* ── Init ── */
refreshFiles();
refreshGradientFiles();
