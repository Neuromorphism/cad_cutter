import * as THREE from 'https://unpkg.com/three@0.161.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.161.0/examples/jsm/controls/OrbitControls.js';

const fileList = document.getElementById('file-list');
const statusEl = document.getElementById('status');
const mainCanvas = document.getElementById('main-canvas');
const thumbs = document.getElementById('thumbnails');
const axisSelect = document.getElementById('axis-select');
const gapInput = document.getElementById('gap-input');
const startSimBtn = document.getElementById('start-sim');

let sceneData = null;
let tileView = false;
let activeAnimation = null;

function setStatus(msg) { statusEl.textContent = msg; }

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: {'content-type':'application/json'}, ...opts });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function buildRenderer(container) {
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.innerHTML = '';
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f141d);
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 10000);
  camera.position.set(160, 140, 160);
  const controls = new OrbitControls(camera, renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff,0x444444,1.4));
  const dl = new THREE.DirectionalLight(0xffffff,1.0); dl.position.set(80,120,30); scene.add(dl);
  return {renderer, scene, camera, controls};
}

function meshFromPayload(payload, color=[0.6,0.7,0.8]) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(payload.vertices, 3));
  geo.setIndex(payload.indices);
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color:new THREE.Color(...color), metalness:0.2, roughness:0.55}));
}

function fitCamera(camera, controls, group) {
  const box = new THREE.Box3().setFromObject(group);
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  camera.position.copy(center.clone().add(new THREE.Vector3(size*0.9,size*0.7,size*0.9)));
  controls.update();
}

function animate(ctx, frameHook = null){
  if (activeAnimation) {
    cancelAnimationFrame(activeAnimation);
    activeAnimation = null;
  }
  let last = performance.now();
  const tick = ()=>{
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (frameHook) frameHook(dt);
    ctx.renderer.render(ctx.scene, ctx.camera);
    activeAnimation = requestAnimationFrame(tick);
  };
  tick();
}

function renderMain() {
  if (!sceneData) return;
  const ctx = buildRenderer(mainCanvas);
  const group = new THREE.Group();
  if (!tileView) {
    sceneData.combined.forEach((p)=> group.add(meshFromPayload(p.mesh, p.color)));
  } else {
    sceneData.parts.forEach((p, i)=> {
      const mesh = meshFromPayload(p.mesh);
      mesh.position.x = i*120;
      group.add(mesh);
    });
  }
  ctx.scene.add(group);
  fitCamera(ctx.camera, ctx.controls, group);
  animate(ctx);
}

function startPhysicsSim() {
  if (!sceneData || tileView) {
    setStatus('Switch to Combined View to run the physics sim.');
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
  setStatus('Physics sim running...');

  const gravity = -260;
  let simDone = false;
  animate(ctx, (dt) => {
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
      setStatus('Physics sim complete. Parts dropped into place.');
    }
  });
}

function buildThumb(part, idx){
  const wrap = document.createElement('div'); wrap.className='thumb';
  wrap.innerHTML = `<strong>${part.name}</strong><div class="thumb-canvas"></div>
  <div class="thumb-controls">
  <label>Rx <input type="number" value="${part.rot[0]}" data-k="x"/></label>
  <label>Ry <input type="number" value="${part.rot[1]}" data-k="y"/></label>
  <label>Rz <input type="number" value="${part.rot[2]}" data-k="z"/></label>
  <label>Scale <input type="number" step="0.1" value="${part.scale}" data-k="scale"/></label>
  <button data-act="focus">Zoom to Main</button>
  </div>`;

  const canvas = wrap.querySelector('.thumb-canvas');
  const ctx = buildRenderer(canvas);
  const mesh = meshFromPayload(part.mesh);
  ctx.scene.add(mesh);
  fitCamera(ctx.camera, ctx.controls, mesh);
  animate(ctx);

  wrap.querySelectorAll('input').forEach((inp)=>inp.addEventListener('change', async ()=>{
    const rotation = {x:+wrap.querySelector('input[data-k="x"]').value, y:+wrap.querySelector('input[data-k="y"]').value, z:+wrap.querySelector('input[data-k="z"]').value};
    const scale = +wrap.querySelector('input[data-k="scale"]').value;
    await api(`/api/part/${idx}`, {method:'PATCH', body: JSON.stringify({rotation, scale})});
    await refreshScene();
  }));

  wrap.querySelector('[data-act="focus"]').addEventListener('click', ()=>{
    tileView = true;
    renderMain();
  });

  thumbs.appendChild(wrap);
}

function renderThumbs(){
  thumbs.innerHTML='';
  sceneData.parts.forEach(buildThumb);
}

async function refreshScene(){
  sceneData = await api('/api/scene');
  renderMain();
  renderThumbs();
}

async function refreshFiles(){
  const data = await api('/api/files');
  fileList.innerHTML='';
  data.files.forEach((f)=>{
    const row = document.createElement('div'); row.className='file-row';
    row.innerHTML=`<input type="checkbox" value="${f}"/><span>${f}</span>`;
    fileList.appendChild(row);
  });
}

document.getElementById('refresh-files').addEventListener('click', refreshFiles);
document.getElementById('load-selected').addEventListener('click', async ()=>{
  const files=[...fileList.querySelectorAll('input:checked')].map((x)=>x.value);
  await api('/api/load',{method:'POST', body:JSON.stringify({files})});
  setStatus(`Loaded ${files.length} parts.`);
  await refreshScene();
});

axisSelect.addEventListener('change', async ()=>{ await api('/api/config',{method:'PATCH', body:JSON.stringify({axis:axisSelect.value})}); await refreshScene(); });
gapInput.addEventListener('change', async ()=>{ await api('/api/config',{method:'PATCH', body:JSON.stringify({gap:+gapInput.value})}); await refreshScene(); });

document.querySelectorAll('[data-stage]').forEach((btn)=>btn.addEventListener('click', async ()=>{
  const out = await api(`/api/stage/${btn.dataset.stage}`, {method:'POST', body:'{}'});
  setStatus(out.message || 'Done');
  await refreshScene();
}));

document.getElementById('combined-view').addEventListener('click', ()=>{tileView=false; renderMain();});
document.getElementById('tile-view').addEventListener('click', ()=>{tileView=true; renderMain();});
startSimBtn.addEventListener('click', startPhysicsSim);

refreshFiles();
