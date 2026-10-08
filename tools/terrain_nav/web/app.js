// Huphy trail teleop page: terrain as surfaces (collider heights), robot from MuJoCo body poses, LiDAR scan + map.
// Server: tools/terrain_nav/trail_teleop.py.  World frame is the simulation frame, z up.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const bin = async (u) => (await fetch(u)).arrayBuffer();

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xcdd6e0);
scene.fog = new THREE.Fog(0xcdd6e0, 45, 160);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 600);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
scene.add(new THREE.HemisphereLight(0xffffff, 0x5d5040, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.7);
sun.position.set(30, -25, 60);
scene.add(sun);
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const meta = await (await fetch('/meta.json')).json();
$('stride').textContent = Math.round(meta.terrain.stride_m * 100);

// ---- whole-map surface; fragments inside the exact-collider patch are discarded so the 10 cm patch shows there
const patchRect = { value: new THREE.Vector4(1e9, 1e9, -1e9, -1e9) };
function clipOutside(mat) {
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uRect = patchRect;
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vW;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vW;\nuniform vec4 uRect;')
      .replace('void main() {', 'void main() {\nif (vW.x > uRect.x && vW.x < uRect.z && vW.y > uRect.y && vW.y < uRect.w) discard;');
  };
  return mat;
}
let terrain;
{
  const b = await bin('/terrain.bin');
  const nv = meta.terrain.nv, nf = meta.terrain.nf;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(b, 0, nv * 3), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(b, nv * 12, nv * 3), 3, true));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(b, nv * 12 + Math.ceil(nv * 3 / 4) * 4, nf * 3), 1));
  g.computeVertexNormals();
  terrain = new THREE.Mesh(g, clipOutside(new THREE.MeshLambertMaterial({ vertexColors: true })));
  scene.add(terrain);
}

// ---- grid surface (exact collider patch, LiDAR map) from a [x0,dx,y0,dy,ni,nj] float header
const gridIndex = {};
function gridGeometry(Z, ni, nj, x0, dx, y0, dy, colorOf, valid) {
  const pos = new Float32Array(ni * nj * 3), col = new Float32Array(ni * nj * 3);
  for (let i = 0; i < ni; i++) for (let j = 0; j < nj; j++) {
    const k = i * nj + j;
    pos[3 * k] = x0 + j * dx; pos[3 * k + 1] = y0 + i * dy; pos[3 * k + 2] = valid(k) ? Z[k] : 0;
    colorOf(k, col);
  }
  let idx;
  const key = `${ni}x${nj}`;
  if (!valid.sparse && gridIndex[key]) idx = gridIndex[key];
  else {
    const a = [];
    for (let i = 0; i < ni - 1; i++) for (let j = 0; j < nj - 1; j++) {
      const k = i * nj + j;
      if (valid.sparse && !(valid(k) && valid(k + 1) && valid(k + nj) && valid(k + nj + 1))) continue;
      a.push(k, k + 1, k + nj + 1, k, k + nj + 1, k + nj);
    }
    idx = new Uint32Array(a);
    if (!valid.sparse) gridIndex[key] = idx;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  return g;
}
const patch = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true }));
scene.add(patch);
let patchC = null, patchBusy = false;
async function loadPatch(x, y) {
  patchBusy = true;
  const b = await bin(`/patch.bin?x=${x.toFixed(2)}&y=${y.toFixed(2)}`);
  const h = new Float32Array(b, 0, 6), ni = h[4], nj = h[5];
  const Z = new Float32Array(b, 24, ni * nj), C = new Uint8Array(b, 24 + ni * nj * 4, ni * nj * 3);
  const all = () => true;
  patch.geometry.dispose();
  patch.geometry = gridGeometry(Z, ni, nj, h[0], h[1], h[2], h[3], (k, col) => {
    col[3 * k] = C[3 * k] / 255; col[3 * k + 1] = C[3 * k + 1] / 255; col[3 * k + 2] = C[3 * k + 2] / 255;
  }, all);
  // hide the coarse surface just inside the patch border (half a cell) so the two never overlap
  patchRect.value.set(h[0] + 0.05, h[2] + 0.05, h[0] + (nj - 1) * h[1] - 0.05, h[2] + (ni - 1) * h[3] - 0.05);
  patchC = new THREE.Vector2(x, y);
  patchBusy = false;
}

// ---- trees (collider trunks) and the planned path
const trees = new THREE.Group();
const trunkMat = new THREE.MeshLambertMaterial({ color: 0x7a5232, transparent: true, opacity: 0.7 });
for (const [x, y, z0, r, z1] of meta.trees) {
  const c = new THREE.Mesh(new THREE.CylinderGeometry(r, r, z1 - z0, 18).rotateX(Math.PI / 2), trunkMat);
  c.position.set(x, y, (z0 + z1) / 2);
  trees.add(c);
}
scene.add(trees);
const pathMesh = new THREE.Mesh(
  new THREE.TubeGeometry(new THREE.CatmullRomCurve3(meta.path.map((p) => new THREE.Vector3(...p))), meta.path.length * 2, 0.045, 6),
  new THREE.MeshBasicMaterial({ color: 0x22c55e }));
scene.add(pathMesh);

// ---- robot
const bodies = [];
{
  const b = await bin('/robot.bin');
  const R = meta.robot;
  const Vall = new Float32Array(b, 0, R.nv * 3), Fall = new Uint32Array(b, R.nv * 12, R.nf * 3);
  for (let i = 0; i < R.nbody; i++) { const g = new THREE.Group(); scene.add(g); bodies.push(g); }
  for (const e of R.geoms) {
    let geo;
    if (e.type === 'mesh') {
      geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(Vall.slice(e.vo * 3, (e.vo + e.vn) * 3), 3));
      geo.setIndex(new THREE.BufferAttribute(Fall.slice(e.fo * 3, (e.fo + e.fn) * 3), 1));
      geo.computeVertexNormals();
    } else {
      geo = new THREE.CylinderGeometry(e.size[0], e.size[0], 2 * e.size[1], 20).rotateX(Math.PI / 2);
    }
    const c = e.type === 'mesh' ? new THREE.Color(e.rgba[0], e.rgba[1], e.rgba[2]) : new THREE.Color(0x202024);
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: c, roughness: 0.6, metalness: 0.15 }));
    mesh.position.set(...e.pos);
    mesh.quaternion.set(e.quat[1], e.quat[2], e.quat[3], e.quat[0]);
    bodies[e.b].add(mesh);
  }
}

// ---- LiDAR: this scan's points + beams; terrain map surface
const scanGeo = new THREE.BufferGeometry();
const scan = new THREE.Points(scanGeo, new THREE.PointsMaterial({ size: 0.05, vertexColors: true }));
scene.add(scan);
const beamGeo = new THREE.BufferGeometry();
const beams = new THREE.LineSegments(beamGeo, new THREE.LineBasicMaterial({ color: 0xf2c94c, transparent: true, opacity: 0.55 }));
beams.visible = false;
scene.add(beams);
const elev = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide,
  polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -8 }));
elev.position.z = 0.03; // drawn just above the terrain surface it was measured from
elev.visible = false;
scene.add(elev);
let lidarSeq = -1;
async function updateScan() {
  const b = await bin('/lidar.bin');
  if (b.byteLength < 8) return;
  const [n, nb] = new Uint32Array(b, 0, 2);
  const P = new Float32Array(b, 8, n * 3), B = new Float32Array(b, 8 + n * 12, nb * 6), G = new Uint8Array(b, 8 + n * 12 + nb * 24, n);
  const col = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    if (G[k]) { col[3 * k] = 0.20; col[3 * k + 1] = 0.78; col[3 * k + 2] = 1.0; } else { col[3 * k] = 1.0; col[3 * k + 1] = 0.48; col[3 * k + 2] = 0.10; }
  }
  scanGeo.setAttribute('position', new THREE.BufferAttribute(P.slice(), 3));
  scanGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  scanGeo.computeBoundingSphere();
  beamGeo.setAttribute('position', new THREE.BufferAttribute(B.slice(), 3));
  beamGeo.computeBoundingSphere();
}
function heightColor(t, out, k) { // blue (low) -> green -> yellow -> red (high), t in 0..1
  const s = Math.min(Math.max(t, 0), 1) * 3;
  const r = Math.min(Math.max(s - 1, 0), 1), g = s < 2 ? Math.min(s, 1) : Math.max(3 - s, 0), b = Math.max(1 - s, 0);
  out[3 * k] = 0.15 + 0.85 * r; out[3 * k + 1] = 0.2 + 0.75 * g; out[3 * k + 2] = 0.25 + 0.75 * b;
}
async function updateElev(x, y, zc) {
  const b = await bin(`/elev.bin?x=${x.toFixed(2)}&y=${y.toFixed(2)}`);
  const h = new Float32Array(b, 0, 6), ni = h[4], nj = h[5];
  const Z = new Float32Array(b, 24, ni * nj), O = new Uint8Array(b, 24 + ni * nj * 4, ni * nj);
  const valid = (k) => !Number.isNaN(Z[k]);
  valid.sparse = true;
  elev.geometry.dispose();
  elev.geometry = gridGeometry(Z, ni, nj, h[0], h[1], h[2], h[3], (k, col) => {
    if (O[k]) { col[3 * k] = 1.0; col[3 * k + 1] = 0.45; col[3 * k + 2] = 0.1; } else heightColor((Z[k] - zc + 2.0) / 4.0, col, k);
  }, valid);
}

// ---- controls: keyboard (on this page), buttons, sliders
const KEYS = { w: 'dvx=0.1', arrowup: 'dvx=0.1', s: 'dvx=-0.1', arrowdown: 'dvx=-0.1', a: 'dwz=0.1', arrowleft: 'dwz=0.1',
               d: 'dwz=-0.1', arrowright: 'dwz=-0.1', ' ': 'stop=1', r: 'reset=1' };
const cmd = (q) => fetch('/cmd?' + q);
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type === 'range') return;
  const q = KEYS[e.key.toLowerCase()];
  if (q) { e.preventDefault(); cmd(q); }
});
document.querySelectorAll('button[data-k]').forEach((b) => b.addEventListener('click', () => cmd(KEYS[b.dataset.k])));
$('start').onclick = () => cmd('start=1');
$('clear').onclick = () => cmd('clear=1');
let dragging = false;
for (const id of ['vx', 'wz']) {
  $(id).addEventListener('pointerdown', () => (dragging = true));
  $(id).addEventListener('pointerup', () => (dragging = false));
  $(id).addEventListener('input', () => cmd(`${id}=${$(id).value}`));
}
const show = (id, obj) => { const f = () => (obj.visible = $(id).checked); $(id).onchange = f; f(); };
show('s_terrain', terrain); show('s_patch', patch); show('s_trees', trees); show('s_path', pathMesh);
show('s_elev', elev); show('s_scan', scan); show('s_beams', beams);

// ---- joystick: a gamepad (browser Gamepad API - only on secure pages, hence the https port) or the on-screen stick.
// Both give an absolute command: stick up = forward up to 1.2 m/s, down = backward up to 0.4 m/s, sideways = turn 0.6 rad/s.
const DEAD = 0.12;
const dz = (v) => (Math.abs(v) < DEAD ? 0 : (v - Math.sign(v) * DEAD) / (1 - DEAD));
const stickToCmd = (fwd, turn) => [fwd > 0 ? fwd * 1.2 : fwd * 0.4, -turn * 0.6];
let sent = [0, 0], tSent = 0, padActive = false;
function sendStick(vx, wz) {
  const now = performance.now();
  const moved = Math.abs(vx - sent[0]) > 0.02 || Math.abs(wz - sent[1]) > 0.02;
  const back = (vx === 0 && wz === 0) && (sent[0] !== 0 || sent[1] !== 0);
  if ((moved && now - tSent > 50) || back) {
    cmd(`vx=${vx.toFixed(2)}&wz=${wz.toFixed(2)}`);
    sent = [vx, wz]; tSent = now;
  }
}
const prevBtn = [];
function pollPad() {
  const pads = navigator.getGamepads ? [...navigator.getGamepads()].filter(Boolean) : [];
  if (!window.isSecureContext) {
    $('pad').innerHTML = `<span class="bad">gamepad needs the https page:</span> <a href="https://${location.hostname}:8097/">https://${location.hostname}:8097/</a> (accept the certificate warning once). The on-screen stick works here.`;
  } else if (!pads.length) {
    $('pad').innerHTML = 'no gamepad yet - plug it in and <b>press any button</b>';
  } else {
    const g = pads[0], ax = g.axes;
    const fwd = -dz(ax[1] || 0), rx = dz(ax[2] || 0), lx = dz(ax[0] || 0);
    const turn = Math.abs(rx) > 0 ? rx : lx;
    const [vx, wz] = stickToCmd(fwd, turn);
    if (fwd !== 0 || turn !== 0 || padActive) sendStick(vx, wz);
    padActive = fwd !== 0 || turn !== 0;
    const press = (i) => g.buttons[i] && g.buttons[i].pressed && !prevBtn[i];
    if (press(0)) cmd('reset=1');
    if (press(1)) { cmd('stop=1'); sent = [0, 0]; }
    if (press(2)) { $('s_elev').checked = !$('s_elev').checked; $('s_elev').onchange(); }
    if (press(3)) $('s_follow').checked = !$('s_follow').checked;
    if (press(9)) cmd('start=1');
    g.buttons.forEach((b, i) => (prevBtn[i] = b.pressed));
    $('pad').innerHTML = `<span class="ok">${g.id.slice(0, 40)}</span><br>stick: forward ${fwd.toFixed(2)}, turn ${turn.toFixed(2)} → ${vx.toFixed(2)} m/s, ${wz.toFixed(2)} rad/s`;
  }
  requestAnimationFrame(pollPad);
}
pollPad();
{ // on-screen stick (mouse or touch)
  const base = $('vstick'), knob = $('knob');
  let id = null;
  const move = (e) => {
    const r = base.getBoundingClientRect(), R = r.width / 2;
    let x = (e.clientX - r.left - R) / R, y = (e.clientY - r.top - R) / R;
    const n = Math.hypot(x, y);
    if (n > 1) { x /= n; y /= n; }
    knob.style.transform = `translate(${x * R * 0.7}px, ${y * R * 0.7}px)`;
    const [vx, wz] = stickToCmd(-dz(y), dz(x));
    sendStick(vx, wz);
  };
  base.addEventListener('pointerdown', (e) => { id = e.pointerId; base.setPointerCapture(id); move(e); });
  base.addEventListener('pointermove', (e) => { if (e.pointerId === id) move(e); });
  const end = (e) => { if (e.pointerId !== id) return; id = null; knob.style.transform = ''; sendStick(0, 0); };
  base.addEventListener('pointerup', end);
  base.addEventListener('pointercancel', end);
}

// ---- state loop
const pRobot = new THREE.Vector3();
let lastP = null, first = true, elevBusy = false, tElev = 0;
const fmtFeet = (f) => (f.length ? f.map((k) => `<span class="ok">${k}</span>`).join(' + ') : '<span class="dim">in the air</span>');
async function poll() {
  for (;;) {
    let s;
    try { s = await (await fetch('/state')).json(); } catch { $('status').innerHTML = '<span class="bad">disconnected</span>'; await sleep(1000); continue; }
    for (let i = 0; i < bodies.length; i++) {
      const o = 7 * i, v = s.bodies;
      bodies[i].position.set(v[o], v[o + 1], v[o + 2]);
      bodies[i].quaternion.set(v[o + 4], v[o + 5], v[o + 6], v[o + 3]);
    }
    pRobot.set(...s.p);
    if (first) {
      camera.position.set(s.p[0] - 4.5 * Math.cos(s.yaw), s.p[1] - 4.5 * Math.sin(s.yaw), s.p[2] + 2.2);
      controls.target.set(s.p[0], s.p[1], s.p[2] + 0.1);
      first = false;
      $('load').remove();
    }
    if ($('s_follow').checked && lastP) {
      camera.position.add(pRobot.clone().sub(lastP));
      controls.target.set(s.p[0], s.p[1], s.p[2] + 0.1);
    }
    lastP = pRobot.clone();
    if (!patchBusy && (!patchC || patchC.distanceTo(new THREE.Vector2(s.p[0], s.p[1])) > 7)) loadPatch(s.p[0], s.p[1]);
    if (!dragging) { $('vx').value = s.vx; $('wz').value = s.wz; }
    $('vxv').textContent = s.vx.toFixed(2); $('wzv').textContent = s.wz.toFixed(2);
    $('status').innerHTML = (s.fallen ? '<span class="bad">FALLEN - press R (get up here)</span>' : `<span class="ok">walking - RL policy ${meta.policy}</span>`) +
      `<br>command ${s.vx.toFixed(2)} m/s, turn ${s.wz.toFixed(2)} rad/s<br>walked ${s.walked.toFixed(1)} m · base ${s.above.toFixed(2)} m above ground · sim ${s.t.toFixed(0)} s`;
    const c = s.contact;
    $('contact').innerHTML = `left foot: ${fmtFeet(s.feet.L)}<br>right foot: ${fmtFeet(s.feet.R)}<br>` +
      `force on trail heightfield <b>${c.heightfield} N</b> (robot weight ${meta.weight} N)<br>` +
      `tree trunks ${c.tree} N · old flat floor ${c['flat floor']} N`;
    const L = s.lidar;
    $('lidar').innerHTML = `${meta.lidar.beams} beams × 360 = ${L.n.toLocaleString()} rays, ${L.hz.toFixed(1)} scans/s<br>` +
      `returns ${meta.lidar.rmin}-${meta.lidar.rmax} m: ${L.ret.toLocaleString()} → 10 cm voxel: ${L.kept.toLocaleString()}<br>` +
      `ground <span style="color:#1d9bd6">${L.ground.toLocaleString()}</span> · obstacle <span style="color:#e8710a">${L.obst.toLocaleString()}</span> · dropped on unsurveyed walls ${L.wall}<br>` +
      `terrain map: ${L.cells.toLocaleString()} cells of 10 cm`;
    if ((scan.visible || beams.visible) && L.seq !== lidarSeq) { lidarSeq = L.seq; updateScan(); }
    if (elev.visible && !elevBusy && performance.now() - tElev > 1000) {
      elevBusy = true; tElev = performance.now();
      updateElev(s.p[0], s.p[1], s.p[2] - s.above).finally(() => (elevBusy = false));
    }
    await sleep(30);
  }
}
poll();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
