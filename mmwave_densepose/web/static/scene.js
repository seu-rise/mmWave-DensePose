/** Dataset canvas presentation copied from the current docs/index.html Pages viewer.
 * Stored XYZ is passed through unchanged. FPS, image decoding and UI live elsewhere.
 */
import * as THREE from "./vendor/three.module.js";
import {PaperViewControls} from "./paper-view-controls.js";

export const PAGES_VIEW_CONFIG = {
  rotX: 76, rotY: 0, rotZ: 0, distance: 7, targetOffsetZ: -0.95,
  xlim: [-2, 2], ylim: [0.15, 7.15], zlim: [-0.1, 4],
};

export const PAGES_RENDER_CONFIG = {
  camera: {fov: 45, near: 0.01, far: 5000},
  scene: {backgroundColor: 0x111111},
  lighting: {
    ambient: {color: 0xffffff, intensity: 0.9},
    directional: {color: 0xffffff, intensity: 1.15, position: [3, 5, 4]},
  },
  points: {size: 0.3, opacity: 0.95, alphaTest: 0.35, defaultColor: "#ff8d1a",
    fallbackColor: "#4ec3ff", circleTextureSize: 64},
  keypoints: {size: 0.4, glowScale: 2, mainOpacity: 1, mainAlphaTest: 0.02,
    glowOpacity: 0.48, glowAlphaTest: 0.01, glowTextureSize: 128, color: "#00fa9a"},
  bones: {color: 0x8dc8ff, opacity: 0.9},
  helpers: {axisScaleRatio: 0.22, boundsColor: 0x00bfff, boundsOpacity: 0.35,
    groundPlaneColor: 0x1f2d3d, groundPlaneOpacity: 0.22, gridDivisions: 20,
    gridColorCenter: 0x4f6f8f, gridColor: 0x2f4154, gridOpacity: 0.55, gridZOffset: 0},
  palette: ["#2b005c", "#0057d9", "#00d5ff", "#f8f32b", "#ff7a00", "#b00020"],
};

function finitePoint(point) {
  return point && point.length >= 3 && point.slice(0, 3).every(Number.isFinite);
}

function createCircleTexture(size) {
  const canvas = document.createElement("canvas");
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, size, size);
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size * 0.45, 0, Math.PI * 2);
  ctx.closePath(); ctx.fillStyle = "#ffffff"; ctx.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

export function createGlowTexture(size) {
  const canvas = document.createElement("canvas");
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext("2d"), center = size * 0.5;
  const gradient = ctx.createRadialGradient(center, center, 0, center, center, center);
  gradient.addColorStop(0, "rgba(255,255,255,1.0)");
  gradient.addColorStop(0.35, "rgba(255,255,255,0.95)");
  gradient.addColorStop(0.7, "rgba(255,255,255,0.28)");
  gradient.addColorStop(1, "rgba(255,255,255,0.0)");
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function geometry() {
  const result = new THREE.BufferGeometry();
  result.setAttribute("position", new THREE.Float32BufferAttribute([], 3));
  result.setDrawRange(0, 0);
  return result;
}

function setPositions(target, values, colors) {
  // Geometry objects and materials persist; attributes only grow to peak demand.
  const count = values.length / 3;
  let attribute = target.getAttribute("position");
  if (attribute.count < count) {
    let capacity = Math.max(1, attribute.count);
    while (capacity < count) capacity *= 2;
    attribute = new THREE.BufferAttribute(new Float32Array(capacity * 3), 3);
    attribute.setUsage(THREE.DynamicDrawUsage);
    target.setAttribute("position", attribute);
    if (colors) target.setAttribute("color", new THREE.BufferAttribute(new Float32Array(capacity * 3), 3)
      .setUsage(THREE.DynamicDrawUsage));
  }
  attribute.array.set(values); attribute.needsUpdate = true;
  if (colors) {
    let color = target.getAttribute("color");
    if (!color) {
      color = new THREE.BufferAttribute(new Float32Array(attribute.count * 3), 3);
      color.setUsage(THREE.DynamicDrawUsage); target.setAttribute("color", color);
    }
    color.array.set(colors); color.needsUpdate = true;
  }
  target.setDrawRange(0, count);
  // Three.js also uses boundingSphere.center to sort transparent objects. The
  // Pages viewer computes it from exact-sized position arrays, so spare/stale
  // capacity must not affect either sort depth or the visible glow layering.
  // Match BufferGeometry.computeBoundingSphere's box-center/max-radius method
  // using only the active Float32 vertices while retaining reusable attributes.
  if (target.boundingSphere === null) target.boundingSphere = new THREE.Sphere();
  const bounds = new THREE.Box3(), vertex = new THREE.Vector3();
  for (let index = 0; index < count; index++) {
    bounds.expandByPoint(vertex.fromBufferAttribute(attribute, index));
  }
  bounds.getCenter(target.boundingSphere.center);
  let radiusSquared = 0;
  for (let index = 0; index < count; index++) {
    vertex.fromBufferAttribute(attribute, index);
    radiusSquared = Math.max(radiusSquared, target.boundingSphere.center.distanceToSquared(vertex));
  }
  target.boundingSphere.radius = Math.sqrt(radiusSquared);
}

export class SpatialView {
  constructor(container) {
    const config = PAGES_RENDER_CONFIG;
    this.container = container; this.disposed = false;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(config.scene.backgroundColor);
    this.camera = new THREE.PerspectiveCamera(config.camera.fov, 1, config.camera.near, config.camera.far);
    this.camera.up.set(0,0,1);
    this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    // Pages deliberately uses the renderer's default NoToneMapping/exposure1.
    this.renderer.domElement.setAttribute("aria-label", "3D radar and skeleton view");
    this.renderer.domElement.dataset.renderer = "three-r183";
    container.prepend(this.renderer.domElement);
    this.controls = new PaperViewControls(this.camera, this.renderer.domElement);
    this.controls.addEventListener('change',()=>this.render());
    this.renderer.domElement.title='Drag to rotate · Pinch or scroll to zoom · Double-click to reset';
    this.scene.add(new THREE.AmbientLight(config.lighting.ambient.color, config.lighting.ambient.intensity));
    const light = new THREE.DirectionalLight(config.lighting.directional.color, config.lighting.directional.intensity);
    light.position.set(...config.lighting.directional.position); this.scene.add(light);

    this.pointTexture = createCircleTexture(config.points.circleTextureSize);
    this.jointTexture = createGlowTexture(config.keypoints.glowTextureSize);
    this.pointGeometry = geometry();
    this.pointMaterial = new THREE.PointsMaterial({
      size: config.points.size, map: this.pointTexture, alphaMap: this.pointTexture,
      vertexColors: true, transparent: true, opacity: config.points.opacity,
      alphaTest: config.points.alphaTest, sizeAttenuation: true,
    });
    this.points = new THREE.Points(this.pointGeometry, this.pointMaterial);
    this.points.frustumCulled = false; this.scene.add(this.points);
    this.skeleton = new THREE.Group(); this.scene.add(this.skeleton);
    this.jointGeometry = geometry();
    this.jointMaterial = new THREE.PointsMaterial({
      size: config.keypoints.size, map: this.jointTexture, alphaMap: this.jointTexture,
      vertexColors: true, transparent: true, opacity: config.keypoints.mainOpacity,
      alphaTest: config.keypoints.mainAlphaTest, depthWrite: false, sizeAttenuation: true,
    });
    this.jointPoints = new THREE.Points(this.jointGeometry, this.jointMaterial);
    this.jointPoints.frustumCulled = false;
    this.jointGlowMaterial = new THREE.PointsMaterial({
      size: config.keypoints.size * config.keypoints.glowScale,
      map: this.jointTexture, alphaMap: this.jointTexture, vertexColors: true,
      transparent: true, opacity: config.keypoints.glowOpacity, alphaTest: config.keypoints.glowAlphaTest,
      depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
    });
    this.jointGlow = new THREE.Points(this.jointGeometry, this.jointGlowMaterial);
    this.jointGlow.frustumCulled = false;
    this.boneGeometry = geometry();
    this.boneMaterial = new THREE.LineBasicMaterial({
      color: config.bones.color, transparent: true, opacity: config.bones.opacity,
    });
    this.bones = new THREE.LineSegments(this.boneGeometry, this.boneMaterial);
    this.bones.frustumCulled = false;
    this.skeleton.add(this.jointPoints, this.jointGlow, this.bones);

    this.axes = new THREE.AxesHelper(1); this.scene.add(this.axes);
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({
      color: config.helpers.groundPlaneColor, transparent: true, opacity: config.helpers.groundPlaneOpacity,
      side: THREE.DoubleSide, depthWrite: false,
    }));
    this.scene.add(this.ground);
    this.grid = new THREE.GridHelper(1, config.helpers.gridDivisions,
      config.helpers.gridColorCenter, config.helpers.gridColor);
    this.grid.rotation.x = Math.PI / 2;
    for (const material of Array.isArray(this.grid.material) ? this.grid.material : [this.grid.material]) {
      material.transparent = true; material.opacity = config.helpers.gridOpacity;
    }
    this.scene.add(this.grid);
    // Pages' fixed scene extent is a presentation helper, not the person's box.
    this.sceneBounds = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({color: config.helpers.boundsColor, transparent: true, opacity: config.helpers.boundsOpacity}));
    this.scene.add(this.sceneBounds);
    this.boundary = new THREE.Box3Helper(new THREE.Box3(), config.helpers.boundsColor);
    this.boundary.material.transparent = true;
    this.boundary.material.opacity = 0.65;
    this.boundary.visible = false;
    this.boundaryVertex = new THREE.Vector3();
    this.scene.add(this.boundary);
    this.options = {color: "solid", radar: true, skeleton: true, boundary: true,
      pointSize: 1, dopplerMin: -150, dopplerMax: 150};
    this.scalarPalette = config.palette.map(value => new THREE.Color(value));
    this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(container);
    this.reset(); this.resize();
    const animate = () => {
      if (this.disposed) return;
      this.controls.update(); this.render();
      this.animationId = requestAnimationFrame(animate);
    };
    this.animationId = requestAnimationFrame(animate);
  }

  updateBounds() {
    const config = PAGES_VIEW_CONFIG, style = PAGES_RENDER_CONFIG.helpers;
    const [sx, sy, sz] = ["xlim", "ylim", "zlim"].map(key => Math.max(0.01, config[key][1] - config[key][0]));
    const [cx, cy, cz] = ["xlim", "ylim", "zlim"].map(key => (config[key][0] + config[key][1]) * 0.5);
    this.sceneBounds.scale.set(sx, sy, sz); this.sceneBounds.position.set(cx, cy, cz);
    this.ground.scale.set(sx, sy, 1); this.ground.position.set(cx, cy, 0);
    this.grid.scale.set(sx, 1, sy); this.grid.position.set(cx, cy, style.gridZOffset);
    const length = Math.max(0.5, Math.max(sx, sy, sz) * style.axisScaleRatio);
    this.axes.position.set(config.xlim[0], config.ylim[0], config.zlim[0]);
    this.axes.scale.setScalar(length);
  }

  reset() {
    const config = PAGES_VIEW_CONFIG;
    const center = new THREE.Vector3((config.xlim[0] + config.xlim[1]) * 0.5,
      (config.ylim[0] + config.ylim[1]) * 0.5,
      (config.zlim[0] + config.zlim[1]) * 0.5 + config.targetOffsetZ);
    const euler = new THREE.Euler(...[config.rotX, config.rotY, config.rotZ].map(THREE.MathUtils.degToRad), "XYZ");
    this.camera.position.copy(center).add(new THREE.Vector3(0, 0, config.distance).applyEuler(euler));
    this.camera.up.set(0,0,1);this.camera.zoom=1;
    this.camera.lookAt(center); this.controls.target.copy(center);
    const span = Math.max(config.xlim[1] - config.xlim[0], config.ylim[1] - config.ylim[0],
      config.zlim[1] - config.zlim[0], 1);
    this.camera.near = Math.max(0.01, config.distance * 0.001);
    this.camera.far = Math.max(100, config.distance + span * 8);
    this.camera.updateProjectionMatrix(); this.controls.update(); this.controls.saveState(); this.updateBounds(); this.render();
  }

  resize() {
    const width = this.container.clientWidth, height = this.container.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false); this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix(); this.render();
  }

  fit() {
    if (!this.frame) return;
    const values = [...this.frame.radar, ...(this.frame.joints || [])].filter(finitePoint);
    if (!values.length) return;
    const bounds = new THREE.Box3().setFromPoints(values.map(value => new THREE.Vector3(...value.slice(0, 3))));
    const center = bounds.getCenter(new THREE.Vector3()), size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z, 0.5) * 0.75 + 0.25;
    const distance = Math.max(this.controls.minDistance,
      radius / Math.sin(this.camera.fov * Math.PI / 360) / Math.min(this.camera.aspect, 1));
    const direction = this.camera.position.clone().sub(this.controls.target).normalize();
    this.controls.target.copy(center); this.camera.position.copy(center).addScaledVector(direction, distance);this.camera.zoom=1;
    this.camera.updateProjectionMatrix();this.controls.update();this.controls.saveState();this.render();
  }

  colorFromValue(value, low, high) {
    const span = high - low;
    const amount = span <= 1e-8 ? 0.5 : THREE.MathUtils.clamp((value - low) / span, 0, 1);
    const scaled = amount * (this.scalarPalette.length - 1), index = Math.floor(scaled), local = scaled - index;
    return this.scalarPalette[Math.min(index, this.scalarPalette.length - 1)].clone()
      .lerp(this.scalarPalette[Math.min(index + 1, this.scalarPalette.length - 1)], local);
  }

  setFrame(frame, summary) {
    this.frame = frame; this.summary = summary;
    this.setOptions({});
    this.renderer.domElement.dataset.frameIndex = String(frame.index);
    this.renderer.domElement.dataset.generation = String(frame.generation);
  }

  setOptions(options) {
    Object.assign(this.options, options);
    const opts = this.options, config = PAGES_RENDER_CONFIG;
    this.points.visible = opts.radar; this.skeleton.visible = opts.skeleton;
    this.boundary.visible = false;
    this.pointMaterial.size = config.points.size * opts.pointSize;
    this.jointMaterial.size = config.keypoints.size * opts.pointSize;
    this.jointGlowMaterial.size = config.keypoints.size * config.keypoints.glowScale * opts.pointSize;
    if (this.frame) {
      const points = this.frame.radar.filter(finitePoint), channel = {doppler: 3, snr: 4}[opts.color];
      const available = channel !== undefined && points.some(point => point.length > channel);
      const [low, high] = channel === 3 ? [opts.dopplerMin, opts.dopplerMax] : [0, 50];
      const positions = [], colors = [];
      for (const point of points) {
        positions.push(...point.slice(0, 3));
        const value = point[channel];
        const color = channel === undefined ? new THREE.Color(config.points.defaultColor)
          : !Number.isFinite(value) ? new THREE.Color(config.points.fallbackColor)
          : this.colorFromValue(value, low, high);
        colors.push(color.r, color.g, color.b);
      }
      setPositions(this.pointGeometry, positions, colors);
      const joints = this.frame.joints || [], valid = joints.filter(finitePoint), jointPositions = [], jointColors = [];
      // Only the current frame's finite GT joints define the human box. Radar
      // outliers and the configured scene extents must not change its bounds.
      this.boundary.box.makeEmpty();
      for (const point of valid) this.boundary.box.expandByPoint(this.boundaryVertex.set(...point));
      if (valid.length) {
        this.boundary.box.expandByScalar(0.2);
        this.boundary.visible = opts.boundary;
      }
      const jointColor = new THREE.Color(config.keypoints.color);
      for (const point of valid) { jointPositions.push(...point); jointColors.push(jointColor.r, jointColor.g, jointColor.b); }
      setPositions(this.jointGeometry, jointPositions, jointColors);
      const lines = [];
      for (const [start, end] of this.frame.edges || []) {
        if (finitePoint(joints[start]) && finitePoint(joints[end])) lines.push(...joints[start], ...joints[end]);
      }
      setPositions(this.boneGeometry, lines);
      this.legend = {available, low, high, mode: opts.color, stops: config.palette};
      this.renderer.domElement.dataset.points = String(points.length);
      this.renderer.domElement.dataset.joints = String(valid.length);
    }
    this.renderer.domElement.dataset.pointStyle = JSON.stringify({
      primitive: "points_sprite", baseSize: config.points.size, sizeScale: opts.pointSize,
      size: this.pointMaterial.size, material: this.pointMaterial.type,
      opacity: this.pointMaterial.opacity, alphaTest: this.pointMaterial.alphaTest,
      sizeAttenuation: this.pointMaterial.sizeAttenuation, depthWrite: this.pointMaterial.depthWrite,
      toneMapped: this.pointMaterial.toneMapped, reference: "docs/index.html",
    });
    this.renderer.domElement.dataset.skeletonStyle = JSON.stringify({
      joints: "Points", jointMaterial: this.jointMaterial.type, size: this.jointMaterial.size,
      glowSize: this.jointGlowMaterial.size, glowBlending: this.jointGlowMaterial.blending,
      color: config.keypoints.color, boneColor: config.bones.color, lineOpacity: this.boneMaterial.opacity,
    });
    this.renderer.domElement.dataset.layers = JSON.stringify({
      radar: this.points.visible, skeleton: this.skeleton.visible, boundary: this.boundary.visible,
      sceneBounds: this.sceneBounds.visible,
    });
    this.render();
  }

  render() {
    if (this.disposed) return;
    this.renderer.render(this.scene, this.camera);
    this.renderer.domElement.dataset.camera = JSON.stringify(this.camera.position.toArray());
    this.renderer.domElement.dataset.cameraTarget = JSON.stringify(this.controls.target.toArray());
    this.renderer.domElement.dataset.cameraUp = JSON.stringify(this.camera.up.toArray());
    this.renderer.domElement.dataset.cameraZoom = String(this.camera.zoom);
    this.renderer.domElement.dataset.viewControls = 'paper-orbit';
    this.renderer.domElement.dataset.ready = "true";
  }

  clear() {
    this.frame = null; this.summary = null;
    this.boundary.box.makeEmpty(); this.boundary.visible = false;
    for (const target of [this.pointGeometry, this.jointGeometry, this.boneGeometry]) {
      target.setDrawRange(0, 0);
      if (target.boundingSphere) target.boundingSphere.set(new THREE.Vector3(), 0);
    }
    this.renderer.domElement.dataset.points = "0"; this.renderer.domElement.dataset.joints = "0";
    delete this.renderer.domElement.dataset.frameIndex; delete this.renderer.domElement.dataset.generation;
    this.render();
  }

  dispose() {
    this.disposed = true; cancelAnimationFrame(this.animationId);
    this.observer.disconnect(); this.controls.dispose();
    const geometries = new Set(), materials = new Set(), textures = new Set();
    this.scene.traverse(object => {
      if (object.geometry) geometries.add(object.geometry);
      if (object.material) for (const material of Array.isArray(object.material) ? object.material : [object.material]) materials.add(material);
    });
    for (const item of geometries) item.dispose();
    for (const item of materials) {
      if (item.map) textures.add(item.map);
      if (item.alphaMap) textures.add(item.alphaMap);
      item.dispose();
    }
    for (const item of textures) item.dispose();
    this.renderer.dispose();
  }
}
