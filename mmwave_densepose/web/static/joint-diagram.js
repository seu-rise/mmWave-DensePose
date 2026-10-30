import * as THREE from "./vendor/three.module.js";
import {PaperViewControls} from "./paper-view-controls.js";

// Figure 7 topology. This pose is a diagram, not a predicted test frame.
const POSE = [
  [0, .89, 0], [0, 1.15, 0], [0, 1.67, 0], [0, 1.88, 0],
  [-.29, 1.46, 0], [-.45, 1.24, .015], [-.57, .96, .04],
  [.29, 1.46, 0], [.46, 1.24, -.015], [.6, .98, .035],
  [-.21, .73, 0], [-.27, .43, .015], [-.33, .14, 0], [-.47, .055, .11],
  [.21, .73, 0], [.27, .43, -.015], [.33, .14, 0], [.47, .055, .11],
  [0, 1.53, 0],
];
const BONES = [[0,1],[1,18],[18,2],[2,3],[18,4],[4,5],[5,6],[18,7],[7,8],[8,9],[0,10],[10,11],[11,12],[12,13],[0,14],[14,15],[15,16],[16,17]];
const NAMES = ["Spine base","Spine mid","Neck","Head","Shoulder L","Elbow L","Hand L","Shoulder R","Elbow R","Hand R","Hip L","Knee L","Ankle L","Foot L","Hip R","Knee R","Ankle R","Foot R","Spine shoulder"];
const SVG_NS = "http://www.w3.org/2000/svg";
const ERROR_COLORS = ["#43d99a", "#a6d873", "#edd269", "#f5a259", "#f27058"];
const MISSING_COLOR = "#738394";

export function jointErrorScale(values) {
  const valid = values.filter(value => Number.isFinite(value) && value >= 0);
  const min = valid.length ? Math.min(...valid) : null;
  const max = valid.length ? Math.max(...valid) : null;
  function color(value) {
    if (!Number.isFinite(value) || value < 0 || min === null) return MISSING_COLOR;
    const t = max === min ? 0 : Math.max(0, Math.min(1, (value - min) / (max - min)));
    const position = t * (ERROR_COLORS.length - 1), index = Math.min(Math.floor(position), ERROR_COLORS.length - 2);
    const channels = hex => [1,3,5].map(offset => parseInt(hex.slice(offset,offset+2),16));
    const from = channels(ERROR_COLORS[index]), to = channels(ERROR_COLORS[index+1]);
    return "#" + from.map((channel,i) => Math.round(channel+(to[i]-channel)*(position-index)).toString(16).padStart(2,"0")).join("");
  }
  return {min,max,color,gradient:min === null ? MISSING_COLOR : max === min ? ERROR_COLORS[0] : `linear-gradient(90deg, ${ERROR_COLORS.join(", ")})`};
}

function element(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function ensureStyles() {
  if (document.getElementById("joint-diagram-css")) return;
  const link = document.createElement("link");
  link.id = "joint-diagram-css";
  link.rel = "stylesheet";
  link.href = new URL("./joint-diagram.css", import.meta.url).href;
  document.head.append(link);
}

function glowTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const context = canvas.getContext("2d");
  const gradient = context.createRadialGradient(32,32,0,32,32,32);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(.2, "rgba(255,255,255,.8)");
  gradient.addColorStop(.5, "rgba(255,255,255,.18)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  context.fillStyle = gradient;
  context.fillRect(0,0,64,64);
  return new THREE.CanvasTexture(canvas);
}

/** Mount the 19-joint schematic using per-joint MLE values in centimeters. */
export function mountJointDiagram(container, {rows = []} = {}) {
  ensureStyles();
  const root = element("div", "joint-diagram");
  root.dataset.jointDiagram = "";
  root.append(element("span", "joint-diagram-label", "19-joint layout"));
  const viewport = element("div", "joint-diagram-viewport");
  const stage = element("div", "joint-diagram-stage");
  const overlay = document.createElementNS(SVG_NS, "svg");
  overlay.classList.add("joint-diagram-leaders");
  overlay.setAttribute("aria-hidden", "true");
  viewport.append(stage, overlay);root.append(viewport);
  container.append(root);

  const byIndex = new Map();
  rows.forEach((row, fallback) => {
    const index = Number(row.index ?? row.joint_index ?? fallback);
    if (Number.isInteger(index) && index >= 0 && index < 19) byIndex.set(index,row);
  });
  const scale = jointErrorScale([...byIndex.values()].map(row => row.value));
  const legend = element("div", "joint-error-legend");
  legend.setAttribute("role","img");
  legend.setAttribute("aria-label",scale.min === null ? "Joint MLE unavailable" : `Joint MLE in centimeters, ${scale.min.toFixed(2)} to ${scale.max.toFixed(2)}, green is lower and orange-red is higher`);
  const legendHead = element("div","joint-error-legend-head");
  legendHead.append(element("span","joint-error-legend-title","MLE · cm"),element("span","joint-error-legend-direction",scale.min === null ? "No joint errors" : "Low → High"));
  const ramp = element("div","joint-error-ramp");ramp.style.background=scale.gradient;
  const ticks = element("div","joint-error-ticks");
  const tickValues = scale.min === null ? [null,null,null] : [scale.min,(scale.min+scale.max)/2,scale.max];
  tickValues.forEach((value,index)=>{const tick=element("span","",value === null ? "—" : value.toFixed(2));if(scale.min === scale.max && index !== 0)tick.textContent="";ticks.append(tick)});
  legend.append(legendHead,ramp,ticks);root.append(legend);
  const labels = POSE.map((_, index) => {
    const row = byIndex.get(index);
    const value = Number.isFinite(row?.value) && row.value >= 0 ? row.value : null;
    const color = scale.color(value);
    const name = row?.label || NAMES[index];
    const label = element("button", "joint-diagram-callout");
    label.type = "button";
    label.dataset.jointIndex = String(index);
    label.dataset.errorColor = color;
    label.style.setProperty("--joint-error-color",color);
    label.setAttribute("aria-label", `${name}, MLE ${value === null ? "unavailable" : `${value.toFixed(2)} centimeters`}`);
    label.title = `${name} · ${value === null ? "—" : `${value.toFixed(2)} cm`}`;
    label.append(element("span", "joint-diagram-index", String(index).padStart(2,"0")), element("span", "joint-diagram-name", name), element("strong", "joint-diagram-value", value === null ? "—" : value.toFixed(2)), element("small", "joint-diagram-unit", "cm"));
    const line = document.createElementNS(SVG_NS,"path");
    line.dataset.jointLeader = String(index);
    line.style.setProperty("--joint-error-color",color);
    overlay.append(line);
    viewport.append(label);
    return {index,label,line,value,color};
  });

  let renderer = null, controls = null, observer = null, active = true, disposed = false, dragging = false;
  let frame = 0, focused = -1, hovered = -1, highlighted = -1;
  let width = 0, height = 0, projected = [];
  let fallback = null, fallbackBones = null;
  const fallbackNodes = [];
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1.5,1.5,1.2,-1.2,.01,20);
  camera.position.set(0,.97,5);
  camera.lookAt(0,.97,0);
  const nodes = [], resources = new Set(), textures = new Set();
  const own = object => { resources.add(object); return object; };
  const points = POSE.map(point => new THREE.Vector3(...point));
  const nodeGeometry = own(new THREE.SphereGeometry(.021,16,12));
  const haloGeometry = own(new THREE.RingGeometry(.031,.035,32));
  const haloTexture = glowTexture();
  textures.add(haloTexture);

  for (const [index, position] of points.entries()) {
    const color = labels[index].color;
    const material = own(new THREE.MeshBasicMaterial({color}));
    const mesh = new THREE.Mesh(nodeGeometry,material);
    mesh.position.copy(position);
    const haloMaterial = own(new THREE.MeshBasicMaterial({color,transparent:true,opacity:.65,side:THREE.DoubleSide}));
    const ring = new THREE.Mesh(haloGeometry,haloMaterial);
    ring.position.copy(position); ring.position.z += .025;
    const glowMaterial = own(new THREE.SpriteMaterial({map:haloTexture,color,transparent:true,opacity:.46,depthWrite:false,blending:THREE.AdditiveBlending}));
    const glow = new THREE.Sprite(glowMaterial);
    glow.position.copy(position); glow.scale.setScalar(.15);
    scene.add(mesh,ring,glow);
    nodes.push({index,mesh,ring,glow});
  }

  const bonePositions = [], dustPositions = [];
  BONES.forEach(([from,to],boneIndex) => {
    bonePositions.push(...points[from].toArray(),...points[to].toArray());
    const length = points[from].distanceTo(points[to]);
    const count = Math.max(9,Math.round(length*160));
    for (let i=0;i<count;i++) {
      const t=i/(count-1), point=points[from].clone().lerp(points[to],t);
      // Deterministic particles describe schematic bone geometry only.
      const angle=i*2.399963+boneIndex, radius=.008+.014*((i*7+boneIndex*3)%11)/11;
      point.x+=Math.cos(angle)*radius;
      point.z+=Math.sin(angle)*radius;
      dustPositions.push(...point.toArray());
    }
  });
  const boneGeometry = own(new THREE.BufferGeometry());
  boneGeometry.setAttribute("position",new THREE.Float32BufferAttribute(bonePositions,3));
  const boneMaterial = own(new THREE.LineBasicMaterial({color:0x7797a4,transparent:true,opacity:.65}));
  scene.add(new THREE.LineSegments(boneGeometry,boneMaterial));
  const dustGeometry = own(new THREE.BufferGeometry());
  dustGeometry.setAttribute("position",new THREE.Float32BufferAttribute(dustPositions,3));
  const dustMaterial = own(new THREE.PointsMaterial({color:0x7797a4,size:.012,map:haloTexture,transparent:true,opacity:.42,depthWrite:false,blending:THREE.AdditiveBlending}));
  scene.add(new THREE.Points(dustGeometry,dustMaterial));

  const floor = new THREE.Group();
  floor.position.set(0,.005,0);
  for (const radius of [.25,.43,.68]) {
    const shape=[];
    for(let i=0;i<=96;i++){const angle=i/96*Math.PI*2;shape.push(Math.cos(angle)*radius,0,Math.sin(angle)*radius*.35)}
    const geometry=own(new THREE.BufferGeometry());
    geometry.setAttribute("position",new THREE.Float32BufferAttribute(shape,3));
    const material=own(new THREE.LineBasicMaterial({color:0x26738c,transparent:true,opacity:.38}));
    floor.add(new THREE.Line(geometry,material));
  }
  floor.rotation.x=.55;
  scene.add(floor);

  function paint() {
    frame=0;
    if(disposed || !active || width<1 || height<1) return;
    controls?.update();
    camera.updateMatrixWorld();
    positionAnnotations();
    const offset = camera.getWorldDirection(new THREE.Vector3()).multiplyScalar(-.025);
    for (const item of nodes) {
      item.ring.quaternion.copy(camera.quaternion);
      item.ring.position.copy(item.mesh.position).add(offset);
    }
    renderer?.render(scene,camera);
  }
  function requestPaint() {
    if (!frame && active && !disposed) frame=requestAnimationFrame(paint);
  }

  function highlight(index) {
    if (disposed || index===highlighted) return;
    highlighted=index;
    root.dataset.activeJoint=String(index);
    for (const item of labels) {
      const selected=item.index===index;
      item.label.classList.toggle("is-active",selected);
      item.line.classList.toggle("is-active",selected);
    }
    for (const item of nodes) {
      const selected=item.index===index;
      item.mesh.scale.setScalar(selected?1.7:1);
      item.ring.material.color.set(selected?0xffffff:labels[item.index].color);
      item.ring.material.opacity=selected?1:.65;
      item.ring.scale.setScalar(selected?1.5:1);
      item.glow.material.opacity=selected ? .95 : .46;
      item.glow.scale.setScalar(selected ? .24 : .15);
    }
    fallbackNodes.forEach((node,index) => node.classList.toggle("is-active",index===highlighted));
    requestPaint();
  }

  function layout() {
    if(disposed) return;
    width=viewport.clientWidth; height=viewport.clientHeight;
    if(!width || !height) return;
    const compact=height<350;
    root.classList.toggle("is-compact",compact);
    const aspect=width/height, horizontal=Math.max(2.6,2.14*aspect), vertical=horizontal/aspect;
    camera.left=-horizontal/2;camera.right=horizontal/2;
    camera.top=vertical/2;camera.bottom=-vertical/2;
    camera.updateProjectionMatrix();
    renderer?.setSize(width,height,false);
    overlay.setAttribute("viewBox",`0 0 ${width} ${height}`);
    requestPaint();
  }

  function positionAnnotations() {
    projected=points.map(point=>{const p=point.clone().project(camera);return {x:(p.x+1)*width/2,y:(1-p.y)*height/2,z:p.z};});
    if(fallback){
      fallback.setAttribute("viewBox",`0 0 ${width} ${height}`);
      fallbackBones.setAttribute("d",BONES.map(([from,to])=>`M ${projected[from].x} ${projected[from].y} L ${projected[to].x} ${projected[to].y}`).join(" "));
      fallbackNodes.forEach((node,index)=>{node.setAttribute("cx",String(projected[index].x));node.setAttribute("cy",String(projected[index].y))});
    }
    const placed=[], labelWidth=labels[0].label.offsetWidth || 84, labelHeight=labels[0].label.offsetHeight || 22;
    const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
    const overlap=(a,b)=>Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))
      *Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));
    // Search only close to each joint. Collision handling never turns labels
    // into remote columns; camera rotation always reprojects their own anchors.
    for (const item of [...labels].sort((a,b)=>projected[a.index].y-projected[b.index].y)) {
      const node=projected[item.index];
      const visible=node.z>=-1 && node.z<=1 && node.x>=0 && node.x<=width && node.y>=0 && node.y<=height;
      item.label.hidden=!visible;item.line.style.display=visible?"":"none";
      if(!visible)continue;
      const side=Math.abs(node.x-projected[0].x)<8 ? ([1,18].includes(item.index)?-1:1) : Math.sign(node.x-projected[0].x);
      const dy=[12,16].includes(item.index)?-10:[13,17].includes(item.index)?8:0;
      const candidates=[];
      for(const direction of [side,-side])for(const shift of [dy,dy-24,dy+24,dy-42,dy+42]) {
        candidates.push({x:node.x+(direction<0?-labelWidth-12:12),y:node.y-labelHeight/2+shift,w:labelWidth,h:labelHeight});
      }
      let best=null,bestScore=Infinity;
      for(const candidate of candidates) {
        const box={...candidate,x:clamp(candidate.x,8,width-labelWidth-8),y:clamp(candidate.y,34,height-labelHeight-8)};
        const end={x:clamp(node.x,box.x,box.x+box.w),y:clamp(node.y,box.y,box.y+box.h)};
        const score=placed.reduce((sum,other)=>sum+overlap(box,other)*8,0)
          +projected.reduce((sum,point)=>sum+overlap(box,{x:point.x-6,y:point.y-6,w:12,h:12})*16,0)
          +Math.hypot(end.x-node.x,end.y-node.y)+(Math.sign(box.x+box.w/2-node.x)===side?0:8);
        if(score<bestScore){bestScore=score;best={box,end}}
      }
      placed.push(best.box);
      item.label.style.left=`${best.box.x}px`;item.label.style.top=`${best.box.y}px`;
      item.line.setAttribute("d",`M ${node.x} ${node.y} L ${best.end.x} ${best.end.y}`);
    }
  }

  function pointerMove(event) {
    if(dragging)return;
    const label=event.target.closest("[data-joint-index]");
    if(label && root.contains(label)){hovered=Number(label.dataset.jointIndex);highlight(hovered);return}
    const bounds=root.getBoundingClientRect(),x=event.clientX-bounds.left,y=event.clientY-bounds.top;
    let nearest=-1,distance=15;
    projected.forEach((point,index)=>{const d=Math.hypot(point.x-x,point.y-y);if(d<distance){distance=d;nearest=index}});
    hovered=nearest;highlight(hovered>=0?hovered:focused);
    stage.style.cursor=nearest>=0?"pointer":renderer?"grab":"default";
  }
  function pointerLeave(){hovered=-1;highlight(focused)}
  function focusIn(event){const label=event.target.closest("[data-joint-index]");if(label){focused=Number(label.dataset.jointIndex);highlight(focused)}}
  function focusOut(event){
    const label=event.relatedTarget?.closest?.("button[data-joint-index]");
    focused=label && root.contains(label)?Number(label.dataset.jointIndex):-1;
    highlight(focused>=0?focused:hovered);
  }
  function click(event){const label=event.target.closest("[data-joint-index]");if(label){focused=Number(label.dataset.jointIndex);highlight(focused)}}
  function keyDown(event){
    const label=event.target.closest("[data-joint-index]");
    if(!label || !["ArrowDown","ArrowUp","Home","End"].includes(event.key))return;
    event.preventDefault();
    const available=labels.filter(item=>!item.label.hidden);
    if(!available.length)return;
    const index=available.findIndex(item=>item.label===label);
    const next=event.key==="Home"?0:event.key==="End"?available.length-1
      :(index+(event.key==="ArrowDown"?1:available.length-1))%available.length;
    available[next].label.focus();
  }
  function dragStart(){dragging=true;hovered=-1;highlight(focused);stage.style.cursor="grabbing"}
  function dragEnd(){dragging=false;stage.style.cursor="grab"}
  function resetView(){
    if(controls){
      controls.reset();
    }
    requestPaint();
  }
  root.addEventListener("pointermove",pointerMove);
  root.addEventListener("pointerleave",pointerLeave);
  root.addEventListener("focusin",focusIn);
  root.addEventListener("focusout",focusOut);
  root.addEventListener("click",click);
  root.addEventListener("keydown",keyDown);
  stage.addEventListener("dblclick",resetView);
  try {
    renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:"low-power"});
    renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));
    renderer.setClearColor(0x000000,0);
    renderer.domElement.setAttribute("aria-hidden","true");
    stage.append(renderer.domElement);
    stage.title="Left drag to rotate · Middle drag to pan · Scroll to zoom · Double-click to reset";
    stage.style.cursor="grab";
    controls=new PaperViewControls(camera,renderer.domElement,{resetOnDoubleClick:false});
    controls.target.set(0,.97,0);
    controls.update();controls.saveState();
    controls.addEventListener("change",requestPaint);
    controls.addEventListener("start",dragStart);controls.addEventListener("end",dragEnd);
  } catch (_error) {
    renderer?.dispose();renderer=null;
    root.classList.add("joint-diagram-no-webgl");
    root.querySelector(".joint-diagram-label").textContent="19-joint layout · 2D";
    fallback=document.createElementNS(SVG_NS,"svg");
    fallback.classList.add("joint-diagram-fallback");
    fallback.setAttribute("role","img");
    fallback.setAttribute("aria-label","19-joint schematic, 2D view");
    fallbackBones=document.createElementNS(SVG_NS,"path");
    fallback.append(fallbackBones);
    POSE.forEach((_,index)=>{
      const node=document.createElementNS(SVG_NS,"circle");
      node.dataset.jointIndex=String(index);node.setAttribute("r","4");
      node.style.setProperty("--joint-error-color",labels[index].color);
      fallback.append(node);fallbackNodes.push(node);
    });
    stage.replaceChildren(fallback);
  }
  observer=new ResizeObserver(layout);observer.observe(root);
  layout();

  return {
    setActive(value){active=Boolean(value);if(controls)controls.enabled=active;if(active)layout();else if(frame){cancelAnimationFrame(frame);frame=0}},
    dispose(){
      if(disposed)return;
      disposed=true;observer?.disconnect();if(frame)cancelAnimationFrame(frame);
      root.removeEventListener("pointermove",pointerMove);root.removeEventListener("pointerleave",pointerLeave);
      root.removeEventListener("focusin",focusIn);root.removeEventListener("focusout",focusOut);
      root.removeEventListener("click",click);root.removeEventListener("keydown",keyDown);
      stage.removeEventListener("dblclick",resetView);
      controls?.removeEventListener("change",requestPaint);controls?.removeEventListener("start",dragStart);controls?.removeEventListener("end",dragEnd);controls?.dispose();
      resources.forEach(resource=>resource.dispose());textures.forEach(texture=>texture.dispose());
      renderer?.dispose();renderer?.forceContextLoss();scene.clear();root.remove();
    },
  };
}
