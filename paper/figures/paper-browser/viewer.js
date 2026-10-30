'use strict';
// Numeric snapshots stay frozen. Presentation follows ENG-2474 and the source exporters.
const THREE=globalThis.THREE;
const query=new URLSearchParams(location.search),figureId=Number(query.get('figure')||6);
const interactive=query.get('mode')==='interactive',selectedPanel=query.get('panel');
const sourceFigures=[figureId,...(query.has('include')?[Number(query.get('include'))]:[])];
const panels=sourceFigures.flatMap(sourceFigure=>(globalThis.PAPER_FIGURES?.[sourceFigure]?.panels||[]).filter(p=>!interactive||(p.id===selectedPanel&&p.kind!=='image')).map(panel=>({...panel,source_figure:sourceFigure})));
const matlab=figureId===6;
const root=document.getElementById('figure'),figures=[];
let scheduled=0,disposed=false;
document.body.classList.add(interactive?'subplot-view':'preview-view',`figure-id-${figureId}`,matlab?'matlab-style':'export-style');
window.figureRenderState={id:figureId,renderer:'threejs',frozen:true,mode:interactive?'interactive':'preview',interactive,panels:0,panelIds:[],axes:[],ready:false};
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function axesFor(panel){
 if(matlab)return {...panel.axes,view:{...panel.axes.view,convention:'matlab'}};
 return {...panel.axes,
  x:{limits:[-1.5,1.5],ticks:[-1.5,0,1.5],tick_labels:['−1.5','0','1.5'],label:'X (m)'},
  y:{limits:[0,4],ticks:[0,2,4],tick_labels:['0','2','4'],label:'Y (m)'},
  z:{limits:[-.1,2],ticks:[-.1,.95,2],tick_labels:['−0.10','0.95','2.00'],label:'Z (m)'},
  // The final PDF composites reshape the exported frame; the stored XYZ is unchanged.
  view:{elevation:16,azimuth:-60,convention:'matplotlib'},source_box_aspect:panel.axes.box_aspect,export_box_aspect:[3,4,2.1],box_aspect:figureId===4?[4,4,3]:[1,1,1],presentation_aspect:'edited PDF frame proportions',projection:'perspective',
  grid:{x:'off',y:'off',z:'off',alpha:0},font:{name:'Arial'}};
}
function titleFor(panel){
 if(matlab)return panel.label||'';
 if(panel.kind==='radar'){if(figureId===11){const frame=(panel.label||panel.id||'').match(/frame[- ]?(\d+)/i)?.[1]||'1';return `Radar point cloud (frame ${frame})`}return 'Radar point cloud'}
 return figureId===11?(panel.label||''):'Ground-truth skeleton';
}
function colorValue(value){return Array.isArray(value)?new THREE.Color(...value).convertSRGBToLinear():new THREE.Color(value||'#545454')}
function cssColor(value){return '#'+colorValue(value).getHexString()}
function element(tag,cls,text){const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=text;return node}
if(!panels.length){root.append(element('div','empty','Unavailable'));window.figureRenderState.ready=true}else buildFigure();

function buildFigure(){
 const filter=element('div');filter.className='display-filter';filter.innerHTML='<svg width="0" height="0" aria-hidden="true"><defs><filter id="paper-rb-swap" color-interpolation-filters="sRGB"><feColorMatrix type="matrix" values="0 0 1 0 0  0 1 0 0 0  1 0 0 0 0  0 0 0 1 0"/></filter></defs></svg>';document.body.append(filter);
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.domElement.className='three-figure';renderer.setClearColor(0xffffff,1);renderer.setScissorTest(true);document.body.prepend(renderer.domElement);
 const grid=element('div',`figure-grid figure-${figureId}`);root.append(grid);
 if(matlab&&!interactive){for(const panel of panels.filter(p=>p.kind==='radar')){const heading=element('div','dataset-heading'+(/ours/i.test(panel.dataset)?' ours':''),panel.dataset);heading.append(element('span','dataset-rule'));grid.append(heading)}}
 for(const [index,panel] of panels.entries()){
  if(!interactive&&[9,10].includes(figureId)&&index%6===0){for(const p of [panel,panels[index+3]])grid.append(element('div','motion-group-title'+(index===0?' first-row':''),p.sample_label))}
  if(!interactive&&figureId===11&&index%10===0){const heading=element('div','sample-title',`Five-frame temporal window—sample ${Math.floor(index/10)+1}`);heading.append(element('span','','Target frame'));grid.append(heading)}
  const card=element('section',`plot-panel kind-${panel.kind}`);grid.append(card);
  const title=element('h3','plot-title',titleFor(panel));card.append(title);
  if(panel.kind==='image'){
   const mount=element('div','plot-image image-mount'),crop=element('div','rgb-crop'),img=element('img');img.alt=`${panel.sample_label||'Frame 145'} RGB`;crop.append(img);mount.append(crop);card.append(mount);
   const display=panel.display||{},output=display.output_size||[256,256];crop.style.width=output[0]+'px';crop.style.height=output[1]+'px';
   img.onload=()=>{img.style.width=img.naturalWidth+'px';img.style.height=img.naturalHeight+'px';const m=display.native_to_paper||display.native_to_paper_256||[[output[0]/img.naturalWidth,0,0],[0,output[1]/img.naturalHeight,0],[0,0,1]];img.style.transform=`matrix(${m[0][0]},${m[1][0]},${m[0][1]},${m[1][1]},${m[0][2]},${m[1][2]})`+(display.mirror_x?` translate(${img.naturalWidth}px,0) scaleX(-1)`:'');if(display.swap_rb)img.style.filter='url(#paper-rb-swap)'};img.src=panel.image;
   const fit=()=>{const scale=Math.min(mount.clientWidth/output[0],mount.clientHeight/output[1]);crop.style.transform=`translate(${(mount.clientWidth-output[0]*scale)/2}px,${(mount.clientHeight-output[1]*scale)/2}px) scale(${scale})`};new ResizeObserver(fit).observe(mount);fit();
   if(figureId===4)card.append(element('div','subfigure-letter','(c)'));window.figureRenderState.images=(window.figureRenderState.images||0)+1;continue;
  }
  const mount=element('div','plot-mount');mount.dataset.panelId=panel.id;mount.dataset.panel=String(index);card.append(mount);
  if(!interactive){card.dataset.openPanel=panel.id;card.tabIndex=0;card.setAttribute('role','button');card.setAttribute('aria-haspopup','dialog');card.setAttribute('aria-label','Open '+[panel.dataset,panel.sample_label,titleFor(panel)].filter(Boolean).join(' · '));const open=event=>{event.preventDefault();parent.postMessage({type:'paper:open-subplot',renderer:'threejs',figure:figureId,sourceFigure:panel.source_figure||figureId,panel:panel.id},'*')};card.addEventListener('click',open);card.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key))open(event)})}
  const axes=axesFor(panel),ranges=['x','y','z'].map(k=>axes[k].limits),raw=axes.box_aspect||axes.plot_box_aspect_ratio||[1,1,1],max=Math.max(...raw),aspect=raw.map(n=>n/max*2),low=aspect.map(n=>-n/2),high=aspect.map(n=>n/2);
  const mapPoint=p=>new THREE.Vector3(...p.slice(0,3).map((n,i)=>(n-(ranges[i][0]+ranges[i][1])/2)/(ranges[i][1]-ranges[i][0])*aspect[i]));
  const scene=new THREE.Scene(),camera=matlab?new THREE.OrthographicCamera(-2,2,2,-2,.01,100):new THREE.PerspectiveCamera(30,1,.01,100);camera.up.set(0,0,1);
  const item={panelId:panel.id,scene,camera,mount,card,axes,ranges,aspect,low,high,mapPoint,labels:[],markers:[],azimuth:matlab?Number(axes.view.azimuth)-90:-60,elevation:matlab?Number(axes.view.elevation):16,zoom:1,distance:10};
  item.initialView=[item.azimuth,item.elevation];figures.push(item);window.figureRenderState.axes.push(axes);drawAxes(item);
  const legend=[];
  if(panel.series?.length){for(const series of panel.series){const positions=series.points||[],color=series.color||'#00BAAF';if(series.kind==='line'||series.type==='line')addLine(item,positions,color,series.line_width,series.alpha??.8);else{addPoints(item,positions,color,series.marker_edge_color||series.edge_color||'#545454',series.marker_face_alpha??series.alpha??.92);if(series.type==='skeleton'||series.edges?.length)addBones(item,positions,series.edges,series.line_color||'#545454');if(series.label||series.name)legend.push([series.label||series.name,color])}}}
  else{if(panel.points?.length){addPoints(item,panel.points,panel.color||(panel.kind==='radar'?'#6AB4E6':'#00BAAF'),'#545454',.92);if(panel.kind!=='radar')addBones(item,panel.points,panel.edges,'#545454');legend.push([panel.kind==='radar'?'Radar points':'Ground-truth joints',panel.color||'#6AB4E6'])}if(panel.gt?.length){addPoints(item,panel.gt,'#00BAAF','#545454',.9);addBones(item,panel.gt,panel.edges,'#545454');legend.push(['Ground-truth keypoints','#00BAAF'])}if(panel.prediction?.length){addPoints(item,panel.prediction,'#E58220','#545454',.9);addBones(item,panel.prediction,panel.edges,'#545454');legend.push(['Predicted keypoints','#E58220'])}}
  if((matlab||interactive&&figureId===11)&&legend.length){const box=element('div','plot-legend');box.innerHTML=legend.map(([name,color])=>`<span><i></i>${esc(name)}</span>`).join('');box.querySelectorAll('i').forEach((node,i)=>node.style.background=cssColor(legend[i][1]));mount.append(box)}
  if(figureId===11&&!interactive){if(panel.kind==='radar'){title.classList.add('radar-inline-title');title.style.setProperty('--marker-color',cssColor(legend[0]?.[1]||'#6AB4E6'))}else{const box=element('div','plot-legend-above');box.innerHTML=legend.filter(([name])=>!/radar/i.test(name)).map(([name,color])=>`<span><i></i>${esc(/ground.?truth/i.test(name)?'Ground-truth keypoints':/predict/i.test(name)?'Predicted keypoints':name)}</span>`).join('');box.querySelectorAll('i').forEach((node,i)=>node.style.background=cssColor(legend.filter(([name])=>!/radar/i.test(name))[i][1]));card.insertBefore(box,mount)}}
  if(figureId===4&&!interactive)card.append(element('div','subfigure-letter',index===0?'(a)':'(b)'));
  if(interactive){
   const pointers=new Map();let drag=null,pinch=null;
   mount.addEventListener('pointerdown',event=>{
    if(event.pointerType==='touch'){
     event.preventDefault();if(pointers.size>=2)return;
     pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});mount.setPointerCapture(event.pointerId);
     if(pointers.size===1)drag={pointerId:event.pointerId,x:event.clientX,y:event.clientY,azimuth:item.azimuth,elevation:item.elevation};
     else{const ids=[...pointers.keys()],a=pointers.get(ids[0]),b=pointers.get(ids[1]);pinch={ids,distance:Math.max(1,Math.hypot(b.x-a.x,b.y-a.y)),zoom:item.zoom};drag=null}
     return;
    }
    drag={pointerId:event.pointerId,x:event.clientX,y:event.clientY,azimuth:item.azimuth,elevation:item.elevation};mount.setPointerCapture(event.pointerId);
   });
   mount.addEventListener('pointermove',event=>{
    if(event.pointerType==='touch'&&pointers.has(event.pointerId)){
     const point=pointers.get(event.pointerId);point.x=event.clientX;point.y=event.clientY;
     if(pinch?.ids.includes(event.pointerId)){
      const a=pointers.get(pinch.ids[0]),b=pointers.get(pinch.ids[1]);if(a&&b){const distance=Math.hypot(b.x-a.x,b.y-a.y);item.zoom=Math.max(.4,Math.min(4,pinch.zoom*distance/pinch.distance));requestRender()}
      return;
     }
    }
    if(!drag||event.pointerId!==drag.pointerId)return;item.azimuth=drag.azimuth+(event.clientX-drag.x)*.35;item.elevation=Math.max(-80,Math.min(80,drag.elevation+(event.clientY-drag.y)*.25));requestRender();
   });
   const endPointer=event=>{
    if(event.pointerType==='touch'&&pointers.has(event.pointerId)){
     pointers.delete(event.pointerId);if(mount.hasPointerCapture?.(event.pointerId))mount.releasePointerCapture(event.pointerId);
     if(pinch?.ids.includes(event.pointerId)){
      const remaining=[...pointers.entries()];pinch=null;
      if(remaining.length>=2){const ids=remaining.slice(0,2).map(([id])=>id),a=pointers.get(ids[0]),b=pointers.get(ids[1]);pinch={ids,distance:Math.max(1,Math.hypot(b.x-a.x,b.y-a.y)),zoom:item.zoom};drag=null}
      else if(remaining.length===1){const [pointerId,point]=remaining[0];drag={pointerId,x:point.x,y:point.y,azimuth:item.azimuth,elevation:item.elevation}}
      else drag=null;
     }else if(drag?.pointerId===event.pointerId)drag=null;
     return;
    }
    if(drag?.pointerId===event.pointerId)drag=null;
   };
   mount.addEventListener('pointerup',endPointer);mount.addEventListener('pointercancel',endPointer);mount.addEventListener('lostpointercapture',endPointer);
   mount.addEventListener('dblclick',()=>{[item.azimuth,item.elevation]=item.initialView;item.zoom=1;requestRender()});mount.addEventListener('wheel',event=>{event.preventDefault();item.zoom=Math.max(.4,Math.min(4,item.zoom*Math.exp(-Math.max(-120,Math.min(120,event.deltaY))*.0018)));requestRender()},{passive:false});
  }
 }
 if(matlab&&!interactive)for(let i=0;i<5;i++)grid.append(element('div','subfigure-letter',`(${String.fromCharCode(97+i)})`));
 window.figureRenderState.panels=figures.length;window.figureRenderState.panelIds=figures.map(item=>item.panelId);window.figureRenderState.contexts=1;
 function requestRender(){if(!scheduled&&!disposed)scheduled=requestAnimationFrame(render)}
 function render(){
  scheduled=0;renderer.setSize(innerWidth,innerHeight,false);renderer.setScissorTest(false);renderer.clear();renderer.setScissorTest(true);
  for(const item of figures){const rect=item.mount.getBoundingClientRect();if(rect.bottom<=0||rect.top>=innerHeight||rect.right<=0||rect.left>=innerWidth)continue;const w=rect.width,h=rect.height;
   fitCamera(item,w,h);const diameter=interactive?9:Math.max(2.4,w*(figureId===4?.018:.024));for(const material of item.markers){material.uniforms.pointSize.value=diameter;material.uniforms.pixelRatio.value=renderer.getPixelRatio()}
   renderer.setViewport(rect.left,innerHeight-rect.bottom,w,h);renderer.setScissor(Math.max(0,rect.left),Math.max(0,innerHeight-rect.bottom),Math.max(0,Math.min(rect.right,innerWidth)-Math.max(0,rect.left)),Math.max(0,Math.min(rect.bottom,innerHeight)-Math.max(0,rect.top)));renderer.render(item.scene,item.camera);positionLabels(item,w,h);
  }
  window.figureRenderState.views=figures.map(item=>({id:item.panelId,azimuth:item.azimuth,elevation:item.elevation,distance:item.distance,orthoSpan:item.camera.isOrthographicCamera?2/item.camera.zoom:null,zoom:item.zoom}));window.figureRenderState.ready=true;window.figureRenderState.renderCount=(window.figureRenderState.renderCount||0)+1;
 }
 window.addEventListener('resize',requestRender);window.addEventListener('scroll',requestRender,{passive:true});new ResizeObserver(requestRender).observe(root);requestRender();
 window.addEventListener('pagehide',()=>{disposed=true;cancelAnimationFrame(scheduled);for(const item of figures)item.scene.traverse(object=>{object.geometry?.dispose();for(const material of object.material?[].concat(object.material):[])material.dispose()});renderer.dispose()},{once:true});
}

function addPoints(item,positions,color,edge,opacity){
 const points=positions.filter(p=>p?.length>=3&&p.slice(0,3).every(Number.isFinite));if(!points.length)return;
 const geometry=new THREE.BufferGeometry().setFromPoints(points.map(item.mapPoint));
 const material=new THREE.ShaderMaterial({uniforms:{pointSize:{value:3},pixelRatio:{value:1},fill:{value:colorValue(color)},edge:{value:colorValue(edge)},opacity:{value:opacity}},transparent:true,depthWrite:false,
  vertexShader:'uniform float pointSize;uniform float pixelRatio;void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=pointSize*pixelRatio;}',
  fragmentShader:'uniform vec3 fill;uniform vec3 edge;uniform float opacity;void main(){float r=length(gl_PointCoord-vec2(.5));float a=1.0-smoothstep(.45,.5,r);if(a<.01)discard;float inner=1.0-smoothstep(.40,.46,r);gl_FragColor=vec4(mix(edge,fill,inner),opacity*a);\n#include <colorspace_fragment>\n}'});
 item.markers.push(material);item.scene.add(new THREE.Points(geometry,material));
}
function addLine(item,positions,color,width=1,opacity=.8){const valid=positions.filter(p=>p?.length>=3&&p.slice(0,3).every(Number.isFinite));if(valid.length<2)return;item.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(valid.map(item.mapPoint)),new THREE.LineBasicMaterial({color:colorValue(color),transparent:true,opacity,linewidth:width||1})))}
function addBones(item,points,edges,color){const vertices=[];for(const [a,b] of edges||[])if(points[a]&&points[b])vertices.push(item.mapPoint(points[a]),item.mapPoint(points[b]));if(vertices.length)item.scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(vertices),new THREE.LineBasicMaterial({color:colorValue(color),transparent:true,opacity:.8})))}
function drawAxes(item){
 const {low:l,high:h,scene,axes}=item,line=(a,b,color,opacity=1)=>scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a),new THREE.Vector3(...b)]),new THREE.LineBasicMaterial({color,transparent:opacity<1,opacity})));
 const A=[l[0],l[1],l[2]],B=[h[0],l[1],l[2]],C=[h[0],h[1],l[2]],D=[l[0],h[1],l[2]],E=[l[0],l[1],h[2]],F=[l[0],h[1],h[2]],G=[h[0],h[1],h[2]];
 const border=matlab?'#6f6f73':'#303038';
 for(const [a,b] of [[A,B],[B,C],[C,D],[D,A],[A,E],[D,F],[C,G],[E,F],[F,G]])line(a,b,border,matlab?.8:1);
 if(matlab)for(let axis=0;axis<3;axis++)for(const tick of axes[['x','y','z'][axis]].ticks){const at=l[axis]+(tick-item.ranges[axis][0])/(item.ranges[axis][1]-item.ranges[axis][0])*(h[axis]-l[axis]);if(axis===0){line([at,l[1],l[2]],[at,h[1],l[2]],'#888888',.18);line([at,h[1],l[2]],[at,h[1],h[2]],'#888888',.18)}if(axis===1){line([l[0],at,l[2]],[h[0],at,l[2]],'#888888',.18);line([l[0],at,l[2]],[l[0],at,h[2]],'#888888',.18)}if(axis===2){line([l[0],h[1],at],[h[0],h[1],at],'#888888',.18);line([l[0],l[1],at],[l[0],h[1],at],'#888888',.18)}}
 item.axisEdges=[[A,B],[B,C],matlab?[A,E]:[C,G]];
 for(const [axis,key] of ['x','y','z'].entries()){
  const [a,b]=item.axisEdges[axis];
  for(const [index,tick] of axes[key].ticks.entries()){const t=(tick-item.ranges[axis][0])/(item.ranges[axis][1]-item.ranges[axis][0]),position=a.map((v,i)=>v+(b[i]-v)*t),node=element('span','axis-text axis-tick',axes[key].tick_labels?.[index]??String(tick));item.mount.append(node);item.labels.push({node,position:new THREE.Vector3(...position),axis,index,label:false})}
  const node=element('span','axis-text axis-label',axes[key].label||`${key.toUpperCase()} (m)`);item.mount.append(node);item.labels.push({node,position:new THREE.Vector3(...a.map((v,i)=>(v+b[i])/2)),axis,label:true});
 }
 window.figureRenderState.edgeCount=9;
}
function pixel(position,camera,w,h){const p=position.clone().project(camera);return {x:(p.x+1)*w/2,y:(1-p.y)*h/2}}
function fitCamera(item,w,h){
 const camera=item.camera,ratio=w/h,az=THREE.MathUtils.degToRad(item.azimuth),el=THREE.MathUtils.degToRad(item.elevation);camera.clearViewOffset();camera.zoom=1;
 if(camera.isOrthographicCamera){camera.left=-2*ratio;camera.right=2*ratio;camera.top=2;camera.bottom=-2}else camera.aspect=ratio;
 camera.position.set(item.distance*Math.cos(el)*Math.cos(az),item.distance*Math.cos(el)*Math.sin(az),item.distance*Math.sin(el));camera.lookAt(0,0,0);camera.updateMatrixWorld(true);camera.updateProjectionMatrix();
 const corners=[];for(const x of [item.low[0],item.high[0]])for(const y of [item.low[1],item.high[1]])for(const z of [item.low[2],item.high[2]])corners.push(new THREE.Vector3(x,y,z));
 const box=()=>{const p=corners.map(v=>pixel(v,camera,w,h));return {left:Math.min(...p.map(v=>v.x)),right:Math.max(...p.map(v=>v.x)),top:Math.min(...p.map(v=>v.y)),bottom:Math.max(...p.map(v=>v.y))}};
 const area=interactive?{left:w*.08,right:w*.92,top:h*.04,bottom:h*.90}:matlab?{left:w*.17,right:w*.89,top:h*.035,bottom:h*.82}:figureId===4?{left:w*.05,right:w*.84,top:h*.02,bottom:h*.90}:{left:w*.085,right:w*.785,top:h*.025,bottom:h*.87};
 const b=box();camera.zoom=Math.min((area.right-area.left)/(b.right-b.left),(area.bottom-area.top)/(b.bottom-b.top))*item.zoom;camera.updateProjectionMatrix();
 const fit=box(),dx=(fit.left+fit.right-area.left-area.right)/2,dy=(fit.top+fit.bottom-area.top-area.bottom)/2;camera.setViewOffset(w,h,dx,dy,w,h);camera.updateProjectionMatrix();
}
function positionLabels(item,w,h){
 const camera=item.camera,center=pixel(new THREE.Vector3(),camera,w,h),tickSize=interactive?16:Math.max(8,w*(matlab?.039:figureId===4?.037:.062)),labelSize=interactive?18:Math.max(9,w*(matlab?.047:figureId===4?.04:.065));
 const edges=item.axisEdges.map(([a,b])=>{const p=pixel(new THREE.Vector3(...a),camera,w,h),q=pixel(new THREE.Vector3(...b),camera,w,h),length=Math.hypot(q.x-p.x,q.y-p.y)||1,t={x:(q.x-p.x)/length,y:(q.y-p.y)/length};let n={x:-t.y,y:t.x};if(n.x*((p.x+q.x)/2-center.x)+n.y*((p.y+q.y)/2-center.y)<0)n={x:-n.x,y:-n.y};return {t,n}});
 for(const label of item.labels){const {node,axis}=label,edge=edges[axis],point=pixel(label.position,camera,w,h),font=label.label?labelSize:tickSize;node.style.fontSize=font+'px';
  const pad=label.label?(axis===2?tickSize*2.8:axis===1&&!matlab?tickSize*3.1:tickSize*2.6):tickSize*.85;let x=point.x+edge.n.x*pad,y=point.y+edge.n.y*pad;
  if(!label.label&&axis===0&&label.index===item.axes.x.ticks.length-1){x-=edge.t.x*(node.offsetWidth/2+3);y-=edge.t.y*(node.offsetWidth/2+3)}if(!label.label&&axis===1&&label.index===0){x+=edge.t.x*(node.offsetWidth/2+3);y+=edge.t.y*(node.offsetWidth/2+3)}
  if(!label.label&&axis===2&&label.index===0){x+=edge.t.x*tickSize*.5;y+=edge.t.y*tickSize*.5}
  const rotation=label.label?(axis===2?-90:axis===1&&!matlab?Math.atan2(edge.t.y,edge.t.x)*180/Math.PI:0):0;node.style.transform=`translate(-50%,-50%) rotate(${rotation}deg)`;node.style.left=x+'px';node.style.top=y+'px';
 }
}
