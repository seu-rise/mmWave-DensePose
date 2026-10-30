import * as THREE from './vendor/three.module.js';

// Same gestures as paper/figures/paper-browser/viewer.js interactive subplots.
export const PAPER_VIEW_GESTURES=Object.freeze({azimuthPerPixel:.35,elevationPerPixel:.25,maxElevation:80,minZoom:.4,maxZoom:4,wheelLimit:120,wheelScale:.0018});

export class PaperViewControls extends THREE.EventDispatcher {
  constructor(camera,element,{resetOnDoubleClick=true}={}){
    super();this.camera=camera;this.element=element;this.target=new THREE.Vector3();this.enabled=true;
    this.isPaperViewControls=true;this.enableDamping=false;this.enablePan=true;this.zoomToCursor=false;
    this.minDistance=2.5;this.minZoom=PAPER_VIEW_GESTURES.minZoom;this.maxZoom=PAPER_VIEW_GESTURES.maxZoom;
    this.up=camera.up.clone().normalize();this.toZ=new THREE.Quaternion().setFromUnitVectors(this.up,new THREE.Vector3(0,0,1));this.fromZ=this.toZ.clone().invert();
    this.drag=null;this.pointers=new Map();this.disposed=false;this.oldTouchAction=element.style.touchAction;this.oldCursor=element.style.cursor;
    element.style.touchAction='none';element.style.cursor='grab';
    this.handlers={pointerdown:event=>this.pointerDown(event),pointermove:event=>this.pointerMove(event),pointerup:event=>this.pointerEnd(event),pointercancel:event=>this.pointerEnd(event),lostpointercapture:event=>this.pointerEnd(event),wheel:event=>this.wheel(event),contextmenu:event=>event.preventDefault()};
    if(resetOnDoubleClick)this.handlers.dblclick=()=>{if(this.enabled)this.reset()};
    for(const [type,handler] of Object.entries(this.handlers))element.addEventListener(type,handler,type==='wheel'?{passive:false}:undefined);
    this.saveState();
  }
  view(){
    const offset=this.camera.position.clone().sub(this.target).applyQuaternion(this.toZ),distance=offset.length();
    return {azimuth:THREE.MathUtils.radToDeg(Math.atan2(offset.y,offset.x)),elevation:distance?THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(offset.z/distance,-1,1))):0,distance,zoom:this.camera.zoom};
  }
  pointerDown(event){
    if(!this.enabled)return;
    if(event.pointerType==='touch'){
      event.preventDefault();if(this.pointers.size>=2)return;
      this.pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});this.element.setPointerCapture(event.pointerId);
      if(this.pointers.size===1){this.drag={pointerId:event.pointerId,x:event.clientX,y:event.clientY,mode:'rotate',...this.view()};this.element.style.cursor='grabbing';this.dispatchEvent({type:'start'});return}
      const pointerIds=[...this.pointers.keys()],distance=this.pointerDistance(pointerIds);
      if(distance>0){this.drag={mode:'pinch',pointerIds,startDistance:distance,startZoom:this.camera.zoom};this.element.style.cursor='grabbing';this.dispatchEvent({type:'start'})}
      return;
    }
    if(this.drag||event.isPrimary===false||![0,1].includes(event.button))return;
    const pan=event.button===1&&event.pointerType!=='touch';event.preventDefault();if(pan&&!this.enablePan)return;
    const start={pointerId:event.pointerId,x:event.clientX,y:event.clientY,mode:pan?'pan':'rotate',...this.view()};
    if(pan){
      const width=this.element.clientWidth,height=this.element.clientHeight,projection=this.camera.projectionMatrix.elements;
      if(!(width>0&&height>0&&Number.isFinite(projection[0])&&projection[0]!==0&&Number.isFinite(projection[5])&&projection[5]!==0))return;
      this.update();
      const depth=this.camera.isPerspectiveCamera?start.distance:1;
      Object.assign(start,{eye:this.camera.position.clone(),target:this.target.clone(),
        right:new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld,0),up:new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld,1),
        unitsX:2*depth/(width*projection[0]),unitsY:2*depth/(height*projection[5])});
    }
    this.drag=start;this.element.setPointerCapture(event.pointerId);this.element.style.cursor=pan?'move':'grabbing';this.dispatchEvent({type:'start'});
  }
  pointerMove(event){
    if(event.pointerType==='touch'){
      const pointer=this.pointers.get(event.pointerId);if(!this.enabled||!pointer)return;
      pointer.x=event.clientX;pointer.y=event.clientY;
      if(this.drag?.mode==='pinch'&&this.drag.pointerIds.includes(event.pointerId)){
        const distance=this.pointerDistance(this.drag.pointerIds);
        if(distance>0){this.camera.zoom=THREE.MathUtils.clamp(this.drag.startZoom*distance/this.drag.startDistance,this.minZoom,this.maxZoom);this.camera.updateProjectionMatrix();this.dispatchEvent({type:'change'})}
        return;
      }
    }
    const start=this.drag;if(!this.enabled||!start||start.mode==='pinch'||event.pointerId!==start.pointerId)return;
    if(start.mode==='pan'){
      if(!this.enablePan)return;
      // Moving eye and target together keeps the orbit unchanged. Projection
      // scale includes perspective depth or orthographic extent and zoom, so
      // the scene follows the pointer by the same number of screen pixels.
      const shift=start.right.clone().multiplyScalar(-(event.clientX-start.x)*start.unitsX)
        .addScaledVector(start.up,(event.clientY-start.y)*start.unitsY);
      this.camera.position.copy(start.eye).add(shift);this.target.copy(start.target).add(shift);this.update();this.dispatchEvent({type:'change'});return;
    }
    const azimuth=THREE.MathUtils.degToRad(start.azimuth+(event.clientX-start.x)*PAPER_VIEW_GESTURES.azimuthPerPixel);
    const elevation=THREE.MathUtils.degToRad(THREE.MathUtils.clamp(start.elevation+(event.clientY-start.y)*PAPER_VIEW_GESTURES.elevationPerPixel,-PAPER_VIEW_GESTURES.maxElevation,PAPER_VIEW_GESTURES.maxElevation));
    const offset=new THREE.Vector3(Math.cos(elevation)*Math.cos(azimuth),Math.cos(elevation)*Math.sin(azimuth),Math.sin(elevation)).multiplyScalar(start.distance).applyQuaternion(this.fromZ);
    this.camera.position.copy(this.target).add(offset);this.update();this.dispatchEvent({type:'change'});
  }
  pointerEnd(event){
    if(event.pointerType==='touch'&&this.pointers.has(event.pointerId)){
      this.pointers.delete(event.pointerId);if(this.element.hasPointerCapture?.(event.pointerId))this.element.releasePointerCapture(event.pointerId);
      if(this.drag?.mode==='pinch'&&this.drag.pointerIds.includes(event.pointerId)){
        const remaining=[...this.pointers.entries()];
        if(remaining.length>=2){const pointerIds=remaining.slice(0,2).map(([id])=>id);this.drag={mode:'pinch',pointerIds,startDistance:this.pointerDistance(pointerIds),startZoom:this.camera.zoom};return}
        if(remaining.length===1){const [pointerId,point]=remaining[0];this.drag={pointerId,x:point.x,y:point.y,mode:'rotate',...this.view()};return}
        this.drag=null;this.element.style.cursor='grab';this.dispatchEvent({type:'end'});return;
      }
    }
    if(!this.drag||event.pointerId!==this.drag.pointerId)return;
    const pointerId=this.drag.pointerId;this.drag=null;
    if(this.element.hasPointerCapture?.(pointerId))this.element.releasePointerCapture(pointerId);
    this.element.style.cursor='grab';this.dispatchEvent({type:'end'});
  }
  pointerDistance(pointerIds){
    const first=this.pointers.get(pointerIds[0]),second=this.pointers.get(pointerIds[1]);
    return first&&second?Math.hypot(second.x-first.x,second.y-first.y):0;
  }
  wheel(event){
    if(!this.enabled)return;event.preventDefault();
    const delta=THREE.MathUtils.clamp(event.deltaY,-PAPER_VIEW_GESTURES.wheelLimit,PAPER_VIEW_GESTURES.wheelLimit);
    this.camera.zoom=THREE.MathUtils.clamp(this.camera.zoom*Math.exp(-delta*PAPER_VIEW_GESTURES.wheelScale),this.minZoom,this.maxZoom);
    this.camera.updateProjectionMatrix();this.dispatchEvent({type:'change'});
  }
  update(){this.camera.up.copy(this.up);this.camera.lookAt(this.target);this.camera.updateMatrixWorld();return false}
  saveState(){this.saved={position:this.camera.position.clone(),target:this.target.clone(),zoom:this.camera.zoom}}
  reset(){
    if(this.drag)this.pointerEnd({pointerId:this.drag.pointerId});
    this.camera.position.copy(this.saved.position);this.target.copy(this.saved.target);this.camera.zoom=this.saved.zoom;
    this.camera.updateProjectionMatrix();this.update();this.dispatchEvent({type:'change'});
  }
  dispose(){
    if(this.disposed)return;this.disposed=true;
    if(this.pointers.size){
      for(const pointerId of this.pointers.keys())if(this.element.hasPointerCapture?.(pointerId))this.element.releasePointerCapture(pointerId);
      this.pointers.clear();this.drag=null;this.element.style.cursor='grab';
      this.dispatchEvent({type:'end'});
    }else if(this.drag)this.pointerEnd({pointerId:this.drag.pointerId});
    for(const [type,handler] of Object.entries(this.handlers))this.element.removeEventListener(type,handler);
    this.element.style.touchAction=this.oldTouchAction;this.element.style.cursor=this.oldCursor;
  }
}
