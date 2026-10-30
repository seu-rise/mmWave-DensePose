/** Offline ECharts scalar views over the original fetched log records. */
const COLORS=['#4ec3ff','#eabc78','#c19fef','#78d9b2','#ff9b9b','#91d7e3','#ed8796','#a6da95'];
const AXES={step:'Step',epoch:'Epoch',relative_time:'Relative time',wall_time:'Wall time'};
const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
const number=value=>Number.isFinite(value)?Number(value.toPrecision(6)).toString():'—';
const axisNumber=value=>!Number.isFinite(value)?'—':value!==0&&(Math.abs(value)<.001||Math.abs(value)>=1e5)?value.toExponential(2).replace('e+','e'):Number(value.toPrecision(4)).toString();
const esc=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const xText=(value,axis)=>axis==='wall_time'?new Date(value*1000).toLocaleString('sv-SE'):number(value);
function make(tag,className,text){const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=text;return node;}

export function chooseChartMetrics(tags,current=[]){
  const available=[...new Set(tags)],valid=new Set(available);
  const slots=Array.from({length:4},(_,i)=>valid.has(current[i])?current[i]:null);
  const used=new Set(slots.filter(Boolean));
  const preferences=[
    ['train/loss','loss','train_loss','val_loss','val/loss'],
    ['MLE (cm)','train/MLE (cm)','mle','mle_m','train_mle','val/MLE (cm)','Chamfer distance (cm)','cd','cd_m'],
    ['val_loss','val/loss','validation/loss'],
    ['val/MLE (cm)','val_mle','val_mle_m','val_cd','val_cd_m','val/Chamfer distance (cm)'],
  ];
  const ordinary=available.filter(tag=>tag!=='hp_metric'&&!tag.startsWith('hparam/'));
  for(let i=0;i<4;i++)if(!slots[i]){
    const choice=preferences[i].find(tag=>valid.has(tag)&&!used.has(tag));
    if(choice){slots[i]=choice;used.add(choice);}
  }
  for(let i=0;i<4;i++)if(!slots[i]){
    const choice=ordinary.find(tag=>!used.has(tag));
    if(choice){slots[i]=choice;used.add(choice);}
  }
  return slots;
}

let libraryPromise;
export function loadECharts(){
  if(window.echarts)return Promise.resolve(window.echarts);
  if(!libraryPromise)libraryPromise=new Promise((resolve,reject)=>{
    const script=document.createElement('script');script.src=new URL('./vendor/echarts.min.js',import.meta.url).href;
    script.onload=()=>window.echarts?resolve(window.echarts):reject(new Error('ECharts unavailable'));
    script.onerror=()=>reject(new Error('Unable to load the local ECharts renderer'));
    document.head.append(script);
  });
  return libraryPromise;
}

export function prepareScalarSeries(items,selected=[]){
  return items.map((item,index)=>{
    const points=[];
    for(let i=0;i<Math.min(item.x?.length||0,item.values?.length||0);i++){
      const x=item.x[i],y=item.values[i];if(!Number.isFinite(x)||!Number.isFinite(y))continue;
      const raw=item.raw_values?item.raw_values[i]:y;
      points.push({x,y,raw:Number.isFinite(raw)?raw:null});
    }
    points.sort((a,b)=>a.x-b.x);
    const order=selected.indexOf(item.run_id),colorIndex=(order<0?index:order)%COLORS.length;
    return {...item,points,colorIndex,color:COLORS[colorIndex]};
  }).filter(item=>item.points.length);
}

export function nearestSample(points,x){
  if(!points.length||!Number.isFinite(x))return null;
  const upper=value=>{let low=0,high=points.length;while(low<high){const mid=(low+high)>>>1;if(points[mid].x<=value)low=mid+1;else high=mid;}return low;};
  const right=upper(x),left=Math.max(0,right-1);
  const chosen=right<points.length&&Math.abs(points[right].x-x)<Math.abs(points[left].x-x)?right:left;
  return points[upper(points[chosen].x)-1];
}

export function scalarDomains(series){
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
  const values=[];
  for(const item of series)for(const p of item.points){xmin=Math.min(xmin,p.x);xmax=Math.max(xmax,p.x);ymin=Math.min(ymin,p.y);ymax=Math.max(ymax,p.y);values.push(p.y);}
  if(!Number.isFinite(xmin))return null;
  const padded=(low,high)=>{const pad=((high-low)||Math.max(Math.abs(high)*.1,1))*.06;return [low-pad,high+pad];};
  const xspan=xmax-xmin||1;
  const bounds={x:[xmin,xmax+xspan*.08],y:padded(ymin,ymax)};
  // Match TensorBoard's ignore-outliers scaling: pool visible curves and use
  // the inner 90% of finite displayed values. Keep the full extent for zoom-out.
  values.sort((a,b)=>a-b);
  const low=values.length>2?values[Math.ceil(.05*(values.length-1))]:ymin;
  const high=values.length>2?values[Math.floor(.95*(values.length-1))]:ymax;
  return {bounds,focus:{x:[...bounds.x],y:padded(low,high)}};
}

export class ScalarChart{
  constructor({tag,onMetricChange,chartIndex=0}){
    this.tag=tag;this.series=[];this.active=true;this.disposed=false;this.zoomed=false;
    this.element=make('section','logs-chart-card');this.element.dataset.scalarTag=tag;
    const heading=make('div','logs-chart-title');
    if(onMetricChange){
      this.metricSelect=make('select','logs-chart-metric');
      this.metricSelect.setAttribute('aria-label',`Chart ${chartIndex+1} metric`);
      this.metricSelect.addEventListener('change',()=>onMetricChange(this.metricSelect.value));
      heading.append(this.metricSelect);this.element.dataset.chartSlot=String(chartIndex);
    }else{this.title=make('h2','',tag);this.title.title=tag;heading.append(this.title);}
    this.plot=make('div','logs-chart');this.plot.setAttribute('role','img');this.plot.tabIndex=0;
    this.plot.setAttribute('aria-label',`${tag} comparison`);
    this.plot.setAttribute('aria-description','Default scaling ignores the outer 5% of values on each side. Scroll out to see the full range. Double-click or press 0 to restore automatic scaling.');
    this.empty=make('div','logs-chart-empty','Loading chart…');
    this.legend=make('div','logs-legend');this.legend.setAttribute('aria-label',`${tag} runs`);
    this.element.append(heading,this.plot,this.empty,this.legend);
    this.resetHandler=()=>this.reset();
    this.keyHandler=event=>{if(event.key==='0'||event.key==='Home'){event.preventDefault();this.reset();}};
    this.scrollHandler=()=>{this.pointer=null;this.chart?.dispatchAction({type:'hideTip'});};
    this.pointerHandler=event=>{if(this.active)this.pointer={x:event.clientX,y:event.clientY};};
    this.leaveHandler=()=>{this.pointer=null;};
    this.plot.addEventListener('mousemove',this.pointerHandler,{capture:true,passive:true});
    this.plot.addEventListener('wheel',this.pointerHandler,{capture:true,passive:true});
    this.plot.addEventListener('pointerleave',this.leaveHandler);
    this.plot.addEventListener('dblclick',this.resetHandler);this.plot.addEventListener('keydown',this.keyHandler);
    document.addEventListener('scroll',this.scrollHandler,true);
    this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(this.plot);
    this.ready=loadECharts().then(library=>{if(!this.disposed){this.library=library;this.draw();}}).catch(error=>{
      if(this.disposed)return;this.plot.hidden=true;this.empty.hidden=false;this.empty.textContent=error.message;
    });
  }

  setMetric(tag,options=[]){
    tag=tag||'';
    if(this.tag!==tag){this.tag=tag;this.zoomed=false;this.chart?.dispatchAction({type:'hideTip'});}
    this.element.dataset.scalarTag=tag;
    this.plot.setAttribute('aria-label',`${tag||'Metric'} comparison`);
    this.legend.setAttribute('aria-label',`${tag||'Metric'} runs`);
    if(this.title){this.title.textContent=tag;this.title.title=tag;}
    if(this.metricSelect){
      const key=JSON.stringify(options);
      if(key!==this.metricOptions){
        const placeholder=make('option','','Select metric');placeholder.value='';
        this.metricSelect.replaceChildren(placeholder,...options.map(value=>{const option=make('option','',value==='train/loss'?'loss':value);option.value=value;return option;}));
        this.metricOptions=key;
      }
      if(this.metricSelect.value!==tag)this.metricSelect.value=tag;
      this.metricSelect.title=tag||'Select metric';
    }
  }

  update(items,selected=[]){
    if(this.disposed)return;
    const previous=this.signature;
    this.series=prepareScalarSeries(items,selected);
    const domains=scalarDomains(this.series);this.bounds=domains?.bounds;this.focus=domains?.focus;
    this.signature=this.series.map(item=>`${item.run_id}:${item.x_axis||'epoch'}`).join('|');
    if(previous!==this.signature)this.zoomed=false;
    this.axis=this.series[0]?.x_axis||'epoch';
    this.plot.hidden=!this.bounds;this.empty.hidden=Boolean(this.bounds);this.legend.hidden=!this.bounds;
    if(!this.bounds)this.empty.textContent=this.tag?'No time-series values':'Select a metric';
    const key=this.series.map(item=>`${item.run_id}:${item.name}:${item.legend_name}:${item.color}`).join('|');
    if(key!==this.legendKey){
      const scroll=this.legend.scrollTop;
      this.legend.replaceChildren(...this.series.map(item=>{
        const row=make('span','logs-legend-entry'),swatch=make('i',`logs-legend-swatch logs-color-bg-${item.colorIndex}`);
        const label=make('span','logs-legend-name',item.legend_name||item.name||item.run_id);label.title=item.name||item.run_id;
        row.append(swatch,label);return row;
      }));this.legend.scrollTop=scroll;this.legendKey=key;
    }
    this.draw();
  }

  draw(){
    if(!this.active||this.disposed||!this.library||!this.bounds)return;
    const rect=this.plot.getBoundingClientRect();if(!rect.width||!rect.height)return;
    if(!this.chart){
      this.chart=this.library.init(this.plot,null,{renderer:'canvas'});
      this.plot.dataset.renderer='echarts';
      this.chart.on('datazoom',()=>{this.zoomed=true;this.syncDomains();});
      this.chart.on('finished',()=>this.syncDomains());
    }
    const view=this.zoomed&&this.view?{x:[...this.view.x],y:[...this.view.y]}:this.focus;
    const zoom=['x','y'].map(axis=>({id:`scalar-${axis}`,type:'inside',
      [axis==='x'?'xAxisIndex':'yAxisIndex']:0,filterMode:'none',
      ...(axis==='x'?{minSpan:1}:{minValueSpan:(this.focus.y[1]-this.focus.y[0])*.01}),
      zoomOnMouseWheel:true,moveOnMouseWheel:false,moveOnMouseMove:false,throttle:16,
      ...(view?{rangeMode:['value','value'],startValue:view[axis][0],endValue:view[axis][1]}:
        {rangeMode:['percent','percent'],start:0,end:100,startValue:null,endValue:null})}));
    this.chart.setOption({animation:false,backgroundColor:'transparent',
      textStyle:{fontFamily:'Cascadia Code, Consolas, monospace',fontSize:10,color:'#8299b5'},
      grid:{left:52,right:16,top:12,bottom:29,containLabel:false},
      xAxis:{type:'value',scale:true,min:this.bounds.x[0],max:this.bounds.x[1],splitNumber:4,
        name:AXES[this.axis]||'X',nameLocation:'middle',nameGap:18,nameTextStyle:{fontSize:9},
        axisLine:{lineStyle:{color:'#354258'}},axisTick:{show:false},splitLine:{show:false},axisPointer:{snap:false},
        axisLabel:{fontSize:9,hideOverlap:true,formatter:value=>this.axis==='wall_time'?new Date(value*1000).toLocaleTimeString('en-GB'):number(value)}},
      yAxis:{type:'value',scale:true,min:this.bounds.y[0],max:this.bounds.y[1],splitNumber:4,
        axisLine:{show:false},axisTick:{show:false},axisLabel:{fontSize:9,formatter:axisNumber},
        splitLine:{lineStyle:{color:'#2a374b',width:1}}},
      dataZoom:zoom,
      tooltip:{trigger:'axis',triggerOn:'mousemove',renderMode:'html',appendToBody:true,confine:false,
        enterable:false,transitionDuration:0,showDelay:0,hideDelay:0,className:'logs-chart-tooltip',
        backgroundColor:'rgba(16,24,39,.72)',borderColor:'#35475f',borderWidth:1,padding:[8,10],
        textStyle:{color:'#d5e2f1',fontFamily:'Cascadia Code, Consolas, monospace',fontSize:11},
        axisPointer:{type:'line',axis:'x',snap:false,animation:false,lineStyle:{color:'#91a5be',type:'dashed'}},
        formatter:params=>this.tooltipMarkup(params),
        position:(point,params,dom,_rect,size)=>{
          dom.setAttribute('role','tooltip');
          const chartRect=this.plot.getBoundingClientRect(),[width,height]=size.contentSize;
          const x=chartRect.left+point[0],y=chartRect.top+point[1];
          const left=clamp(x+14,10,Math.max(10,innerWidth-width-10));
          const top=clamp(y+14+height>innerHeight-10?y-height-14:y+14,10,Math.max(10,innerHeight-height-10));
          return [left-chartRect.left,top-chartRect.top];
        }},
      series:this.series.map((item,index)=>({id:item.run_id||`series-${index}`,name:item.name||item.run_id,
        type:'line',showSymbol:item.points.length===1,symbol:'circle',symbolSize:5,clip:true,
        lineStyle:{width:1.8,color:item.color},itemStyle:{color:item.color},
        emphasis:{scale:false},encode:{x:0,y:1},data:item.points.map(point=>[point.x,point.y,point.raw])}))
    },{replaceMerge:['series','dataZoom'],lazyUpdate:false});
    this.syncDomains();
  }

  tooltipMarkup(params){
    const first=Array.isArray(params)?params[0]:params;
    let x=Number(first?.axisValue);
    if(this.pointer&&this.chart){
      const box=this.plot.getBoundingClientRect(),pixel=[this.pointer.x-box.left,this.pointer.y-box.top];
      if(this.chart.containPixel({gridIndex:0},pixel))x=this.chart.convertFromPixel({xAxisIndex:0},pixel[0]);
    }
    if(!Number.isFinite(x))return '';
    const lastX=Math.max(...this.series.map(item=>item.points[item.points.length-1].x));
    x=Math.min(x,lastX);
    this.plot.dataset.hoverX=String(x);
    const rows=this.series.map(item=>{
      const point=nearestSample(item.points,x);
      return `<tr class="logs-tooltip-row" data-run-id="${esc(item.run_id)}" data-x="${point.x}" data-raw="${point.raw===null?'':point.raw}" data-smoothed="${point.y}"><td class="logs-tooltip-run"><span class="logs-tooltip-run-label"><i class="logs-tooltip-swatch logs-color-bg-${item.colorIndex}"></i><span title="${esc(item.name||item.run_id)}">${esc(item.name||item.run_id)}</span></span></td><td class="logs-tooltip-value">${number(point.y)}</td><td class="logs-tooltip-raw">${number(point.raw)}</td><td class="logs-tooltip-x">${esc(xText(point.x,this.axis))}</td></tr>`;
    }).join('');
    return `<div class="logs-tooltip-title">${esc(AXES[this.axis]||'X')} ${esc(xText(x,this.axis))}</div><table class="logs-tooltip-table"><thead><tr><th>Run</th><th>Value</th><th>Raw</th><th>${esc(AXES[this.axis]||'X')}</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  syncDomains(){
    if(!this.chart||this.disposed||!this.bounds)return;
    const width=this.chart.getWidth(),height=this.chart.getHeight();
    const plot={left:52,right:width-16,top:12,bottom:height-29};
    const lower=this.chart.convertFromPixel({gridIndex:0},[plot.left,plot.bottom]);
    const upper=this.chart.convertFromPixel({gridIndex:0},[plot.right,plot.top]);
    if(!Array.isArray(lower)||!Array.isArray(upper)||![...lower,...upper].every(Number.isFinite))return;
    this.view={x:[lower[0],upper[0]],y:[lower[1],upper[1]]};
    for(const axis of ['x','y']){this.plot.dataset[`view${axis.toUpperCase()}`]=JSON.stringify(this.view[axis]);this.plot.dataset[`bounds${axis.toUpperCase()}`]=JSON.stringify(this.bounds[axis]);this.plot.dataset[`focus${axis.toUpperCase()}`]=JSON.stringify(this.focus[axis]);}
    this.plot.dataset.plot=JSON.stringify(plot);
  }
  resize(){if(!this.active||this.disposed)return;if(this.chart){this.chart.resize();this.syncDomains();}else this.draw();}
  reset(){this.zoomed=false;this.draw();}
  setActive(value){
    const active=Boolean(value);if(this.active===active)return;this.active=active;
    if(this.chart)this.chart.setOption({tooltip:{show:active},dataZoom:[{id:'scalar-x',disabled:!active},{id:'scalar-y',disabled:!active}]});
    if(active)this.resize();else this.scrollHandler();
  }
  destroy(){if(this.disposed)return;this.disposed=true;this.observer.disconnect();document.removeEventListener('scroll',this.scrollHandler,true);this.plot.removeEventListener('mousemove',this.pointerHandler,true);this.plot.removeEventListener('wheel',this.pointerHandler,true);this.plot.removeEventListener('pointerleave',this.leaveHandler);this.plot.removeEventListener('dblclick',this.resetHandler);this.plot.removeEventListener('keydown',this.keyHandler);this.chart?.dispose();this.element.remove();}
}
