/** Table 5 values rendered with hover inspection and optional popup controls. */
import {loadECharts} from './scalar-chart.js';

const COLORS=['#eabc78','#a59ee8','#78d9b2','#ed8796','#91bedf','#d2ac87','#abd58f','#ce95c5'];
const PAST_COLOR='#4edcff';
const esc=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

function numeric(value){
  if(typeof value!=='number'&&typeof value!=='string')return null;
  if(typeof value==='string'&&!value.trim())return null;
  const result=Number(value);
  return Number.isFinite(result)?result:null;
}

function shortModel(name){
  return String(name).replace(/\s*\(\d{4}\).*$/,'').replace(/\s*\[[^\]]+\].*$/,'')
    .replace(/\s*\(ours\)\s*$/i,'').replace(/\s+-\s+/g,'-').trim();
}

export function buildJointComparisonOption(data,{interactive=false}={}){
  const seen=new Set();
  const joints=(data?.columns||[]).flatMap((label,column)=>{
    const match=String(label).match(/^J(\d+)(?:\s|\(|$)/i),index=match?Number(match[1]):-1;
    if(index<0||index>18||seen.has(index))return [];
    seen.add(index);return [{column,index,label:`J${index}`}];
  }).sort((a,b)=>a.index-b.index);
  const names=new Map();
  let colorIndex=0;
  const series=(data?.rows||[]).filter(row=>Array.isArray(row)&&row.length).map(row=>{
    const name=String(row[0]??''),past=/PAST[-\s]?Net/i.test(name),color=past?PAST_COLOR:COLORS[colorIndex++%COLORS.length];
    names.set(name,shortModel(name));
    return {name,type:'line',data:joints.map(joint=>numeric(row[joint.column])),connectNulls:false,
      smooth:false,symbol:'circle',symbolSize:past?4.5:3,showSymbol:interactive,
      silent:false,legendHoverLink:interactive,z:past?10:2,
      lineStyle:{width:past?3:1.4,color,opacity:past?1:.76},
      itemStyle:{color,opacity:past?1:.86},
      emphasis:{disabled:!interactive,focus:interactive?'series':'none',lineStyle:{width:past?3.5:2.5}}};
  });
  return {
    animation:false,backgroundColor:'transparent',
    textStyle:{fontFamily:'Cascadia Code, Consolas, monospace',fontSize:interactive?12:10,color:'#91a5be'},
    grid:{left:49,right:18,top:25,bottom:interactive?83:54,containLabel:false},
    xAxis:{type:'category',data:joints.map(joint=>joint.label),boundaryGap:false,
      axisLine:{lineStyle:{color:'#354258'}},axisTick:{show:false},
      axisLabel:{interval:0,fontSize:interactive?11:9,hideOverlap:true,color:'#91a5be'}},
    yAxis:{type:'value',name:'MLE (cm)',nameGap:9,min:0,splitNumber:3,
      nameTextStyle:{fontSize:interactive?12:10,color:'#91a5be'},axisLine:{show:false},axisTick:{show:false},
      axisLabel:{fontSize:interactive?11:9,color:'#91a5be',formatter:value=>Number(value.toPrecision(3)).toString()},
      splitLine:{lineStyle:{color:'#2a374b',type:'dashed',opacity:.65}}},
    legend:{type:interactive?'scroll':'plain',bottom:3,left:12,right:12,itemWidth:15,itemHeight:6,itemGap:13,
      textStyle:{color:'#9db1c9',fontSize:interactive?12:10},selectedMode:interactive,formatter:name=>names.get(name)||name,
      pageIconColor:'#91a5be',pageIconInactiveColor:'#38485d',pageTextStyle:{color:'#91a5be'}},
    tooltip:{show:true,trigger:'axis',triggerOn:'mousemove|click',renderMode:'html',
      appendToBody:!interactive,confine:interactive,enterable:false,transitionDuration:0,
      showDelay:0,hideDelay:0,className:'paper-joint-tooltip',
      backgroundColor:'#101827',borderColor:'#35475f',textStyle:{color:'#d5e2f1',fontSize:interactive?12:11},
      axisPointer:{type:'line',snap:true,animation:false,lineStyle:{color:'#91a5be',type:'dashed'}},
      formatter:params=>{
        const items=Array.isArray(params)?params:[];
        if(!items.length)return '';
        const joint=items[0]?.axisValueLabel||items[0]?.name||'';
        const index=Number.isInteger(items[0].dataIndex)?items[0].dataIndex:joints.findIndex(item=>item.label===joint);
        const byName=new Map(items.map(item=>[item.seriesName,item]));
        const visible=interactive?items:series.map(item=>byName.get(item.name)||{seriesName:item.name,data:item.data[index]});
        return `<div class="paper-joint-tooltip-title">${esc(joint)}</div><table><thead><tr><th>Model</th><th>MLE</th></tr></thead><tbody>${visible.map(item=>{
          const color=series.find(row=>row.name===item.seriesName)?.lineStyle.color||'#91a5be';
          const value=typeof item.data==='number'&&Number.isFinite(item.data)?item.data.toFixed(2)+' cm':'-';
          return `<tr><td title="${esc(item.seriesName)}"><span class="paper-joint-tooltip-model"><svg viewBox="0 0 10 4" aria-hidden="true"><rect width="10" height="4" fill="${color}"/></svg><span>${esc(shortModel(item.seriesName))}</span></span></td><td>${value}</td></tr>`;
        }).join('')}</tbody></table>`;
      }},
    dataZoom:interactive?[{type:'inside',xAxisIndex:0,filterMode:'none',zoomOnMouseWheel:true,moveOnMouseWheel:false},
      {type:'slider',xAxisIndex:0,filterMode:'none',bottom:31,height:14,borderColor:'#354258',
        backgroundColor:'#141d2b',fillerColor:'#4ec3ff22',dataBackground:{lineStyle:{color:'#4b657c'},areaStyle:{color:'#24384b'}},
        textStyle:{color:'#91a5be',fontSize:9},handleStyle:{color:'#4ec3ff'}}]:[],
    series,
  };
}

export function mountJointComparison(container,data,{interactive=false}={}){
  const option=buildJointComparisonOption(data,{interactive});
  let chart=null,disposed=false,active=true;
  container.classList.add('paper-joint-chart');
  container.classList.toggle('paper-joint-chart-preview',!interactive);
  container.setAttribute('role','img');
  container.setAttribute('aria-label','Per-joint MLE comparison in centimeters');
  container.dataset.jointCount=String(option.xAxis.data.length);
  container.dataset.modelCount=String(option.series.length);
  container.dataset.interactive=String(interactive);
  container.dataset.hover='true';
  container.dataset.missingValues=String(option.series.reduce((sum,item)=>sum+item.data.filter(value=>value===null).length,0));
  const available=option.series.some(item=>item.data.some(value=>typeof value==='number'));
  function hideTip(){chart?.dispatchAction({type:'hideTip'});}
  function resize(){if(active&&!disposed){hideTip();chart?.resize();}}
  document.addEventListener('scroll',hideTip,true);
  if(!interactive)option.tooltip.position=(point,_params,_dom,_rect,size)=>{
    const box=container.getBoundingClientRect(),[width,height]=size.contentSize;
    const x=Math.max(10,Math.min(box.left+point[0]+14,innerWidth-width-10));
    const pointerY=box.top+point[1],y=Math.max(10,Math.min(pointerY+height+14>innerHeight?pointerY-height-14:pointerY+14,innerHeight-height-10));
    return [x-box.left,y-box.top];
  };
  const observer=new ResizeObserver(resize);observer.observe(container);
  const ready=available?loadECharts().then(library=>{
    if(disposed)return;
    chart=library.init(container,null,{renderer:'canvas'});
    container.dataset.paperJointRenderer='echarts';
    chart.setOption(option);if(active)resize();
  }).catch(error=>{
    if(disposed)return;
    const message=document.createElement('div');message.className='paper-empty';
    message.textContent=error.message||'Chart unavailable';container.replaceChildren(message);
  }):Promise.resolve().then(()=>{
    if(disposed)return;
    const message=document.createElement('div');message.className='paper-empty';message.textContent='No per-joint metrics';
    container.replaceChildren(message);
  });
  return {ready,setActive(value){active=Boolean(value);if(active)resize();else chart?.dispatchAction({type:'hideTip'});},
    dispose(){if(disposed)return;disposed=true;observer.disconnect();document.removeEventListener('scroll',hideTip,true);chart?.dispose();chart=null;container.replaceChildren();}};
}
