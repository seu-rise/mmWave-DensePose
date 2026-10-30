'use strict';
(() => {
 const payload=window.FIGURE_05_DATA;
 const parameters=new URLSearchParams(location.search);
 const mode=parameters.get('mode')==='interactive'?'interactive':'preview';
 const interactive=mode==='interactive';
 window.figure05Mode=Object.freeze({mode,interactive});
 document.body.classList.add(mode);
 const panel=parameters.get('panel');
 document.querySelectorAll('[data-open-panel]').forEach(card=>{
  if(interactive)return;
  card.tabIndex=0;card.setAttribute('role','button');card.setAttribute('aria-label',`Open subplot (${card.dataset.openPanel})`);card.setAttribute('aria-haspopup','dialog');
  const open=event=>{event.preventDefault();window.parent.postMessage({type:'paper:open-subplot',renderer:'echarts',figure:5,panel:card.dataset.openPanel},'*')};
  card.addEventListener('click',open);card.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key))open(event)});
 });
 if(panel==='a'||panel==='b'){
  document.body.classList.add('single');
  document.getElementById('panel-'+(panel==='a'?'b':'a')).hidden=true;
 }
 const charts={};
 const rgba=(hex,alpha)=>{const n=parseInt(hex.slice(1),16);return `rgba(${n>>16},${n>>8&255},${n&255},${alpha})`;};
 const scaled=(width,value,min=10,max=value)=>Math.max(min,Math.min(max,Math.round(value*width/1120)));
 function histogramOption(width){
  const d=payload.a,label=scaled(width,17,10,17),name=scaled(width,22,14,22),legend=scaled(width,18,10,18);
  const bar='#467ab6',line='#ce7948',mark='#805b2e';
  return {backgroundColor:'#fff',animation:false,tooltip:{show:false},legend:{top:0,right:30,itemGap:18,selectedMode:false,textStyle:{fontSize:legend,color:'#344054',fontFamily:'Arial, sans-serif'},data:['Frame proportion','Cumulative proportion']},grid:{left:scaled(width,94,52,94),right:scaled(width,104,64,104),top:scaled(width,82,48,82),bottom:scaled(width,83,56,83),containLabel:true},xAxis:{type:'category',name:'Radar points per frame',nameLocation:'middle',nameGap:scaled(width,54,34,54),nameTextStyle:{fontFamily:'Times New Roman',fontWeight:'bold',fontSize:name,color:'#000'},data:d.labels,axisTick:{alignWithLabel:true,lineStyle:{color:'#c6c8cc'}},axisLine:{lineStyle:{color:'#c6c8cc'}},axisLabel:{fontFamily:'Arial, sans-serif',color:'#475467',fontSize:label,interval:index=>index%5===0,margin:10}},yAxis:[{type:'value',name:'Frame proportion (%)',nameLocation:'middle',nameRotate:90,nameGap:scaled(width,58,35,58),min:0,max:d.yMax,interval:2,nameTextStyle:{fontFamily:'Times New Roman',fontWeight:'bold',fontSize:name,color:'#000'},axisLabel:{color:'#475467',fontFamily:'Arial, sans-serif',fontSize:label,formatter:'{value}%',showMaxLabel:true},splitLine:{lineStyle:{color:'#e0e2e5',type:'dashed'}}},{type:'value',name:'Cumulative proportion (%)',nameLocation:'middle',nameRotate:-90,nameGap:scaled(width,64,39,64),min:0,max:100,interval:20,nameTextStyle:{fontFamily:'Times New Roman',fontWeight:'bold',fontSize:name,color:'#000'},axisLabel:{color:'#475467',fontFamily:'Arial, sans-serif',fontSize:label,formatter:'{value}%'},splitLine:{show:false}}],series:[{name:'Frame proportion',type:'bar',data:d.percentages,barMaxWidth:22,itemStyle:{borderRadius:[5,5,0,0],color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:rgba(bar,.98)},{offset:1,color:rgba(bar,.43)}])},markLine:{silent:true,symbol:'none',data:[{xAxis:d.meanLabel,lineStyle:{color:mark,type:'dashed',width:2},label:{formatter:`Mean = ${d.mean.toFixed(1)}; Median = ${d.median.toFixed(1)}`,fontSize:scaled(width,18,10,18),fontWeight:'bold',color:mark,position:'end',rotate:0,align:'left',verticalAlign:'bottom',distance:8}}]}},{name:'Cumulative proportion',type:'line',yAxisIndex:1,data:d.cumulativePercentages,smooth:true,showSymbol:false,lineStyle:{color:line,width:3},areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:rgba(line,.16)},{offset:1,color:rgba(line,.02)}])}}]};
 }
 function boxOption(width){
  const d=payload.b,label=scaled(width,20,11,20),name=scaled(width,26,15,26),box='#2c819b',mean='#db875a',pct='#9b713e';
  return {backgroundColor:'#fff',animation:false,tooltip:{show:false},grid:{left:scaled(width,84,47,84),right:scaled(width,28,12,28),top:scaled(width,25,12,25),bottom:scaled(width,91,64,91),containLabel:true},xAxis:{type:'category',name:'Dataset',nameLocation:'middle',nameGap:scaled(width,58,37,58),nameTextStyle:{fontFamily:'Times New Roman',fontWeight:'bold',fontSize:name,color:'#000'},data:['Ours','MARS','mRI','MiliPoint','mmPoseNLP'],boundaryGap:true,axisTick:{show:false},axisLine:{lineStyle:{color:'#bfc4cb'}},axisLabel:{fontFamily:'Times New Roman',fontWeight:'bold',fontSize:label,color:'#000',margin:14,interval:0}},yAxis:{type:'value',name:'Radar points / frame',nameLocation:'middle',nameRotate:90,nameGap:scaled(width,58,36,58),min:d.yMin,max:d.yMax,interval:100,nameTextStyle:{fontFamily:'Times New Roman',fontWeight:'bold',fontSize:name,color:'#000'},axisLabel:{fontFamily:'Arial, sans-serif',color:'#475467',fontSize:label,showMaxLabel:true},splitLine:{lineStyle:{color:'#e0e2e5',type:'dashed'}}},series:[{name:'P5-P95',type:'custom',data:d.percentileData,renderItem:(params,api)=>{const low=api.coord([api.value(0),api.value(1)]),high=api.coord([api.value(0),api.value(2)]),half=Math.max(10,width*.031);const style={stroke:pct,lineWidth:2.2,opacity:.94};return {type:'group',children:[{type:'line',shape:{x1:low[0],y1:low[1],x2:high[0],y2:high[1]},style},{type:'line',shape:{x1:low[0]-half,y1:low[1],x2:low[0]+half,y2:low[1]},style},{type:'line',shape:{x1:high[0]-half,y1:high[1],x2:high[0]+half,y2:high[1]},style}]};},z:2},{name:'Min-Max / IQR',type:'boxplot',data:d.boxData,boxWidth:['28%','44%'],itemStyle:{color:rgba(box,.19),borderColor:box,borderWidth:2.2},z:3},{name:'Mean',type:'scatter',data:d.meanData,symbol:'diamond',symbolSize:scaled(width,20,10,20),itemStyle:{color:mean,borderColor:'#fff',borderWidth:2},z:4}]};
 }
 function withMode(option,id){
  option.tooltip={show:interactive,trigger:id==='a'?'axis':'item',confine:true,backgroundColor:'#fffffffa',borderColor:'#d6dde5',textStyle:{color:'#263442',fontFamily:'Arial, sans-serif',fontSize:12},formatter:params=>{
   const value=number=>Number(number).toLocaleString('en-US',{maximumFractionDigits:4});
   if(id==='a'){
    const item=Array.isArray(params)?params[0]:params,row=payload.a.rows[item?.dataIndex];
    return row?`<strong>${row.label}</strong><br>Frames: ${value(row.frame_count)}<br>Frame proportion: ${value(row.frame_percentage)}%<br>Cumulative proportion: ${value(row.cumulative_frame_percentage)}%`:'';
   }
   const row=payload.b.statsRows[params.dataIndex];
   return row?`<strong>${option.xAxis.data[params.dataIndex]}</strong><br>Min: ${value(row.min)}<br>Q1: ${value(row.q1)}<br>Median: ${value(row.median)}<br>Q3: ${value(row.q3)}<br>Max: ${value(row.max)}<br>Mean: ${value(row.mean)}<br>P5–P95: ${value(row.p5)}–${value(row.p95)}`:'';
  }};
  if(option.legend)option.legend.selectedMode=interactive;
  option.series.forEach(series=>{series.silent=!interactive;series.emphasis={...(series.emphasis||{}),disabled:!interactive};});
  option.dataZoom=interactive&&id==='a'?[{type:'inside',xAxisIndex:0,filterMode:'none',start:0,end:100,zoomOnMouseWheel:true,moveOnMouseMove:true}]:[];
  return option;
 }
 function mount(id,option){const el=document.getElementById('chart-'+id);if(el.parentElement.hidden)return;el.inert=!interactive;el.tabIndex=interactive?0:-1;const chart=echarts.init(el,null,{renderer:'canvas'});charts[id]=chart;const draw=()=>{chart.resize();chart.setOption(withMode(option(el.clientWidth),id),true)};new ResizeObserver(draw).observe(el);draw();}
 mount('a',histogramOption);mount('b',boxOption);
 window.figure05Charts=charts;
 window.figure05Axes={a:{x:'Radar points per frame',left:[0,9.2],right:[0,100]},b:{x:'Dataset',y:[0,460],order:['Ours','MARS','mRI','MiliPoint','mmPoseNLP']}};
})();
