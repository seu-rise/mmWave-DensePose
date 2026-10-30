const BAR_TABLES=new Set([4,6,7,8,9,10]);

export function metricValue(value){
  if(typeof value==='number')return Number.isFinite(value)&&value>=0?value:null;
  if(typeof value!=='string')return null;
  const match=value.trim().match(/^(\d+(?:\.\d+)?)(?:\s*±\s*\d+(?:\.\d+)?)?$/);
  return match?Number(match[1]):null;
}

export function tableMetricColumns(id,data){
  const columns=new Map();
  if(!BAR_TABLES.has(id)||!data)return columns;
  const first=id===7?2:1;
  data.columns.forEach((label,column)=>{
    if(column<first)return;
    const values=data.rows.map(row=>metricValue(row[column])).filter(value=>value!==null);
    if(!values.length)return;
    const higher=/PCK|AUC/i.test(label),percentage=/%|PCK|AUC/i.test(label);
    columns.set(column,{maximum:percentage?100:Math.max(...values),
      best:higher?Math.max(...values):Math.min(...values),higher});
  });
  return columns;
}

export function metricPercentage(value,maximum){
  const number=metricValue(value);
  return number===null||!Number.isFinite(maximum)||maximum<=0?0:Math.max(0,Math.min(100,number/maximum*100));
}
