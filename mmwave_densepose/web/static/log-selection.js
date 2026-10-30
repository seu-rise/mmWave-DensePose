/** Bulk log selection is independent of the eight-run scalar comparison limit. */
export const chartRunIds=selected=>selected.slice(0,8);

export function selectedUserRuns(runs,selected,details={}){
  const byId=new Map([...Object.values(details),...runs].map(run=>[run.run_id||run.id,run]));
  return [...new Set(selected)].map(id=>byId.get(id)).filter(run=>
    run?.origin==='user'&&run.can_delete===true&&(run.action||run.task)==='train');
}
