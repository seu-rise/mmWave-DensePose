/** Shared presentation of explicit release-installation readiness. */
export function datasetNeedsDownload(status){
  return Boolean(status&&(status.availability==='missing'||status.availability==='partial'||status.installation_incomplete||status.can_resume));
}
export function datasetBlocksModelTasks(status){
  return Boolean(status&&status.requires_full_dataset!==false&&(status.training_ready===false||status.availability==='missing'||status.availability==='partial'||status.installation_incomplete));
}
export function datasetStatusText(status,{compact=false}={}){
  if(!status)return '';
  if(status.requires_full_dataset===false)return 'Custom dataset';
  const sequences=Number.isInteger(status.installed_sequences)&&Number.isInteger(status.total_sequences)?`${status.installed_sequences} / ${status.total_sequences} sequences installed`:'';
  if(compact&&datasetBlocksModelTasks(status))return ['Dataset incomplete',sequences,'Continue download in Dataset to enable model tasks'].filter(Boolean).join(' · ');
  const cache=Number(status.cached_archives)>0?`${status.cached_archives} cached package${Number(status.cached_archives)===1?'':'s'} (checked when continuing)`:'';
  if(datasetBlocksModelTasks(status))return [status.availability==='missing'?'Dataset not installed':'Dataset incomplete',sequences,cache,status.browsing_available?'Installed sequences can be browsed':'', 'Complete the dataset download to enable model tasks'].filter(Boolean).join(' · ');
  return ['Dataset ready',sequences].filter(Boolean).join(' · ');
}
