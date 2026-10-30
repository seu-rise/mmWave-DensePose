/** Compact context help and the official public resource entry point. */
export const SCIENCEDB_URL='https://www.scidb.cn/detail?dataSetId=96bb16b514eb4f69b79293d230666c10';
const escape=value=>String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const question='<svg viewBox="0 0 20 20" aria-hidden="true" focusable="false"><circle cx="10" cy="10" r="7.25"/><path d="M7.7 7.4a2.4 2.4 0 0 1 4.6 1c0 1.6-2.3 1.7-2.3 3.2"/><path d="M10 14.1h.01"/></svg>';
const database='<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><ellipse cx="9" cy="5.5" rx="6" ry="2.5"/><path d="M3 5.5v12c0 1.4 2.7 2.5 6 2.5 1.2 0 2.3-.1 3.2-.4M3 11.5C3 12.9 5.7 14 9 14c1.2 0 2.3-.1 3.2-.4M15 5.5v4M15.5 18.5 22 12m-5 0h5v5"/></svg>';

export function helpIcon(text,label='Help'){
  return `<button type="button" class="ui-help-icon" data-ui-help="${escape(text)}" data-loading="false" aria-label="${escape(label)}" aria-pressed="false">${question}</button>`;
}
export function scienceDbLink(label='Open dataset, weights and predictions on ScienceDB'){
  return `<a class="ui-sciencedb-link" href="${SCIENCEDB_URL}" target="_blank" rel="noopener noreferrer" data-loading="false" aria-label="${escape(label)}" title="${escape(label)}">${database}</a>`;
}
function elementFrom(markup){const template=document.createElement('template');template.innerHTML=markup;return template.content.firstElementChild;}
export function createHelpIcon(text,label='Help'){return elementFrom(helpIcon(text,label));}
export function createScienceDbLink(label){return elementFrom(scienceDbLink(label));}

let installed=false;
export function installContextHelp(){
  if(installed)return;
  installed=true;
  const tooltip=document.createElement('div');
  tooltip.id='ui-context-tooltip';tooltip.className='ui-context-tooltip';tooltip.setAttribute('role','tooltip');
  tooltip.setAttribute('popover','manual');tooltip.hidden=true;document.body.append(tooltip);
  let trigger=null,pinned=false;
  const target=event=>event.target instanceof Element?event.target.closest('[data-ui-help]'):null;
  function close(){
    if(trigger){trigger.removeAttribute('aria-describedby');trigger.setAttribute('aria-pressed','false');}
    if(typeof tooltip.hidePopover==='function'&&tooltip.matches(':popover-open'))tooltip.hidePopover();
    tooltip.hidden=true;trigger=null;pinned=false;
  }
  function place(){
    if(!trigger?.isConnected||!trigger.getClientRects().length){close();return;}
    const anchor=trigger.getBoundingClientRect(),box=tooltip.getBoundingClientRect();
    const width=document.documentElement.clientWidth,height=window.innerHeight,gap=8;
    const left=Math.max(12,Math.min(width-box.width-12,anchor.left+anchor.width/2-box.width/2));
    let top=anchor.bottom+gap;
    if(top+box.height>height-12)top=anchor.top-box.height-gap;
    tooltip.style.left=`${left}px`;tooltip.style.top=`${Math.max(12,Math.min(height-box.height-12,top))}px`;
  }
  function show(button,pin=false){
    if(trigger!==button)close();
    trigger=button;pinned=pin;tooltip.textContent=button.dataset.uiHelp;
    tooltip.hidden=false;
    if(typeof tooltip.showPopover==='function'&&!tooltip.matches(':popover-open'))tooltip.showPopover();
    trigger.setAttribute('aria-describedby',tooltip.id);trigger.setAttribute('aria-pressed',String(pin));place();
  }
  document.addEventListener('pointerover',event=>{
    const button=target(event);if(!button||button.contains(event.relatedTarget)||pinned)return;show(button);
  });
  document.addEventListener('pointerout',event=>{
    const button=target(event);if(button&&button===trigger&&!button.contains(event.relatedTarget)&&!pinned&&document.activeElement!==button)close();
  });
  document.addEventListener('focusin',event=>{const button=target(event);if(button&&!pinned)show(button);});
  document.addEventListener('focusout',event=>{if(target(event)===trigger&&!pinned)close();});
  document.addEventListener('click',event=>{
    const button=target(event);
    if(button){event.preventDefault();event.stopPropagation();if(trigger===button&&pinned)close();else show(button,true);}
    else close();
  });
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&trigger){event.preventDefault();event.stopPropagation();close();}},true);
  document.addEventListener('scroll',close,{capture:true,passive:true});
  window.addEventListener('resize',close);
  window.addEventListener('pagehide',close);
  new MutationObserver(()=>{
    if(!trigger||trigger.isConnected&&trigger.getClientRects().length)return;
    const label=trigger.getAttribute('aria-label'),copy=trigger.dataset.uiHelp;
    const replacements=[...document.querySelectorAll('[data-ui-help]')]
      .filter(button=>button.getAttribute('aria-label')===label&&button.dataset.uiHelp===copy&&button.getClientRects().length);
    if(replacements.length===1&&(pinned||replacements[0].matches(':hover')))show(replacements[0],pinned);
    else close();
  })
    .observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:['hidden','aria-hidden']});
}
