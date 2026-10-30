/** Four top-level workbench areas share the existing application window. */
export function setupTabs(onChange) {
  const tabs = [...document.querySelectorAll('.workbench-header .tabs [role="tab"]')];
  function select(tab, focus = false) {
    for (const item of tabs) {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
    }
    const dataset = tab.id === "tab-dataset";
    document.getElementById("panel-dataset").hidden = !dataset;
    const panel = document.getElementById("panel-workbench");
    panel.hidden = dataset;
    panel.setAttribute("aria-labelledby", tab.id);
    if (focus) tab.focus();
    onChange(dataset ? "dataset" : tab.id.replace("tab-", ""));
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", event => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); select(tabs[next], true); }
    });
  });
  const requested=new URLSearchParams(location.search).get('view');
  const mapped={official:'paper',tasks:'training',logs:'training','result-metrics':'results',compare:'results',overview:'dataset',verification:'dataset'}[requested]||requested||'dataset';
  select(tabs.find(tab=>tab.id==='tab-'+mapped)||tabs[0]);
}
