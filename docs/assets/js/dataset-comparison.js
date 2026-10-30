(() => {
  const container = document.getElementById('datasetPointComparison');
  const data = window.DATASET_POINT_COMPARISON;
  if (!container || !data) return;
  if (!window.echarts) {
    container.textContent = 'Chart unavailable';
    return;
  }

  const chart = window.echarts.init(container);
  const render = () => {
    const compact = container.clientWidth < 600;
    chart.resize();
    chart.setOption({
      backgroundColor: '#fff',
      animation: false,
      color: ['#9b713e', '#2c819b', '#db875a'],
      legend: {
        top: 16, textStyle: { color: '#475467', fontSize: compact ? 11 : 14 },
        data: ['P5–P95', 'Min–max / IQR', 'Mean'],
      },
      grid: { left: compact ? 66 : 92, right: 24, top: 65, bottom: compact ? 82 : 92 },
      tooltip: {
        trigger: 'axis', confine: true,
        formatter: items => {
          const row = data.statsRows[items[0]?.dataIndex];
          if (!row) return '';
          const value = number => Number(number).toLocaleString('en-US', { maximumFractionDigits: 2 });
          return `<strong>${row.label}</strong><br>Mean: ${value(row.mean)}<br>Median: ${value(row.median)}<br>Q1–Q3: ${value(row.q1)}–${value(row.q3)}<br>Min–max: ${value(row.min)}–${value(row.max)}<br>P5–P95: ${value(row.p5)}–${value(row.p95)}`;
        },
      },
      xAxis: {
        type: 'category', data: data.labels, name: 'Dataset', nameLocation: 'middle',
        nameGap: compact ? 52 : 58, axisTick: { show: false },
        axisLabel: { interval: 0, rotate: compact ? 20 : 0, fontSize: compact ? 11 : 15, color: '#475467' },
        nameTextStyle: { fontSize: compact ? 13 : 17, color: '#344054' },
      },
      yAxis: {
        type: 'value', min: data.yMin, max: data.yMax, interval: 100,
        name: 'Radar points / frame', nameLocation: 'middle', nameGap: compact ? 42 : 54,
        nameTextStyle: { fontSize: compact ? 13 : 17, color: '#344054' },
        axisLabel: { fontSize: compact ? 11 : 14, color: '#475467' },
        splitLine: { lineStyle: { color: '#e0e2e5', type: 'dashed' } },
      },
      series: [
        {
          name: 'P5–P95', type: 'custom', data: data.percentileData, z: 2,
          renderItem: (_parameters, api) => {
            const low = api.coord([api.value(0), api.value(1)]);
            const high = api.coord([api.value(0), api.value(2)]);
            const half = Math.max(7, container.clientWidth * .018);
            const style = { stroke: '#9b713e', lineWidth: 2 };
            return { type: 'group', children: [
              { type: 'line', shape: { x1: low[0], y1: low[1], x2: high[0], y2: high[1] }, style },
              { type: 'line', shape: { x1: low[0] - half, y1: low[1], x2: low[0] + half, y2: low[1] }, style },
              { type: 'line', shape: { x1: high[0] - half, y1: high[1], x2: high[0] + half, y2: high[1] }, style },
            ] };
          },
        },
        {
          name: 'Min–max / IQR', type: 'boxplot', data: data.boxData,
          boxWidth: ['22%', '38%'], z: 3,
          itemStyle: { color: 'rgba(44,129,155,.19)', borderColor: '#2c819b', borderWidth: 2 },
        },
        {
          name: 'Mean', type: 'scatter', data: data.meanData,
          symbol: 'diamond', symbolSize: compact ? 10 : 16, z: 4,
          itemStyle: { color: '#db875a', borderColor: '#fff', borderWidth: 1 },
        },
      ],
    });
  };
  const observer = new ResizeObserver(render);
  observer.observe(container);
  render();
  window.addEventListener('pagehide', () => { observer.disconnect(); chart.dispose(); }, { once: true });
})();
