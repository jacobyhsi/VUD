window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  // === Bulma Carousel ===
  var options = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 5000,
  };
  bulmaCarousel.attach('.carousel', options);
  bulmaSlider.attach();

  // === Helper Functions for plot actions ===

  // === Plotly axis and heatmap preparation ===
  const xmin = -4.5;
  const xmax = 4.5;
  const ymin = -4.5;
  const ymax = 4.5;
  const step = 0.5;

  const x = [];
  const y = [];
  for (let val = xmin; val <= xmax; val += step) {
    x.push(val);
  }
  for (let val = ymin; val <= ymax; val += step) {
    y.push(val);
  }

  const z = [];
  for (let i = 0; i < y.length; i++) {
    const row = [];
    for (let j = 0; j < x.length; j++) {
      row.push(0);
    }
    z.push(row);
  }

  const trace1 = {
    x: [],
    y: [],
    mode: 'markers',
    marker: {
      color: [],
      size: [],
      line: { width: 2, color: 'white' }
    },
    type: 'scatter',
    hoverinfo: 'none'
  };

  const totalScale = [
    [0, 'rgba(230, 247, 255, 1)'],
    [0.2, 'rgba(166, 213, 245, 1)'],
    [0.4, 'rgba(101, 179, 232, 1)'],
    [0.6, 'rgba(57, 148, 219, 1)'],
    [0.8, 'rgba(17, 112, 185, 1)'],
    [1, 'rgba(8, 69, 148, 1)']
  ];

  const aleatoricScale = [
    [0, 'rgba(255, 247, 230, 1)'],
    [0.2, 'rgba(255, 227, 166, 1)'],
    [0.4, 'rgba(255, 204, 101, 1)'],
    [0.6, 'rgba(255, 180, 57, 1)'],
    [0.8, 'rgba(255, 157, 17, 1)'],
    [1, 'rgba(230, 125, 0, 1)']
  ];

  const epistemicScale = [
    [0, 'rgba(255, 230, 230, 1)'],
    [0.2, 'rgba(255, 189, 189, 1)'],
    [0.4, 'rgba(255, 140, 140, 1)'],
    [0.6, 'rgba(255, 94, 94, 1)'],
    [0.8, 'rgba(255, 48, 48, 1)'],
    [1, 'rgba(204, 0, 0, 1)']
  ];

  let heatmapData = {
    total: {
      z: JSON.parse(JSON.stringify(z)),
      x: x,
      y: y,
      type: 'heatmap',
      colorscale: totalScale,
      zmin: 0,
      zmax: 1,
      opacity: 0.6,
      hoverinfo: 'none',
      interpolate: true
    },
    aleatoric: {
      z: JSON.parse(JSON.stringify(z)),
      x: x,
      y: y,
      type: 'heatmap',
      colorscale: aleatoricScale,
      zmin: 0,
      zmax: 1,
      opacity: 0.6,
      hoverinfo: 'none'
    },
    epistemic: {
      z: JSON.parse(JSON.stringify(z)),
      x: x,
      y: y,
      type: 'heatmap',
      colorscale: epistemicScale,
      zmin: 0,
      zmax: 1,
      opacity: 0.6,
      hoverinfo: 'none'
    }
  };
  let currentHeatmapType = 'total'

  // === Plotly axis and layout setup ===
  const allTickValsX = [];
  const allTickTextX = [];
  const allTickValsY = [];
  const allTickTextY = [];

  for (let val = xmin; val <= xmax; val += step) {
    allTickValsX.push(val);
    allTickTextX.push(Number.isInteger(val) ? val.toString() : '');
  }
  for (let val = ymin; val <= ymax; val += step) {
    allTickValsY.push(val);
    allTickTextY.push(Number.isInteger(val) ? val.toString() : '');
  }

  const layout = {
    height: 600,
    width: '100%',
    margin: { t: 0, b: 55 },
    xaxis: {
      range: [xmin, xmax],
      fixedrange: true,
      automargin: true,
      tickvals: allTickValsX,
      ticktext: allTickTextX,
      gridcolor: 'rgba(200, 200, 200, 0.8)',
      tickcolor: 'rgba(200, 200, 200, 0.8)',
      ticks: 'outside',
      ticklen: 8,
      tickwidth: 1.5
    },
    yaxis: {
      range: [ymin, ymax],
      fixedrange: true,
      automargin: true,
      tickvals: allTickValsY,
      ticktext: allTickTextY,
      gridcolor: 'rgba(200, 200, 200, 0.8)',
      tickcolor: 'rgba(200, 200, 200, 0.8)',
      ticks: 'outside',
      ticklen: 8,
      tickwidth: 1.5
    }
  };
  const config = { displayModeBar: false };

  // === Helper Functions ===
  function roundToHalf(num) {
    return Math.round(num * 2) / 2;
  }

  function isPointExists(x, y) {
    return trace1.x.some((xi, i) => Math.abs(xi - x) < 1e-6 && Math.abs(trace1.y[i] - y) < 1e-6);
  }

  function updatePlot() {
    if (!Array.isArray(trace1.marker.size)) {
      trace1.marker.size = Array(trace1.x.length).fill(trace1.marker.size || 12);
    }

    Plotly.deleteTraces('plot', 1);
    Plotly.addTraces('plot', trace1);

    updatePointInfo();
    sendPointsToBackend()
  }

  function updatePointInfo() {
    // const infoDiv = document.getElementById('point-info');
    // let infoHTML = '<strong>Added dots:</strong><ul>';
    // for (let i = 0; i < trace1.x.length; i++) {
    //   infoHTML += `<li>Point ${i + 1}: (${trace1.x[i].toFixed(2)}, ${trace1.y[i].toFixed(2)}), ${trace1.marker.color[i]}</li>`;
    // }
    // infoHTML += '</ul>';
    // infoDiv.innerHTML = infoHTML;
  }

  // === Drag balls ===
  document.querySelectorAll('.draggable-ball').forEach(ball => {
    ball.addEventListener('dragstart', e => {
      e.dataTransfer.setData('label', ball.dataset.color);
    });
  });

  const plotDiv = document.getElementById('plot');

  plotDiv.addEventListener('dragover', e => {
    e.preventDefault();
  });

  plotDiv.addEventListener('drop', e => {
    e.preventDefault();

    const color = e.dataTransfer.getData('label');
    const [xPix, yPix] = [e.offsetX, e.offsetY];

    const xaxis = plotDiv._fullLayout.xaxis;
    const yaxis = plotDiv._fullLayout.yaxis;

    const xVal = roundToHalf(xaxis.p2c(xPix - xaxis._offset));
    const yVal = roundToHalf(yaxis.p2c(yPix - yaxis._offset));

    if (isPointExists(xVal, yVal) || xVal <= xmin || xVal >= xmax || yVal <= ymin || yVal >= ymax) {
      return;
    }

    // add point
    trace1.x.push(xVal);
    trace1.y.push(yVal);
    trace1.marker.color.push(color);
    trace1.marker.size.push(12);

    updatePlot();
  });

  // === Click to delete existing point ===
  Plotly.newPlot('plot', [heatmapData.total, trace1], layout, config).then(plot => { setupPlotlyClickHandler(plot, plotDiv, trace1, updatePlot); });

  // TODO: Other functions
  // document.getElementById('reset-btn').addEventListener('click', function () {
  //   trace1.x = [];
  //   trace1.y = [];
  //   trace1.marker.color = [];
  //   trace1.marker.size = [];

  //   updatePlot();
  // });

  // document.getElementById('heatmap-intensity').addEventListener('input', function (e) {
  //   const intensity = parseFloat(e.target.value);
  //   heatmap.opacity = intensity;
  //   Plotly.restyle(plotDiv, 'opacity', intensity, 0); 
  // });

  // document.getElementById('zoom-reset').addEventListener('click', function () {
  //   Plotly.relayout(plotDiv, {
  //     'xaxis.range': [xmin, xmax],
  //     'yaxis.range': [ymin, ymax]
  //   });
  // });

  // document.getElementById('save-plot').addEventListener('click', function () {
  //   Plotly.downloadImage(plotDiv, { format: 'png', width: 1200, height: 800 });
  // });

  // === Communicate with backend python script === 
  async function sendPointsToBackend() {
    // Preparation of data  
    const points = [];
    for (let i = 0; i < trace1.x.length; i++) {
      points.push({
        x: trace1.x[i],
        y: trace1.y[i],
        label: trace1.marker.color[i]
      });
    }

    // send & obtain responses
    try {
      const response = await fetch('http://localhost:5000/process_points', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ points })
      });

      if (!response.ok) {
        throw new Error('Server Error');
      }

      console.log("result retrieved!");
      const result = await response.json();
      heatmapData.total.z = result.total;
      heatmapData.total.x = result.x_range;
      heatmapData.total.y = result.y_range;

      heatmapData.aleatoric.z = result.aleatoric;
      heatmapData.aleatoric.x = result.x_range;
      heatmapData.aleatoric.y = result.y_range;

      heatmapData.epistemic.z = result.epistemic;
      heatmapData.epistemic.x = result.x_range;
      heatmapData.epistemic.y = result.y_range;

      renderHeatmap(currentHeatmapType)

    } catch (error) {
      console.error('Error with point data handling:', error);
      alert('Error with point data handling: ' + error.message);
    }
  }

  function renderHeatmap(type) {
    if (!heatmapData[type]) return;
    // both heatmap and points are refreshed to avoid losing point when just alter uncertainty
    // it seems points are refreshed twice but this makes the animation better
    Plotly.newPlot('plot', [heatmapData[type], trace1], layout, config).then(plot => { setupPlotlyClickHandler(plot, plotDiv, trace1, updatePlot); })
    heatmapData.currentType = type;
  }

  function setupPlotlyClickHandler(plot, plotDiv, trace1, updatePlot) {
    plot.on('plotly_click', function (data) {
      if (!data.points || data.points.length === 0) return;
      const xPix = data.event.offsetX;
      const yPix = data.event.offsetY;

      const xaxis = plotDiv._fullLayout.xaxis;
      const yaxis = plotDiv._fullLayout.yaxis;
      const clickX = xaxis.p2c(xPix - xaxis._offset);
      const clickY = yaxis.p2c(yPix - yaxis._offset);

      let minDistance = Infinity;
      let closestPointIndex = -1;

      trace1.x.forEach((x, i) => {
        const y = trace1.y[i];
        const distance = Math.sqrt(Math.pow(x - clickX, 2) + Math.pow(y - clickY, 2));

        if (distance < minDistance) {
          minDistance = distance;
          closestPointIndex = i;
        }
      });

      const threshold = 0.15;
      if (closestPointIndex !== -1 && minDistance <= threshold) {
        const newMarker = JSON.parse(JSON.stringify(trace1.marker));
        newMarker.size[closestPointIndex] = 16;
        Plotly.restyle(plotDiv, 'marker', [newMarker]);

        setTimeout(() => {
          trace1.x.splice(closestPointIndex, 1);
          trace1.y.splice(closestPointIndex, 1);
          trace1.marker.color.splice(closestPointIndex, 1);
          trace1.marker.size.splice(closestPointIndex, 1);

          updatePlot();
        }, 200);
      }
    });
  }

  // === Select to alter uncertainty type ===
  const heatmapTypeSelect = document.getElementById('heatmap-type-select');
  heatmapTypeSelect.addEventListener('change', function () {
    const selectedType = this.value;
    currentHeatmapType = selectedType;
    renderHeatmap(selectedType);
  });
// === Calculate single point ===
document.getElementById('calculate-btn').addEventListener('click', async function() {
  console.error('Pressed');
  const inputX = document.getElementById('input-x');
  const inputY = document.getElementById('input-y');
  const xValue = parseFloat(inputX.value);
  const yValue = parseFloat(inputY.value);

  // 1. 保留原输入校验逻辑
  if (isNaN(xValue) || xValue < -4.5 || xValue > 4.5) {
    alert('Please enter a valid X value (range: -4.5 ~ 4.5)');
    inputX.focus();
    return;
  }
  if (isNaN(yValue) || yValue < -4.5 || yValue > 4.5) {
    alert('Please enter a valid Y value (range: -4.5 ~ 4.5)');
    inputY.focus();
    return;
  }

  // 2. 获取选择的配置（仅传给后端，不用于本地生成Prompt）
  const { prompt_structure, model } = getSelectedConfig();

  // 3. 准备参考点数据（传给后端，用于后端生成Prompt）
  const points = [];
  for (let i = 0; i < trace1.x.length; i++) {
    points.push({
      x: trace1.x[i],
      y: trace1.y[i],
      label: trace1.marker.color[i] 
    });
  }

  console.error(prompt_structure, model, xValue, yValue);

  // 4. 准备请求数据（无本地Prompt生成逻辑，直接传原始配置给后端）
  const requestData = {
    prompt_structure: prompt_structure,  
    model: model,                        
    x: xValue,                           
    y: yValue,                           
    points: points                    
  };

  try {
    // 5. 发送请求到后端
    const response = await fetch('http://localhost:5000/calculate_single', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    // 6. 从后端获取所有数据（包括Prompt和三种不确定性值）
    const result = await response.json();

    console.error(result);

    const backendPrompt = result.prompt; // 后端返回的实际Prompt
    const totalUncertainty = result.total; // 后端返回的总不确定性
    const aleatoricUncertainty = result.aleatoric; // 后端返回的随机不确定性
    const epistemicUncertainty = result.epistemic; // 后端返回的认知不确定性

    // 7. 更新 PromptViewer 区域（展示后端返回的Prompt，这是核心修改）
    const promptElement = document.getElementById('generated-prompt');
    promptElement.textContent = backendPrompt || 'No valid prompt returned from server.';
    if (window.hljs) {
      hljs.highlightElement(promptElement);
    }

    // // 8. 更新 Debug 区域（不确定性值来自后端）
    // const pointInfoDiv = document.getElementById('point-info');
    // let existingPointsHTML = '<strong>Added dots:</strong><ul>';
    // points.forEach((p, i) => {
    //   existingPointsHTML += `<li>Point ${i+1}: (${p.x.toFixed(2)}, ${p.y.toFixed(2)}), ${p.label}</li>`;
    // });
    // existingPointsHTML += '</ul>';

    // const calculationHTML = `
    //   <strong>Uncertainty for (${xValue.toFixed(2)}, ${yValue.toFixed(2)}):</strong><br>
    //   - Prompt Structure: ${prompt_structure}<br>
    //   - Model: ${model}<br>
    //   - Total Uncertainty: ${totalUncertainty.toFixed(4)}<br>
    //   - Aleatoric Uncertainty: ${aleatoricUncertainty.toFixed(4)}<br>
    //   - Epistemic Uncertainty: ${epistemicUncertainty.toFixed(4)}<br>
    // `;
    // pointInfoDiv.innerHTML = calculationHTML + existingPointsHTML;

    // 9. 更新三个只读展示栏（不确定性值来自后端，同步更新）
    document.getElementById('total-uncertainty').value = totalUncertainty.toFixed(4);
    document.getElementById('aleatoric-uncertainty').value = aleatoricUncertainty.toFixed(4);
    document.getElementById('epistemic-uncertainty').value = epistemicUncertainty.toFixed(4);

  } catch (error) {
    console.error('Error calculating single point uncertainty:', error);
    alert('Failed to calculate uncertainty: ' + error.message);
  }
});

// 保留原配置获取函数（无修改）
function getSelectedConfig() {
  const promptStruct = document.querySelector('input[name="promptStructure"]:checked').value;
  const selectedModel = document.querySelector('input[name="modelSelect"]:checked').value;
  return {
    prompt_structure: promptStruct,  
    model: selectedModel           
  };
}

// 保留原Prompt复制功能（仅优化容错逻辑，确保复制的是后端返回的Prompt）
document.getElementById('copy-prompt-btn').addEventListener('click', () => {
  const promptText = document.getElementById('generated-prompt').textContent;
  // 容错：排除“未生成”或“后端未返回”的提示文本
  const invalidTexts = [
    'No prompt generated yet.\nClick "Calculate" to generate a prompt and view it here.',
    'No valid prompt returned from server.'
  ];
  if (invalidTexts.includes(promptText.trim())) {
    alert('No prompt to copy yet!');
    return;
  }
  
  navigator.clipboard.writeText(promptText).then(() => {
    const copyBtn = document.getElementById('copy-prompt-btn');
    const originalText = copyBtn.textContent;
    copyBtn.textContent = 'Copied!';
    copyBtn.style.color = '#22c55e';
    setTimeout(() => {
      copyBtn.textContent = originalText;
      copyBtn.style.color = '#3273dc';
    }, 1500);
  }).catch(err => {
    console.error('Failed to copy: ', err);
    alert('Failed to copy prompt. Please try again.');
  });
});

})