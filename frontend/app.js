/**
 * AttentionDTA Frontend Dashboard Controller
 * Group 17 - Statistical Learning (DSC4804), BML Munjal University
 */

// Dynamically use current host if served via Flask, or default to http://localhost:5000 if opened directly
const API_BASE_URL = window.location.origin.includes("5000") || window.location.protocol.startsWith("http")
  ? window.location.origin
  : "http://localhost:5000";

let targetsData = [];
let currentHeatmapData = null;
let currentFullProteinLen = 1200;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSlider();
  checkHealth();
  loadTargets();
  loadBenchmarks();

  document.getElementById("predictBtn").addEventListener("click", () => {
    runPrediction();
  });

  document.getElementById("targetSelect").addEventListener("change", (e) => {
    onTargetSelected(e.target.value);
  });
});

// --- TABS CONTROLLER ---
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
        // Trigger Plotly relayout to fit newly visible container
        window.dispatchEvent(new Event('resize'));
      }
    });
  });
}

// --- WINDOW SLIDER ---
function initSlider() {
  const slider = document.getElementById("windowSlider");
  slider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    document.getElementById("windowRangeLabel").innerText = `${val} - ${val + 60}`;
  });

  slider.addEventListener("change", (e) => {
    // Re-run prediction with the updated window position
    runPrediction();
  });
}

// --- BACKEND HEALTH CHECK ---
async function checkHealth() {
  const statusEl = document.getElementById("backendStatus");
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (res.ok) {
      statusEl.innerText = "Backend Connected (Port 5000)";
      statusEl.className = "status-indicator online";
    } else {
      statusEl.innerText = "Backend Error";
    }
  } catch (err) {
    statusEl.innerText = "Backend Offline (Start python backend/server.py)";
    statusEl.className = "status-indicator";
  }
}

// --- LOAD CANCER TARGETS ---
async function loadTargets() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/targets`);
    if (!res.ok) return;
    targetsData = await res.json();
    const select = document.getElementById("targetSelect");
    select.innerHTML = "";

    targetsData.forEach((t, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.innerText = `${t.Target_Name} — ${t.Disease_Context} (${t.Drug_Name})`;
      select.appendChild(opt);
    });

    // Add custom option
    const customOpt = document.createElement("option");
    customOpt.value = "custom";
    customOpt.innerText = "Custom Drug SMILES & Kinase Sequence";
    select.appendChild(customOpt);

    // Trigger initial select
    if (targetsData.length > 0) {
      onTargetSelected(0);
      runPrediction(); // Auto-run initial benchmark
    }
  } catch (err) {
    console.error("Failed to load targets:", err);
  }
}

function onTargetSelected(val) {
  const details = document.getElementById("customInputDetails");
  if (val === "custom") {
    details.open = true;
    return;
  }
  const idx = parseInt(val);
  if (targetsData[idx]) {
    const t = targetsData[idx];
    document.getElementById("drugSmilesInput").value = t.Drug_SMILES;
    document.getElementById("proteinSeqInput").value = t.Protein_Sequence;
    details.open = false;
  }
}

// --- LOAD BENCHMARKS COMPARISON TABLE ---
async function loadBenchmarks() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/benchmarks`);
    if (!res.ok) return;
    const benchmarks = await res.json();
    const tbody = document.getElementById("benchmarksBody");
    tbody.innerHTML = "";

    const chartLabels = [];
    const chartR2 = [];
    const chartColors = [];

    benchmarks.forEach(b => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><b>${b.condition}</b></td>
        <td>${b.source}</td>
        <td>${b.mse.toFixed(4)}</td>
        <td>${b.mae.toFixed(4)}</td>
        <td><span style="font-weight:600; color:${b.r2 > 0.5 ? '#16a34a' : (b.r2 > 0 ? '#ca8a04' : '#dc2626')}">${b.r2.toFixed(4)}</span></td>
        <td>${b.notes}</td>
      `;
      tbody.appendChild(tr);

      // Add to chart if key milestone
      if (b.condition.includes("Reproduced") || b.condition.includes("Original") || b.condition.includes("Held-Out") || b.condition.includes("Full Cold-Start")) {
        chartLabels.push(b.condition.split("(")[0].trim());
        chartR2.push(b.r2);
        chartColors.push(b.r2 > 0.5 ? '#2563eb' : (b.r2 > 0 ? '#f59e0b' : '#ef4444'));
      }
    });

    renderComparisonChart(chartLabels, chartR2, chartColors);
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}

function renderComparisonChart(labels, values, colors) {
  const data = [{
    x: labels,
    y: values,
    type: 'bar',
    marker: { color: colors },
    text: values.map(v => v.toFixed(3)),
    textposition: 'auto'
  }];

  const layout = {
    title: { text: "R² Generalization Comparison (Davis Dataset)", font: { size: 14 } },
    yaxis: { title: "R² Metric (Higher = Better)" },
    margin: { l: 40, r: 20, t: 30, b: 60 },
    height: 280
  };

  Plotly.newPlot('comparisonBarChart', data, layout, { responsive: true, displayModeBar: false });
}

// --- RUN PREDICTION API CALL ---
async function runPrediction() {
  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.innerHTML = `<span class="btn-icon">⏳</span> Computing Attention...`;

  const smiles = document.getElementById("drugSmilesInput").value.trim();
  const protein_seq = document.getElementById("proteinSeqInput").value.trim();
  const mode = document.getElementById("modeSelect").value;
  const windowStart = parseInt(document.getElementById("windowSlider").value) || 0;

  if (!smiles || !protein_seq) {
    alert("Please enter both Drug SMILES and Protein Sequence.");
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">⚡</span> Run Prediction & Attention`;
    return;
  }

  try {
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        smiles: smiles,
        protein_seq: protein_seq,
        mode: mode,
        window_start: windowStart,
        window_size: 60
      })
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.error || "Server error");
    }

    const data = await res.json();
    renderPredictionResults(data);
  } catch (err) {
    alert(`Prediction failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">⚡</span> Run Prediction & Attention`;
  }
}

// --- RENDER PREDICTION OUTPUTS ---
function renderPredictionResults(data) {
  // 1. KPI Cards
  document.getElementById("pkdValue").innerText = data.predicted_pKd.toFixed(2);
  document.getElementById("kdValue").innerText = `${data.Kd_nM.toFixed(1)} nM`;

  const isCold = data.mode === "cold_start";
  document.getElementById("conditionValue").innerText = isCold ? "Cold-Start" : "Standard S1";
  document.getElementById("conditionStatus").innerText = isCold ? "Unseen Novel Drug (R² ~ -0.01)" : "Known Kinase Scaffold (R² ~ 0.69)";
  document.getElementById("conditionStatus").style.color = isCold ? "#ef4444" : "#16a34a";

  const potencyEl = document.getElementById("potencyValue");
  if (data.predicted_pKd >= 8.0) {
    potencyEl.innerText = "High (Potent)";
    potencyEl.style.color = "#16a34a";
  } else if (data.predicted_pKd >= 7.0) {
    potencyEl.innerText = "Moderate";
    potencyEl.style.color = "#ca8a04";
  } else {
    potencyEl.innerText = "Weak";
    potencyEl.style.color = "#dc2626";
  }

  // Update window slider max
  const slider = document.getElementById("windowSlider");
  currentFullProteinLen = data.protein_length;
  slider.max = Math.max(0, currentFullProteinLen - 60);

  // 2. Render Heatmap with Plotly
  renderHeatmap(data.heatmap);

  // 3. Render 1D Profiles & Tables
  renderActiveSites(data);
}

// --- RENDER 2D HEATMAP ---
function renderHeatmap(hm) {
  const data = [{
    z: hm.z,
    x: hm.x,
    y: hm.y,
    type: 'heatmap',
    colorscale: 'Viridis',
    hoverongaps: false,
    hovertemplate: 'Drug Atom: %{y}<br>Kinase Residue: %{x}<br>Attention Weight: %{z:.4f}<extra></extra>'
  }];

  const layout = {
    title: {
      text: `Cross-Attention Heatmap (Residues ${hm.window_start + 1} to ${hm.window_end} of ${hm.total_protein_len})`,
      font: { size: 14 }
    },
    xaxis: { title: "Target Cancer Kinase Residues", tickangle: -45 },
    yaxis: { title: "Drug Chemical Tokens (SMILES)" },
    margin: { l: 60, r: 30, t: 40, b: 60 },
    height: 500
  };

  Plotly.newPlot('heatmapPlot', data, layout, { responsive: true });
}

// --- RENDER 1D PROFILES & TABLES ---
function renderActiveSites(data) {
  // Protein Profile Plot
  const protIndices = Array.from({ length: data.alpha_prot.length }, (_, i) => i + 1);
  const protPlotData = [{
    x: protIndices,
    y: data.alpha_prot,
    type: 'scatter',
    mode: 'lines',
    line: { color: '#2563eb', width: 2 },
    hovertemplate: 'Position %{x}: %{y:.4f}<extra></extra>'
  }];

  Plotly.newPlot('proteinProfilePlot', protPlotData, {
    title: { text: "Cancer Kinase Residue Importance Profile", font: { size: 13 } },
    xaxis: { title: "Amino Acid Position" },
    yaxis: { title: "Attention Score" },
    margin: { l: 40, r: 20, t: 30, b: 40 },
    height: 240
  }, { responsive: true, displayModeBar: false });

  // Drug Profile Plot
  const drugIndices = Array.from({ length: data.alpha_drug.length }, (_, i) => i + 1);
  const drugPlotData = [{
    x: drugIndices,
    y: data.alpha_drug,
    type: 'bar',
    marker: { color: '#10b981' },
    hovertemplate: 'Drug Token %{x}: %{y:.4f}<extra></extra>'
  }];

  Plotly.newPlot('drugProfilePlot', drugPlotData, {
    title: { text: "Drug Pharmacophore Token Importance", font: { size: 13 } },
    xaxis: { title: "SMILES Character Index" },
    yaxis: { title: "Attention Score" },
    margin: { l: 40, r: 20, t: 30, b: 40 },
    height: 240
  }, { responsive: true, displayModeBar: false });

  // Update Protein Top Table
  const protTbody = document.getElementById("proteinTopTable").querySelector("tbody");
  protTbody.innerHTML = "";
  data.top_residues.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${r.position}</b></td>
      <td><span class="badge badge-primary">${r.residue}</span></td>
      <td><code>...${r.local_window}...</code></td>
      <td>${r.attention_score.toFixed(4)}</td>
    `;
    protTbody.appendChild(tr);
  });

  // Update Drug Top Table
  const drugTbody = document.getElementById("drugTopTable").querySelector("tbody");
  drugTbody.innerHTML = "";
  data.top_substructures.forEach(d => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${d.index}</b></td>
      <td><span class="badge badge-success">${d.atom_char}</span></td>
      <td><code>${d.substructure}</code></td>
      <td>${d.attention_score.toFixed(4)}</td>
    `;
    drugTbody.appendChild(tr);
  });
}
