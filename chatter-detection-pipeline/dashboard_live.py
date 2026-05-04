import asyncio
import base64
import csv
import hashlib
import json
import os
import pathlib
import struct
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HOST = os.environ.get("CHATTER_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("CHATTER_DASHBOARD_PORT", "8765"))
BASE_DIR = pathlib.Path(__file__).resolve().parent
PLAYBACK_INTERVAL = 0.08

DASHBOARD_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Rolling Monitor</title>
<style>
  :root {
    --bg: #eef2f6;
    --panel: #ffffff;
    --panel-soft: #f7f9fb;
    --line: #d8e0e8;
    --line-strong: #c5cfda;
    --text: #18212b;
    --muted: #5d6a78;
    --blue: #2e5b8f;
    --blue-soft: #edf3f9;
    --green: #1d7f57;
    --green-soft: #e7f4ee;
    --amber: #a36c0c;
    --amber-soft: #fbf3df;
    --orange: #b95f12;
    --orange-soft: #fcecdf;
    --red: #b23a2b;
    --red-soft: #fbe8e6;
    --shadow: 0 10px 24px rgba(17, 34, 51, 0.06);
    --radius: 14px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  .shell {
    max-width: 1600px;
    margin: 0 auto;
    padding: 18px;
    display: grid;
    gap: 16px;
  }
  .top {
    display: grid;
    grid-template-columns: 1.5fr 1fr auto auto;
    gap: 16px;
    align-items: stretch;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
  }
  .pad { padding: 18px 20px; }
  .eyebrow {
    font-size: 11px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  .title {
    margin-top: 8px;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.03em;
  }
  .subtitle {
    margin-top: 6px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.45;
  }
  .subtitle:empty, .panel-note:empty, .small-note:empty, .footer:empty { display:none; }
  .roll-name {
    margin-top: 10px;
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -0.04em;
  }
  .roll-detail {
    margin-top: 8px;
    color: var(--muted);
    font-size: 14px;
  }
  .clock-box, .feed-box {
    min-width: 190px;
    display: grid;
    align-content: center;
  }
  .metric-value {
    margin-top: 10px;
    font-size: 22px;
    font-weight: 800;
  }
  .toolbar {
    display: grid;
    grid-template-columns: 1.25fr repeat(4, 1fr);
    gap: 16px;
  }
  .selector-grid {
    display: grid;
    gap: 14px;
  }
  .selector-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.04em;
  }
  .run-controls {
    display: flex;
    gap: 10px;
    align-items: center;
  }
  select {
    flex: 1;
    min-width: 0;
    height: 48px;
    padding: 0 14px;
    border-radius: 12px;
    border: 1px solid var(--line-strong);
    background: #fff;
    color: var(--text);
    font-size: 15px;
  }
  button {
    height: 48px;
    padding: 0 16px;
    border-radius: 12px;
    border: 1px solid var(--line-strong);
    background: var(--panel-soft);
    color: var(--text);
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
  }
  button.primary {
    background: var(--blue);
    border-color: var(--blue);
    color: #fff;
  }
  .state-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 136px;
    padding: 10px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .stable { background: var(--green-soft); color: var(--green); }
  .watch { background: var(--amber-soft); color: var(--amber); }
  .warning { background: var(--orange-soft); color: var(--orange); }
  .alarm { background: var(--red-soft); color: var(--red); }
  .state-text { margin-top: 12px; color: var(--muted); font-size: 14px; line-height: 1.5; }
  .big-number {
    margin-top: 10px;
    font-size: 22px;
    font-weight: 800;
  }
  .small-note { margin-top: 6px; font-size: 14px; color: var(--muted); }
  .layout {
    display: grid;
    grid-template-columns: 1.45fr 0.95fr;
    gap: 16px;
  }
  .stack { display: grid; gap: 16px; }
  .panel-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 14px;
    padding: 18px 20px 0;
  }
  .panel-title {
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  .panel-note { color: var(--muted); font-size: 13px; }
  .chart-wrap { padding: 14px 20px 20px; }
  canvas {
    width: 100%;
    height: 300px;
    display: block;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fff;
  }
  .instant-grid {
    padding: 0 20px 20px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  .kv {
    padding: 14px 14px 12px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--panel-soft);
  }
  .kv .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; }
  .kv .v { margin-top: 8px; font-size: 19px; font-weight: 800; }
  .bands {
    padding: 0 20px 20px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .band-box {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--panel-soft);
    padding: 14px;
  }
  .band-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 12px;
  }
  .band-title { font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; }
  .band-mini {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 12px;
  }
  .band-bar {
    position: relative;
    height: 12px;
    border-radius: 999px;
    background: #e7edf4;
    overflow: hidden;
  }
  .band-bar > span {
    position: absolute;
    inset: 0 auto 0 0;
    width: 0%;
    background: var(--blue);
    border-radius: inherit;
  }
  .band-mark {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    background: #b5c0cb;
  }
  .band-legend {
    margin-top: 8px;
    display: flex;
    justify-content: space-between;
    color: var(--muted);
    font-size: 11px;
  }
  .summary-table {
    width: 100%;
    border-collapse: collapse;
  }
  .summary-table td {
    padding: 12px 0;
    border-bottom: 1px solid var(--line);
    font-size: 14px;
  }
  .summary-table td:first-child { color: var(--muted); }
  .summary-table td:last-child { text-align: right; font-weight: 800; }
  .summary-table tr:last-child td { border-bottom: none; }
  .advisory {
    padding: 16px 18px;
    min-height: 180px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--panel-soft);
    font-size: 15px;
    line-height: 1.6;
    color: var(--text);
    white-space: normal;
    overflow-wrap: anywhere;
  }
  .log {
    display: grid;
    gap: 10px;
  }
  .log-item {
    display: grid;
    grid-template-columns: 84px 1fr;
    gap: 10px;
    padding: 12px 14px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: var(--panel-soft);
  }
  .log-time { color: var(--muted); font-variant-numeric: tabular-nums; }
  .footer { color: var(--muted); font-size: 12px; padding: 0 2px 2px; }
  .chart-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 0 20px 10px;
    font-size: 12px;
    color: var(--muted);
  }
  .chart-legend .lg {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .chart-legend .swatch {
    width: 14px;
    height: 3px;
    border-radius: 2px;
    display: inline-block;
  }
  .chart-legend .dash {
    width: 14px;
    height: 0;
    border-top: 2px dashed;
    display: inline-block;
  }
  .instant-legend {
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 0 20px 14px;
    font-size: 11px;
    color: var(--muted);
    line-height: 1.5;
  }
  .instant-legend b { color: var(--text); font-weight: 700; }
  @media (max-width: 1300px) {
    .top, .toolbar, .layout, .instant-grid, .bands { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <div class="card pad">
        <div class="eyebrow">Cold rolling mill · replay console</div>
        <div class="title">Chatter monitoring console</div>
        
      </div>
      <div class="card pad">
        <div class="eyebrow">Currently replaying</div>
        <div class="roll-name" id="current-roll">--</div>
        <div class="roll-detail" id="current-roll-detail">Select a roll</div>
      </div>
      <div class="card pad clock-box">
        <div class="eyebrow">System clock</div>
        <div class="metric-value" id="clock">--:--:--</div>
        
      </div>
      <div class="card pad feed-box">
        <div class="eyebrow">Feed status</div>
        <div class="metric-value" id="feed-status">CONNECTING</div>
        
      </div>
    </div>

    <div class="toolbar">
      <div class="card pad selector-grid">
        <div class="eyebrow">Roll selection</div>
        <div class="selector-title" id="selected-roll">Select roll</div>
        
        <div class="run-controls">
          <select id="run-select"></select>
          <button class="primary" id="load-run">Load</button>
          <button id="restart-run">Restart</button>
        </div>
      </div>
      <div class="card pad">
        <div class="eyebrow">Machine state</div>
        <div style="margin-top: 12px"><span id="state-pill" class="state-pill stable">STABLE</span></div>
        <div class="state-text" id="state-msg">Stable</div>
      </div>
      <div class="card pad">
        <div class="eyebrow">Risk index</div>
        <div class="big-number" id="risk-score">0.000</div>
        <div class="small-note" id="risk-caption">Nominal</div>
      </div>
      <div class="card pad">
        <div class="eyebrow">Frame / time</div>
        <div class="big-number"><span id="frame-idx">0</span></div>
        <div class="small-note"><span id="time-sec">0.00</span>s</div>
      </div>
      <div class="card pad">
        <div class="eyebrow">Acceleration RMS</div>
        <div class="big-number" id="vib-rms">0.000</div>
        
      </div>
    </div>

    <div class="layout">
      <div class="stack">
        <div class="card">
          <div class="panel-head">
            <div>
              <div class="panel-title">Risk trend</div>
              
            </div>
            <div class="panel-note" id="risk-meta"></div>
          </div>
          <div class="chart-wrap"><canvas id="risk-chart" width="960" height="300"></canvas></div>
          <div class="chart-legend">
            <span class="lg"><span class="swatch" style="background:#4f86d9"></span>Risk score (0–1)</span>
            <span class="lg"><span class="dash" style="border-color:#c79a2a"></span>Warning threshold (0.55)</span>
            <span class="lg"><span class="dash" style="border-color:#cc6e6e"></span>Alarm threshold (0.70)</span>
          </div>
        </div>

        <div class="card">
          <div class="panel-head">
            <div>
              <div class="panel-title">Process variables</div>
              
            </div>
          </div>
          <div class="chart-wrap"><canvas id="process-chart" width="960" height="300"></canvas></div>
          <div class="chart-legend">
            <span class="lg"><span class="swatch" style="background:#2e5b8f"></span>Rolling speed</span>
            <span class="lg"><span class="swatch" style="background:#8b6d28"></span>Force / tension</span>
            <span class="lg"><span class="swatch" style="background:#2d8a74"></span>Strip thickness</span>
          </div>
        </div>

        <div class="card">
          <div class="panel-head">
            <div>
              <div class="panel-title">Acceleration signals</div>
              
            </div>
          </div>
          <div class="chart-wrap"><canvas id="accel-chart" width="960" height="300"></canvas></div>
          <div class="chart-legend">
            <span class="lg"><span class="swatch" style="background:#285e9c"></span>Operator-side (OS) accel RMS</span>
            <span class="lg"><span class="swatch" style="background:#7f4f9a"></span>Drive-side (DS) accel RMS</span>
          </div>
        </div>
      </div>

      <div class="stack">
        <div class="card">
          <div class="panel-head">
            <div>
              <div class="panel-title">Instantaneous values</div>
              
            </div>
          </div>
          <div class="instant-grid">
            <div class="kv"><div class="k">Speed</div><div class="v" id="speed-now">--</div></div>
            <div class="kv"><div class="k">Force / tension</div><div class="v" id="force-now">--</div></div>
            <div class="kv"><div class="k">Thickness</div><div class="v" id="thickness-now">--</div></div>
            <div class="kv"><div class="k">OS accel RMS</div><div class="v" id="os-now">--</div></div>
            <div class="kv"><div class="k">DS accel RMS</div><div class="v" id="ds-now">--</div></div>
            <div class="kv"><div class="k">Dominant band</div><div class="v" id="band-now">--</div></div>
          </div>
          <div class="instant-legend">
            <span><b>OS / DS accel RMS</b> — broadband vibration at each bearing.</span>
            <span><b>Dominant band</b> — which chatter regime (3rd-oct ~80–260 Hz or 5th-oct ~500–1000 Hz) is currently leading.</span>
          </div>
        </div>

        <div class="card">
          <div class="panel-head">
            <div>
              <div class="panel-title">Band tracking</div>
              
            </div>
          </div>
          <div class="bands">
            <div class="band-box">
              <div class="band-top"><div class="band-title">3rd band</div><div class="panel-note" id="third-state">STABLE</div></div>
              <div class="band-mini">
                <div class="kv"><div class="k">Frequency</div><div class="v" id="third-freq">--</div></div>
                <div class="kv"><div class="k">Score</div><div class="v" id="third-score">0.00</div></div>
              </div>
              <div class="band-bar"><span id="third-fill"></span><div class="band-mark" style="left:55%"></div><div class="band-mark" style="left:70%; background:#d17b7b"></div></div>
              <div class="band-legend"><span>Quiet</span><span>Warning</span><span>Alarm</span></div>
            </div>
            <div class="band-box">
              <div class="band-top"><div class="band-title">5th band</div><div class="panel-note" id="fifth-state">STABLE</div></div>
              <div class="band-mini">
                <div class="kv"><div class="k">Frequency</div><div class="v" id="fifth-freq">--</div></div>
                <div class="kv"><div class="k">Score</div><div class="v" id="fifth-score">0.00</div></div>
              </div>
              <div class="band-bar"><span id="fifth-fill"></span><div class="band-mark" style="left:55%"></div><div class="band-mark" style="left:70%; background:#d17b7b"></div></div>
              <div class="band-legend"><span>Quiet</span><span>Warning</span><span>Alarm</span></div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="panel-head">
            <div>
              <div class="panel-title">Advisory output</div>
              
            </div>
          </div>
          <div class="pad" style="padding-top: 14px; display:grid; gap:12px;">
            <div class="instant-grid" style="padding:0; grid-template-columns:1fr 1fr 1fr;">
              <div class="kv"><div class="k">Control mode</div><div class="v" id="control-mode">HOLD</div></div>
              <div class="kv"><div class="k">Recommended speed</div><div class="v" id="speed-sp">--</div></div>
              <div class="kv"><div class="k">Recommended tension</div><div class="v" id="tension-sp">--</div></div>
            </div>
            <div class="advisory" id="control-text">Stable. Hold nominal settings.</div>
          </div>
        </div>

        <div class="card">
          <div class="panel-head">
            <div>
              <div class="panel-title">Run summary</div>
              
            </div>
          </div>
          <div class="pad" style="padding-top: 10px;">
            <table class="summary-table">
              <tr><td>Detected events</td><td id="kpi-events">--</td></tr>
              <tr><td>Mean lead time</td><td id="kpi-lead">--</td></tr>
              <tr><td>False warnings / hour</td><td id="kpi-false">--</td></tr>
              <tr><td>Detection rate</td><td id="kpi-detect">--</td></tr>
              <tr><td>Warning frames / hour</td><td id="kpi-warning-rate">--</td></tr>
              <tr><td>Alarm frames / hour</td><td id="kpi-alarm-rate">--</td></tr>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="panel-head">
            <div>
              <div class="panel-title">Event log</div>
              
            </div>
          </div>
          <div class="pad" style="padding-top: 12px;">
            <div class="log" id="log"></div>
          </div>
        </div>
      </div>
    </div>

    
  </div>

<script>
const WS_PORT = (parseInt(location.port || '8765', 10) + 1).toString();
const WS_URL = `ws://${location.hostname}:${WS_PORT}`;
const riskHistory = [];
const speedHistory = [];
const forceHistory = [];
const thicknessHistory = [];
const osHistory = [];
const dsHistory = [];
const MAX_POINTS = 180;
let frameIndex = 0;
let lastState = '';
let ws = null;

function byId(id) { return document.getElementById(id); }
function num(v) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
}
function fmt(v, d=2) {
  const n = num(v);
  return n === null ? '--' : n.toFixed(d);
}
function clockTick() { byId('clock').textContent = new Date().toLocaleTimeString(); }
setInterval(clockTick, 1000); clockTick();

function friendlyRun(name) {
  if (!name) return '--';
  const key = String(name).toLowerCase();
  if (key === 'run_main') return 'Main roll';
  if (key === 'roll3') return 'Roll 3';
  if (key === 'book1') return 'Roll 2';
  return String(name).replace(/_/g, ' ');
}

function addLog(text) {
  const row = document.createElement('div');
  row.className = 'log-item';
  row.innerHTML = `<div class="log-time">${new Date().toLocaleTimeString()}</div><div>${text}</div>`;
  const log = byId('log');
  log.prepend(row);
  while (log.children.length > 12) log.removeChild(log.lastChild);
}

function stateClass(state) {
  const s = String(state || 'STABLE').toUpperCase();
  if (s === 'STABLE') return 'stable';
  if (s === 'WATCH') return 'watch';
  if (s === 'WARNING') return 'warning';
  return 'alarm';
}

function setState(state, msg) {
  const s = String(state || 'STABLE').toUpperCase();
  const pill = byId('state-pill');
  pill.className = 'state-pill ' + stateClass(s);
  pill.textContent = s;
  byId('state-msg').textContent = msg || 'System update';
  if (s !== lastState) {
    addLog(`State changed to ${s}. ${msg || ''}`);
    lastState = s;
  }
}

function setRisk(value) {
  const v = Math.max(0, Math.min(1, num(value) || 0));
  byId('risk-score').textContent = v.toFixed(3);
  let text = 'Nominal';
  if (v >= 0.70) text = 'Alarm';
  else if (v >= 0.55) text = 'Warning';
  else if (v >= 0.35) text = 'Watch';
  byId('risk-caption').textContent = text;
}

function resetHistories() {
  frameIndex = 0;
  riskHistory.length = 0;
  speedHistory.length = 0;
  forceHistory.length = 0;
  thicknessHistory.length = 0;
  osHistory.length = 0;
  dsHistory.length = 0;
  drawRiskChart();
  drawProcessChart();
  drawAccelChart();
}

function pushHist(arr, value) {
  arr.push(value);
  if (arr.length > MAX_POINTS) arr.shift();
}

function drawSeriesChart(canvasId, seriesList, options = {}) {
  const canvas = byId(canvasId);
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = '#e6edf3';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = (h / 5) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  for (let i = 0; i <= 8; i++) {
    const x = (w / 8) * i;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }

  const values = [];
  seriesList.forEach(s => s.data.forEach(v => { if (v !== null && Number.isFinite(v)) values.push(v); }));
  let minY = options.minY !== undefined ? options.minY : (values.length ? Math.min(...values) : 0);
  let maxY = options.maxY !== undefined ? options.maxY : (values.length ? Math.max(...values) : 1);
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  const padding = (maxY - minY) * 0.08;
  minY -= padding;
  maxY += padding;
  const yOf = v => h - ((v - minY) / (maxY - minY)) * h;

  if (options.thresholds) {
    ctx.setLineDash([6, 6]);
    options.thresholds.forEach(th => {
      ctx.strokeStyle = th.color;
      ctx.beginPath();
      ctx.moveTo(0, yOf(th.value));
      ctx.lineTo(w, yOf(th.value));
      ctx.stroke();
    });
    ctx.setLineDash([]);
  }

  seriesList.forEach(s => {
    const data = s.data;
    if (!data.length) return;
    ctx.lineWidth = s.width || 2.2;
    ctx.strokeStyle = s.color;
    ctx.beginPath();
    let started = false;
    data.forEach((v, i) => {
      if (v === null || !Number.isFinite(v)) return;
      const x = (i / Math.max(1, MAX_POINTS - 1)) * w;
      const y = yOf(v);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    if (started) ctx.stroke();
  });
}

function drawRiskChart() {
  drawSeriesChart('risk-chart', [
    { data: riskHistory, color: '#4f86d9', width: 2.6 }
  ], {
    minY: 0,
    maxY: 1,
    thresholds: [
      { value: 0.55, color: '#c79a2a' },
      { value: 0.70, color: '#cc6e6e' }
    ]
  });
}

function drawProcessChart() {
  drawSeriesChart('process-chart', [
    { data: speedHistory, color: '#2e5b8f' },
    { data: forceHistory, color: '#8b6d28' },
    { data: thicknessHistory, color: '#2d8a74' }
  ]);
}

function drawAccelChart() {
  drawSeriesChart('accel-chart', [
    { data: osHistory, color: '#285e9c' },
    { data: dsHistory, color: '#7f4f9a' }
  ]);
}

function applyMeta(meta) {
  const run = meta && meta.current_run ? meta.current_run : null;
  const pretty = friendlyRun(run);
  byId('current-roll').textContent = pretty;
  byId('selected-roll').textContent = pretty;
  const frames = meta && meta.frames != null ? meta.frames : '--';
  const duration = meta && meta.duration_sec != null ? `${Number(meta.duration_sec).toFixed(1)} s` : '--';
  const raw = run ? `Internal name: ${run}` : 'No run loaded';
  byId('current-roll-detail').textContent = run ? `Frames: ${frames} · Duration: ${duration}` : 'Select a roll';
  const note = byId('selected-roll-note'); if (note) note.textContent = run ? `${frames} frames · ${duration}` : '';
  byId('risk-meta').textContent = run ? `${pretty}` : '';
}

function applyKpi(kpi) {
  if (!kpi) return;
  byId('kpi-events').textContent = kpi.n_events ?? '--';
  byId('kpi-lead').textContent = kpi.mean_lead_time_sec == null ? '--' : `${Number(kpi.mean_lead_time_sec).toFixed(2)} s`;
  byId('kpi-false').textContent = kpi.false_warnings_per_hour == null ? '--' : Number(kpi.false_warnings_per_hour).toFixed(2);
  byId('kpi-detect').textContent = kpi.detection_rate_pct == null ? '--' : `${Number(kpi.detection_rate_pct).toFixed(1)}%`;
  byId('kpi-warning-rate').textContent = kpi.warning_frames_per_hour == null ? '--' : Number(kpi.warning_frames_per_hour).toFixed(2);
  byId('kpi-alarm-rate').textContent = kpi.alarm_frames_per_hour == null ? '--' : Number(kpi.alarm_frames_per_hour).toFixed(2);
}

function applyFrame(f) {
  frameIndex += 1;
  const state = f.state || 'STABLE';
  const msg = f.recommended_control_text || f.control_message || 'System update';
  const risk = num(f.risk_score) || 0;
  const speed = num(f.speed_mean);
  const force = num(f.tension_mean);
  const thickness = num(f.thickness_mean);
  const os = num(f.vib_rms_os);
  const ds = num(f.vib_rms_ds);
  const rms = num(f.vib_rms_mean);

  byId('frame-idx').textContent = frameIndex;
  byId('time-sec').textContent = fmt(f.t_sec, 2);
  byId('vib-rms').textContent = fmt(rms, 3);
  byId('speed-now').textContent = fmt(speed, 1);
  byId('force-now').textContent = fmt(force, 1);
  byId('thickness-now').textContent = fmt(thickness, 3);
  byId('os-now').textContent = fmt(os, 3);
  byId('ds-now').textContent = fmt(ds, 3);
  byId('band-now').textContent = String(f.dominant_band || '--').toUpperCase();
  byId('control-mode').textContent = f.recommended_control_mode || 'HOLD';
  byId('speed-sp').textContent = fmt(f.recommended_speed_setpoint, 1);
  byId('tension-sp').textContent = fmt(f.recommended_tension_setpoint, 1);
  byId('control-text').textContent = msg;
  byId('third-freq').textContent = fmt(f.dom_third_hz, 1);
  byId('third-score').textContent = fmt(f.score_third, 2);
  byId('fifth-freq').textContent = fmt(f.dom_fifth_hz, 1);
  byId('fifth-score').textContent = fmt(f.score_fifth, 2);
  byId('third-state').textContent = String(f.third_state || state).toUpperCase();
  byId('fifth-state').textContent = String(f.fifth_state || state).toUpperCase();
  byId('third-fill').style.width = `${Math.max(0, Math.min(100, (num(f.score_third) || 0) * 100))}%`;
  byId('fifth-fill').style.width = `${Math.max(0, Math.min(100, (num(f.score_fifth) || 0) * 100))}%`;

  setState(state, msg);
  setRisk(risk);

  pushHist(riskHistory, risk);
  pushHist(speedHistory, speed);
  pushHist(forceHistory, force);
  pushHist(thicknessHistory, thickness);
  pushHist(osHistory, os);
  pushHist(dsHistory, ds);

  drawRiskChart();
  drawProcessChart();
  drawAccelChart();
}

async function loadRuns() {
  const res = await fetch('/api/runs');
  const data = await res.json();
  const sel = byId('run-select');
  sel.innerHTML = '';
  (data.runs || []).forEach(run => {
    const opt = document.createElement('option');
    opt.value = run;
    opt.textContent = friendlyRun(run);
    if (run === data.current_run) opt.selected = true;
    sel.appendChild(opt);
  });
  applyMeta(data.meta || {});
  applyKpi(data.kpi || {});
  addLog('Run list loaded.');
}

async function chooseRun(runName) {
  const res = await fetch(`/api/select?run=${encodeURIComponent(runName)}`);
  const data = await res.json();
  resetHistories();
  applyMeta(data.meta || {});
  applyKpi(data.kpi || {});
  addLog(`Loaded ${friendlyRun(runName)}.`);
}

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    byId('feed-status').textContent = 'CONNECTED';
    addLog('Replay feed connected.');
  };
  ws.onclose = () => {
    byId('feed-status').textContent = 'RECONNECTING';
    addLog('Feed disconnected. Reconnecting...');
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'meta') {
      resetHistories();
      applyMeta(msg.meta || {});
    } else if (msg.type === 'kpi') {
      applyKpi(msg.kpi || {});
    } else if (msg.type === 'frame') {
      applyFrame(msg.data || {});
    } else if (msg.type === 'end') {
      addLog('Replay restarted from the beginning.');
      resetHistories();
    }
  };
}

byId('load-run').addEventListener('click', () => chooseRun(byId('run-select').value));
byId('restart-run').addEventListener('click', () => chooseRun(byId('run-select').value));
byId('run-select').addEventListener('change', () => { byId('selected-roll').textContent = friendlyRun(byId('run-select').value); });

drawRiskChart();
drawProcessChart();
drawAccelChart();
loadRuns().then(connect).catch(err => {
  addLog(`Failed to load run catalogue: ${err}`);
  connect();
});
</script>
</body>
</html>'''

_WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_handshake(key: str) -> str:
    sha = hashlib.sha1((key + _WS_MAGIC).encode()).digest()
    accept = base64.b64encode(sha).decode()
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )


def ws_encode(data: str) -> bytes:
    payload = data.encode("utf-8")
    length = len(payload)
    if length <= 125:
        header = bytes([0x81, length])
    elif length <= 65535:
        header = struct.pack(">BBH", 0x81, 126, length)
    else:
        header = struct.pack(">BBQ", 0x81, 127, length)
    return header + payload


@dataclass
class RunBundle:
    name: str
    path: pathlib.Path
    timeline_csv: pathlib.Path
    kpi_json: pathlib.Path | None = None
    summary_json: pathlib.Path | None = None
    rows: list[dict] = field(default_factory=list)
    kpi: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def load(self) -> None:
        with open(self.timeline_csv, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if self.kpi_json and self.kpi_json.exists():
            with open(self.kpi_json, encoding="utf-8") as f:
                self.kpi = json.load(f)
        else:
            self.kpi = {}
        if self.summary_json and self.summary_json.exists():
            with open(self.summary_json, encoding="utf-8") as f:
                self.summary = json.load(f)
        else:
            self.summary = {}

    def meta(self) -> dict:
        duration = self.summary.get("duration_sec")
        if duration is None and self.rows:
            try:
                duration = float(self.rows[-1].get("t_sec", 0))
            except Exception:
                duration = None
        return {
            "current_run": self.name,
            "frames": len(self.rows),
            "duration_sec": duration,
            "path": str(self.path),
        }


class DashboardState:
    def __init__(self) -> None:
        self.runs: dict[str, RunBundle] = {}
        self.current_run: str | None = None
        self.frame_index = 0
        self.version = 0
        self.discover_runs()

    def discover_runs(self) -> None:
        preferred = pathlib.Path(os.environ.get("CHATTER_RESULTS_DIR", "")).expanduser()
        default_run = None
        roots: list[pathlib.Path] = []
        if preferred and preferred.exists():
            if (preferred / "timeline.csv").exists():
                default_run = preferred.name
                roots.append(preferred.parent)
            elif (preferred / "results_by_run").exists():
                roots.append(preferred / "results_by_run")
        roots.append(BASE_DIR / "results_by_run")
        roots.append(BASE_DIR)

        seen = set()
        for root in roots:
            if root in seen or not root.exists():
                continue
            seen.add(root)
            if root.name == "results_by_run":
                for child in sorted(root.iterdir()):
                    if child.is_dir() and (child / "timeline.csv").exists():
                        self.runs[child.name] = RunBundle(
                            name=child.name,
                            path=child,
                            timeline_csv=child / "timeline.csv",
                            kpi_json=child / "early_warning_kpi.json",
                            summary_json=child / "summary.json",
                        )
                continue
            if (root / "timeline.csv").exists():
                self.runs[root.name] = RunBundle(
                    name=root.name,
                    path=root,
                    timeline_csv=root / "timeline.csv",
                    kpi_json=root / "early_warning_kpi.json",
                    summary_json=root / "summary.json",
                )
            nested = root / "results_by_run"
            if nested.exists():
                for child in sorted(nested.iterdir()):
                    if child.is_dir() and (child / "timeline.csv").exists():
                        self.runs[child.name] = RunBundle(
                            name=child.name,
                            path=child,
                            timeline_csv=child / "timeline.csv",
                            kpi_json=child / "early_warning_kpi.json",
                            summary_json=child / "summary.json",
                        )
        if not self.runs:
            raise FileNotFoundError("No processed runs found.")
        self.current_run = None

    def load_run(self, run_name: str | None) -> None:
        if not run_name:
            raise RuntimeError("No run selected")
        bundle = self.runs[run_name]
        bundle.load()
        self.current_run = run_name
        self.frame_index = 0
        self.version += 1
        print(f"[dashboard] Loaded run '{run_name}' with {len(bundle.rows)} frames")

    def current_bundle(self) -> RunBundle | None:
        if not self.current_run:
            return None
        return self.runs[self.current_run]

    def runs_payload(self) -> dict:
        bundle = self.current_bundle()
        return {
            "runs": list(self.runs.keys()),
            "current_run": self.current_run,
            "meta": bundle.meta() if bundle else None,
            "kpi": bundle.kpi if bundle else {},
        }


STATE: DashboardState | None = None


class HTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/runs":
            self._send_json(STATE.runs_payload())
            return
        if path == "/api/select":
            query = parse_qs(parsed.query)
            run_name = query.get("run", [None])[0]
            if not run_name or run_name not in STATE.runs:
                self._send_json({"error": "Unknown run"}, status=404)
                return
            STATE.load_run(run_name)
            bundle = STATE.current_bundle()
            self._send_json({"ok": True, "meta": bundle.meta(), "kpi": bundle.kpi})
            return
        self.send_response(404)
        self.end_headers()


class WSServer:
    def __init__(self):
        self.clients = set()
        self.lock = asyncio.Lock()

    async def broadcast(self, msg: dict):
        data = ws_encode(json.dumps(msg))
        async with self.lock:
            dead = set()
            for writer in self.clients:
                try:
                    writer.write(data)
                    await writer.drain()
                except Exception:
                    dead.add(writer)
            self.clients -= dead

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = await reader.read(4096)
            if not chunk:
                writer.close()
                return
            raw += chunk
        ws_key = None
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if line.lower().startswith("sec-websocket-key:"):
                ws_key = line.split(":", 1)[1].strip()
                break
        if not ws_key:
            writer.close()
            return
        writer.write(ws_handshake(ws_key).encode())
        await writer.drain()
        async with self.lock:
            self.clients.add(writer)
        bundle = STATE.current_bundle()
        if bundle is not None:
            try:
                writer.write(ws_encode(json.dumps({"type": "meta", "meta": bundle.meta()})))
                writer.write(ws_encode(json.dumps({"type": "kpi", "kpi": bundle.kpi})))
                await writer.drain()
            except Exception:
                pass
        try:
            while True:
                chunk = await reader.read(256)
                if not chunk:
                    break
        finally:
            async with self.lock:
                self.clients.discard(writer)
            try:
                writer.close()
            except Exception:
                pass


async def stream_frames(ws_server: WSServer, state: DashboardState):
    last_version = -1
    while True:
        bundle = state.current_bundle()
        if bundle is None:
            await asyncio.sleep(0.25)
            continue
        if state.version != last_version:
            last_version = state.version
            await ws_server.broadcast({"type": "meta", "meta": bundle.meta()})
            await ws_server.broadcast({"type": "kpi", "kpi": bundle.kpi})
        if not bundle.rows:
            await asyncio.sleep(0.25)
            continue
        if state.frame_index >= len(bundle.rows):
            state.frame_index = 0
            await ws_server.broadcast({"type": "end"})
            await asyncio.sleep(0.25)
            continue
        row = bundle.rows[state.frame_index]
        state.frame_index += 1
        await ws_server.broadcast({"type": "frame", "data": row})
        await asyncio.sleep(PLAYBACK_INTERVAL)


def run_http():
    server = ThreadingHTTPServer((HOST, PORT), HTTPHandler)
    print(f"[http] http://{HOST}:{PORT}")
    server.serve_forever()


async def main_async():
    global STATE
    STATE = DashboardState()
    ws_server = WSServer()
    ws_port = PORT + 1
    async_server = await asyncio.start_server(ws_server.handle, HOST, ws_port)
    print(f"[ws] ws://{HOST}:{ws_port}")
    async with async_server:
        await asyncio.gather(async_server.serve_forever(), stream_frames(ws_server, STATE))


def main():
    import threading
    http_thread = threading.Thread(target=run_http, daemon=True)
    http_thread.start()
    url = f"http://{HOST}:{PORT}"
    print("=" * 64)
    print("Rolling monitor")
    print(url)
    print("=" * 64)
    time.sleep(0.8)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")


if __name__ == "__main__":
    main()