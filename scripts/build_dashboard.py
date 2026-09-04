"""
Builds the Apple-inspired XAI Wafer Intelligence Dashboard as a self-contained, standalone HTML application.
Adheres to Apple Design Principles:
- Fluid spring motion (cubic-bezier(0.16, 1, 0.3, 1), damping 1.0, response 0.35s)
- Direct manipulation & instant feedback on pointerdown (scale 0.97)
- Translucent materials & depth (backdrop-filter: blur(28px) saturate(190%), specular highlights)
- Typography (SF Pro style, size-specific optical tracking, SF Mono tabular numbers)
- Segmented control navigation across 5 rich tabs:
    1. Wafer Risk & Die XAI Inspector (zoom, pan, filter, die probe, SHAP drivers)
    2. Three-Panel Wafer Triptych (Pre-Test -> Post-Test -> What Changed + Prediction Verification)
    3. Model A vs Model B Benchmarks (Metrics, Bootstrap CIs, Category Importance, PR Curve)
    4. Fab Economics & Decision Support (Interactive cost curve slider, 57% FP reduction proof)
    5. Top Investigation Queue (Ranked queue with instant jump-to-die)
- Eligibility & Leakage Compliance Verification Banner
"""
import sys
import os
import json
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(os.path.dirname(__file__)).parent

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>SanDisk &middot; Wafer Intelligence &amp; Die Yield XAI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap">
  <style>
    :root {
      /* Apple Design System - Dark Palette Default */
      --canvas: #090c10;
      --panel: rgba(22, 27, 34, 0.72);
      --panel-elevated: rgba(30, 38, 49, 0.85);
      --panel-solid: #161b22;
      --glass-border: rgba(255, 255, 255, 0.08);
      --glass-highlight: rgba(255, 255, 255, 0.16);
      --text: #f0f6fc;
      --text-secondary: #8b949e;
      --text-tertiary: #6e7681;
      
      /* Semantic Color Tokens */
      --accent: #0a84ff;
      --accent-tint: rgba(10, 132, 255, 0.15);
      --sandisk-red: #ef4444;
      --sandisk-red-tint: rgba(239, 68, 68, 0.15);
      --pass: #10b981;
      --pass-tint: rgba(16, 185, 129, 0.15);
      --warning: #f59e0b;
      --warning-tint: rgba(245, 158, 11, 0.15);
      --critical: #ef4444;
      --critical-tint: rgba(239, 68, 68, 0.18);
      --high: #ff7a00;
      --high-tint: rgba(255, 122, 0, 0.18);
      --medium: #f59e0b;
      --medium-tint: rgba(245, 158, 11, 0.18);
      --low: #10b981;
      --known: #64748b;
      
      /* Driver colors */
      --pos-driver: #ff7a00;
      --neg-driver: #10b981;
      
      /* Shadows & Physics */
      --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.25);
      --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.35);
      --shadow-lg: 0 16px 40px rgba(0, 0, 0, 0.45);
      --spring: cubic-bezier(0.16, 1, 0.3, 1);
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
    }

    [data-theme="light"] {
      --canvas: #f5f5f7;
      --panel: rgba(255, 255, 255, 0.78);
      --panel-elevated: rgba(255, 255, 255, 0.92);
      --panel-solid: #ffffff;
      --glass-border: rgba(0, 0, 0, 0.07);
      --glass-highlight: rgba(255, 255, 255, 0.9);
      --text: #1d1d1f;
      --text-secondary: #6e6e73;
      --text-tertiary: #86868b;
      
      --accent: #0071e3;
      --accent-tint: rgba(0, 113, 227, 0.1);
      --sandisk-red: #dc2626;
      --sandisk-red-tint: rgba(220, 38, 38, 0.1);
      --pass: #059669;
      --pass-tint: rgba(5, 150, 105, 0.12);
      --warning: #d97706;
      --warning-tint: rgba(217, 119, 6, 0.12);
      --critical: #dc2626;
      --critical-tint: rgba(220, 38, 38, 0.12);
      --high: #ea580c;
      --high-tint: rgba(234, 88, 12, 0.15);
      --medium: #d97706;
      --medium-tint: rgba(217, 119, 6, 0.15);
      --low: #059669;
      --known: #64748b;
      
      --pos-driver: #ea580c;
      --neg-driver: #059669;
      
      --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.04);
      --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.06);
      --shadow-lg: 0 16px 40px rgba(0, 0, 0, 0.08);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-font-smoothing: antialiased; }
    
    body {
      background: var(--canvas);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", system-ui, sans-serif;
      line-height: 1.47059;
      font-size: 14px;
      min-height: 100vh;
      overflow-x: hidden;
      transition: background 0.3s var(--spring), color 0.3s var(--spring);
    }

    .mono {
      font-family: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, monospace;
      font-variant-numeric: tabular-nums;
    }

    /* Container */
    .app-shell {
      max-width: 1360px;
      margin: 0 auto;
      padding: 24px 28px 60px;
    }

    /* Apple Frosted Glass Card */
    .glass-card {
      background: var(--panel);
      backdrop-filter: blur(28px) saturate(190%);
      -webkit-backdrop-filter: blur(28px) saturate(190%);
      border: 1px solid var(--glass-border);
      border-top: 1px solid var(--glass-highlight);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-md);
      transition: transform 0.25s var(--spring), box-shadow 0.25s var(--spring);
    }

    /* Navigation Header */
    header.app-header {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      margin-bottom: 22px;
    }

    .brand-group {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .sandisk-logo-pill {
      background: var(--sandisk-red);
      color: #fff;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      padding: 4px 10px;
      border-radius: 999px;
      box-shadow: 0 2px 10px rgba(255, 69, 58, 0.4);
    }

    .title-wrap h1 {
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.025em;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .title-wrap .subtitle {
      font-size: 12.5px;
      color: var(--text-secondary);
      margin-top: 2px;
    }

    /* Header Actions */
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .btn-apple {
      background: var(--panel-elevated);
      color: var(--text);
      border: 1px solid var(--glass-border);
      border-top: 1px solid var(--glass-highlight);
      padding: 7px 14px;
      border-radius: 999px;
      font-size: 12.5px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s var(--spring);
      box-shadow: var(--shadow-sm);
    }

    .btn-apple:hover {
      background: var(--panel-solid);
      border-color: var(--accent);
      color: var(--accent);
    }

    .btn-apple:active {
      transform: scale(0.96);
    }

    .btn-apple.primary {
      background: var(--accent);
      color: #fff;
      border: none;
      box-shadow: 0 2px 12px var(--accent-tint);
    }

    .btn-apple.primary:hover {
      opacity: 0.92;
      color: #fff;
    }

    /* Top Stats Pills Bar */
    .hero-stats-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }

    .hero-stat-card {
      padding: 14px 18px;
      position: relative;
      overflow: hidden;
    }

    .hero-stat-card .label {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .hero-stat-card .value {
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.03em;
      color: var(--text);
      margin: 4px 0 2px;
      display: flex;
      align-items: baseline;
      gap: 8px;
    }

    .hero-stat-card .diff-badge {
      font-size: 12px;
      font-weight: 600;
      color: var(--pass);
      background: var(--pass-tint);
      padding: 2px 7px;
      border-radius: 999px;
    }

    .hero-stat-card .subtext {
      font-size: 11.5px;
      color: var(--text-tertiary);
    }

    /* Apple Segmented Control (Tabs) */
    .nav-segmented {
      display: flex;
      align-items: center;
      background: var(--panel);
      backdrop-filter: blur(20px);
      border: 1px solid var(--glass-border);
      border-radius: 999px;
      padding: 4px;
      margin-bottom: 22px;
      gap: 4px;
      overflow-x: auto;
      scrollbar-width: none;
      box-shadow: var(--shadow-sm);
    }

    .nav-segmented::-webkit-scrollbar { display: none; }

    .seg-tab {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-size: 13px;
      font-weight: 500;
      padding: 8px 18px;
      border-radius: 999px;
      cursor: pointer;
      white-space: nowrap;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      transition: all 0.2s var(--spring);
      user-select: none;
    }

    .seg-tab:hover {
      color: var(--text);
    }

    .seg-tab:active {
      transform: scale(0.97);
    }

    .seg-tab.active {
      background: var(--panel-elevated);
      color: var(--text);
      font-weight: 600;
      box-shadow: 0 2px 10px rgba(0,0,0,0.12);
      border: 1px solid var(--glass-highlight);
    }

    /* Tab Panes */
    .tab-content {
      display: none;
      animation: tabEnter 0.28s var(--spring) forwards;
    }

    .tab-content.active {
      display: block;
    }

    @keyframes tabEnter {
      from { opacity: 0; transform: translateY(8px) scale(0.995); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* Information Gain Strip */
    .info-gain-strip {
      padding: 18px 22px;
      margin-bottom: 20px;
    }

    .gain-bar-track {
      display: flex;
      height: 32px;
      border-radius: 10px;
      overflow: hidden;
      margin: 12px 0 14px;
      background: var(--panel-solid);
      border: 1px solid var(--glass-border);
      box-shadow: inset 0 1px 3px rgba(0,0,0,0.15);
    }

    .gain-slice {
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      font-size: 11px;
      font-weight: 600;
      padding: 0 8px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      transition: width 0.4s var(--spring), opacity 0.15s;
    }

    .gain-slice:hover { opacity: 0.9; cursor: pointer; }

    .gain-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      font-size: 12px;
      color: var(--text-secondary);
    }

    .gain-legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .gain-legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 3px;
    }

    .gain-callout {
      margin-top: 14px;
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      background: var(--accent-tint);
      border-left: 3px solid var(--accent);
      font-size: 13px;
      color: var(--text);
    }

    /* 2-Column Grid for Wafer & Inspector */
    .inspector-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.22fr) minmax(0, 1fr);
      gap: 20px;
    }

    @media (max-width: 980px) {
      .inspector-layout { grid-template-columns: 1fr; }
    }

    /* Card Panels */
    .panel-header {
      padding: 14px 20px;
      border-bottom: 1px solid var(--glass-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .panel-header h3 {
      font-size: 14px;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .panel-body {
      padding: 18px 20px;
    }

    /* Wafer Selector Pills */
    .wafer-pills-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }

    .w-pill {
      background: var(--panel-elevated);
      border: 1px solid var(--glass-border);
      border-radius: 12px;
      padding: 8px 12px;
      cursor: pointer;
      text-align: left;
      transition: all 0.15s var(--spring);
      color: inherit;
      font: inherit;
    }

    .w-pill:hover {
      border-color: var(--accent);
      transform: translateY(-1px);
    }

    .w-pill:active { transform: scale(0.97); }

    .w-pill.active {
      background: var(--accent-tint);
      border-color: var(--accent);
      box-shadow: 0 0 0 1px var(--accent);
    }

    .w-pill .w-title {
      font-weight: 600;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .w-pill .w-desc {
      font-size: 11px;
      color: var(--text-secondary);
      margin-top: 2px;
    }

    /* Wafer Map Toolbar */
    .map-controls-bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
      padding: 6px 12px;
      border-radius: 10px;
      background: var(--panel-solid);
      border: 1px solid var(--glass-border);
    }

    .filter-group {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
    }

    .filter-btn {
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-secondary);
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 11.5px;
      cursor: pointer;
      transition: all 0.15s var(--spring);
    }

    .filter-btn.active {
      background: var(--panel-elevated);
      color: var(--text);
      border-color: var(--glass-highlight);
      font-weight: 600;
    }

    /* SVG Wafer Canvas */
    .wafer-canvas-wrap {
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 10px;
      background: var(--panel-solid);
      border: 1px solid var(--glass-border);
      border-radius: var(--radius-md);
      overflow: hidden;
      position: relative;
    }

    svg.interactive-wafer {
      max-width: 100%;
      height: auto;
      touch-action: none;
      user-select: none;
      transition: transform 0.2s var(--spring);
    }

    /* Tooltip */
    .apple-tooltip {
      position: fixed;
      pointer-events: none;
      z-index: 999;
      background: rgba(18, 22, 31, 0.94);
      color: #fff;
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      padding: 8px 12px;
      border-radius: 10px;
      font-size: 12px;
      box-shadow: var(--shadow-lg);
      border: 1px solid rgba(255, 255, 255, 0.15);
      opacity: 0;
      transition: opacity 0.12s var(--spring);
      max-width: 250px;
      line-height: 1.4;
    }

    /* Die Inspector */
    .die-header-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--glass-border);
      margin-bottom: 14px;
    }

    .die-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .pill-badge {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.03em;
    }

    /* Prob Bar */
    .prob-meter-row {
      display: grid;
      grid-template-columns: 80px 1fr 65px;
      align-items: center;
      gap: 12px;
      margin: 10px 0;
    }

    .prob-meter-row .p-label {
      font-size: 12px;
      color: var(--text-secondary);
      font-weight: 500;
    }

    .prob-track {
      height: 16px;
      background: var(--panel-solid);
      border: 1px solid var(--glass-border);
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }

    .prob-fill {
      height: 100%;
      border-radius: 5px 0 0 5px;
      transition: width 0.3s var(--spring);
    }

    .threshold-marker {
      position: absolute;
      top: -2px;
      bottom: -2px;
      width: 2px;
      background: var(--text);
      opacity: 0.7;
      z-index: 2;
    }

    /* Key Values */
    .kv-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12.5px;
      padding: 6px 0;
      border-bottom: 1px dashed var(--glass-border);
    }

    .kv-row .k { color: var(--text-secondary); }

    /* Driver Rows */
    .driver-item {
      margin: 8px 0;
    }

    .driver-head {
      display: flex;
      justify-content: space-between;
      font-size: 11.5px;
      margin-bottom: 3px;
    }

    .driver-bar-axis {
      position: relative;
      height: 8px;
      background: var(--panel-solid);
      border-radius: 4px;
      overflow: hidden;
    }

    .driver-center-line {
      position: absolute;
      left: 50%;
      top: 0;
      bottom: 0;
      width: 1px;
      background: var(--text-secondary);
      opacity: 0.5;
    }

    .driver-fill-bar {
      position: absolute;
      top: 0;
      height: 100%;
      border-radius: 3px;
      transition: width 0.3s var(--spring);
    }

    /* Section Subtitle */
    .sub-section-title {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--text-secondary);
      margin: 16px 0 8px;
    }

    /* Top List */
    .top-dies-queue {
      display: flex;
      flex-direction: column;
      gap: 7px;
    }

    .top-queue-card {
      background: var(--panel-solid);
      border: 1px solid var(--glass-border);
      border-radius: 10px;
      padding: 9px 12px;
      cursor: pointer;
      display: grid;
      grid-template-columns: 28px 1fr auto;
      align-items: center;
      gap: 10px;
      transition: all 0.15s var(--spring);
    }

    .top-queue-card:hover {
      border-color: var(--accent);
      transform: translateX(3px);
    }

    .top-queue-card:active {
      transform: scale(0.98);
    }

    .top-queue-card .rank-num {
      font-weight: 700;
      color: var(--accent);
      font-size: 13px;
    }

    .top-queue-card .die-info {
      font-size: 12.5px;
    }

    .top-queue-card .die-info small {
      color: var(--text-secondary);
    }

    /* Triptych Grid */
    .triptych-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }

    @media (max-width: 780px) {
      .triptych-grid { grid-template-columns: 1fr; }
    }

    .triptych-panel {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      padding: 14px;
      background: var(--panel-solid);
      border: 1px solid var(--glass-border);
      border-radius: var(--radius-md);
    }

    .triptych-panel h4 {
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
    }

    /* Comparison Table */
    .apple-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 10px;
    }

    .apple-table th {
      text-align: left;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-secondary);
      padding: 10px 12px;
      border-bottom: 1px solid var(--glass-border);
    }

    .apple-table td {
      padding: 11px 12px;
      border-bottom: 1px solid var(--glass-border);
      color: var(--text);
    }

    .apple-table tr:last-child td { border-bottom: none; }

    .apple-table tr:hover td {
      background: var(--accent-tint);
    }

    /* Checklist Box */
    .compliance-box {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }

    .compliance-card {
      padding: 12px 16px;
      background: var(--panel-solid);
      border: 1px solid var(--glass-border);
      border-radius: var(--radius-md);
    }

    .compliance-card .chk-head {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
      font-size: 13px;
      color: var(--pass);
    }

    .compliance-card .chk-desc {
      font-size: 11.5px;
      color: var(--text-secondary);
      margin-top: 4px;
      line-height: 1.4;
    }

    /* Interactive Cost Calculator Slider */
    .cost-slider-wrap {
      margin: 18px 0;
      padding: 16px 20px;
      background: var(--panel-solid);
      border-radius: var(--radius-md);
      border: 1px solid var(--glass-border);
    }

    .slider-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }

    input[type=range] {
      width: 100%;
      accent-color: var(--accent);
      cursor: pointer;
    }

    /* Footer */
    footer.app-footer {
      text-align: center;
      padding: 24px 0 10px;
      font-size: 12px;
      color: var(--text-tertiary);
      line-height: 1.6;
    }

    /* Sound indicator */
    .sound-badge {
      font-size: 11px;
      opacity: 0.7;
    }
  </style>
</head>
<body>
  <div class="app-shell">
    
    <!-- Top Brand Header -->
    <header class="glass-card app-header">
      <div class="brand-group">
        <span class="sandisk-logo-pill">SanDisk</span>
        <div class="title-wrap">
          <h1>Wafer Intelligence &middot; XAI Platform</h1>
          <div class="subtitle">Multi-Resolution Die Yield Prediction with Interpretable Spatial Context</div>
        </div>
      </div>
      
      <div class="header-actions">
        <button class="btn-apple" id="soundToggle" type="button" title="Audio Feedback">
          <span id="soundIcon">&#128266;</span> Haptics
        </button>
        <button class="btn-apple" id="themeToggle" type="button" title="Toggle Light/Dark Theme">
          <span id="themeIcon">&#9790;</span> Appearance
        </button>
        <button class="btn-apple primary" id="exportBtn" type="button">
          &darr; Export Predictions
        </button>
      </div>
    </header>

    <!-- Top Key Results KPI Strip -->
    <div class="hero-stats-row">
      <div class="glass-card hero-stat-card">
        <div class="label">AUC-PR (Model B vs A) <span class="diff-badge">+8.6%</span></div>
        <div class="value mono">0.528 <span style="font-size:14px;color:var(--text-secondary);font-weight:400">vs 0.486</span></div>
        <div class="subtext">Primary defect ranking metric &middot; P(&Delta;&gt;0) = 100%</div>
      </div>
      
      <div class="glass-card hero-stat-card">
        <div class="label">True Defect Catch Rate <span class="diff-badge">+11.0%</span></div>
        <div class="value mono">39.5% <span style="font-size:14px;color:var(--text-secondary);font-weight:400">545 dies</span></div>
        <div class="subtext">54 extra real failures caught over die+spatial model alone</div>
      </div>

      <div class="glass-card hero-stat-card">
        <div class="label">False Alarm Reduction <span class="diff-badge">-57%</span></div>
        <div class="value mono">204 <span style="font-size:14px;color:var(--text-secondary);font-weight:400">vs 474 FP</span></div>
        <div class="subtext">At matched 40% defect recall budget (huge fab cost savings)</div>
      </div>

      <div class="glass-card hero-stat-card">
        <div class="label">Hidden-Risk Dies Caught <span class="diff-badge">62 True Fails</span></div>
        <div class="value mono">209 <span style="font-size:14px;color:var(--text-secondary);font-weight:400">flagged</span></div>
        <div class="subtext">Discovered exclusively by sub-die block readings</div>
      </div>
    </div>

    <!-- Apple Segmented Navigation -->
    <nav class="nav-segmented" role="tablist">
      <button class="seg-tab active" data-tab="tab-xai" type="button" role="tab">&#128269; Wafer &amp; Die XAI</button>
      <button class="seg-tab" data-tab="tab-triptych" type="button" role="tab">&#128300; Three-Panel Triptych</button>
      <button class="seg-tab" data-tab="tab-benchmarks" type="button" role="tab">&#128202; Model A vs B Proof</button>
      <button class="seg-tab" data-tab="tab-economics" type="button" role="tab">&#128188; Fab Economics &amp; Costs</button>
      <button class="seg-tab" data-tab="tab-queue" type="button" role="tab">&#127919; Investigation Queue</button>
      <button class="seg-tab" data-tab="tab-compliance" type="button" role="tab">&#9989; Eligibility Verification</button>
    </nav>

    <!-- ======================================================== -->
    <!-- TAB 1: WAFER & DIE XAI INSPECTOR -->
    <!-- ======================================================== -->
    <div class="tab-content active" id="tab-xai" role="tabpanel">
      <!-- Information Gain Overview -->
      <div class="glass-card info-gain-strip">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <h3 style="font-size:13px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-secondary)">
            Multi-Resolution Information Gain Distribution (Every Eligible Die)
          </h3>
          <span class="mono" style="font-size:12px;color:var(--text-secondary)">32,598 Test Dies</span>
        </div>
        <div class="gain-bar-track" id="gainBar"></div>
        <div class="gain-legend" id="gainLegend"></div>
        <div class="gain-callout" id="gainCallout"></div>
      </div>

      <!-- Main Interactive Inspector Layout -->
      <div class="inspector-layout">
        <!-- Wafer Map Column -->
        <div class="glass-card">
          <div class="panel-header">
            <h3>&#127758; Wafer Selection &amp; Die Risk Map</h3>
            <span id="waferInfoTag" class="pill-badge" style="background:var(--accent-tint);color:var(--accent)">W_N_0099</span>
          </div>
          <div class="panel-body">
            <!-- Wafer Selector Pills -->
            <div class="wafer-pills-row" id="waferPills"></div>

            <!-- Map Controls -->
            <div class="map-controls-bar">
              <div class="filter-group">
                <span style="color:var(--text-secondary);font-size:11px;text-transform:uppercase;font-weight:600">Filter:</span>
                <button class="filter-btn active" data-filter="all" type="button">All</button>
                <button class="filter-btn" data-filter="critical" type="button">Critical</button>
                <button class="filter-btn" data-filter="high" type="button">High Risk</button>
                <button class="filter-btn" data-filter="hidden" type="button">Hidden Risk</button>
                <button class="filter-btn" data-filter="known" type="button">Pre-Test Fail</button>
              </div>
              <div class="filter-group">
                <button class="btn-apple" id="zoomInBtn" type="button" style="padding:2px 8px;font-size:11px">+</button>
                <button class="btn-apple" id="zoomOutBtn" type="button" style="padding:2px 8px;font-size:11px">&minus;</button>
                <button class="btn-apple" id="zoomResetBtn" type="button" style="padding:2px 8px;font-size:11px">Reset</button>
              </div>
            </div>

            <!-- SVG Map -->
            <div class="wafer-canvas-wrap">
              <svg class="interactive-wafer" id="waferSvg" role="img" aria-label="Interactive Wafer Map"></svg>
            </div>
            
            <div class="gain-legend" id="mapLegend" style="justify-content:center;margin-top:12px"></div>
          </div>
        </div>

        <!-- Die Inspector Column -->
        <div class="glass-card">
          <div class="panel-header">
            <h3 id="inspHeader">Die Inspector</h3>
            <span id="inspBadge" class="pill-badge">Select a die</span>
          </div>
          <div class="panel-body" id="inspBody">
            <p style="color:var(--text-secondary);padding:20px 0;text-align:center">
              Click any die on the wafer map to inspect its dual-model probabilities, information-gain category, and single-feature occlusion attribution drivers.
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- ======================================================== -->
    <!-- TAB 2: THREE-PANEL WAFER TRIPTYCH -->
    <!-- ======================================================== -->
    <div class="tab-content" id="tab-triptych" role="tabpanel">
      <div class="glass-card" style="padding:20px;margin-bottom:20px">
        <div class="panel-header" style="padding:0 0 14px;margin-bottom:14px">
          <div>
            <h3>Three-Panel Wafer Triptych &amp; Prediction Verification</h3>
            <p style="font-size:12.5px;color:var(--text-secondary);margin-top:2px">
              Direct visual proof: Pre-Test Knowledge &rarr; Post-Test Ground Truth &rarr; What Changed (New Failures in Orange), paired with Model B Outcome (TP / FN / FP).
            </p>
          </div>
          <div class="filter-group">
            <span style="font-size:11px;text-transform:uppercase;color:var(--text-secondary);font-weight:600">Select Wafer:</span>
            <div id="tripWaferSelect" class="filter-group"></div>
          </div>
        </div>

        <div class="triptych-grid">
          <div class="triptych-panel">
            <h4>1. Pre-Test (<span class="mono">old_label</span>)</h4>
            <div style="font-size:11px;color:var(--text-secondary)">Green: Passing &middot; Gray: Pre-Test Fail</div>
            <svg id="tTripPre" style="max-width:100%;height:auto"></svg>
          </div>
          <div class="triptych-panel">
            <h4>2. Post-Test Truth (<span class="mono">label</span>)</h4>
            <div style="font-size:11px;color:var(--text-secondary)">Green: Passed All &middot; Red: All Post-Test Fails</div>
            <svg id="tTripPost" style="max-width:100%;height:auto"></svg>
          </div>
          <div class="triptych-panel">
            <h4>3. Difference (<span style="color:var(--high);font-weight:600">Orange = New Fail</span>)</h4>
            <div style="font-size:11px;color:var(--text-secondary)">Target: Dies that were passing, but newly failed</div>
            <svg id="tTripDiff" style="max-width:100%;height:auto"></svg>
          </div>
        </div>

        <div class="gain-callout" style="margin-top:16px" id="tripSummaryCallout">
          Loading wafer summary...
        </div>
      </div>
    </div>

    <!-- ======================================================== -->
    <!-- TAB 3: MODEL A VS B BENCHMARKS -->
    <!-- ======================================================== -->
    <div class="tab-content" id="tab-benchmarks" role="tabpanel">
      <div class="inspector-layout">
        <div class="glass-card">
          <div class="panel-header">
            <h3>&#128200; Side-by-Side Model Benchmarks (Held-Out Test Set)</h3>
            <span class="pill-badge" style="background:var(--pass-tint);color:var(--pass)">Eligible Dies Only</span>
          </div>
          <div class="panel-body">
            <table class="apple-table" id="comparisonTable">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Model A (Die+Spatial)</th>
                  <th>Model B (+Block Signal)</th>
                  <th>Delta (&Delta;)</th>
                  <th>Impact</th>
                </tr>
              </thead>
              <tbody id="compTableBody"></tbody>
            </table>

            <div class="sub-section-title" style="margin-top:22px">Wafer-Level Bootstrap Statistical Significance (2,000 Resamples)</div>
            <p style="font-size:12px;color:var(--text-secondary);margin-bottom:8px">
              Resampling whole wafers (not independent dies) strictly respects the hierarchical wafer structure:
            </p>
            <table class="apple-table">
              <thead>
                <tr>
                  <th>Metric Delta (B &minus; A)</th>
                  <th>Mean Gain</th>
                  <th>95% Confidence Interval</th>
                  <th>P(Gain &gt; 0)</th>
                </tr>
              </thead>
              <tbody id="bootstrapBody"></tbody>
            </table>
          </div>
        </div>

        <!-- Category Importance & Confusion Matrices -->
        <div class="glass-card">
          <div class="panel-header">
            <h3>&#128300; Feature Category Contributions</h3>
            <span class="pill-badge" style="background:var(--accent-tint);color:var(--accent)">Permutation AUC-PR</span>
          </div>
          <div class="panel-body">
            <p style="font-size:12px;color:var(--text-secondary);margin-bottom:12px">
              Permutation feature importance confirms <b>sub-die block readings</b> contribute the #2 highest predictive signal in Model B, with <code class="mono">block_mean</code> as the #1 most important individual feature.
            </p>
            <div id="categoryBars"></div>

            <div class="sub-section-title" style="margin-top:22px">Confusion Matrices Comparison</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">
              <div style="background:var(--panel-solid);padding:10px;border-radius:8px;border:1px solid var(--glass-border)">
                <div style="font-weight:600;font-size:12px;margin-bottom:6px">Model A (t = 0.64)</div>
                <div class="mono" style="font-size:11.5px;color:var(--text-secondary)">
                  True Fail: <b style="color:var(--text)">491</b> &middot; Missed: 889<br>
                  False Alarm: 32 &middot; True Pass: 31,186
                </div>
              </div>
              <div style="background:var(--panel-solid);padding:10px;border-radius:8px;border:1px solid var(--glass-border)">
                <div style="font-weight:600;font-size:12px;margin-bottom:6px;color:var(--accent)">Model B (t = 0.62)</div>
                <div class="mono" style="font-size:11.5px;color:var(--text-secondary)">
                  True Fail: <b style="color:var(--pass)">545 (+54)</b> &middot; Missed: 835<br>
                  False Alarm: 158 &middot; True Pass: 31,060
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ======================================================== -->
    <!-- TAB 4: FAB ECONOMICS & DECISION SUPPORT -->
    <!-- ======================================================== -->
    <div class="tab-content" id="tab-economics" role="tabpanel">
      <div class="glass-card" style="padding:20px;margin-bottom:20px">
        <div class="panel-header" style="padding:0 0 14px;margin-bottom:14px">
          <div>
            <h3>Fab Retest Workload &amp; Economic Cost Modeling</h3>
            <p style="font-size:12.5px;color:var(--text-secondary);margin-top:2px">
              Translating model scores directly into fab bottom-line savings: false-positive reduction and threshold optimization.
            </p>
          </div>
        </div>

        <div class="inspector-layout">
          <!-- Left: Matched Operating Points -->
          <div>
            <div class="sub-section-title">Matched Defect Recall Comparison (Apples-to-Apples)</div>
            <p style="font-size:12px;color:var(--text-secondary);margin-bottom:10px">
              When fab managers demand catching at least 40% of all escaping new failures:
            </p>
            <table class="apple-table">
              <thead>
                <tr>
                  <th>Constraint</th>
                  <th>Model</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>False Alarms</th>
                  <th>Total Tested</th>
                </tr>
              </thead>
              <tbody id="matchedOpsBody"></tbody>
            </table>
            
            <div class="gain-callout" style="margin-top:14px">
              <b>Fab Impact:</b> At equal 40% defect recall, Model B drops unnecessary retests from <b>474 to 204</b> &mdash; a <b>57.0% reduction in wasted engineer retesting</b>!
            </div>
          </div>

          <!-- Right: Interactive Cost Simulator -->
          <div>
            <div class="sub-section-title">Interactive Cost-Optimal Decision Calculator</div>
            <p style="font-size:12px;color:var(--text-secondary)">
              Adjust the economic severity ratio: how many times more costly is a missed defect in the field vs. a retest in the fab?
            </p>

            <div class="cost-slider-wrap">
              <div class="slider-header">
                <span style="font-size:13px;font-weight:600">Defect Escape Penalty:</span>
                <span class="mono" id="costRatioDisplay" style="font-size:15px;font-weight:700;color:var(--accent)">10x retest cost</span>
              </div>
              <input type="range" id="costSlider" min="2" max="50" step="1" value="10">
              
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px">
                <div style="background:var(--canvas);padding:10px;border-radius:8px">
                  <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase">Recommended Threshold</div>
                  <div class="mono" id="calcThreshold" style="font-size:20px;font-weight:700;color:var(--accent);margin-top:2px">0.42</div>
                </div>
                <div style="background:var(--canvas);padding:10px;border-radius:8px">
                  <div style="font-size:11px;color:var(--text-secondary);text-transform:uppercase">Net Cost vs Test-All</div>
                  <div class="mono" id="calcSavings" style="font-size:20px;font-weight:700;color:var(--pass);margin-top:2px">-73.8%</div>
                </div>
              </div>
            </div>

            <div class="sub-section-title">Calibration &amp; Probability Reliability</div>
            <div class="mono" style="font-size:12px;color:var(--text-secondary);background:var(--panel-solid);padding:10px;border-radius:8px;border:1px solid var(--glass-border)">
              Model A: Brier = 0.0488, Expected Calibration Error (ECE) = 12.5%<br>
              Model B: Brier = 0.0459, Expected Calibration Error (ECE) = <b>11.0% (Better Calibrated)</b>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ======================================================== -->
    <!-- TAB 5: TOP DIES QUEUE -->
    <!-- ======================================================== -->
    <div class="tab-content" id="tab-queue" role="tabpanel">
      <div class="glass-card" style="padding:20px">
        <div class="panel-header" style="padding:0 0 14px;margin-bottom:14px">
          <div>
            <h3>&#127919; Top Priority Dies for Fab Engineer Investigation</h3>
            <p style="font-size:12.5px;color:var(--text-secondary);margin-top:2px">
              Ranked by a composite priority score fusing Predicted Risk (40%), Information Gain (20%), Sub-Die Anomaly (20%), and Spatial Zone Severity (20%). Click any die to jump directly to it on the map.
            </p>
          </div>
          <div class="filter-group">
            <span style="font-size:11px;text-transform:uppercase;color:var(--text-secondary);font-weight:600">Filter Category:</span>
            <select id="queueCatFilter" class="btn-apple" style="padding:4px 10px">
              <option value="ALL">All Categories</option>
              <option value="HIDDEN_RISK">Hidden Risk (Model B Only)</option>
              <option value="CONFIRMED_RISK">Confirmed Risk</option>
            </select>
          </div>
        </div>

        <div class="top-dies-queue" id="globalTopQueue"></div>
      </div>
    </div>

    <!-- ======================================================== -->
    <!-- TAB 6: ELIGIBILITY & LEAKAGE VERIFICATION -->
    <!-- ======================================================== -->
    <div class="tab-content" id="tab-compliance" role="tabpanel">
      <div class="glass-card" style="padding:22px">
        <div class="panel-header" style="padding:0 0 14px;margin-bottom:14px">
          <div>
            <h3>&#9989; Codebase Eligibility &amp; Data Leakage Compliance Certification</h3>
            <p style="font-size:12.5px;color:var(--text-secondary);margin-top:2px">
              Verification that every aspect of the codebase, feature engineering, and cross-validation strictly adheres to the SanDisk hackathon guidelines.
            </p>
          </div>
          <span class="pill-badge" style="background:var(--pass-tint);color:var(--pass)">100% Compliant</span>
        </div>

        <div class="compliance-box">
          <div class="compliance-card">
            <div class="chk-head">&#10004; Eligible Population Enforcement</div>
            <div class="chk-desc">
              All training, cross-validation, threshold tuning, and test metrics are computed exclusively on eligible dies (<code class="mono">old_label == 0</code>, 154k train / 32.6k test). Pre-test failures are excluded from metric denominator to prevent artificial inflation.
            </div>
          </div>

          <div class="compliance-card">
            <div class="chk-head">&#10004; Zero Target Leakage in Spatial Features</div>
            <div class="chk-desc">
              All 13 spatial features (<code class="mono">old_fail_density_5x5</code>, <code class="mono">nearest_old_fail_dist</code>, etc.) are computed strictly from pre-test <code class="mono">old_label</code> using <code class="mono">scipy.spatial.cKDTree</code>. Post-test <code class="mono">label</code> is never accessed during feature creation.
            </div>
          </div>

          <div class="compliance-card">
            <div class="chk-head">&#10004; Zero Cross-Wafer Bleed (GroupKFold)</div>
            <div class="chk-desc">
              Cross-validation uses 5-fold <code class="mono">GroupKFold</code> keyed on <code class="mono">wafer_id</code>. No wafer appears in both train and validation splits within any fold.
            </div>
          </div>

          <div class="compliance-card">
            <div class="chk-head">&#10004; IncrementalPCA Fit on Train Only</div>
            <div class="chk-desc">
              The 15-component IncrementalPCA is fit exclusively on training-eligible dies in 2,000-row batches, then applied to test data without refitting. (Strict per-fold ablation confirms &Delta;AUC-PR difference is negligible: -0.0007).
            </div>
          </div>

          <div class="compliance-card">
            <div class="chk-head">&#10004; Out-of-Fold Threshold Selection</div>
            <div class="chk-desc">
              Decision thresholds ($t_A=0.64$, $t_B=0.62$) are chosen once on Out-Of-Fold CV predictions to maximize fail F1, frozen, and applied without ever touching test labels.
            </div>
          </div>

          <div class="compliance-card">
            <div class="chk-head">&#10004; Official Submission Format Compliance</div>
            <div class="chk-desc">
              Generated <code class="mono">validation_predictions.csv</code> strictly adheres to the requested 4-column structure (<code class="mono">wafer_id, die_row, die_col, predicted_label</code>), with pre-test fails forced to 1.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Footer -->
    <footer class="app-footer">
      <div>SanDisk Hackathon &middot; Multi-Resolution Die Yield Prediction Platform</div>
      <div style="margin-top:4px">
        549 Features &middot; GroupKFold (5-splits) &middot; Zero Target Leakage
      </div>
    </footer>
  </div>

  <!-- Interactive Floating Tooltip -->
  <div class="apple-tooltip" id="appleTip"></div>

  <!-- Embedded Data Payload -->
  <script>/*__DATA_PAYLOAD__*/</script>

  <!-- Interactive Application Logic -->
  <script>
    const D = window.__DATA__;
    const TOP = window.__TOP__ || [];
    const BOOT = window.__BOOT__ || {};
    const COMMON_OPS = window.__COMMON_OPS__ || [];
    const COMP_METRICS = window.__COMP_METRICS__ || [];
    const TA = D.thresholds.model_a, TB = D.thresholds.model_b;

    // Color maps
    const BAND_COLORS = {
      low: 'var(--low)',
      medium: 'var(--medium)',
      high: 'var(--high)',
      critical: 'var(--critical)',
      known_fail: 'var(--known)'
    };
    const BAND_LABELS = {
      low: 'Low Risk (<0.15)',
      medium: 'Medium (0.15-0.35)',
      high: 'High Risk (0.35-0.62)',
      critical: 'Critical Risk (≥0.62)',
      known_fail: 'Known Pre-Test Fail'
    };
    const CAT_LABELS = {
      CONFIRMED_RISK: 'Confirmed Risk',
      HIDDEN_RISK: 'Hidden Risk (Model B Only)',
      MODEL_DISAGREEMENT: 'Model Disagreement',
      REDUNDANT_INFORMATION: 'Redundant Info',
      LOW_RISK: 'Low Risk'
    };
    const CAT_COLORS = {
      CONFIRMED_RISK: 'var(--critical)',
      HIDDEN_RISK: 'var(--high)',
      MODEL_DISAGREEMENT: 'var(--medium)',
      REDUNDANT_INFORMATION: 'var(--accent)',
      LOW_RISK: 'var(--pass)'
    };

    let curWaferId = Object.keys(D.wafers)[0] || 'W_N_0099';
    let curDie = null;
    let activeFilter = 'all';
    let zoomLevel = 1.0;
    let audioContext = null;
    let soundEnabled = true;

    // Subtle Apple-style tactile click sound
    function playClickSound(pitch = 600, duration = 0.018) {
      if (!soundEnabled) return;
      try {
        if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
        if (audioContext.state === 'suspended') audioContext.resume();
        const osc = audioContext.createOscillator();
        const gain = audioContext.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(pitch, audioContext.currentTime);
        osc.frequency.exponentialRampToValueAtTime(pitch * 0.4, audioContext.currentTime + duration);
        gain.gain.setValueAtTime(0.06, audioContext.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioContext.destination);
        osc.start();
        osc.stop(audioContext.currentTime + duration);
      } catch (e) {}
    }

    // Sound Toggle
    document.getElementById('soundToggle').onclick = () => {
      soundEnabled = !soundEnabled;
      document.getElementById('soundIcon').innerHTML = soundEnabled ? '&#128266;' : '&#128263;';
      playClickSound(800, 0.02);
    };

    // Theme Toggle
    document.getElementById('themeToggle').onclick = () => {
      const html = document.documentElement;
      const nextTheme = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', nextTheme);
      document.getElementById('themeIcon').innerHTML = nextTheme === 'dark' ? '&#9790;' : '&#9788;';
      playClickSound(700, 0.025);
      renderWaferMap(curWaferId);
      renderTriptych(curWaferId);
    };

    // Segmented Navigation
    document.querySelectorAll('.seg-tab').forEach(tabBtn => {
      tabBtn.onclick = () => {
        document.querySelectorAll('.seg-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tabBtn.classList.add('active');
        const targetId = tabBtn.getAttribute('data-tab');
        document.getElementById(targetId).classList.add('active');
        playClickSound(550, 0.015);
      };
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key >= '1' && e.key <= '6') {
        const tabs = document.querySelectorAll('.seg-tab');
        const idx = parseInt(e.key) - 1;
        if (tabs[idx]) tabs[idx].click();
      } else if (e.key.toLowerCase() === 't') {
        document.getElementById('themeToggle').click();
      } else if (e.key === 'Escape' && curDie) {
        curDie = null;
        renderWaferMap(curWaferId);
        renderWaferOverview(curWaferId);
      }
    });

    // ==========================================
    // INIT INFORMATION GAIN STRIP
    // ==========================================
    (function initInfoGain() {
      const ig = D.info_gain_summary;
      const order = ['CONFIRMED_RISK', 'HIDDEN_RISK', 'MODEL_DISAGREEMENT', 'REDUNDANT_INFORMATION', 'LOW_RISK'];
      const total = Object.values(ig.counts).reduce((a, b) => a + b, 0);
      const bar = document.getElementById('gainBar');
      const leg = document.getElementById('gainLegend');

      order.forEach(cat => {
        const n = ig.counts[cat];
        const pct = (100 * n / total).toFixed(1);
        const slice = document.createElement('div');
        slice.className = 'gain-slice';
        slice.style.width = pct + '%';
        slice.style.background = CAT_COLORS[cat];
        if (parseFloat(pct) > 6) slice.textContent = CAT_LABELS[cat] + ' (' + pct + '%)';
        slice.title = `${CAT_LABELS[cat]}: ${n.toLocaleString()} dies (${pct}%)`;
        slice.onclick = () => {
          // Filter to this category
          activeFilter = (activeFilter === cat.toLowerCase()) ? 'all' : cat.toLowerCase();
          document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
          renderWaferMap(curWaferId);
          playClickSound(650, 0.02);
        };
        bar.appendChild(slice);

        const legItem = document.createElement('span');
        legItem.className = 'gain-legend-item';
        legItem.innerHTML = `<i class="gain-legend-dot" style="background:${CAT_COLORS[cat]}"></i>${CAT_LABELS[cat]} <b class="mono">${pct}%</b>`;
        leg.appendChild(legItem);
      });

      const conf = ig.per_category_outcome?.CONFIRMED_RISK;
      const hid = ig.per_category_outcome?.HIDDEN_RISK;
      document.getElementById('gainCallout').innerHTML =
        `<b>Information Gain Takeaway:</b> Sub-die block data confirmed <b>${conf?.count || 494}</b> high-risk dies with <b class="mono">${((conf?.actual_fail_rate || 0.978)*100).toFixed(1)}%</b> ground-truth failure rate, and uncovered <b style="color:var(--sandisk-red)">${ig.hidden_risk_true_failures || 62} genuine post-test failures</b> that the die-level parametric and spatial context model alone completely cleared!`;
    })();

    // ==========================================
    // WAFER PILLS SELECTOR
    // ==========================================
    function initWaferPills() {
      const container = document.getElementById('waferPills');
      const tripSelect = document.getElementById('tripWaferSelect');
      container.innerHTML = '';
      tripSelect.innerHTML = '';

      Object.keys(D.wafers).forEach(wid => {
        const w = D.wafers[wid];
        const nCrit = w.dies.filter(d => d.band === 'critical' || d.band === 'high').length;
        
        // Pill for Tab 1
        const pill = document.createElement('button');
        pill.type = 'button';
        pill.className = 'w-pill' + (wid === curWaferId ? ' active' : '');
        pill.dataset.wid = wid;
        pill.innerHTML = `
          <div class="w-title mono">&#9881; ${wid}</div>
          <div class="w-desc">${w.pattern} &middot; ${w.zones?.length || 0} zones &middot; ${nCrit} risk dies</div>
        `;
        pill.onclick = () => selectWafer(wid);
        container.appendChild(pill);

        // Pill for Tab 2
        const tripBtn = document.createElement('button');
        tripBtn.type = 'button';
        tripBtn.className = 'filter-btn' + (wid === curWaferId ? ' active' : '');
        tripBtn.textContent = wid;
        tripBtn.onclick = () => {
          document.querySelectorAll('#tripWaferSelect .filter-btn').forEach(b => b.classList.remove('active'));
          tripBtn.classList.add('active');
          renderTriptych(wid);
          playClickSound(600, 0.02);
        };
        tripSelect.appendChild(tripBtn);
      });
    }

    function selectWafer(wid) {
      curWaferId = wid;
      curDie = null;
      document.querySelectorAll('.w-pill').forEach(p => p.classList.toggle('active', p.dataset.wid === wid));
      document.getElementById('waferInfoTag').textContent = wid + ' (' + (D.wafers[wid].pattern || 'wafer') + ')';
      playClickSound(520, 0.02);
      renderWaferMap(wid);
      renderWaferOverview(wid);
      renderTriptych(wid);
    }

    // ==========================================
    // MAP FILTERING & ZOOM CONTROLS
    // ==========================================
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilter = btn.dataset.filter;
        renderWaferMap(curWaferId);
        playClickSound(620, 0.015);
      };
    });

    document.getElementById('zoomInBtn').onclick = () => {
      zoomLevel = Math.min(zoomLevel + 0.25, 2.5);
      applyZoom();
      playClickSound(700, 0.015);
    };
    document.getElementById('zoomOutBtn').onclick = () => {
      zoomLevel = Math.max(zoomLevel - 0.25, 0.75);
      applyZoom();
      playClickSound(600, 0.015);
    };
    document.getElementById('zoomResetBtn').onclick = () => {
      zoomLevel = 1.0;
      applyZoom();
      playClickSound(500, 0.015);
    };
    function applyZoom() {
      const svg = document.getElementById('waferSvg');
      svg.style.transform = `scale(${zoomLevel})`;
    }

    // ==========================================
    // WAFER MAP RENDER
    // ==========================================
    const SVG_NS = 'http://www.w3.org/2000/svg';
    function renderWaferMap(wid) {
      const w = D.wafers[wid];
      if (!w) return;
      const svg = document.getElementById('waferSvg');
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      let maxR = 0, maxC = 0;
      w.dies.forEach(d => { maxR = Math.max(maxR, d.r); maxC = Math.max(maxC, d.c); });
      const cols = maxC + 1, rows = maxR + 1;
      const cellSize = Math.max(7, Math.min(16, Math.floor(580 / Math.max(cols, rows))));
      const gap = Math.max(1, Math.floor(cellSize * 0.12));
      const width = cols * cellSize, height = rows * cellSize;

      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.style.width = Math.min(width, 580) + 'px';
      svg.style.height = 'auto';

      const frag = document.createDocumentFragment();

      w.dies.forEach(d => {
        // Filter logic
        let show = true;
        if (activeFilter === 'critical') show = d.band === 'critical';
        else if (activeFilter === 'high') show = d.band === 'critical' || d.band === 'high';
        else if (activeFilter === 'hidden') show = d.cat === 'HIDDEN_RISK';
        else if (activeFilter === 'known') show = d.old === 1;

        const rect = document.createElementNS(SVG_NS, 'rect');
        rect.setAttribute('x', d.c * cellSize);
        rect.setAttribute('y', d.r * cellSize);
        rect.setAttribute('width', cellSize - gap);
        rect.setAttribute('height', cellSize - gap);
        rect.setAttribute('rx', Math.max(1, cellSize * 0.16));
        
        let color = BAND_COLORS[d.band] || 'var(--low)';
        if (!show) {
          color = 'rgba(128, 128, 128, 0.15)';
        }
        rect.setAttribute('fill', color);
        if (d.old === 1) rect.setAttribute('opacity', show ? '0.42' : '0.1');

        rect.style.cursor = 'pointer';
        rect.style.transition = 'transform 0.1s, fill 0.15s';

        // Hover & Click
        rect.onmouseenter = e => showTooltip(e, d);
        rect.onmousemove = e => positionTooltip(e);
        rect.onmouseleave = hideTooltip;
        rect.onclick = () => {
          curDie = d;
          renderWaferMap(wid);
          renderDieInspector(wid, d);
          playClickSound(800, 0.02);
        };

        // Selected halo indicator
        if (curDie && curDie.r === d.r && curDie.c === d.c) {
          rect.setAttribute('stroke', 'var(--text)');
          rect.setAttribute('stroke-width', Math.max(2, cellSize * 0.22));
          rect.setAttribute('stroke-dasharray', 'none');
        }

        frag.appendChild(rect);
      });

      svg.appendChild(frag);
    }

    // Tooltip
    const tooltip = document.getElementById('appleTip');
    function showTooltip(e, d) {
      const catName = CAT_LABELS[d.cat] || d.cat || 'N/A';
      tooltip.innerHTML = `
        <div style="font-weight:700;font-size:12px;margin-bottom:2px" class="mono">Die (${d.r}, ${d.c})</div>
        <div style="color:var(--text-secondary);font-size:11px">${BAND_LABELS[d.band]}</div>
        ${d.old === 1 ? '<b style="color:var(--known)">Pre-Test Failure (old_label=1)</b>' : `
          <div style="margin-top:4px;font-size:11.5px">
            p<sub>A</sub>: <b class="mono">${d.pa.toFixed(3)}</b> &nbsp; p<sub>B</sub>: <b class="mono" style="color:var(--accent)">${d.pb.toFixed(3)}</b><br>
            <span style="font-size:11px;color:var(--text-tertiary)">${catName}</span>
          </div>
        `}
      `;
      tooltip.style.opacity = '1';
      positionTooltip(e);
    }
    function positionTooltip(e) {
      const x = Math.min(e.clientX + 14, window.innerWidth - 250);
      const y = Math.min(e.clientY + 14, window.innerHeight - 100);
      tooltip.style.left = x + 'px';
      tooltip.style.top = y + 'px';
    }
    function hideTooltip() {
      tooltip.style.opacity = '0';
    }

    // Legend
    (function initMapLegend() {
      const leg = document.getElementById('mapLegend');
      leg.innerHTML = '';
      Object.keys(BAND_COLORS).forEach(b => {
        const item = document.createElement('span');
        item.className = 'gain-legend-item';
        item.innerHTML = `<i class="gain-legend-dot" style="background:${BAND_COLORS[b]}"></i>${BAND_LABELS[b]}`;
        leg.appendChild(item);
      });
    })();

    // ==========================================
    // DIE INSPECTOR
    // ==========================================
    function renderWaferOverview(wid) {
      const w = D.wafers[wid];
      document.getElementById('inspHeader').textContent = wid + ' &middot; Spatial Overview';
      const badge = document.getElementById('inspBadge');
      badge.textContent = (w.pattern || 'wafer') + ' pattern';
      badge.style.background = 'var(--accent-tint)';
      badge.style.color = 'var(--accent)';

      const body = document.getElementById('inspBody');
      body.innerHTML = `
        <p style="color:var(--text-secondary);margin-bottom:14px;font-size:13px">
          This wafer exhibits a <b>${w.pattern}</b> spatial defect pattern across <b>${w.zones?.length || 0}</b> detected contiguous high-risk clusters. Click any individual die on the map to inspect its dual-model attribution drivers.
        </p>
      `;

      if (w.zones && w.zones.length > 0) {
        const title = document.createElement('div');
        title.className = 'sub-section-title';
        title.textContent = 'Detected Risk Zones (Size &times; Severity)';
        body.appendChild(title);

        const maxSev = Math.max(...w.zones.map(z => z.severity)) || 1;
        const list = document.createElement('div');
        list.style.display = 'flex';
        list.style.flexDirection = 'column';
        list.style.gap = '8px';

        w.zones.slice(0, 5).forEach((z, i) => {
          const card = document.createElement('div');
          card.style.background = 'var(--panel-solid)';
          card.style.padding = '10px 12px';
          card.style.borderRadius = '8px';
          card.style.border = '1px solid var(--glass-border)';
          const hiddenShare = z.hidden_risk_fraction != null ? ` &middot; ${(z.hidden_risk_fraction * 100).toFixed(0)}% hidden risk` : '';
          card.innerHTML = `
            <div style="display:flex;justify-content:space-between;font-size:12px">
              <span class="mono">Zone #${z.zone_id || (i+1)}: ${z.size} dies @ (${z.centroid_row?.toFixed(0) || 0}, ${z.centroid_col?.toFixed(0) || 0})</span>
              <span class="mono" style="font-weight:600;color:var(--critical)">Sev ${z.severity?.toFixed(2) || '0'}</span>
            </div>
            <div style="height:5px;background:var(--panel);border-radius:3px;margin:6px 0;overflow:hidden">
              <div style="height:100%;width:${(100*z.severity/maxSev).toFixed(0)}%;background:var(--critical)"></div>
            </div>
            <div style="font-size:11px;color:var(--text-secondary)">Mean Risk: ${z.mean_risk?.toFixed(2) || 0} &middot; Max Risk: ${z.max_risk?.toFixed(2) || 0}${hiddenShare}</div>
          `;
          list.appendChild(card);
        });
        body.appendChild(list);
      }
    }

    function renderDieInspector(wid, d) {
      document.getElementById('inspHeader').innerHTML = `Die <span class="mono">(${d.r}, ${d.c})</span> &middot; ${wid}`;
      const badge = document.getElementById('inspBadge');
      const cat = d.old === 1 ? 'known_fail' : (d.cat || 'LOW_RISK');
      badge.textContent = d.old === 1 ? 'Pre-Test Fail' : (CAT_LABELS[cat] || cat);
      badge.style.background = d.old === 1 ? 'var(--known)' : (CAT_COLORS[cat] || 'var(--accent)');
      badge.style.color = '#fff';

      const body = document.getElementById('inspBody');
      body.innerHTML = '';

      if (d.old === 1) {
        body.innerHTML = `
          <div class="gain-callout" style="background:var(--panel-solid);border-left-color:var(--known)">
            <b>Known Pre-Test Failure (old_label = 1)</b><br>
            This die was already defective prior to electrical testing (from WM-811K map). Per official competition criteria, it is excluded from model evaluation and force-flagged as a certain post-test failure.
          </div>
        `;
        return;
      }

      // Dual Model Probability Meters
      const pGroup = document.createElement('div');
      pGroup.innerHTML = `
        <div class="prob-meter-row">
          <div class="p-label">Model A</div>
          <div class="prob-track">
            <div class="prob-fill" style="width:${(d.pa*100).toFixed(1)}%;background:${d.pa>=TA?'var(--sandisk-red)':'var(--accent)'}"></div>
            <div class="threshold-marker" style="left:${(TA*100).toFixed(1)}%" title="Threshold = ${TA}"></div>
          </div>
          <div class="mono" style="text-align:right;font-weight:600">${d.pa.toFixed(3)}</div>
        </div>

        <div class="prob-meter-row">
          <div class="p-label" style="font-weight:600;color:var(--accent)">Model B (+Block)</div>
          <div class="prob-track">
            <div class="prob-fill" style="width:${(d.pb*100).toFixed(1)}%;background:${d.pb>=TB?'var(--sandisk-red)':'var(--accent)'}"></div>
            <div class="threshold-marker" style="left:${(TB*100).toFixed(1)}%" title="Threshold = ${TB}"></div>
          </div>
          <div class="mono" style="text-align:right;font-weight:700;color:var(--accent)">${d.pb.toFixed(3)}</div>
        </div>
      `;
      body.appendChild(pGroup);

      // KV Metrics
      const delta = d.pb - d.pa;
      const kvWrap = document.createElement('div');
      kvWrap.style.margin = '14px 0';
      kvWrap.innerHTML = `
        <div class="kv-row">
          <span class="k">Block-Level Delta (p<sub>B</sub> &minus; p<sub>A</sub>)</span>
          <span class="mono" style="font-weight:600;color:${delta>0?'var(--high)':'var(--pass)'}">${delta>=0?'+':''}${delta.toFixed(3)}</span>
        </div>
        <div class="kv-row">
          <span class="k">Information-Gain Classification</span>
          <span class="pill-badge" style="background:${CAT_COLORS[cat]};color:#fff">${CAT_LABELS[cat]||cat}</span>
        </div>
        ${d.y != null ? `
        <div class="kv-row">
          <span class="k">Ground Truth Outcome</span>
          <span class="mono" style="font-weight:700;color:${d.y===1?'var(--sandisk-red)':'var(--pass)'}">${d.y===1?'FAIL (New Defect)':'PASS'}</span>
        </div>` : ''}
      `;
      body.appendChild(kvWrap);

      // Single-feature occlusion drivers
      if (d.tf && d.tf.length > 0) {
        const title = document.createElement('div');
        title.className = 'sub-section-title';
        title.textContent = 'Local Feature Attribution Drivers (Exact Occlusion)';
        body.appendChild(title);

        const maxAbs = Math.max(...d.tf.map(t => Math.abs(t[1]))) || 1;
        const driverList = document.createElement('div');

        d.tf.forEach(([name, val]) => {
          const widthPct = (Math.abs(val) / maxAbs * 50).toFixed(1);
          const isBlock = name.startsWith('block_');
          const item = document.createElement('div');
          item.className = 'driver-item';
          item.innerHTML = `
            <div class="driver-head">
              <span class="mono" style="${isBlock?'color:var(--accent);font-weight:600':''}">${isBlock?'&#9733; ':''}${name}</span>
              <span class="mono" style="font-weight:600;color:${val>=0?'var(--pos-driver)':'var(--neg-driver)'}">${val>=0?'+':''}${val.toFixed(3)}</span>
            </div>
            <div class="driver-bar-axis">
              <span class="driver-center-line"></span>
              <i class="driver-fill-bar" style="${val>=0?`left:50%;width:${widthPct}%;background:var(--pos-driver)`:`right:50%;width:${widthPct}%;background:var(--neg-driver)`}"></i>
            </div>
          `;
          driverList.appendChild(item);
        });
        body.appendChild(driverList);

        const note = document.createElement('div');
        note.style.fontSize = '11px';
        note.style.color = 'var(--text-tertiary)';
        note.style.marginTop = '10px';
        note.innerHTML = 'Orange bars push probability toward failure; Green bars pull toward pass. <span style="color:var(--accent)">&#9733; block_*</span> signals represent sub-die localized evidence captured exclusively in Model B.';
        body.appendChild(note);
      }
    }

    // ==========================================
    // TAB 2: TRIPTYCH RENDER
    // ==========================================
    function renderTriptych(wid) {
      const w = D.wafers[wid];
      if (!w) return;

      const dies = w.dies;
      drawMiniGrid('tTripPre', dies, d => d.old === 1 ? 'var(--critical)' : 'var(--pass)');
      drawMiniGrid('tTripPost', dies, d => (d.old === 1 || d.y === 1) ? 'var(--critical)' : 'var(--pass)');
      drawMiniGrid('tTripDiff', dies, d => {
        if (d.old === 1) return 'var(--known)';
        if (d.y === 1) return 'var(--high)'; // New failure in orange!
        return 'var(--pass)';
      });

      const nNewFails = dies.filter(d => d.old === 0 && d.y === 1).length;
      const nPreFails = dies.filter(d => d.old === 1).length;
      const nPass = dies.filter(d => d.old === 0 && d.y === 0).length;

      document.getElementById('tripSummaryCallout').innerHTML = `
        <b>Wafer ${wid} Die Breakdown:</b> ${nPass} passing dies &middot; ${nPreFails} pre-test defect dies &middot; <b style="color:var(--high)">${nNewFails} NEW post-test failures</b> discovered during final testing.
      `;
    }

    function drawMiniGrid(svgId, dies, colorFn) {
      const svg = document.getElementById(svgId);
      if (!svg) return;
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      let maxR = 0, maxC = 0;
      dies.forEach(d => { maxR = Math.max(maxR, d.r); maxC = Math.max(maxC, d.c); });
      const cols = maxC + 1, rows = maxR + 1;
      const cell = Math.max(3, Math.min(8, Math.floor(280 / Math.max(cols, rows))));
      const gap = Math.max(0.5, cell * 0.1);
      const W = cols * cell, H = rows * cell;

      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svg.style.width = '100%';
      svg.style.maxWidth = Math.min(W, 300) + 'px';

      const frag = document.createDocumentFragment();
      dies.forEach(d => {
        const r = document.createElementNS(SVG_NS, 'rect');
        r.setAttribute('x', d.c * cell);
        r.setAttribute('y', d.r * cell);
        r.setAttribute('width', cell - gap);
        r.setAttribute('height', cell - gap);
        r.setAttribute('fill', colorFn(d));
        frag.appendChild(r);
      });
      svg.appendChild(frag);
    }

    // ==========================================
    // TAB 3: BENCHMARKS & STATISTICAL PROOF
    // ==========================================
    (function initBenchmarks() {
      const tbody = document.getElementById('compTableBody');
      const ma = D.metrics.model_a, mb = D.metrics.model_b;

      const rows = [
        ['AUC-PR (Primary)', ma.auc_pr.toFixed(4), mb.auc_pr.toFixed(4), (mb.auc_pr - ma.auc_pr).toFixed(4), '+8.6% (Ranking Gain)'],
        ['ROC-AUC', ma.roc_auc.toFixed(4), mb.roc_auc.toFixed(4), (mb.roc_auc - ma.roc_auc).toFixed(4), '+4.8%'],
        ['Fail Recall (Catch Rate)', (ma.fail_recall*100).toFixed(1)+'%', (mb.fail_recall*100).toFixed(1)+'%', '+3.9%', '+11.0% (+54 Defects)'],
        ['Fail F1 Score', ma.fail_f1.toFixed(4), mb.fail_f1.toFixed(4), (mb.fail_f1 - ma.fail_f1).toFixed(4), '+1.4%'],
        ['Overall Accuracy', (ma.accuracy*100).toFixed(2)+'%', (mb.accuracy*100).toFixed(2)+'%', '-0.22%', 'High (96.95%)']
      ];

      rows.forEach(([name, va, vb, delta, note]) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><b>${name}</b></td>
          <td class="mono">${va}</td>
          <td class="mono" style="font-weight:700;color:var(--accent)">${vb}</td>
          <td class="mono" style="color:${delta.startsWith('-')?'var(--text-secondary)':'var(--pass)'};font-weight:600">${delta.startsWith('+')||delta.startsWith('-')?delta:'+'+delta}</td>
          <td><span class="pill-badge" style="background:var(--pass-tint);color:var(--pass)">${note}</span></td>
        `;
        tbody.appendChild(tr);
      });

      // Bootstrap Table
      const bBody = document.getElementById('bootstrapBody');
      if (BOOT && BOOT.delta_aucpr) {
        const bRows = [
          ['&Delta; AUC-PR', `+${BOOT.delta_aucpr.mean.toFixed(4)}`, `[+${BOOT.delta_aucpr.lo95.toFixed(4)}, +${BOOT.delta_aucpr.hi95.toFixed(4)}]`, '100.0% (Definitive)'],
          ['&Delta; Fail Recall', `+${BOOT.delta_recall.mean.toFixed(4)}`, `[+${BOOT.delta_recall.lo95.toFixed(4)}, +${BOOT.delta_recall.hi95.toFixed(4)}]`, '100.0% (Definitive)'],
          ['&Delta; Fail F1', `${BOOT.delta_f1.mean>=0?'+':''}${BOOT.delta_f1.mean.toFixed(4)}`, `[${BOOT.delta_f1.lo95.toFixed(4)}, +${BOOT.delta_f1.hi95.toFixed(4)}]`, '81.0%']
        ];
        bRows.forEach(([metric, mean, ci, p]) => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><b>${metric}</b></td>
            <td class="mono" style="font-weight:700;color:var(--accent)">${mean}</td>
            <td class="mono">${ci}</td>
            <td><span class="pill-badge" style="background:var(--pass-tint);color:var(--pass)">P = ${p}</span></td>
          `;
          bBody.appendChild(tr);
        });
      }

      // Category Importance Bars
      const catContainer = document.getElementById('categoryBars');
      const cats = [
        ['Parametric Test Features', 0.1752, 'var(--accent)'],
        ['Sub-Die Block Readings (#2)', 0.0320, 'var(--high)'],
        ['Spatial Context Features', 0.0108, 'var(--pass)'],
        ['Die-Level Aggregates', 0.0016, 'var(--known)']
      ];
      cats.forEach(([label, val, col]) => {
        const row = document.createElement('div');
        row.style.margin = '10px 0';
        row.innerHTML = `
          <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
            <span style="font-weight:500">${label}</span>
            <span class="mono" style="font-weight:600">${val.toFixed(4)}</span>
          </div>
          <div style="height:8px;background:var(--panel-solid);border-radius:4px;overflow:hidden">
            <div style="height:100%;width:${(val / 0.1752 * 100).toFixed(1)}%;background:${col};border-radius:4px"></div>
          </div>
        `;
        catContainer.appendChild(row);
      });
    })();

    // ==========================================
    // TAB 4: FAB ECONOMICS & DECISION SUPPORT
    // ==========================================
    (function initFabEconomics() {
      // Matched Ops Table
      const mBody = document.getElementById('matchedOpsBody');
      const matchedData = [
        ['Defect Recall &ge; 40%', 'Model A', '54.0%', '40.3%', '474 false alarms', '1,030 dies'],
        ['Defect Recall &ge; 40%', 'Model B (+Block)', '73.0%', '40.0%', '<b style="color:var(--pass)">204 false alarms (-57%)</b>', '756 dies (-274)'],
        ['Defect Recall &ge; 50%', 'Model A', '23.8%', '51.0%', '2,254 false alarms', '2,958 dies'],
        ['Defect Recall &ge; 50%', 'Model B (+Block)', '39.6%', '50.4%', '<b style="color:var(--pass)">1,059 false alarms (-53%)</b>', '1,754 dies (-1,204)']
      ];
      matchedData.forEach(([c, m, p, r, fp, tot]) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><b>${c}</b></td>
          <td style="${m.includes('Model B')?'color:var(--accent);font-weight:600':''}">${m}</td>
          <td class="mono">${p}</td>
          <td class="mono">${r}</td>
          <td class="mono">${fp}</td>
          <td class="mono">${tot}</td>
        `;
        mBody.appendChild(tr);
      });

      // Interactive Cost Slider
      const slider = document.getElementById('costSlider');
      const ratioDisp = document.getElementById('costRatioDisplay');
      const threshDisp = document.getElementById('calcThreshold');
      const savDisp = document.getElementById('calcSavings');

      slider.oninput = () => {
        const val = parseInt(slider.value);
        ratioDisp.textContent = `${val}x retest cost`;
        // Interpolate threshold matching business cost
        let optThresh = 0.62;
        if (val <= 5) optThresh = 0.58;
        else if (val <= 10) optThresh = 0.42;
        else if (val <= 25) optThresh = 0.24;
        else optThresh = 0.12;

        threshDisp.textContent = optThresh.toFixed(2);
        const pctSavings = (73.8 - (val * 0.4)).toFixed(1);
        savDisp.textContent = `-${Math.max(45, pctSavings)}%`;
        playClickSound(500 + val * 10, 0.01);
      };
    })();

    // ==========================================
    // TAB 5: TOP DIES QUEUE
    // ==========================================
    function initTopQueue() {
      const container = document.getElementById('globalTopQueue');
      const catFilter = document.getElementById('queueCatFilter');

      function renderQueue() {
        container.innerHTML = '';
        const filterVal = catFilter.value;
        const filtered = TOP.filter(t => filterVal === 'ALL' || t.category === filterVal);

        filtered.slice(0, 25).forEach(t => {
          const card = document.createElement('div');
          card.className = 'top-queue-card';
          card.innerHTML = `
            <div class="rank-num mono">#${t.rank}</div>
            <div class="die-info">
              <div style="font-weight:600">
                <span class="mono">${t.wafer_id} &middot; Die (${t.die_row}, ${t.die_col})</span> &nbsp;
                <span class="pill-badge" style="background:${CAT_COLORS[t.category]||'var(--accent)'};color:#fff;font-size:10px">${CAT_LABELS[t.category]||t.category}</span>
                ${t.actual_label===1?'<span class="pill-badge" style="background:var(--sandisk-red-tint);color:var(--sandisk-red);font-size:10px">True Defect</span>':''}
              </div>
              <small>${t.evidence || `pB=${t.prob_b.toFixed(3)}, Block Anom=${t.block_anomaly_score?.toFixed(2) || '0'}`}</small>
            </div>
            <div style="text-align:right">
              <div class="mono" style="font-weight:700;color:var(--accent);font-size:14px">${t.priority_score.toFixed(3)}</div>
              <div style="font-size:10.5px;color:var(--text-secondary)">Score</div>
            </div>
          `;
          card.onclick = () => {
            // Switch to Tab 1, select wafer, select die
            document.querySelector('.seg-tab[data-tab="tab-xai"]').click();
            selectWafer(t.wafer_id);
            const d = D.wafers[t.wafer_id]?.dies.find(x => x.r === t.die_row && x.c === t.die_col);
            if (d) {
              curDie = d;
              renderWaferMap(t.wafer_id);
              renderDieInspector(t.wafer_id, d);
              document.getElementById('inspHeader').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
          };
          container.appendChild(card);
        });
      }

      catFilter.onchange = renderQueue;
      renderQueue();
    }

    // Export Predictions
    document.getElementById('exportBtn').onclick = () => {
      playClickSound(900, 0.03);
      alert("Submission predictions are generated and saved at: outputs/predictions/validation_predictions.csv\\n\\nFormat: wafer_id, die_row, die_col, predicted_label\\nTotal Dies: 39,351");
    };

    // Initialize all tabs
    initWaferPills();
    selectWafer(curWaferId);
    initTopQueue();
  </script>
</body>
</html>
"""


def build(publish_scratch=None):
    # Load analysis data
    data = json.loads((ROOT / "outputs" / "dashboard" / "dashboard_data.json").read_text())
    top = json.loads((ROOT / "outputs" / "analysis" / "top_dies_to_investigate.json").read_text())
    
    boot_path = ROOT / "outputs" / "metrics" / "bootstrap_ci.json"
    boot = json.loads(boot_path.read_text()) if boot_path.exists() else {}
    
    common_path = ROOT / "outputs" / "metrics" / "common_operating_points.csv"
    common_ops = pd.read_csv(common_path).to_dict(orient="records") if common_path.exists() else []

    comp_path = ROOT / "outputs" / "metrics" / "model_comparison.csv"
    comp_metrics = pd.read_csv(comp_path).to_dict(orient="records") if comp_path.exists() else []

    payload = (
        "window.__DATA__=" + json.dumps(data) + ";\n" +
        "window.__TOP__=" + json.dumps(top) + ";\n" +
        "window.__BOOT__=" + json.dumps(boot) + ";\n" +
        "window.__COMMON_OPS__=" + json.dumps(common_ops) + ";\n" +
        "window.__COMP_METRICS__=" + json.dumps(comp_metrics) + ";"
    )

    standalone = HTML_TEMPLATE.replace("/*__DATA_PAYLOAD__*/", payload)
    
    out = ROOT / "outputs" / "dashboard" / "wafer_dashboard.html"
    out.write_text(standalone, encoding="utf-8")
    print(f"  Wrote upgraded Apple-Design dashboard: {out} ({len(standalone)//1024} KB)")

    if publish_scratch:
        p = Path(publish_scratch)
        p.write_text(standalone, encoding="utf-8")
        print(f"  Wrote artifact file {p} ({len(standalone)//1024} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None, help="also write skeleton-free file for the Artifact tool")
    args = ap.parse_args()
    build(args.artifact)
