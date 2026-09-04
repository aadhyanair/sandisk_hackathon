"""
Builds the XAI Wafer Intelligence Dashboard (Feature 1) as a self-contained HTML file.

Reads the real analysis outputs (outputs/dashboard/dashboard_data.json and
outputs/analysis/top_dies_to_investigate.json), inlines them, and writes:
  - outputs/dashboard/wafer_dashboard.html   (full standalone document)
  - <scratch>/wafer_dashboard_artifact.html  (skeleton-free, for the Artifact tool)

No network calls, no libraries; the browser can't fetch local files under the artifact
CSP, so all data is embedded inline.
"""
import sys, os, json, argparse
from pathlib import Path

ROOT = Path(os.path.dirname(__file__)).parent

INNER = r"""<title>Wafer Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --bg:#eef1f4; --panel:#ffffff; --panel-2:#f6f8fa; --ink:#131a22; --ink-soft:#4a5563;
  --line:#d7dee6; --line-strong:#c2ccd6; --accent:#0e8f9c; --accent-ink:#0a6b75;
  --low:#2f9e6b; --medium:#c69214; --high:#df6c26; --critical:#cf3646; --known:#788392;
  --pos:#d0603a; --neg:#2f8f9e; --shadow:0 1px 2px rgba(18,26,34,.06),0 8px 24px rgba(18,26,34,.06);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0c1116; --panel:#141b23; --panel-2:#0f151b; --ink:#e6edf4; --ink-soft:#98a4b3;
  --line:#232c37; --line-strong:#2e3947; --accent:#22b5c4; --accent-ink:#7fd6e0;
  --low:#3fb27c; --medium:#d9a52b; --high:#ef7f3c; --critical:#e8556a; --known:#6b7684;
  --pos:#e57a52; --neg:#3fa9ba; --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}}
:root[data-theme="dark"]{
  --bg:#0c1116; --panel:#141b23; --panel-2:#0f151b; --ink:#e6edf4; --ink-soft:#98a4b3;
  --line:#232c37; --line-strong:#2e3947; --accent:#22b5c4; --accent-ink:#7fd6e0;
  --low:#3fb27c; --medium:#d9a52b; --high:#ef7f3c; --critical:#e8556a; --known:#6b7684;
  --pos:#e57a52; --neg:#3fa9ba; --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.5}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1280px;margin:0 auto;padding:22px 22px 40px}
header.top{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:18px}
.brand h1{font-size:26px;font-weight:700;letter-spacing:-.01em;margin:0;text-wrap:balance}
.brand .sub{color:var(--ink-soft);font-size:13.5px;margin-top:3px;max-width:60ch}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent-ink);font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:7px 11px;box-shadow:var(--shadow)}
.chip .k{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-soft)}
.chip .v{font-size:15px;font-weight:600}
.chip .v .up{color:var(--low);font-size:12.5px}
.gainbar{display:flex;height:34px;border-radius:8px;overflow:hidden;border:1px solid var(--line);box-shadow:var(--shadow)}
.gainbar > div{display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:600;min-width:0;white-space:nowrap}
.summary{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:18px;box-shadow:var(--shadow)}
.summary h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-soft);margin:0 0 12px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:11px;font-size:12px;color:var(--ink-soft)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.callout{margin-top:13px;font-size:13.5px;border-left:3px solid var(--accent);padding:4px 0 4px 12px}
.callout b{color:var(--ink)}
.grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:18px}
@media(max-width:940px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}
.panel .hd{padding:13px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:10px}
.panel .hd h3{margin:0;font-size:14px;font-weight:600}
.panel .bd{padding:16px}
.wsel{display:flex;flex-wrap:wrap;gap:8px}
.wbtn{cursor:pointer;border:1px solid var(--line-strong);background:var(--panel-2);border-radius:9px;padding:8px 11px;text-align:left;color:inherit;font:inherit;transition:border-color .12s,background .12s}
.wbtn:hover{border-color:var(--accent)}
.wbtn.active{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 12%,var(--panel))}
.wbtn .wid{font-weight:600;font-size:13px}
.wbtn .wmeta{font-size:11px;color:var(--ink-soft);margin-top:2px}
.maparea{display:flex;justify-content:center;padding:6px 0 2px}
svg.wafermap{max-width:100%;height:auto;touch-action:none}
.maplegend{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;font-size:11.5px;color:var(--ink-soft);margin-top:8px}
.tip{position:fixed;pointer-events:none;z-index:40;background:var(--ink);color:var(--bg);padding:7px 9px;border-radius:7px;font-size:11.5px;box-shadow:0 6px 20px rgba(0,0,0,.3);opacity:0;transition:opacity .08s;max-width:230px}
.tip .mono{color:var(--bg)}
.pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:11px;font-weight:600;letter-spacing:.02em}
.inspector .empty{color:var(--ink-soft);font-size:13.5px;padding:8px 0}
.dieid{font-size:19px;font-weight:600}
.prow{display:grid;grid-template-columns:70px 1fr 62px;align-items:center;gap:10px;margin:9px 0}
.prow .lab{font-size:12px;color:var(--ink-soft)}
.track{height:16px;background:var(--panel-2);border:1px solid var(--line);border-radius:6px;overflow:hidden;position:relative}
.track > i{display:block;height:100%;border-radius:5px 0 0 5px}
.prow .val{font-size:13px;text-align:right}
.thmark{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink);opacity:.55}
.kv{display:flex;justify-content:space-between;gap:12px;font-size:13px;padding:6px 0;border-bottom:1px dashed var(--line)}
.kv .k{color:var(--ink-soft)}
.drv{margin:7px 0}
.drv .dl{display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:3px}
.drv .dl .nm{color:var(--ink)}
.drv .db{position:relative;height:10px;background:var(--panel-2);border-radius:5px}
.drv .db .axis{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--line-strong)}
.drv .db i{position:absolute;top:0;height:100%;border-radius:3px}
.section-title{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-soft);margin:18px 0 8px}
.toplist{display:flex;flex-direction:column;gap:6px}
.topitem{cursor:pointer;display:grid;grid-template-columns:26px 1fr auto;align-items:center;gap:10px;
  border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:var(--panel-2);transition:border-color .12s}
.topitem:hover{border-color:var(--accent)}
.topitem .rk{font-weight:700;color:var(--accent-ink);font-size:13px}
.topitem .who{font-size:12.5px}
.topitem .who small{color:var(--ink-soft)}
.zonelist{display:flex;flex-direction:column;gap:7px;margin-top:4px}
.zone{border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:var(--panel-2)}
.zone .zt{display:flex;justify-content:space-between;font-size:12px}
.zone .zbar{height:6px;border-radius:3px;background:var(--critical);margin-top:6px;opacity:.85}
.trip{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:760px){.trip{grid-template-columns:1fr}}
.trip .cell{display:flex;flex-direction:column;align-items:center;gap:7px}
.trip .cell h4{margin:0;font-size:12.5px;font-weight:600}
.trip .cell .cap{font-size:11px;color:var(--ink-soft)}
.trip svg{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel-2)}
.triplegend{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;font-size:11.5px;color:var(--ink-soft);margin-top:12px}
.story{font-size:12.5px;color:var(--ink-soft);margin:2px 0 14px;text-align:center}
.foot{margin-top:22px;color:var(--ink-soft);font-size:11.5px;text-align:center;line-height:1.7}
.themetoggle{cursor:pointer;border:1px solid var(--line);background:var(--panel);color:var(--ink-soft);
  border-radius:8px;padding:6px 10px;font:inherit;font-size:12px}
button:focus-visible,.wbtn:focus-visible,.topitem:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style>

<div class="wrap">
  <header class="top">
    <div class="brand">
      <div class="eyebrow">Die Yield &middot; Explainable AI</div>
      <h1>Wafer Intelligence</h1>
      <div class="sub">Die-level failure risk with multi-resolution attribution. Compare the die+spatial model (A) against the block-augmented model (B), trace every prediction to its drivers, and navigate wafer &rarr; risk zone &rarr; die.</div>
    </div>
    <div class="chips" id="metricChips"></div>
  </header>

  <div class="summary">
    <h2>Information gain from block-level data &mdash; every eligible die, one category</h2>
    <div class="gainbar" id="gainBar"></div>
    <div class="legend" id="gainLegend"></div>
    <div class="callout" id="gainCallout"></div>
  </div>

  <section class="panel" style="margin-bottom:18px">
    <div class="hd"><h3 id="tripTitle">Pre-Test &rarr; Post-Test &rarr; What changed</h3></div>
    <div class="bd">
      <p class="story">What was known before test &rarr; what actually happened &rarr; which passing dies became <b style="color:var(--high)">new failures</b>.</p>
      <div class="trip">
        <div class="cell"><h4>Pre-Test <span class="cap mono">old_label</span></h4><svg id="tPre" role="img" aria-label="Pre-test wafer"></svg></div>
        <div class="cell"><h4>Post-Test <span class="cap mono">label</span></h4><svg id="tPost" role="img" aria-label="Post-test wafer"></svg></div>
        <div class="cell"><h4>Difference <span class="cap">new fails highlighted</span></h4><svg id="tDiff" role="img" aria-label="Difference wafer"></svg></div>
      </div>
      <div class="triplegend" id="tripLegend"></div>
    </div>
  </section>

  <div class="grid">
    <section class="panel">
      <div class="hd">
        <h3>Wafer risk map</h3>
        <button class="themetoggle" id="themeToggle" type="button">Toggle theme</button>
      </div>
      <div class="bd">
        <div class="wsel" id="waferSel"></div>
        <div class="maparea"><svg class="wafermap" id="wafermap" role="img" aria-label="Wafer die risk map"></svg></div>
        <div class="maplegend" id="mapLegend"></div>
      </div>
    </section>

    <section class="panel inspector">
      <div class="hd"><h3 id="inspTitle">Inspector</h3><span id="inspPattern" class="pill"></span></div>
      <div class="bd" id="inspBody"></div>
    </section>
  </div>

  <div class="foot" id="foot"></div>
</div>
<div class="tip" id="tip"></div>

<script>/*__DATA__*/</script>
<script>
const D = window.__DATA__, TOP = window.__TOP__;
const TA = D.thresholds.model_a, TB = D.thresholds.model_b;
const BAND = {low:'--low',medium:'--medium',high:'--high',critical:'--critical',known_fail:'--known'};
const BANDLAB = {low:'Low',medium:'Medium',high:'High',critical:'Critical',known_fail:'Known fail (pre-test)'};
const CATLAB = {CONFIRMED_RISK:'Confirmed risk',HIDDEN_RISK:'Hidden risk',MODEL_DISAGREEMENT:'Model disagreement',REDUNDANT_INFORMATION:'Redundant info',LOW_RISK:'Low risk'};
const CATCOL = {CONFIRMED_RISK:'--critical',HIDDEN_RISK:'--high',MODEL_DISAGREEMENT:'--medium',REDUNDANT_INFORMATION:'--accent',LOW_RISK:'--low'};
const cvar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el = (t,c)=>{const e=document.createElement(t); if(c)e.className=c; return e;};
let curWafer=null, curDie=null;

/* ---------- metric chips ---------- */
(function(){
  const ma=D.metrics.model_a, mb=D.metrics.model_b, ig=D.info_gain_summary;
  const box=document.getElementById('metricChips');
  const chips=[
    ['AUC-PR (B)', mb.auc_pr.toFixed(3), '+'+(ig.auc_pr_gain).toFixed(3)+' vs A'],
    ['Fail recall (B)', mb.fail_recall.toFixed(3), '+'+(ig.fail_recall_gain).toFixed(3)+' vs A'],
    ['Hidden failures caught', ''+ig.hidden_risk_true_failures, 'missed by A'],
  ];
  chips.forEach(([k,v,up])=>{
    const c=el('div','chip'); c.innerHTML=`<div class="k">${k}</div><div class="v mono">${v} <span class="up">${up}</span></div>`;
    box.appendChild(c);
  });
})();

/* ---------- info-gain summary bar ---------- */
(function(){
  const ig=D.info_gain_summary, order=['CONFIRMED_RISK','HIDDEN_RISK','MODEL_DISAGREEMENT','REDUNDANT_INFORMATION','LOW_RISK'];
  const total=Object.values(ig.counts).reduce((a,b)=>a+b,0);
  const bar=document.getElementById('gainBar'), leg=document.getElementById('gainLegend');
  order.forEach(c=>{
    const n=ig.counts[c], pct=100*n/total;
    const d=el('div'); d.style.width=pct+'%'; d.style.background=cvar(CATCOL[c]);
    if(pct>7) d.textContent=CATLAB[c]+' · '+n;
    d.title=CATLAB[c]+': '+n+' ('+pct.toFixed(1)+'%)';
    bar.appendChild(d);
    const s=el('span'); s.innerHTML=`<i class="sw" style="background:${cvar(CATCOL[c])}"></i>${CATLAB[c]} <b class="mono">${(100*n/total).toFixed(1)}%</b>`;
    leg.appendChild(s);
  });
  const o=ig.per_category_outcome||{};
  const conf=o.CONFIRMED_RISK, hid=o.HIDDEN_RISK;
  document.getElementById('gainCallout').innerHTML =
    `Block-level data confirmed <b>${conf?conf.count:0}</b> high-risk dies that truly fail <b class="mono">${conf?(conf.actual_fail_rate*100).toFixed(0):0}%</b> of the time, and `+
    `<b>revealed ${ig.hidden_risk_true_failures}</b> genuine failures the die+spatial model alone would have passed &mdash; the <b>hidden-risk</b> dies whose danger only the sub-die signal exposes.`;
})();

/* ---------- wafer selector ---------- */
(function(){
  const sel=document.getElementById('waferSel');
  Object.keys(D.wafers).forEach(wid=>{
    const w=D.wafers[wid];
    const nCrit=w.dies.filter(d=>d.band==='critical'||d.band==='high').length;
    const b=el('button','wbtn'); b.type='button'; b.dataset.wid=wid;
    b.innerHTML=`<div class="wid mono">${wid}</div><div class="wmeta">${w.pattern} pattern · ${w.zones.length} zones · ${nCrit} high-risk</div>`;
    b.onclick=()=>selectWafer(wid);
    sel.appendChild(b);
  });
})();

/* ---------- map legend ---------- */
(function(){
  const ml=document.getElementById('mapLegend');
  ['critical','high','medium','low','known_fail'].forEach(k=>{
    const s=el('span'); s.innerHTML=`<i class="sw" style="background:${cvar(BAND[k])}"></i>${BANDLAB[k]}`;
    s.style.display='inline-flex'; s.style.alignItems='center'; s.style.gap='6px'; ml.appendChild(s);
  });
})();

/* ---------- wafer map render ---------- */
const SVGNS='http://www.w3.org/2000/svg';
function selectWafer(wid){
  curWafer=wid; curDie=null;
  document.querySelectorAll('.wbtn').forEach(b=>b.classList.toggle('active',b.dataset.wid===wid));
  drawTriptych(wid); drawMap(wid); renderWaferInspector(wid);
}
const TCOL={pass:'--low',fail:'--critical',newfail:'--high'};
function drawMini(svg,dies,colorFn){
  let maxR=0,maxC=0; dies.forEach(d=>{maxR=Math.max(maxR,d.r);maxC=Math.max(maxC,d.c);});
  const cols=maxC+1,rows=maxR+1,cell=Math.max(3,Math.min(9,Math.floor(300/Math.max(cols,rows))));
  const gap=Math.max(0.5,cell*0.1),W=cols*cell,H=rows*cell;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.style.width=Math.min(W,300)+'px';
  while(svg.firstChild) svg.removeChild(svg.firstChild);
  const frag=document.createDocumentFragment();
  dies.forEach(d=>{
    const r=document.createElementNS(SVGNS,'rect');
    r.setAttribute('x',d.c*cell); r.setAttribute('y',d.r*cell);
    r.setAttribute('width',cell-gap); r.setAttribute('height',cell-gap);
    r.setAttribute('fill',cvar(colorFn(d)));
    frag.appendChild(r);
  });
  svg.appendChild(frag);
}
function postState(d){ return (d.old===1)?'fail':((d.y===1)?'fail':'pass'); }
function drawTriptych(wid){
  const w=D.wafers[wid], dies=w.dies;
  document.getElementById('tripTitle').textContent=wid+': Pre-Test → Post-Test → What changed';
  drawMini(document.getElementById('tPre'), dies, d=>TCOL[d.old===1?'fail':'pass']);
  drawMini(document.getElementById('tPost'), dies, d=>TCOL[postState(d)]);
  drawMini(document.getElementById('tDiff'), dies, d=>{
    if(d.old===1) return TCOL.fail;
    if(d.y===1) return TCOL.newfail;   // new failure: was passing, now failed
    return TCOL.pass;
  });
  const nNew=dies.filter(d=>d.old===0&&d.y===1).length;
  const leg=document.getElementById('tripLegend'); leg.innerHTML='';
  [['Passing','--low'],['Pre-existing fail','--critical'],['NEW failure ('+nNew+')','--high']].forEach(([t,c])=>{
    const s=el('span'); s.style.display='inline-flex'; s.style.alignItems='center'; s.style.gap='6px';
    s.innerHTML=`<i class="sw" style="background:${cvar(c)}"></i>${t}`; leg.appendChild(s);
  });
}
function drawMap(wid){
  const w=D.wafers[wid], dies=w.dies;
  let maxR=0,maxC=0; dies.forEach(d=>{maxR=Math.max(maxR,d.r);maxC=Math.max(maxC,d.c);});
  const cols=maxC+1, rows=maxR+1;
  const cell=Math.max(6,Math.min(15,Math.floor(560/Math.max(cols,rows))));
  const gap=Math.max(1,Math.floor(cell*0.12));
  const W=cols*cell, H=rows*cell;
  const svg=document.getElementById('wafermap');
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.style.maxWidth=Math.min(W,600)+'px';
  while(svg.firstChild) svg.removeChild(svg.firstChild);
  const zoneSet=new Set();
  const frag=document.createDocumentFragment();
  dies.forEach(d=>{
    const r=document.createElementNS(SVGNS,'rect');
    r.setAttribute('x',d.c*cell); r.setAttribute('y',d.r*cell);
    r.setAttribute('width',cell-gap); r.setAttribute('height',cell-gap);
    r.setAttribute('rx',Math.max(1,cell*0.14));
    r.setAttribute('fill',cvar(BAND[d.band]));
    if(d.old===1) r.setAttribute('opacity','0.45');
    r.style.cursor='pointer';
    r.addEventListener('mousemove',e=>showTip(e,d));
    r.addEventListener('mouseleave',hideTip);
    r.addEventListener('click',()=>{curDie=d; drawMap(wid); renderDie(wid,d);});
    if(curDie && curDie.r===d.r && curDie.c===d.c){
      r.setAttribute('stroke',cvar('--ink')); r.setAttribute('stroke-width',Math.max(1.5,cell*0.18));
    }
    frag.appendChild(r);
  });
  svg.appendChild(frag);
}
const tip=document.getElementById('tip');
function showTip(e,d){
  const cat=d.cat?CATLAB[d.cat]||d.cat:'';
  tip.innerHTML=`<span class="mono">(${d.r},${d.c})</span> &middot; ${BANDLAB[d.band]}<br>`+
    (d.old===1?'<b>Pre-test failure</b>':`p<sub>A</sub> <span class="mono">${d.pa.toFixed(2)}</span> &nbsp; p<sub>B</sub> <span class="mono">${d.pb.toFixed(2)}</span><br>${cat}`);
  tip.style.opacity='1';
  const pad=14; let x=e.clientX+pad, y=e.clientY+pad;
  if(x+240>innerWidth) x=e.clientX-240; if(y+80>innerHeight) y=e.clientY-80;
  tip.style.left=x+'px'; tip.style.top=y+'px';
}
function hideTip(){tip.style.opacity='0';}

/* ---------- inspector: wafer level ---------- */
function renderWaferInspector(wid){
  const w=D.wafers[wid];
  document.getElementById('inspTitle').textContent=wid+' overview';
  const pp=document.getElementById('inspPattern');
  pp.textContent=w.pattern+' pattern';
  pp.style.background='color-mix(in srgb,'+cvar('--accent')+' 16%,transparent)';
  pp.style.color=cvar('--accent-ink');
  const b=document.getElementById('inspBody'); b.innerHTML='';
  b.innerHTML=`<div class="inspector"><p class="empty">Click any die on the map for its full explanation. This wafer shows a <b>${w.pattern}</b> spatial failure pattern with <b>${w.zones.length}</b> detected high-risk zone(s).</p></div>`;

  if(w.zones.length){
    const st=el('div','section-title'); st.textContent='High-risk zones (size × mean risk)'; b.appendChild(st);
    const zl=el('div','zonelist'); const maxSev=Math.max(...w.zones.map(z=>z.severity));
    w.zones.slice(0,6).forEach(z=>{
      const zd=el('div','zone');
      const hid=z.hidden_risk_fraction!=null?` · ${(z.hidden_risk_fraction*100).toFixed(0)}% hidden-risk`:'';
      zd.innerHTML=`<div class="zt"><span class="mono">${z.size} dies @ (${z.centroid_row.toFixed(0)},${z.centroid_col.toFixed(0)})</span><span class="mono">sev ${z.severity.toFixed(2)}</span></div>`+
        `<div class="zbar" style="width:${(100*z.severity/maxSev).toFixed(0)}%"></div>`+
        `<div class="zt" style="color:var(--ink-soft);margin-top:5px"><span>mean risk ${z.mean_risk.toFixed(2)} · max ${z.max_risk.toFixed(2)}${hid}</span></div>`;
      zl.appendChild(zd);
    });
    b.appendChild(zl);
  }
  const st2=el('div','section-title'); st2.textContent='Top dies to investigate (this wafer)'; b.appendChild(st2);
  const items=TOP.filter(t=>t.wafer_id===wid).slice(0,5);
  if(!items.length){b.appendChild(Object.assign(el('p','empty'),{textContent:'No dies from this wafer in the global top list.'}));}
  else b.appendChild(topList(items));
}

/* ---------- inspector: single die ---------- */
function renderDie(wid,d){
  document.getElementById('inspTitle').innerHTML='Die <span class="mono">('+d.r+','+d.c+')</span> — '+wid;
  const pp=document.getElementById('inspPattern');
  const cat=d.old===1?'known_fail':d.cat;
  pp.textContent=d.old===1?'Pre-test fail':(CATLAB[cat]||cat||'');
  pp.style.background='color-mix(in srgb,'+cvar(d.old===1?'--known':CATCOL[cat]||'--accent')+' 18%,transparent)';
  pp.style.color=cvar('--ink');
  const b=document.getElementById('inspBody'); b.innerHTML='';
  if(d.old===1){
    b.innerHTML=`<p class="empty">This die was already failing at pre-test (old_label = 1). It is excluded from model scoring and reported as a certain failure downstream.</p>`;
    return;
  }
  // probability bars
  b.appendChild(probRow('Model A', d.pa, TA, '--neg'));
  b.appendChild(probRow('Model B', d.pb, TB, '--critical'));
  const delta=d.pb-d.pa;
  const kv1=el('div','kv'); kv1.innerHTML=`<span class="k">Block-driven change (p<sub>B</sub> − p<sub>A</sub>)</span><span class="mono">${delta>=0?'+':''}${delta.toFixed(3)}</span>`; b.appendChild(kv1);
  const kv2=el('div','kv'); kv2.innerHTML=`<span class="k">Information-gain category</span><span class="pill" style="background:color-mix(in srgb,${cvar(CATCOL[cat]||'--accent')} 18%,transparent)">${CATLAB[cat]||cat}</span>`; b.appendChild(kv2);
  if(d.y!=null){const kv3=el('div','kv'); kv3.innerHTML=`<span class="k">Actual post-test outcome</span><span class="mono">${d.y===1?'FAIL':'pass'}</span>`; b.appendChild(kv3);}
  // local drivers
  if(d.tf&&d.tf.length){
    const st=el('div','section-title'); st.textContent='Local drivers (single-feature occlusion attribution)'; b.appendChild(st);
    const mx=Math.max(...d.tf.map(t=>Math.abs(t[1])))||1;
    d.tf.forEach(([nm,c])=>{
      const w=Math.abs(c)/mx*50;
      const drv=el('div','drv');
      drv.innerHTML=`<div class="dl"><span class="nm mono">${nm}</span><span class="mono">${c>=0?'+':''}${c.toFixed(3)}</span></div>`+
        `<div class="db"><span class="axis"></span><i style="${c>=0?`left:50%;width:${w}%;background:${cvar('--pos')}`:`right:50%;width:${w}%;background:${cvar('--neg')}`}"></i></div>`;
      b.appendChild(drv);
    });
    const note=el('p','empty'); note.style.fontSize='11.5px'; note.style.marginTop='8px';
    note.innerHTML='Positive (orange) drivers push failure probability up vs a typical passing die; negative (teal) pull it down. <span class="mono">block_*</span> features are the sub-die evidence unique to Model B.';
    b.appendChild(note);
  }
}
function probRow(lab,val,th,col){
  const row=el('div','prow');
  const flagged=val>=th;
  row.innerHTML=`<div class="lab">${lab}</div>`+
    `<div class="track"><i style="width:${(val*100).toFixed(1)}%;background:${cvar(flagged?col:'--accent')}"></i><span class="thmark" style="left:${(th*100).toFixed(1)}%"></span></div>`+
    `<div class="val mono">${val.toFixed(3)}</div>`;
  return row;
}
function topList(items){
  const box=el('div','toplist');
  items.forEach(t=>{
    const it=el('div','topitem'); it.tabIndex=0; it.setAttribute('role','button');
    const lab=t.actual_label==null?'':`<small>actual ${t.actual_label===1?'FAIL':'pass'}</small>`;
    it.innerHTML=`<div class="rk mono">#${t.rank}</div>`+
      `<div class="who"><span class="mono">${t.wafer_id} (${t.die_row},${t.die_col})</span> &middot; ${CATLAB[t.category]||t.category} ${lab}</div>`+
      `<div class="mono" style="font-size:12px;color:var(--accent-ink)">${t.priority_score.toFixed(3)}</div>`;
    const jump=()=>{ if(D.wafers[t.wafer_id]){ selectWafer(t.wafer_id);
        const d=D.wafers[t.wafer_id].dies.find(x=>x.r===t.die_row&&x.c===t.die_col);
        if(d){curDie=d; drawMap(t.wafer_id); renderDie(t.wafer_id,d); document.querySelector('.grid').scrollIntoView({behavior:'smooth',block:'start'});}}};
    it.onclick=jump; it.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();jump();}};
    box.appendChild(it);
  });
  return box;
}

/* ---------- global top list in footer ---------- */
document.getElementById('foot').innerHTML =
  'Model B failure probabilities on held-out test wafers, eligible dies (old_label = 0). '+
  'Local drivers are exact single-feature occlusion contributions against a typical passing die. '+
  'Risk bands: critical ≥ 0.50, high ≥ model-B threshold ('+TB.toFixed(2)+'), medium ≥ 0.15, else low.';

/* ---------- theme toggle ---------- */
document.getElementById('themeToggle').onclick=()=>{
  const r=document.documentElement;
  const cur=r.getAttribute('data-theme');
  const sysDark=matchMedia('(prefers-color-scheme:dark)').matches;
  const next=cur? (cur==='dark'?'light':'dark') : (sysDark?'light':'dark');
  r.setAttribute('data-theme',next);
  if(curWafer){drawTriptych(curWafer); drawMap(curWafer); if(curDie) renderDie(curWafer,curDie); else renderWaferInspector(curWafer);}
  // redraw summary swatches
};

/* init */
selectWafer(Object.keys(D.wafers)[0]);
</script>"""


def build(publish_scratch=None):
    data = json.loads((ROOT / "outputs" / "dashboard" / "dashboard_data.json").read_text())
    top = json.loads((ROOT / "outputs" / "analysis" / "top_dies_to_investigate.json").read_text())
    payload = "window.__DATA__=" + json.dumps(data) + ";window.__TOP__=" + json.dumps(top) + ";"
    inner = INNER.replace("/*__DATA__*/", payload)

    standalone = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                  "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
                  "<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>"
                  "</head><body>" + inner + "</body></html>")
    out = ROOT / "outputs" / "dashboard" / "wafer_dashboard.html"
    out.write_text(standalone, encoding="utf-8")
    print(f"  Wrote {out} ({len(standalone)//1024} KB)")

    if publish_scratch:
        p = Path(publish_scratch)
        p.write_text(inner, encoding="utf-8")
        print(f"  Wrote artifact file {p} ({len(inner)//1024} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None, help="also write skeleton-free file for the Artifact tool")
    args = ap.parse_args()
    build(args.artifact)
