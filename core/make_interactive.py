# -*- coding: utf-8 -*-
"""Build one self-contained interactive HTML with all 5 papers.

Contents:
  - gap-to-target chart (x = percent gap, y = paper bands, color = method, threshold input)
  - dynamic best-so-far overview (5 panels, top-n / focus)
  - per paper: interactive d-vs-attempt + interactive best-so-far (shared top-n / focus)

Everything is inlined (plotly.js + data).  No method-stats table.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
EXP4 = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP4)
for _p in (ROOT, EXP4, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FINAL = os.path.join(EXP4, "results", "final")
OUT = os.path.join(FINAL, "interactive")
BEST = pd.read_csv(os.path.join(FINAL, "runs_best_recomputed.csv"))

TITLES = {
    "bigoni": "Bigoni 2015 (Econometrica)",
    "marshall": "Marshall 2015 (BJPS)",
    "anderson": "Anderson 2011 (AEJ Applied)",
    "jiang": "Jiang 2018 (AJPS)",
    "andreoni": "Andreoni 2017 (JPE)",
}
SHORT = {"bigoni": "Bigoni", "marshall": "Marshall", "anderson": "Anderson",
         "jiang": "Jiang", "andreoni": "Andreoni"}
PAPERS = ["bigoni", "marshall", "anderson", "jiang", "andreoni"]

METHOD_ORDER = ["RandomUniform", "Random", "COMBO", "CASMOPOLITAN", "BoTorch",
                "HEBO", "Bounce", "Summit", "bayes_optim", "MerCBO", "BODi",
                "bo_pr", "BayBE", "Ax", "GP-EI", "GP-UCB", "GP-TS", "GP-PI"]
METHODS = [m for m in METHOD_ORDER if m in set(BEST.algo)]

COLORS = {
    "RandomUniform": "#95a5a6", "Random": "#7f8c8d", "COMBO": "#2471a3",
    "BoTorch": "#8e44ad", "GP-EI": "#c0392b", "GP-UCB": "#e67e22",
    "GP-TS": "#27ae60", "GP-PI": "#d35400", "HEBO": "#16a085",
    "BayBE": "#7d3c98", "Ax": "#b03a2e", "CASMOPOLITAN": "#2c3e50",
    "Bounce": "#e84393", "Summit": "#00b894", "bayes_optim": "#6c5ce7",
    "MerCBO": "#0984e3", "BODi": "#a0522d", "bo_pr": "#fdcb6e",
}
DARK24 = ["#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF", "#222A2A",
          "#B68100", "#750D86", "#EB663B", "#511CFB", "#00A08B", "#FB00D1",
          "#FC0080", "#B2828D", "#6C7C32", "#778AAE", "#862A16", "#A777F1",
          "#620042", "#1616A7", "#DA60CA", "#6C4516", "#0D2A63", "#AF0038"]
YLIM = {"bigoni": (4.5, 8.5), "marshall": (-0.1, 1.15), "anderson": (0.0, 1.0),
        "jiang": (0.0, 0.9), "andreoni": (0.0, 3.0)}

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
       margin: 0; background: #fafafa; color: #222; }
.wrap { max-width: 1280px; margin: 0 auto; padding: 22px 26px 40px; }
h1 { font-size: 22px; margin: 0 0 6px; }
h2 { font-size: 18px; margin: 0 0 4px; }
h3 { font-size: 15px; margin: 20px 0 4px; }
.sub { color: #666; font-size: 13px; margin: 0 0 12px; line-height: 1.5; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center;
             background: #fff; border: 1px solid #e3e3e3; border-radius: 8px;
             padding: 11px 14px; margin: 8px 0 8px; }
.controls label { font-size: 13px; color: #333; }
.controls input[type=number] { width: 64px; padding: 4px 6px; font-size: 14px;
                                border: 1px solid #bbb; border-radius: 5px; }
.controls input[type=range] { width: 260px; vertical-align: middle; accent-color: #c0392b; }
.thr-wrap { display: inline-flex; flex-direction: column; gap: 2px; vertical-align: middle; }
.thr-label { font-weight: 700; color: #c0392b; font-size: 13px; margin-left: 6px; }
.thr-ticks { display: flex; justify-content: space-between; width: 260px; font-size: 10px;
             color: #999; line-height: 1.1; }
.controls select { padding: 5px 7px; font-size: 13px; border: 1px solid #bbb;
                     border-radius: 5px; background: #fff; }
.controls button { padding: 5px 12px; font-size: 13px; border: 1px solid #888;
                     background: #fff; border-radius: 5px; cursor: pointer; }
.controls button:hover { background: #f0f0f0; }
.status { font-size: 12px; color: #777; }
.hint { font-size: 12px; color: #777; margin: 0 0 8px; }
.paper { border-top: 1px solid #e1e1e1; margin-top: 26px; padding-top: 18px;
          scroll-margin-top: 10px; }
nav { margin: 12px 0 18px; font-size: 13px; }
nav a { color: #1a5fb4; margin-right: 14px; text-decoration: none; }
nav a:hover { text-decoration: underline; }
.cohen { background: #fff; border: 1px solid #e3e3e3; border-radius: 8px;
          padding: 12px 16px; margin: 14px 0 6px; font-size: 13px; line-height: 1.7; }
.gap-summary { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 4px; }
.gap-card { background: #fff; border: 1px solid #e3e3e3; border-radius: 7px;
            padding: 8px 10px; font-size: 12px; line-height: 1.55; min-width: 210px; flex: 1; }
.gap-card b { color: #111; }
.gap-paper { font-weight: 700; font-size: 13px; margin-bottom: 2px; }
.note { font-size: 12px; color: #777; line-height: 1.6; margin: 10px 0 0; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }
"""

COHEN_HTML = """
<p class="cohen">Cohen's d: all methods share the same evaluator. The evaluator takes the treatment t from the path's estimator (df = G−1 for cluster, df = n−rank for classical/HC), d = 2t / sqrt(df). The absolute value of d measures effect strength, d_paper is the target effect size; the best-so-far chart plots the historical minimum of |d_hat − d_paper|.</p>
"""


def color_for(algo: str, idx: int) -> str:
    return COLORS.get(algo, DARK24[idx % len(DARK24)])


def load_paper(paper: str):
    rows = BEST[BEST.paper == paper].copy()
    rows = rows.sort_values(["best_loss", "q2t_1"], na_position="last").reset_index(drop=True)
    target = float(rows.target_d.iloc[0])
    tol = float(rows.tol.iloc[0])
    cap = max(1.0, abs(target) * 2.0, tol * 100.0)
    series = []
    max_attempt = 0
    for i, r in rows.iterrows():
        sd = os.path.join(FINAL, paper, r.algo, f"seed{int(r.seed)}")
        df = pd.read_csv(os.path.join(sd, "trace.csv")).sort_values("attempt").reset_index(drop=True)
        x = df["attempt"].astype(int).tolist()
        y = [None if pd.isna(v) else float(v) for v in df["d"]]
        loss = df["loss"].astype(float).to_numpy()
        best = df["best_loss"].astype(float).to_numpy()
        if np.isnan(best).any():
            best = np.minimum.accumulate(np.where(loss < 1000, loss, np.inf))
            best[~np.isfinite(best)] = np.nan
        phase = df["phase"].astype(str).tolist()
        cd_raw, cd_best = [], []
        path = []
        feas = 0
        for j in range(len(df)):
            if not pd.isna(y[j]):
                feas += 1
                gap_j = 100.0 * abs(float(y[j]) - target) / max(abs(target), 0.5)
                path.append([float(gap_j), int(feas)])
            b = None if pd.isna(best[j]) else min(max(float(best[j]), 1e-8), cap)
            cd_raw.append([int(x[j]), None if pd.isna(y[j]) else float(y[j]),
                           float(loss[j]) if np.isfinite(loss[j]) else None,
                           b, phase[j]])
            cd_best.append([int(x[j]), b, phase[j]])
        series.append(dict(
            algo=str(r.algo), color=color_for(r.algo, i), x=x, y=y, best=best,
            cd_raw=cd_raw, cd_best=cd_best, cap=cap, path=path,
        ))
        max_attempt = max(max_attempt, int(max(x)) if x else 0)
    return dict(paper=paper, title=TITLES[paper], short=SHORT[paper],
                target=target, tol=tol, cap=cap, rows=rows, series=series,
                max_attempt=max_attempt, orders=list(range(len(rows))))


def make_trace(s, kind="raw", showlegend=False):
    if kind == "raw":
        y = s["y"]
        cd = s["cd_raw"]
        hover = ("<b>%{fullData.name}</b><br>attempt %{customdata[0]}<br>"
                 "d = %{y:.4f}<br>|d-target| = %{customdata[2]:.5f}<br>"
                 "best |d-target| = %{customdata[3]:.5f}<br>phase = %{customdata[4]}"
                 "<extra></extra>")
    else:
        y = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else
             min(max(float(v), 1e-8), s["cap"]) for v in s["best"]]
        cd = s["cd_best"]
        hover = ("<b>%{fullData.name}</b><br>attempt %{customdata[0]}<br>"
                 "best |d-target| = %{customdata[1]:.6f}<br>phase = %{customdata[2]}"
                 "<extra></extra>")
    return go.Scatter(
        x=s["x"], y=y, mode="lines", name=s["algo"],
        line=dict(color=s["color"], width=1.6), opacity=0.85,
        customdata=cd, hovertemplate=hover, showlegend=showlegend,
    )


def base_layout(p, div_id, height, showlegend, ylog=False):
    return dict(
        template="plotly_white", height=height,
        margin=dict(l=70, r=25, t=20, b=150 if showlegend else 50),
        xaxis=dict(title="attempt (warm-up 0–19, then proposals; rejected also consume an attempt)",
                   range=[-2, p["max_attempt"] + 5], showgrid=True, zeroline=False),
        yaxis=dict(title="best-so-far |d−d_paper| (log)" if ylog else "Cohen's d",
                   type="log" if ylog else "linear",
                   range=None if ylog else list(YLIM.get(p["paper"], (0, 1))),
                   showgrid=True, zeroline=False),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0,
                    font=dict(size=10), itemwidth=30),
    )


def build_paper_section(p):
    raw_fig = go.Figure([make_trace(s, "raw", True) for s in p["series"]])
    raw_fig.update_layout(**base_layout(p, f"chart_d_{p['paper']}", 560, True, False))
    raw_html = raw_fig.to_html(full_html=False, include_plotlyjs=False,
                               div_id=f"chart_d_{p['paper']}",
                               config={"responsive": True, "displaylogo": False,
                                       "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    best_fig = go.Figure([make_trace(s, "best", False) for s in p["series"]])
    best_fig.update_layout(**base_layout(p, f"chart_best_{p['paper']}", 430, False, True))
    best_fig.update_yaxes(range=[np.log10(max(1e-6, p["tol"] / 20)), np.log10(p["cap"])],
                          tickformat=".1e")
    best_fig.add_hline(y=p["tol"], line_dash="dot", line_color="black", line_width=1)
    best_html = best_fig.to_html(full_html=False, include_plotlyjs=False,
                                 div_id=f"chart_best_{p['paper']}",
                                 config={"responsive": True, "displaylogo": False,
                                         "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    tid = f"topn_{p['paper']}"
    fid = f"focus_{p['paper']}"
    bid = f"reset_{p['paper']}"
    sid = f"status_{p['paper']}"
    default_n = min(5, len(p["series"]))
    options = '<option value="">All</option>' + "".join(
        f'<option value="{i}">{p["series"][i]["algo"]}</option>' for i in range(len(p["series"])))

    js = f"""
(function() {{
  var gdRaw = document.getElementById('chart_d_{p['paper']}');
  var gdBest = document.getElementById('chart_best_{p['paper']}');
  var ORDER = {json.dumps(p['orders'])};
  var DEFAULT_N = {default_n};
  function apply() {{
    var maxN = gdRaw.data.length;
    var n = parseInt(document.getElementById('{tid}').value, 10);
    if (!isFinite(n)) n = DEFAULT_N;
    n = Math.max(1, Math.min(maxN, n));
    document.getElementById('{tid}').value = n;
    var focus = document.getElementById('{fid}').value;
    var vis;
    if (focus !== '') {{
      vis = gdRaw.data.map(function(_, i) {{ return String(i) === focus; }});
    }} else {{
      var top = ORDER.slice(0, n);
      vis = gdRaw.data.map(function(_, i) {{ return top.indexOf(i) >= 0; }});
    }}
    Plotly.restyle(gdRaw, {{visible: vis}});
    Plotly.restyle(gdBest, {{visible: vis}});
    document.getElementById('{sid}').textContent =
      'Showing ' + vis.filter(Boolean).length + ' / ' + maxN + ' (both charts synced)';
  }}
  document.getElementById('{tid}').addEventListener('input', apply);
  document.getElementById('{fid}').addEventListener('change', apply);
  document.getElementById('{bid}').addEventListener('click', function() {{
    document.getElementById('{tid}').value = DEFAULT_N;
    document.getElementById('{fid}').value = '';
    apply();
  }});
  apply();
}})();
"""

    section = f"""
<section class="paper" id="{p['paper']}">
  <h2>{p['title']}</h2>
  <p class="sub">target d = <b>{p['target']:.4f}</b>; tol = <b>{p['tol']:.6f}</b> (about 0.1%);
  each line is that method's best seed; the top chart is d vs attempt, the bottom one is best-so-far. The x-axis includes the 20 warm-up points and rejected proposals.</p>
  <div class="controls">
    <label>Show top <input id="{tid}" type="number" min="1" max="{len(p['series'])}" value="{default_n}"> lines</label>
    <label>Focus <select id="{fid}">{options}</select></label>
    <button id="{bid}">Reset</button>
    <span class="status" id="{sid}"></span>
  </div>
  <p class="hint">Sorted by best |d−target|; the controls act on both the top chart (d) and the bottom one (best-so-far).
  Click the legend to hide/show a single line, double-click to isolate it.</p>
  {raw_html}
  <h3>Best-so-far (filterable, synced with the top chart)</h3>
  {best_html}
</section>
"""
    return section, js


def build_overview(data):
    fig = make_subplots(rows=2, cols=3, subplot_titles=[d["title"] for d in data],
                        horizontal_spacing=0.07, vertical_spacing=0.16)
    trace_panel = []
    orders = []
    names = []
    for pi, p in enumerate(data):
        row, col = pi // 3 + 1, pi % 3 + 1
        for s in p["series"]:
            tr = make_trace(s, "best", False)
            fig.add_trace(tr, row=row, col=col)
            trace_panel.append(pi)
            names.append(s["algo"])
        orders.append(list(range(len(trace_panel) - len(p["series"]), len(trace_panel))))
        fig.add_hline(y=p["tol"], line_dash="dot", line_color="black", line_width=1,
                      row=row, col=col)
        fig.update_yaxes(type="log", row=row, col=col,
                         range=[np.log10(max(1e-6, p["tol"] / 20)), np.log10(p["cap"])],
                         tickformat=".1e")
        fig.update_xaxes(range=[-2, p["max_attempt"] + 5], row=row, col=col)
    fig.update_layout(template="plotly_white", height=620,
                      margin=dict(l=70, r=25, t=50, b=50), showlegend=False,
                      hovermode="closest")
    for r in (1, 2):
        fig.update_xaxes(title_text="attempt", row=r, col=1)
    for c in (1, 2, 3):
        fig.update_yaxes(title_text="best |d−target| (log)", row=2, col=c)
    plot_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart_overview",
                            config={"responsive": True, "displaylogo": False,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    options = '<option value="">All</option>' + "".join(
        f'<option value="{m}">{m}</option>' for m in METHODS)
    js = f"""
(function() {{
  var gd = document.getElementById('chart_overview');
  var PANEL = {json.dumps(trace_panel)};
  var ORDERS = {json.dumps(orders)};
  var NAMES = {json.dumps(names)};
  var DEFAULT_N = 3;
  function apply() {{
    var n = parseInt(document.getElementById('topn_overview').value, 10);
    if (!isFinite(n)) n = DEFAULT_N;
    n = Math.max(1, Math.min(18, n));
    document.getElementById('topn_overview').value = n;
    var focus = document.getElementById('focus_overview').value;
    var vis = NAMES.map(function(name, i) {{
      if (focus !== '') return name === focus;
      return ORDERS[PANEL[i]].slice(0, n).indexOf(i) >= 0;
    }});
    Plotly.restyle(gd, {{visible: vis}});
    document.getElementById('status_overview').textContent =
      'Each panel shows ' + n + '; total ' + vis.filter(Boolean).length + ' / ' + NAMES.length + ' lines';
  }}
  document.getElementById('topn_overview').addEventListener('input', apply);
  document.getElementById('focus_overview').addEventListener('change', apply);
  document.getElementById('reset_overview').addEventListener('click', function() {{
    document.getElementById('topn_overview').value = DEFAULT_N;
    document.getElementById('focus_overview').value = '';
    apply();
  }});
  apply();
}})();
"""
    section = f"""
<section class="paper" id="overview">
  <h2>Best-so-far overview (5 papers, dynamic)</h2>
  <p class="sub">One paper per panel; each panel shows only the top n methods sorted by best |d−target|.
  Colors match the per-paper charts; no legend here (repeating 18 methods takes too much space).</p>
  <div class="controls">
    <label>Show top <input id="topn_overview" type="number" min="1" max="18" value="3"> per panel</label>
    <label>Focus <select id="focus_overview">{options}</select></label>
    <button id="reset_overview">Reset</button>
    <span class="status" id="status_overview"></span>
  </div>
  <p class="hint">The y-axis is best-so-far |d−target| (log), lower is better; the black dotted line is the 0.1% threshold.</p>
  {plot_html}
</section>
"""
    return section, js


def build_gap(data):
    offsets = {m: (i - (len(METHODS) - 1) / 2.0) * 0.055 for i, m in enumerate(METHODS)}
    rows = []
    for pi, p in enumerate(data):
        for si, (_, r) in enumerate(p["rows"].iterrows()):
            gap = 100.0 * float(r.abs_err) / max(abs(float(p["target"])), 0.5)
            rows.append(dict(p=pi, paper=p["short"], m=str(r.algo), gap=float(gap),
                             q2t=None if pd.isna(r.q2t_1) else int(r.q2t_1),
                             best_d=float(r.best_d), target=float(p["target"]),
                             status=str(r.status), path=p["series"][si]["path"]))
    data_json = rows

    traces = []
    for m in METHODS:
        pts = [r for r in rows if r["m"] == m]
        if not pts:
            continue
        x = [max(r["gap"], 1e-4) for r in pts]
        y = [r["p"] + offsets[m] for r in pts]
        cd = [[r["paper"], r["m"], r["gap"], r["best_d"], r["target"],
               "—" if r["q2t"] is None else str(r["q2t"]), r["status"]] for r in pts]
        traces.append(go.Scatter(
            x=x, y=y, mode="markers", name=m,
            marker=dict(size=9, color=COLORS.get(m, "#555"), symbol="square", line=dict(width=0)),
            customdata=cd,
            hovertemplate=("<b>%{customdata[1]}</b><br>%{customdata[0]}<br>"
                           "gap = %{customdata[2]:.4f}%<br>best d = %{customdata[3]:.4f}<br>"
                           "target = %{customdata[4]:.4f}<br>q2t = %{customdata[5]}<br>"
                           "status = %{customdata[6]}<extra></extra>"),
        ))
    # Stars = closest method per paper (not threshold-dependent).
    star_x, star_y, star_cd = [], [], []
    for pi, p in enumerate(data):
        r = p["rows"].iloc[0]
        gap = 100.0 * float(r.abs_err) / max(abs(float(p["target"])), 0.5)
        star_x.append(max(gap, 1e-4))
        star_y.append(pi)
        star_cd.append([p["short"], str(r.algo), gap])
    traces.append(go.Scatter(
        x=star_x, y=star_y, mode="markers", name="closest", showlegend=False,
        marker=dict(size=11, color="#111", symbol="star", line=dict(width=0)),
        customdata=star_cd,
        hovertemplate=("<b>%{customdata[0]}</b> closest<br>%{customdata[1]}<br>"
                       "gap = %{customdata[2]:.4f}%<extra></extra>"),
    ))

    # 5 alternating paper bands + hit region + threshold line.
    base_shapes = []
    for i in range(len(data)):
        base_shapes.append(dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
                                y0=i - 0.5, y1=i + 0.5,
                                fillcolor="rgba(0,0,0,0.035)" if i % 2 == 0 else "rgba(0,0,0,0.07)",
                                line_width=0, layer="below"))
    base_shapes.append(dict(type="rect", xref="x", yref="paper", x0=1e-4, x1=0.1,
                            y0=0, y1=1, fillcolor="rgba(39,174,96,0.10)",
                            line_width=0, layer="below"))
    base_shapes.append(dict(type="line", xref="x", yref="paper", x0=0.1, x1=0.1,
                            y0=0, y1=1, line=dict(color="#c0392b", width=2, dash="dash")))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white", height=640,
        margin=dict(l=110, r=30, t=30, b=150),
        xaxis=dict(title="gap to target:  100 × |d_hat − d_paper| / max(|d_paper|, 0.5)  (%)",
                   type="log", range=[-4, 2.0],
                   tickvals=[0.001, 0.01, 0.1, 1, 10, 100],
                   ticktext=["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"],
                   showgrid=True, zeroline=False, gridcolor="#e8e8e8"),
        yaxis=dict(title="paper", tickvals=list(range(len(data))),
                   ticktext=[p["short"] for p in data],
                   range=[len(data) - 0.45, -0.55],
                   showgrid=False, zeroline=False),
        shapes=base_shapes,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.13, xanchor="left", x=0,
                    font=dict(size=9), itemwidth=30),
    )
    plot_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart_gap",
                            config={"responsive": True, "displaylogo": False,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    js = f"""
(function() {{
  var gd = document.getElementById('chart_gap');
  var DATA = {json.dumps(data_json)};
  var SHORT = {json.dumps([p['short'] for p in data])};
  var BASE = {json.dumps(base_shapes)};
  var HIT = {len(data)};
  var LINE = {len(data) + 1};
  function fmtPct(v) {{
    return parseFloat(v.toPrecision(3)) + '%';
  }}
  function applyGap() {{
    var s = parseFloat(document.getElementById('gap_thr').value);
    if (!isFinite(s)) s = -1;
    s = Math.max(-3, Math.min(2, s));
    var thr = Math.pow(10, s);
    document.getElementById('gap_thr_label').textContent = fmtPct(thr);
    var shapes = JSON.parse(JSON.stringify(BASE));
    shapes[HIT].x1 = thr;
    shapes[LINE].x0 = thr;
    shapes[LINE].x1 = thr;
    Plotly.relayout(gd, {{shapes: shapes}});
    var html = '';
    for (var pi = 0; pi < SHORT.length; pi++) {{
      var rs = DATA.filter(function(r) {{ return r.p === pi; }});
      var hits = rs.filter(function(r) {{ return r.gap <= thr; }});
      var closest = rs.reduce(function(a, b) {{ return a.gap <= b.gap ? a : b; }});
      var best = hits.length ? hits.reduce(function(a, b) {{ return a.gap <= b.gap ? a : b; }}) : null;
      var fast = null;
      hits.forEach(function(r) {{
        for (var k = 0; k < r.path.length; k++) {{
          if (r.path[k][0] <= thr) {{
            var cand = {{m: r.m, q2t: r.path[k][1], gap: r.path[k][0]}};
            if (!fast || cand.q2t < fast.q2t ||
                (cand.q2t === fast.q2t && cand.gap < fast.gap)) fast = cand;
            break;
          }}
        }}
      }});
      html += '<div class="gap-card"><div class="gap-paper">' + SHORT[pi] + '</div>';
      if (best) {{
        html += 'Hits: <b>' + hits.length + '</b> / ' + rs.length + '; best <b>' + best.m +
                '</b> (' + best.gap.toFixed(4) + '%)';
      }} else {{
        html += 'No hits; closest <b>' + closest.m + '</b> (' + closest.gap.toFixed(4) + '%)';
      }}
      html += '<br>Fastest (at current threshold)' +
              (fast ? ' <b>' + fast.m + '</b> (q2t ' + fast.q2t + ')' : ' —') + '</div>';
    }}
    document.getElementById('gap-summary').innerHTML = html;
  }}
  document.getElementById('gap_thr').addEventListener('input', applyGap);
  applyGap();
}})();
"""
    section = f"""
<section class="paper" id="gap">
  <h2>Gap to target (percent · who is best under the current requirement)</h2>
  <p class="sub">x-axis = <code>100 × |d_hat − d_paper| / max(|d_paper|, 0.5)</code> (log scale, further left = closer to target);
  each band on the y-axis is one paper; color = method; black star = the method closest to the target for that paper;
  green region + red line = current gap requirement. The denominator uses 0.5 for Jiang, because its tol has a 0.0005 floor;
  this makes the 0.1% line coincide with the success line.</p>
  <div class="controls">
    <div class="thr-wrap">
      <label>Current gap requirement
        <input id="gap_thr" type="range" min="-3" max="2" step="0.01" value="-1">
        <span class="thr-label" id="gap_thr_label">0.1%</span>
      </label>
      <div class="thr-ticks"><span>0.001%</span><span>0.01%</span><span>0.1%</span><span>1%</span><span>10%</span><span>100%</span></div>
    </div>
    <span class="status">Drag the slider; default 0.1%; only points to the left of the line satisfy the requirement</span>
  </div>
  <p class="hint">"Best" = the method with the smallest gap within the current threshold; "fastest" = the method with the fewest
  feasible evaluations (q2t, including warm-up) to first reach that gap within the threshold.
  When no method satisfies the threshold, it shows "no hits" and gives the globally closest method.</p>
  <div class="gap-summary" id="gap-summary"></div>
  {plot_html}
  <p class="note">q2t counts feasible evaluations including the 20 warm-up points; pure BO queries ≈ q2t − 20.
  Rejected proposals do not count toward q2t, but they do occupy attempt numbers on the x-axis.</p>
</section>
"""
    return section, js


def build_all() -> str:
    data = [load_paper(p) for p in PAPERS]
    gap_section, gap_js = build_gap(data)
    overview_section, overview_js = build_overview(data)
    paper_sections, paper_js = [], []
    for p in data:
        s, j = build_paper_section(p)
        paper_sections.append(s)
        paper_js.append(f"<script>{j}</script>")
    nav = ('<a href="#gap">Gap chart</a> <a href="#overview">Best-so-far overview</a> '
           + " ".join(f'<a href="#{p["paper"]}">{p["short"]}</a>' for p in data))
    plotly_js = open(os.path.join(os.path.dirname(__import__("plotly").__file__),
                                  "package_data", "plotly.min.js"), encoding="utf-8").read()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 4 — 5 papers, dynamic charts</title>
<script>{plotly_js}</script>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>Experiment 4 — 5 papers: Cohen's d, best-so-far, gap to target</h1>
  <p class="sub">One self-contained HTML: plotly.js and all traces are inlined, share this single file, no network needed.
  Each section has its own <b>top=n</b> input, focus, and legend; sorting is fixed by best |d−target|.</p>
  <nav>{nav}</nav>
  {COHEN_HTML}
  {gap_section}
  {overview_section}
  {''.join(paper_sections)}
  {''.join(paper_js)}
</div>
<script>{gap_js}</script>
<script>{overview_js}</script>
</body>
</html>
"""
    out = os.path.join(OUT, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


def build_per_paper():
    for paper in PAPERS:
        p = load_paper(paper)
        s, j = build_paper_section(p)
        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLES[paper]} — d vs attempt + best-so-far</title>
<script src="plotly.min.js"></script>
<style>{CSS}</style></head><body><div class="wrap">{s}<script>{j}</script></div></body></html>
"""
        with open(os.path.join(OUT, f"{paper}_d_vs_attempt.html"), "w", encoding="utf-8") as f:
            f.write(html)


def main():
    os.makedirs(OUT, exist_ok=True)
    js_src = os.path.join(os.path.dirname(__import__("plotly").__file__),
                          "package_data", "plotly.min.js")
    shutil.copyfile(js_src, os.path.join(OUT, "plotly.min.js"))
    build_per_paper()
    print(build_all())
    print("index bytes", os.path.getsize(os.path.join(OUT, "index.html")))


if __name__ == "__main__":
    main()
