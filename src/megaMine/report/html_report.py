"""
html_report.py &mdash; megaMine v2.0
MultiQC-style HTML Report Generator
"""

import os
import sys
import json
import html
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


def _pair_plots(html_a: str, html_b: str) -> str:
    """Wrap two plot subsections side by side in a two-column grid."""
    return f'<div class="plot-row">{html_a}{html_b}</div>'


def _choose_df(primary, fallback):
    if primary is not None and len(primary) > 0:
        return primary
    if fallback is not None and len(fallback) > 0:
        return fallback
    return None


COLORS = {
    "navy":      "#1e2d3d",
    "blue":      "#5b4fcf",
    "light_blue":"#93c5fd",
    "teal":      "#00a8a8",
    "green":     "#2ea87e",
    "yellow":    "#e8a020",
    "red":       "#f05a28",
    "gray":      "#7a8799",
    "light_gray":"#f3f4f6",
    "white":     "#ffffff",
}

NODE_COLORS = {
    "Gene":        "#5b4fcf",
    "Drug":        "#00a8a8",
    "Cancer":      "#f05a28",
    "Alteration":  "#c0398e",
    "Evidence":    "#e8a020",
    "Publication": "#7a8799",
}

TREND_COLORS = {
    "stable":              "#2ea87e",
    "rising_resistance":   "#f05a28",
    "declining_efficacy":  "#e8a020",
    "emerging":            "#5b4fcf",
    "conflicted":          "#c0398e",
    "insufficient_data":   "#7a8799",
}

FLAG_COLORS = {
    "none":              "#2ea87e",
    "watch":             "#e8a020",
    "caution":           "#f05a28",
    "conflict":          "#f05a28",
    "insufficient_data": "#9ca3af",
}


def _safe_load_sheet(xl, *names):
    for name in names:
        if name in xl.sheet_names:
            try:
                return xl.parse(name)
            except Exception:
                continue
    return pd.DataFrame()


def _safe_col(df, *names):
    if df is None or not hasattr(df, "columns"):
        return None
    for n in names:
        if n in df.columns:
            return n
    return None


def _h(text):
    return html.escape(str(text))


def _plot_html(fig, height=420):
    if not HAS_PLOTLY or fig is None:
        return '<p class="skip-note">Plot not available</p>'
    try:
        fig.update_layout(
            height=height,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="white",
            plot_bgcolor="#f3f4f6",
            font=dict(family="Inter, system-ui, sans-serif", size=12),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)
    except Exception as e:
        return f'<p class="skip-note">Plot error: {_h(str(e))}</p>'


def _skip(msg):
    return f'<p class="skip-note">Plot skipped: {_h(msg)}</p>'


def _table_html(df, max_rows=300, caption=""):
    if df is None or len(df) == 0:
        return '<p class="skip-note">No data available</p>'
    note = ""
    if len(df) > max_rows:
        note = f'<p class="table-note">Showing first {max_rows:,} of {len(df):,} rows.</p>'
        df = df.head(max_rows)
    cols_html = "".join(f"<th>{_h(c)}</th>" for c in df.columns)
    rows_html = ""
    HTML_LINK_COLS = {"pmid","pmid_link","doi"}
    for _, row in df.iterrows():
        cells = ""
        for col, v in zip(df.columns, row):
            val = str(v) if pd.notna(v) else ""
            # Allow raw HTML for link columns
            if col in HTML_LINK_COLS and val.startswith("<a "):
                cells += f"<td>{val}</td>"
            else:
                cells += f"<td>{_h(val)}</td>"
        rows_html += f"<tr>{cells}</tr>"
    cap = f"<caption>{_h(caption)}</caption>" if caption else ""
    return f"""
{note}
<div class="table-wrap">
<table class="data-table">
{cap}
<thead><tr>{cols_html}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""


def _metric_card(label, value, icon="", color="#17324d"):
    return f"""
<div class="metric-card">
  <div class="metric-value" style="color:{color}">{_h(str(value))}</div>
  <div class="metric-label">{_h(label)}</div>
</div>"""


def _qc_badge(level, msg):
    colors = {"ok":"#2f6b4f","warn":"#8a5a16","fail":"#9b3a3a"}
    c = colors.get(level, "#6b7280")
    return f'<div class="qc-badge" style="border-left-color:{c}">{_h(msg)}</div>'


def _section(title, content, section_id=""):
    sid = section_id or title.lower().replace(" ", "-")
    return f"""
<section id="{_h(sid)}" class="report-section">
  <div style="display:flex;align-items:center;justify-content:space-between;
       padding-bottom:8px;margin-bottom:16px;border-bottom:2px solid #17324d">
    <h2 class="section-title" style="border:none;padding:0;margin:0">{_h(title)}</h2>
  </div>
  {content}
</section>"""


def _subsection(title, content, caption=""):
    cap = f'<p class="fig-caption">{_h(caption)}</p>' if caption else ""
    return f"""
<div class="subsection">
  <h3 class="subsection-title">{_h(title)}</h3>
  {content}
  {cap}
</div>"""


CSS = """
:root {
  --navy-900:#17324d; --navy-700:#28506f;
  --slate-900:#1f2933; --slate-700:#4b5563; --slate-500:#6b7280;
  --slate-300:#d1d5db; --slate-200:#e5e7eb; --slate-100:#f3f4f6;
  --slate-050:#f8fafc; --white:#ffffff;
  --green-700:#2f6b4f; --green-100:#e8f3ed;
  --amber-700:#8a5a16; --amber-100:#fbf3df;
  --red-700:#9b3a3a;   --red-100:#f8e8e8;
  --blue-700:#315f8c;  --blue-100:#eaf1f8;
  --teal:#00a8a8;
  --font:Inter,"Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--font);background:#eef1f4;color:var(--slate-900);
     display:flex;min-height:100vh;font-size:13px;line-height:1.55}
a{color:var(--navy-700)}

/* Sidebar */
#sidebar{width:220px;min-height:100vh;background:var(--navy-900);
         color:#fff;position:fixed;top:0;left:0;overflow-y:auto;
         z-index:100;padding-bottom:40px;border-right:1px solid rgba(255,255,255,0.08)}
#sidebar .logo{padding:22px 18px 14px;font-size:1rem;font-weight:700;
               letter-spacing:-.01em;border-bottom:1px solid rgba(255,255,255,0.1)}
#sidebar .logo span{color:#93c5fd}
#sidebar nav{padding:10px 0}
#sidebar nav a{display:block;padding:8px 18px;color:rgba(255,255,255,0.7);
               text-decoration:none;font-size:0.8rem;
               border-left:3px solid transparent;transition:all 0.12s}
#sidebar nav a:hover{color:#fff;background:rgba(255,255,255,0.06);
                     border-left-color:#93c5fd}
#sidebar nav a.active{color:#fff;background:rgba(255,255,255,0.1);
                      border-left-color:#93c5fd}
#sidebar .nav-section{padding:14px 18px 4px;font-size:9px;font-weight:700;
                       text-transform:uppercase;letter-spacing:.1em;
                       color:rgba(255,255,255,0.35)}

/* Main content */
#main{margin-left:220px;flex:1;min-width:0}

/* Header */
.report-header{background:var(--white);border-top:7px solid var(--navy-900);
               border-bottom:1px solid var(--slate-200);
               padding:28px 34px 24px}
.report-header-grid{display:grid;grid-template-columns:minmax(0,1fr) 320px;
                    gap:28px;margin-bottom:18px}
.report-eyebrow{font-size:10px;font-weight:700;letter-spacing:.12em;
                text-transform:uppercase;color:var(--navy-700);margin-bottom:6px}
.report-title{font-size:22px;font-weight:700;letter-spacing:-.02em;
              color:var(--navy-900);line-height:1.25}
.report-subtitle{margin-top:6px;font-size:12px;color:var(--slate-500)}
.report-meta{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px 16px;font-size:11.5px}
.report-meta div{color:var(--slate-900);font-weight:600}
.report-meta span{display:block;margin-bottom:2px;color:var(--slate-500);
                  font-size:9.5px;font-weight:700;text-transform:uppercase;
                  letter-spacing:.04em}
.disclaimer{margin-top:0;padding:11px 14px;border-left:4px solid var(--amber-700);
            background:var(--amber-100);font-size:11.5px;color:#5f4a27;line-height:1.5}
.disclaimer strong{color:var(--amber-700)}

/* Sections */
.report-section{padding:24px 34px;border-bottom:1px solid var(--slate-200);
                background:var(--white)}
.report-section:nth-child(even){background:var(--slate-050)}
.section-title{font-size:15px;font-weight:700;color:var(--navy-900);
               padding-bottom:8px;margin-bottom:16px;
               border-bottom:2px solid var(--navy-900)}
.subsection{margin-bottom:22px}
.subsection-title{font-size:12.5px;font-weight:700;color:var(--navy-700);
                  margin-bottom:10px;padding-bottom:5px;
                  border-bottom:1px solid var(--slate-200)}
.fig-caption{margin-top:6px;font-size:10.5px;color:var(--slate-500);
             font-style:italic;line-height:1.4}

/* Metric cards */
.metrics-grid,.metric-grid{
  display:flex;flex-direction:row;flex-wrap:wrap;
  gap:1px;margin-bottom:20px;
  border:1px solid var(--slate-200);
  background:var(--slate-200);width:100%}
.metric-card{background:var(--white);padding:11px 14px;
             text-align:center;flex:1 1 120px;min-width:100px;max-width:200px}
.metric-value{font-size:19px;font-weight:700;color:var(--navy-900);
              line-height:1;margin-bottom:4px}
.metric-label{font-size:8.5px;font-weight:700;text-transform:uppercase;
              letter-spacing:.05em;color:var(--slate-500);line-height:1.3}
.metric-icon{display:none}

/* QC badges */
.qc-badge{padding:7px 11px;margin-bottom:6px;border-left:4px solid var(--slate-300);
          background:var(--slate-050);font-size:11.5px;color:var(--slate-700)}

/* Plot row */
.plot-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}

/* Tables */
.table-wrap{overflow-x:auto;border:1px solid var(--slate-200);margin-bottom:16px}
.data-table{width:100%;border-collapse:collapse;font-size:11.5px}
.data-table th{padding:9px 10px;text-align:left;background:var(--navy-900);
               color:#fff;font-weight:600;font-size:10px;
               text-transform:uppercase;letter-spacing:.03em;white-space:nowrap}
.data-table td{padding:8px 10px;border-bottom:1px solid var(--slate-200);
               vertical-align:top}
.data-table tbody tr:nth-child(even){background:var(--slate-050)}
.data-table tbody tr:last-child td{border-bottom:0}
.data-table tbody tr:hover td{background:var(--blue-100)}
table caption{font-size:10.5px;color:var(--slate-500);padding:6px 0;
              text-align:left;font-style:italic}

/* Evidence labels */
.badge{display:inline-block;padding:2px 6px;border-radius:2px;
       font-size:9px;font-weight:700;line-height:1.3}
.badge-green{color:var(--green-700);background:var(--green-100)}
.badge-blue{color:var(--blue-700);background:var(--blue-100)}
.badge-amber{color:var(--amber-700);background:var(--amber-100)}
.badge-red{color:var(--red-700);background:var(--red-100)}
.badge-gray{color:var(--slate-700);background:var(--slate-100)}

/* Footer */
.report-footer{padding:16px 34px;border-top:1px solid var(--slate-200);
               background:var(--white);display:flex;justify-content:space-between;
               color:var(--slate-500);font-size:10.5px;gap:20px}

/* Print */
@media print{
  body{background:#fff}
  #sidebar{display:none}
  #main{margin-left:0}
  .report-section{border:0;padding:14px 0}
  .data-table th{background:#e5e7eb!important;color:#111!important;
                 border:1px solid #9ca3af}
  .data-table td{border:1px solid #d1d5db}
}
@media(max-width:100%){
  #sidebar{display:none}
  #main{margin-left:0}
  .report-header-grid{grid-template-columns:1fr}
  .plot-row{grid-template-columns:1fr}
  .metrics-grid{grid-template-columns:repeat(3,1fr)}
}

/* Layout + plot corrections */
html,body{width:100%;max-width:100%;overflow-x:hidden}
body{display:block;min-height:100vh}
#sidebar{width:220px;position:fixed;top:0;bottom:0;left:0}
#main{margin-left:220px;width:calc(100% - 220px);max-width:none;min-width:0;overflow-x:hidden}
.report-header,.report-section,.subsection{width:100%;max-width:none;min-width:0}
.report-section{overflow:hidden}
.plot-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;align-items:start;width:100%}
.plot-row>*{min-width:0;max-width:100%}
.plot-row-full{display:block;width:100%}
.plot-frame{position:relative;width:100%;min-width:0;overflow:hidden}
.plot-compact{height:280px}
.plot-standard{height:360px}
.plot-large{height:460px}
.plot-matrix{height:min(620px,70vh)}
.plot-network{height:min(680px,75vh)}
.plotly-graph-div,.js-plotly-plot,.plot-container,.svg-container{width:100%!important;max-width:100%!important;min-width:0!important}
.interpretation-summary,.query-comparison,.summary-text{width:100%;max-width:none}
@media(max-width:1200px){.plot-row{grid-template-columns:1fr}}
@media(max-width:900px){
  #sidebar{display:none}
  #main{margin-left:0;width:100%}
  .report-header-grid{grid-template-columns:1fr}
  .metrics-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
}
@media(max-width:600px){
  .report-header,.report-section{padding-left:18px;padding-right:18px}
  .metrics-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
"""



def _sidebar():
    items = [
        ("summary",        "Executive Summary"),
        ("normalization",  "Normalization"),
        ("resistance",     "Resistance"),
        ("evidence",       "Evidence"),
        ("matrix",         "Gene-Drug Matrix"),
        ("temporal",       "Temporal Trends"),
        ("contradictions", "Contradictions"),
        ("trials",         "ClinicalTrials.gov"),
        ("graph",          "Knowledge Graph"),
        ("tables",         "Data Tables"),
    ]
    links = "".join(f'<a href="#{sid}">{_h(label)}</a>' for sid, label in items)
    return f"""<div id="sidebar">
  <div class="logo">mega<span>Mine</span></div>
  <div class="nav-section">Navigation</div>
  <nav>{links}</nav>
</div>"""


def _header(title, run_info):
    query = version = generated = n_rows = n_ver = n_pmids = ""
    if run_info is not None and len(run_info) > 0:
        ri        = run_info.iloc[0]
        query     = str(ri.get("query", ""))
        version   = str(ri.get("version", "megaMine v2.0"))
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        n_rows    = str(ri.get("n_rows_all", ri.get("n_rows", "")))
        n_ver     = str(ri.get("n_rows_verified", ""))
        n_pmids   = str(ri.get("n_pmids", ""))
    return f"""<div class="report-header">
  <div class="report-header-grid">
    <div>
      <div class="report-eyebrow">Literature Evidence Synthesis Report</div>
      <h1 class="report-title">megaMine v2.0 &mdash; Literature Evidence Report</h1>
      <div class="report-subtitle">APML &middot; Ajou University, Department of Biomedical Sciences &middot; {_h(generated)}</div>
    </div>
    <div class="report-meta">
      <div><span>Total rows</span>{_h(n_rows)}</div>
      <div><span>Verified rows</span>{_h(n_ver)}</div>
      <div><span>Unique PMIDs</span>{_h(n_pmids)}</div>
      <div><span>Query</span>{_h(query[:60])}{'...' if len(query)>60 else ''}</div>
    </div>
  </div>
  <div class="disclaimer">
    <strong>Important:</strong>
    This report presents literature-derived evidence extracted by megaMine v2.0.
    It supports expert review and does <strong>not constitute a clinical
    treatment recommendation.</strong>
    All findings require independent validation by a qualified expert.
  </div>
</div>"""


def _full_html(title, header, sidebar, body):
    plotlyjs = ""
    if HAS_PLOTLY:
        from plotly.offline import get_plotlyjs
        plotlyjs = f"<script>{get_plotlyjs()}</script>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_h(title)}</title>
{plotlyjs}
<style>{CSS}</style>
</head>
<body>
{sidebar}
<div id="main">
{header}
{body}
</div>
<script>
const sections = document.querySelectorAll('.report-section');
const links = document.querySelectorAll('#sidebar nav a');
window.addEventListener('scroll', () => {{
  let current = '';
  sections.forEach(s => {{ if (window.scrollY >= s.offsetTop - 100) current = s.id; }});
  links.forEach(l => {{ l.classList.toggle('active', l.getAttribute('href') === '#' + current); }});
}});
</script>
<footer class="report-footer">
  <div>megaMine v2.0 &middot; Literature evidence synthesis &middot; APML, Ajou University</div>
  <div>Research use only &middot; Not for direct clinical decision-making</div>
</footer>
<script>
(function(){{
  function resizeAll(){{
    document.querySelectorAll(".plotly-graph-div").forEach(function(p){{
      if(window.Plotly && p && p.data) Plotly.Plots.resize(p);
    }});
  }}
  window.addEventListener("load",function(){{
    setTimeout(resizeAll,100); setTimeout(resizeAll,500);
  }});
  window.addEventListener("resize",resizeAll);
}})();
</script>
</body>
</html>"""



def _build_summary(rows_all, rows_ver, trend_ver, contra_ver,
                   trials_df, nodes_df, edges_df, run_info):
    def _v(df, col, val=None):
        if df is None or col not in df.columns: return 0
        if val is None: return df[col].nunique()
        return int((df[col].astype(str).str.lower() == str(val).lower()).sum())

    n_all      = len(rows_all) if rows_all is not None else 0
    n_ver      = len(rows_ver) if rows_ver is not None else 0
    canon_col  = _safe_col(rows_all, "canonical_cancer_type", "cancer_type")
    raw_col    = _safe_col(rows_all, "cancer_type")
    n_clean    = int((rows_all[canon_col].fillna("")!="").sum()) if (rows_all is not None and canon_col) else 0
    n_rejected = int((rows_all["cancer_confidence"].fillna("")=="rejected").sum()) if (rows_all is not None and "cancer_confidence" in rows_all.columns) else 0
    n_canon    = rows_all[canon_col].replace("",pd.NA).dropna().nunique() if (rows_all is not None and canon_col) else 0
    n_raw      = rows_all[raw_col].nunique() if (rows_all is not None and raw_col) else 0
    n_genes    = _v(rows_ver,"biomarker")
    n_drugs    = _v(rows_ver,"drug_primary")
    n_pmids    = _v(rows_ver,"pmid")
    n_ro       = _v(rows_all,"resistance_observed","yes")
    n_rc       = _v(rows_all,"resistance_context","yes")
    n_re       = _v(rows_all,"resistance_evidence","yes")
    n_trip     = len(trend_ver) if trend_ver is not None else 0
    n_flag     = int((contra_ver["contradiction_flag"].isin(["watch","caution","conflict"])).sum()) if (contra_ver is not None and "contradiction_flag" in contra_ver.columns) else 0
    n_trials   = len(trials_df) if trials_df is not None else 0
    n_nodes    = len(nodes_df) if nodes_df is not None else 0
    n_edges    = len(edges_df) if edges_df is not None else 0
    n_cn       = int((nodes_df["node_type"]=="Cancer").sum()) if (nodes_df is not None and "node_type" in nodes_df.columns) else 0
    n_ev       = int((nodes_df["node_type"]=="Evidence").sum()) if (nodes_df is not None and "node_type" in nodes_df.columns) else 0
    n_pub      = int((nodes_df["node_type"]=="Publication").sum()) if (nodes_df is not None and "node_type" in nodes_df.columns) else 0

    # Auto-generate interpretation summary
    # Get query from run_info safely
    _query = ""
    _version = "megaMine v2.0"
    if run_info is not None and len(run_info) > 0:
        _query   = str(run_info.iloc[0].get("query", ""))
        _version = str(run_info.iloc[0].get("version", "megaMine v2.0"))
    n_strict = int((rows_all["canonical_cancer_type"].fillna("")
                    .str.contains("Non-Small Cell", na=False)).sum()) if (rows_all is not None and "canonical_cancer_type" in rows_all.columns) else 0
    interp = f"""
<div class="subsection" style="border-left:4px solid #2563eb;background:#eff6ff;max-width:100%;">
  <h3 class="subsection-title" id="interp-summary">Interpretation Summary</h3>
  <p style="font-size:0.82rem;line-height:1.65;color:#1e293b;max-width:100%;">
  This report presents literature-derived evidence for the query:
  <strong>{_h(_query)}</strong>.<br>
  megaMine v2.0 extracted <strong>{n_all:,} evidence rows</strong> from
  <strong>{n_pmids:,} retrieved publications</strong>.
  After cancer normalization and offline rule-based verification,
  <strong>{n_ver:,} evidence rows</strong> from
  <strong>{_v(rows_ver,"pmid"):,} unique PMIDs</strong> were retained.
  Direct resistance evidence was detected in
  <strong>{n_re:,} extracted rows</strong>
  (of which <strong>{_v(rows_ver,"resistance_evidence","yes"):,} verified</strong>).
  Cancer normalization reduced raw labels to <strong>{n_canon:,} canonical types</strong>.
  Offline verifier confidence threshold: <strong>0.70</strong>.
  Model-assisted verification may improve recall but requires independent benchmarking.
  Version: <strong>{_h(_version)}</strong>.
  <br><br>
  <strong>ClinicalTrials results shown as &ge;200 are capped</strong> and may include off-target trials.
  Concordance values for capped results are marked <em>unknown_verify_manually</em>.
  <br><br>
  All evidence is <strong>literature-derived</strong> and has not been independently validated.
  This report is a software demonstration and <strong>not a clinical treatment recommendation</strong>.
  Do not use these results to guide patient care without expert review.
  <br><br>
   <strong>Row vs publication note:</strong>
  One publication may generate multiple evidence rows when it contains multiple
  gene&ndash;drug&ndash;alteration&ndash;cancer&ndash;direction combinations.
  Evidence-row counts are <em>not</em> equivalent to publication counts.
  673 rows &#8800; 673 independent studies.
  </p>
</div>"""

    cards = '<div class="metric-grid">\n' + "\n".join([
        _metric_card("Total Rows", f"{n_all:,}", "", "#1e2d3d"),
        _metric_card("Clean Cancer", f"{n_clean:,}", "", "#00a8a8"),
        _metric_card("Verified", f"{n_ver:,}", "", "#2ea87e"),
        _metric_card("Unique PMIDs", f"{n_pmids:,}", "&#128240;", "#5b4fcf"),
        _metric_card("Unique Genes", f"{n_genes:,}", "", "#1e2d3d"),
        _metric_card("Unique Drugs", f"{n_drugs:,}", "", "#00a8a8"),
        _metric_card("Raw Cancer Labels", f"{n_raw:,}", "&#128292;", "#7a8799"),
        _metric_card("Canonical Types", f"{n_canon:,}", "", "#2ea87e"),
        _metric_card("Rejected Rows", f"{n_rejected:,}", "", "#f05a28"),
        _metric_card("Resist Observed", f"{n_ro:,}", "", "#f05a28"),
        _metric_card("Resist Context", f"{n_rc:,}", "", "#e8a020"),
        _metric_card("Resist Evidence", f"{n_re:,}", "&#127919;", "#f05a28"),
        _metric_card("Temporal Triplets", f"{n_trip:,}", "", "#5b4fcf"),
        _metric_card("Contradiction Flags", f"{n_flag:,}", "", "#e8a020"),
        _metric_card("Trial Pairs", f"{n_trials:,}", "", "#00a8a8"),
        _metric_card("Graph Nodes", f"{n_nodes:,}", "", "#1e2d3d"),
        _metric_card("Graph Edges", f"{n_edges:,}", "", "#5b4fcf"),
        _metric_card("Cancer Nodes", f"{n_cn:,}", "&#127919;", "#f05a28"),
        _metric_card("Evidence Nodes", f"{n_ev:,}", "", "#e8a020"),
        _metric_card("Publication Nodes", f"{n_pub:,}", "&#128218;", "#7a8799"),
    ]) + "\n</div>"

    qc = []
    if n_raw > 0 and n_canon > 0:
        ratio = n_raw / max(n_canon, 1)
        lvl = "warn" if ratio > 5 else "ok"
        qc.append(_qc_badge(lvl, f"Cancer normalization: {n_raw} raw labels &rarr; {n_canon} canonical types (ratio {ratio:.1f}x)"))
    if n_rejected > 0:
        pct = n_rejected / max(n_all, 1) * 100
        qc.append(_qc_badge("warn" if pct > 10 else "ok", f"{n_rejected} rows ({pct:.1f}%) had non-cancer labels rejected"))
    if n_ro > 0 and n_re > 0:
        ratio = n_ro / max(n_re, 1)
        qc.append(_qc_badge("warn" if ratio > 5 else "ok", f"Resistance refinement: {n_ro} broad &rarr; {n_re} direct evidence ({ratio:.0f}x reduction)"))
    if n_all > 0:
        pct = n_ver / n_all * 100
        qc.append(_qc_badge("fail" if pct < 10 else ("warn" if pct < 30 else "ok"), f"Verification rate: {n_ver}/{n_all} ({pct:.1f}%)"))
    if n_cn > 15:
        qc.append(_qc_badge("warn", f"High cancer graph nodes ({n_cn}) &mdash; check canonical_cancer_type"))
    else:
        qc.append(_qc_badge("ok", f"Clean cancer graph nodes: {n_cn}"))
    if n_flag > 0:
        qc.append(_qc_badge("warn", f"{n_flag} triplets show contradiction signals"))
    else:
        qc.append(_qc_badge("ok", "No contradiction signals in verified rows"))

    # Strict vs expanded query separation
    strict_html = ""
    try:
        if rows_ver is not None:
            is_egfr   = rows_ver["biomarker"].str.upper()  == "EGFR"   if "biomarker"   in rows_ver.columns else pd.Series(False, index=rows_ver.index)
            is_erl    = rows_ver["drug_primary"].str.lower() == "erlotinib" if "drug_primary" in rows_ver.columns else pd.Series(False, index=rows_ver.index)
            cc        = _safe_col(rows_ver, "canonical_cancer_type","cancer_type")
            # Use exact canonical match &mdash; substring matching is imprecise
            is_nsclc  = (rows_ver[cc] == "Non-Small Cell Lung Cancer") if cc else pd.Series(False, index=rows_ver.index)
            n_strict  = int((is_egfr & is_erl & is_nsclc).sum())
            n_expanded= len(rows_ver) - n_strict
            strict_html = f"""
<div class="subsection" style="border-left:4px solid #0d9488;background:#f0fdfa;max-width:100%;box-sizing:border-box;">
  <h3 class="subsection-title">&#127919; Query-Matched vs Expanded Evidence</h3>
  <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;max-width:100%;">
    <div class="metric-card" style="border-top-color:#0d9488;flex:1;min-width:160px;max-width:260px;">
      <div class="metric-icon">&#127919;</div>
      <div class="metric-value" style="color:#0d9488">{n_strict:,}</div>
      <div class="metric-label">Strict Query Match<br><small>EGFR + erlotinib + NSCLC</small></div>
    </div>
    <div class="metric-card" style="border-top-color:#6b7280;flex:1;min-width:160px;max-width:260px;">
      <div class="metric-icon">&#128301;</div>
      <div class="metric-value" style="color:#6b7280">{n_expanded:,}</div>
      <div class="metric-label">Expanded Related Evidence<br><small>Other genes/drugs/cancers</small></div>
    </div>
  </div>
  <p style="font-size:0.78rem;color:#6b7280;margin-top:10px;line-height:1.6;max-width:100%;">
  <strong>Strict</strong> = biomarker=EGFR AND drug=erlotinib AND
  canonical_cancer_type = "Non-Small Cell Lung Cancer" (exact canonical match).<br>
  <strong>Expanded</strong> = other genes (MET, ALK), drugs (osimertinib, afatinib),
  or other cancer contexts (SCLC, Breast, Urothelial) from the same literature search.
  </p>
</div>"""
    except Exception as e:
        strict_html = f'<p class="skip-note">Query separation: {_h(str(e))}</p>'

    qc_html = '<div class="qc-grid">' + "".join(qc) + '</div>'
    return _section("Executive Summary",
                    interp + cards + strict_html + _subsection("QC Summary", qc_html), "summary")


def _build_normalization(rows_all, max_rows):
    if not HAS_PLOTLY or rows_all is None:
        return _section("Normalization", _skip("data not available"), "normalization")
    parts = []

    try:
        n_all  = len(rows_all)
        cc     = _safe_col(rows_all, "canonical_cancer_type")
        n_cln  = int((rows_all[cc].fillna("")!="").sum()) if cc else n_all
        vc     = _safe_col(rows_all, "llm_verified")
        n_ver  = int((rows_all[vc]=="yes").sum()) if vc else 0
        n_vc   = int(((rows_all[vc]=="yes") & (rows_all[cc].fillna("")!="")).sum()) if (vc and cc) else 0
        fig = go.Figure(go.Funnel(
            y=["Extracted","Clean Cancer","LLM Verified","Verified+Clean"],
            x=[n_all, n_cln, n_ver, n_vc],
            textinfo="value+percent initial",
            marker=dict(color=["#1e2d3d","#5b4fcf","#00a8a8","#2ea87e"]),
        ))
        fig.update_layout(title="Evidence Filtering Funnel")
        parts.append(_subsection("Filtering Funnel", _plot_html(fig),
            "Progressive filtering from raw PubMed rows to verified clean evidence."))
    except Exception as e:
        parts.append(_subsection("Funnel", _skip(str(e))))

    try:
        conf_col = _safe_col(rows_all, "cancer_confidence")
        if conf_col:
            vc2 = rows_all[conf_col].value_counts().reset_index()
            vc2.columns = ["confidence","count"]
            cmap = {"high":"#2ea87e","medium":"#5b4fcf","low":"#e8a020","rejected":"#f05a28"}
            fig2 = px.bar(vc2, x="confidence", y="count", color="confidence",
                          color_discrete_map=cmap,
                          title="Cancer Confidence Distribution")
            _pa = _subsection("Cancer Confidence", _plot_html(fig2, 320),
                "high=exact, medium=substring, low=suffix, rejected=fragment")
    except Exception as e:
        _pa = _subsection("Cancer Confidence", _skip(str(e)))

    try:
        cc2 = _safe_col(rows_all, "canonical_cancer_type")
        if cc2:
            ct = rows_all[cc2].replace("",pd.NA).dropna().value_counts().reset_index()
            ct.columns = ["cancer","count"]
            fig3 = px.bar(ct, x="count", y="cancer", orientation="h",
                          color_discrete_sequence=["#00a8a8"],
                          title="Canonical Cancer Types")
            fig3.update_yaxes(categoryorder="total ascending")
            _pb = _subsection("Canonical Cancer Types", _plot_html(fig3, 320),
                "Normalized cancer labels after cleaning.")
        else:
            _pb = ""
    except Exception as e:
        _pb = _subsection("Canonical Types", _skip(str(e)))
    parts.append(_pair_plots(_pa, _pb))

    try:
        rc = _safe_col(rows_all, "cancer_type")
        if rc:
            raw = rows_all[rc].value_counts().head(20).reset_index()
            raw.columns = ["label","count"]
            fig4 = px.bar(raw, x="count", y="label", orientation="h",
                          color_discrete_sequence=["#7a8799"],
                          title="Top 20 Raw Cancer Labels (Before Normalization)")
            fig4.update_yaxes(categoryorder="total ascending")
            parts.append(_subsection("Raw Cancer Labels (Top 20)", _plot_html(fig4, 500),
                "Raw noisy labels before normalization. Many are sentence fragments."))
    except Exception as e:
        parts.append(_subsection("Raw Labels", _skip(str(e))))

    return _section("Normalization & Filtering", "".join(parts), "normalization")


def _build_resistance(rows_all, rows_ver, max_rows):
    if not HAS_PLOTLY:
        return _section("Resistance", _skip("plotly not available"), "resistance")
    parts = []
    df = rows_all if rows_all is not None else rows_ver

    try:
        if df is not None:
            ro = int((df.get("resistance_observed", pd.Series()).astype(str).str.lower()=="yes").sum())
            rc = int((df.get("resistance_context",  pd.Series()).astype(str).str.lower()=="yes").sum())
            re = int((df.get("resistance_evidence", pd.Series()).astype(str).str.lower()=="yes").sum())
            fig = go.Figure(go.Bar(
                x=["resistance_observed\n(broad)","resistance_context\n(mention)","resistance_evidence\n(direct)"],
                y=[ro, rc, re],
                marker_color=["#f05a28","#e8a020","#00a8a8"],
                text=[ro, rc, re], textposition="outside",
            ))
            fig.update_layout(title="Resistance Refinement: Broad &rarr; Precise", yaxis_title="Count")
            parts.append(_subsection("Resistance Refinement", _plot_html(fig),
                "resistance_evidence captures only direct statements &mdash; prevents overcounting."))
    except Exception as e:
        parts.append(_subsection("Refinement", _skip(str(e))))

    try:
        if df is not None and "resistance_direction" in df.columns:
            rd = df["resistance_direction"].value_counts().reset_index()
            rd.columns = ["direction","count"]
            cmap = {"sensitivity":"#2ea87e","resistance":"#f05a28",
                    "post_resistance_efficacy":"#5b4fcf","unclear":"#7a8799"}
            fig2 = px.bar(rd, x="direction", y="count", color="direction",
                          color_discrete_map=cmap,
                          title="Resistance Direction Distribution")
            _pa = _subsection("Resistance Direction", _plot_html(fig2, 320),
                "sensitivity=works; resistance=fails; post_resistance_efficacy=works after prior resistance.")
    except Exception as e:
        _pa = _subsection("Direction", _skip(str(e)))

    try:
        dv = rows_ver if rows_ver is not None else rows_all
        if dv is not None and "resistance_evidence" in dv.columns and "drug_primary" in dv.columns:
            re2 = dv[dv["resistance_evidence"]=="yes"]["drug_primary"].value_counts().head(12).reset_index()
            re2.columns = ["drug","count"]
            fig3 = px.bar(re2, x="count", y="drug", orientation="h",
                          color_discrete_sequence=["#f05a28"],
                          title="Resistance Evidence by Drug")
            fig3.update_yaxes(categoryorder="total ascending")
            _pb = _subsection("Resistance by Drug", _plot_html(fig3, 320),
                "Literature-derived &mdash; does not imply clinical failure.")
        else:
            _pb = ""
    except Exception as e:
        _pb = _subsection("Resistance by Drug", _skip(str(e)))
    parts.append(_pair_plots(_pa, _pb))

    return _section("Resistance Interpretation", "".join(parts), "resistance")


def _build_evidence(rows_all, rows_ver, max_rows):
    if not HAS_PLOTLY:
        return _section("Evidence", _skip("plotly not available"), "evidence")
    parts = []
    df = rows_ver if rows_ver is not None else rows_all

    try:
        if rows_all is not None and "llm_verified" in rows_all.columns:
            vc = rows_all["llm_verified"].value_counts().reset_index()
            vc.columns = ["status","count"]
            fig = px.pie(vc, names="status", values="count",
                         color="status",
                         color_discrete_map={"yes":"#2ea87e","no":"#f05a28","skipped":"#7a8799"},
                         title="Evidence Verification Status (Offline Rule-Based)")
            _pa = _subsection("Verification Status", _plot_html(fig, 320),
                "Proportion of rows verified, rejected, or skipped.")
    except Exception as e:
        _pa = _subsection("Verification", _skip(str(e)))

    try:
        if rows_all is not None and "llm_confidence" in rows_all.columns:
            conf = rows_all["llm_confidence"].dropna()
            fig2 = px.histogram(conf, nbins=20,
                                color_discrete_sequence=["#5b4fcf"],
                                title="Offline Verifier Confidence Distribution")
            fig2.update_layout(xaxis_title="Confidence", yaxis_title="Count")
            _pb = _subsection("Offline Verifier Confidence", _plot_html(fig2, 320),
                "Confidence scores from offline rule-based verifier. Scores below 0.70 rejected.")
        else:
            _pb = ""
    except Exception as e:
        _pb = _subsection("Confidence", _skip(str(e)))
    parts.append(_pair_plots(_pa, _pb))

    try:
        if df is not None and "evidence_type" in df.columns:
            et = df["evidence_type"].value_counts().reset_index()
            et.columns = ["type","count"]
            cmap = {"efficacy":"#2ea87e","resistance":"#f05a28",
                    "review":"#5b4fcf","background":"#7a8799"}
            fig3 = px.bar(et, x="type", y="count", color="type",
                          color_discrete_map=cmap,
                          title="Evidence Type Distribution (Verified)")
            parts.append(_subsection("Evidence Types", _plot_html(fig3),
                "Classification of verified evidence rows by type."))
    except Exception as e:
        parts.append(_subsection("Evidence Types", _skip(str(e))))

    try:
        if df is not None and "biomarker" in df.columns:
            tg = df["biomarker"].value_counts().head(15).reset_index()
            tg.columns = ["gene","count"]
            fig4 = px.bar(tg, x="count", y="gene", orientation="h",
                          color_discrete_sequence=["#1e2d3d"],
                          title="Top Genes by Evidence Count")
            fig4.update_yaxes(categoryorder="total ascending")
            _pa = _subsection("Top Genes", _plot_html(fig4, 360),
                "Genes with most verified evidence rows.")
    except Exception as e:
        _pa = _subsection("Top Genes", _skip(str(e)))

    try:
        if df is not None and "drug_primary" in df.columns:
            td = df["drug_primary"].value_counts().head(15).reset_index()
            td.columns = ["drug","count"]
            fig5 = px.bar(td, x="count", y="drug", orientation="h",
                          color_discrete_sequence=["#00a8a8"],
                          title="Top Drugs by Evidence Count")
            fig5.update_yaxes(categoryorder="total ascending")
            _pb = _subsection("Top Drugs", _plot_html(fig5, 360),
                "Drugs with most verified evidence rows.")
        else:
            _pb = ""
    except Exception as e:
        _pb = _subsection("Top Drugs", _skip(str(e)))
    parts.append(_pair_plots(_pa, _pb))

    try:
        if df is not None and "year" in df.columns:
            yr = df["year"].dropna().astype(int).value_counts().sort_index().reset_index()
            yr.columns = ["year","count"]
            fig6 = px.bar(yr, x="year", y="count",
                          color_discrete_sequence=["#5b4fcf"],
                          title="Publication Year Distribution")
            parts.append(_subsection("Publication Year", _plot_html(fig6),
                "Distribution of verified evidence by publication year."))
    except Exception as e:
        parts.append(_subsection("Year", _skip(str(e))))

    return _section("Evidence Extraction & Verification", "".join(parts), "evidence")


def _build_matrix(rows_ver):
    if not HAS_PLOTLY or rows_ver is None:
        return _section("Gene-Drug Matrix", _skip("data not available"), "matrix")
    parts = []
    try:
        if "biomarker" not in rows_ver.columns or "drug_primary" not in rows_ver.columns:
            return _section("Gene-Drug Matrix", _skip("missing columns"), "matrix")
        tg = rows_ver["biomarker"].value_counts().head(12).index.tolist()
        td = rows_ver["drug_primary"].value_counts().head(12).index.tolist()
        sub = rows_ver[rows_ver["biomarker"].isin(tg) & rows_ver["drug_primary"].isin(td)]
        mat = sub.groupby(["biomarker","drug_primary"]).size().unstack(fill_value=0)
        fig = px.imshow(mat, color_continuous_scale="Blues",
                        title="Gene &times; Drug Evidence Count (Verified)",
                        labels=dict(x="Drug", y="Gene", color="Count"))
        parts.append(_subsection("Evidence Heatmap", _plot_html(fig, 480),
            "Number of verified evidence rows per gene-drug pair."))

        if "resistance_evidence" in sub.columns:
            sub_r = sub[sub["resistance_evidence"]=="yes"]
            if len(sub_r) > 0:
                mat_r = sub_r.groupby(["biomarker","drug_primary"]).size().unstack(fill_value=0)
                fig2 = px.imshow(mat_r, color_continuous_scale="Reds",
                                 title="Gene &times; Drug Resistance Evidence",
                                 labels=dict(x="Drug", y="Gene", color="Count"))
                parts.append(_subsection("Resistance Heatmap", _plot_html(fig2, 480),
                    "Direct resistance evidence rows per gene-drug pair."))
    except Exception as e:
        parts.append(_skip(str(e)))
    return _section("Gene&ndash;Drug Evidence Matrix", "".join(parts), "matrix")


def _build_temporal(trend_ver, profile_ver):
    if not HAS_PLOTLY:
        return _section("Temporal", _skip("plotly not available"), "temporal")
    parts = []

    # A. Trend category distribution &mdash; horizontal bar, color coded
    try:
        if trend_ver is not None and "temporal_trend" in trend_ver.columns:
            tc = trend_ver["temporal_trend"].value_counts().reset_index()
            tc.columns = ["trend","count"]
            tc = tc.sort_values("count", ascending=True)
            colors = [TREND_COLORS.get(t, "#7a8799") for t in tc["trend"]]
            fig = go.Figure(go.Bar(
                y=tc["trend"], x=tc["count"],
                orientation="h",
                marker_color=colors,
                text=tc["count"], textposition="outside",
            ))
            fig.update_layout(
                title="Temporal Trend Classification of Gene-Drug-Cancer Triplets",
                xaxis_title="Number of Triplets",
                yaxis_title="",
                showlegend=False,
            )
            parts.append(_subsection("Temporal Trend Distribution",
                _plot_html(fig, 380),
                "Each bar = number of gene-drug-cancer triplets in that trend category. "
                "rising_resistance = resistance evidence growing over time. "
                "declining_efficacy = efficacy evidence shrinking. "
                "stable = consistent evidence. emerging = recent drug with few papers."))
    except Exception as e:
        parts.append(_subsection("Trend Distribution", _skip(str(e))))

    # B. Top triplets ranked by paper count &mdash; horizontal bar
    try:
        if trend_ver is not None and len(trend_ver) > 0 and "total_papers" in trend_ver.columns:
            top = trend_ver.sort_values("total_papers", ascending=False).head(15).copy()
            # Build label: gene + drug + cancer
            cc = _safe_col(top, "canonical_cancer_type", "cancer_type")
            if cc:
                top["triplet"] = (top["biomarker"].astype(str) + " + " +
                                   top["drug_primary"].astype(str) + " (" +
                                   top[cc].astype(str).str.split(";").str[0].str.strip() + ")")
            else:
                top["triplet"] = top["biomarker"].astype(str) + " + " + top["drug_primary"].astype(str)

            top = top.sort_values("total_papers", ascending=True)
            trend_colors_list = [TREND_COLORS.get(t, "#7a8799")
                                  for t in top["temporal_trend"]]

            fig2 = go.Figure(go.Bar(
                y=top["triplet"],
                x=top["total_papers"],
                orientation="h",
                marker_color=trend_colors_list,
                text=top["temporal_trend"],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Total papers: %{x}<br>"
                    "Trend: %{text}<extra></extra>"
                ),
            ))
            fig2.update_layout(
                title="Top Triplets by Evidence Volume (color = trend)",
                xaxis_title="Total Evidence Papers",
                yaxis_title="",
                showlegend=False,
            )
            parts.append(_subsection("Top Gene-Drug-Cancer Triplets by Evidence Count",
                _plot_html(fig2, 500),
                "Top triplets ranked by number of supporting papers. "
                "Color indicates temporal trend direction. "
                "Hover for details."))
    except Exception as e:
        parts.append(_subsection("Top Triplets", _skip(str(e))))

    # C. SINGLE MULTI-LINE PLOT &mdash; all triplets in one chart
    # Each triplet gets its own color pair (efficacy=solid, resistance=dashed)
    # No subplots &mdash; no overlap &mdash; clean and readable
    try:
        if profile_ver is not None and len(profile_ver) > 0:
            yb  = _safe_col(profile_ver, "year_bin")
            ep  = _safe_col(profile_ver, "efficacy_papers")
            rp  = _safe_col(profile_ver, "resistance_papers")
            np_ = _safe_col(profile_ver, "n_papers")

            if yb and ep and rp and "biomarker" in profile_ver.columns:
                cc2      = _safe_col(profile_ver, "canonical_cancer_type","cancer_type")
                grp_cols = ["biomarker","drug_primary"]
                if cc2:
                    grp_cols.append(cc2)

                totals = profile_ver.groupby(grp_cols)[np_ or ep].sum().reset_index()
                totals.columns = grp_cols + ["total"]
                top6 = totals.sort_values("total", ascending=False).head(6)

                bin_order = ["pre-2010","2010-2014","2015-2017",
                              "2018-2020","2021-2023","2024+"]

                # Color palette &mdash; one color per triplet
                PALETTE = [
                    "#5b4fcf","#f05a28","#00a8a8",
                    "#c0398e","#e8a020","#2ea87e",
                ]

                # &#9472;&#9472; Plot 1: Efficacy paper counts &mdash; all triplets &#9472;&#9472;
                fig_eff = go.Figure()
                for idx, (_, trow) in enumerate(top6.iterrows()):
                    mask = ((profile_ver["biomarker"] == trow["biomarker"]) &
                            (profile_ver["drug_primary"] == trow["drug_primary"]))
                    if cc2 and cc2 in trow.index:
                        mask = mask & (profile_ver[cc2] == trow[cc2])
                    sub = profile_ver[mask].copy()
                    if len(sub) == 0:
                        continue
                    sub["_ord"] = sub[yb].apply(
                        lambda x: bin_order.index(x) if x in bin_order else 99
                    )
                    sub = sub.sort_values("_ord")
                    label = f"{trow['biomarker']} + {trow['drug_primary']}"
                    color = PALETTE[idx % len(PALETTE)]

                    fig_eff.add_trace(go.Scatter(
                        name=label,
                        x=sub[yb], y=sub[ep],
                        mode="lines+markers",
                        line=dict(color=color, width=2.5),
                        marker=dict(size=7, symbol="circle"),
                        hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Efficacy papers: %{{y}}<extra></extra>",
                    ))

                fig_eff.update_layout(
                    title="Efficacy Papers Over Time &mdash; Top Gene-Drug Pairs",
                    xaxis_title="Year Bin",
                    yaxis_title="Number of Efficacy Papers",
                    yaxis=dict(rangemode="tozero"),
                    legend=dict(
                        orientation="v",
                        x=1.02, y=1,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#e2e8f0",
                        borderwidth=1,
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                fig_eff.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                fig_eff.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

                parts.append(_subsection(
                    "Efficacy Evidence Trajectory",
                    _plot_html(fig_eff, 420),
                    "Efficacy paper counts per year bin for top gene-drug pairs. "
                    "Each color = one gene-drug pair. "
                    "Declining lines = fewer efficacy papers being published recently."))

                # &#9472;&#9472; Plot 2: Resistance paper counts &mdash; all triplets &#9472;&#9472;
                fig_res = go.Figure()
                for idx, (_, trow) in enumerate(top6.iterrows()):
                    mask = ((profile_ver["biomarker"] == trow["biomarker"]) &
                            (profile_ver["drug_primary"] == trow["drug_primary"]))
                    if cc2 and cc2 in trow.index:
                        mask = mask & (profile_ver[cc2] == trow[cc2])
                    sub = profile_ver[mask].copy()
                    if len(sub) == 0:
                        continue
                    sub["_ord"] = sub[yb].apply(
                        lambda x: bin_order.index(x) if x in bin_order else 99
                    )
                    sub = sub.sort_values("_ord")
                    label = f"{trow['biomarker']} + {trow['drug_primary']}"
                    color = PALETTE[idx % len(PALETTE)]

                    fig_res.add_trace(go.Scatter(
                        name=label,
                        x=sub[yb], y=sub[rp],
                        mode="lines+markers",
                        line=dict(color=color, width=2.5, dash="dash"),
                        marker=dict(size=7, symbol="diamond"),
                        hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Resistance papers: %{{y}}<extra></extra>",
                    ))

                fig_res.update_layout(
                    title="Resistance Papers Over Time &mdash; Top Gene-Drug Pairs",
                    xaxis_title="Year Bin",
                    yaxis_title="Number of Resistance Papers",
                    yaxis=dict(rangemode="tozero"),
                    legend=dict(
                        orientation="v",
                        x=1.02, y=1,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#e2e8f0",
                        borderwidth=1,
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                fig_res.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                fig_res.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

                parts.append(_subsection(
                    "Resistance Evidence Trajectory",
                    _plot_html(fig_res, 420),
                    "Resistance paper counts per year bin for top gene-drug pairs. "
                    "Dashed lines + diamonds = resistance evidence. "
                    "Rising lines = growing resistance signal. "
                    "Same color as efficacy plot = same gene-drug pair."))

                # &#9472;&#9472; Plot 3: Combined efficacy vs resistance for TOP triplet only &#9472;&#9472;
                if len(top6) > 0:
                    trow = top6.iloc[0]
                    mask = ((profile_ver["biomarker"] == trow["biomarker"]) &
                            (profile_ver["drug_primary"] == trow["drug_primary"]))
                    sub = profile_ver[mask].copy()
                    sub["_ord"] = sub[yb].apply(
                        lambda x: bin_order.index(x) if x in bin_order else 99
                    )
                    sub = sub.sort_values("_ord")
                    label = f"{trow['biomarker']} + {trow['drug_primary']}"

                    fig_comb = go.Figure()
                    fig_comb.add_trace(go.Scatter(
                        name="Efficacy papers",
                        x=sub[yb], y=sub[ep],
                        mode="lines+markers",
                        line=dict(color="#2ea87e", width=3),
                        marker=dict(size=9, symbol="circle"),
                        fill="tozeroy",
                        fillcolor="rgba(46,168,126,0.10)",
                        hovertemplate="%{x}<br>Efficacy: %{y} papers<extra></extra>",
                    ))
                    fig_comb.add_trace(go.Scatter(
                        name="Resistance papers",
                        x=sub[yb], y=sub[rp],
                        mode="lines+markers",
                        line=dict(color="#f05a28", width=3),
                        marker=dict(size=9, symbol="diamond"),
                        fill="tozeroy",
                        fillcolor="rgba(240,90,40,0.10)",
                        hovertemplate="%{x}<br>Resistance: %{y} papers<extra></extra>",
                    ))
                    fig_comb.update_layout(
                        title=f"Evidence Balance: {label}",
                        xaxis_title="Year Bin",
                        yaxis_title="Papers",
                        yaxis=dict(rangemode="tozero"),
                        legend=dict(orientation="h", y=1.12),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                    )
                    fig_comb.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
                    fig_comb.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

                    parts.append(_subsection(
                        f"Evidence Balance Spotlight: {label}",
                        _plot_html(fig_comb, 400),
                        f"Detailed view of efficacy vs resistance evidence over time for {label}. "
                        "Green area = efficacy papers. Red area = resistance papers. "
                        "When red rises and green falls = resistance emergence pattern."))

    except Exception as e:
        parts.append(_subsection("Evidence Trajectory", _skip(str(e))))

    # D. Resistance emergence heatmap &mdash; all triplets &times; year bins
    try:
        if profile_ver is not None and len(profile_ver) > 0:
            yb = _safe_col(profile_ver, "year_bin")
            rp = _safe_col(profile_ver, "resistance_papers")
            if yb and rp and "biomarker" in profile_ver.columns:
                cc3 = _safe_col(profile_ver, "canonical_cancer_type","cancer_type")
                profile_ver2 = profile_ver.copy()
                if cc3:
                    profile_ver2["triplet"] = (
                        profile_ver2["biomarker"] + " + " +
                        profile_ver2["drug_primary"] + " (" +
                        profile_ver2[cc3].str.split(";").str[0].str.strip() + ")"
                    )
                else:
                    profile_ver2["triplet"] = (
                        profile_ver2["biomarker"] + " + " +
                        profile_ver2["drug_primary"]
                    )

                # Only top 15 triplets
                top_trip = (profile_ver2.groupby("triplet")[rp].sum()
                            .sort_values(ascending=False).head(15).index.tolist())
                sub2 = profile_ver2[profile_ver2["triplet"].isin(top_trip)]

                pivot = sub2.pivot_table(
                    index="triplet", columns=yb,
                    values=rp, aggfunc="sum", fill_value=0
                )

                # Sort columns by year bin order
                bin_order = ["pre-2010","2010-2014","2015-2017",
                              "2018-2020","2021-2023","2024+"]
                ordered_cols = [c for c in bin_order if c in pivot.columns]
                pivot = pivot[ordered_cols]

                fig4 = px.imshow(
                    pivot,
                    color_continuous_scale="Reds",
                    title="Resistance Paper Count Heatmap by Year Bin",
                    labels=dict(x="Year Bin", y="Triplet", color="Resistance Papers"),
                    aspect="auto",
                )
                fig4.update_xaxes(side="bottom")
                parts.append(_subsection(
                    "Resistance Evidence Heatmap (Triplets &times; Year Bins)",
                    _plot_html(fig4, 500),
                    "Each cell = number of resistance papers for that triplet in that year. "
                    "Darker red = more resistance evidence. "
                    "Shows which triplets have rising resistance over time."))
    except Exception as e:
        parts.append(_subsection("Resistance Heatmap", _skip(str(e))))

    # E. Temporal summary table
    try:
        if trend_ver is not None and len(trend_ver) > 0:
            # Show shares only in main table &mdash; ratios in download only
            # Shares sum to 1.0 &mdash; directly interpretable
            main_cols = [c for c in [
                "biomarker","drug_primary","canonical_cancer_type",
                "temporal_trend","total_papers",
                "early_efficacy_share","latest_efficacy_share",
                "early_resistance_share","latest_resistance_share",
                "peak_evidence_bin","resistance_emergence_bin",
            ] if c in trend_ver.columns]
            top_t = (trend_ver.sort_values("total_papers", ascending=False).head(30)[main_cols]
                     if "total_papers" in trend_ver.columns else trend_ver[main_cols].head(30))
            caption = ("efficacy_share + resistance_share = 1.0 per time bin. "
                       "early = first year bin with data; latest = most recent. "
                       "Old ratio columns (prevalence) available in Excel output.")
            parts.append(_subsection("Temporal Summary Table",
                _table_html(top_t, caption=caption),
                "Shares are directional fractions summing to 1.0 &mdash; more interpretable than ratios. "
                "Full data including prevalence ratios available in TrendSummary_Verified sheet."))
    except Exception as e:
        parts.append(_subsection("Temporal Table", _skip(str(e))))

    return _section("Temporal Evidence Tracking", "".join(parts), "temporal")


def _build_contradiction(contra_ver):
    if not HAS_PLOTLY or contra_ver is None:
        return _section("Contradictions", _skip("data not available"), "contradictions")
    parts = []

    try:
        if "contradiction_flag" in contra_ver.columns:
            cf = contra_ver["contradiction_flag"].value_counts().reset_index()
            cf.columns = ["flag","count"]
            colors = [FLAG_COLORS.get(f, "#7a8799") for f in cf["flag"]]
            fig = go.Figure(go.Bar(x=cf["flag"], y=cf["count"],
                                   marker_color=colors,
                                   text=cf["count"], textposition="outside"))
            fig.update_layout(title="Contradiction Flag Distribution")
            parts.append(_subsection("Flag Distribution", _plot_html(fig),
                "none=consistent; watch=emerging; caution=moderate; conflict=strong"))
    except Exception as e:
        parts.append(_subsection("Flags", _skip(str(e))))

    try:
        if all(c in contra_ver.columns for c in ["efficacy_papers","resistance_papers","contradiction_flag"]):
            hov = [c for c in ["biomarker","drug_primary","canonical_cancer_type","conflict_score"] if c in contra_ver.columns]
            fig2 = px.scatter(contra_ver,
                x="efficacy_papers", y="resistance_papers",
                color="contradiction_flag",
                color_discrete_map=FLAG_COLORS,
                hover_data=hov,
                title="Efficacy vs Resistance per Triplet")
            parts.append(_subsection("Contradiction Scatter", _plot_html(fig2),
                "Each point = one gene-drug-cancer triplet. Top-right = high conflict."))
    except Exception as e:
        parts.append(_subsection("Scatter", _skip(str(e))))

    try:
        flagged = contra_ver[contra_ver["contradiction_flag"].isin(["watch","caution","conflict"])] if "contradiction_flag" in contra_ver.columns else pd.DataFrame()
        if len(flagged) > 0:
            cols = [c for c in ["biomarker","drug_primary","canonical_cancer_type",
                                  "contradiction_flag","conflict_score",
                                  "efficacy_papers","resistance_papers"] if c in flagged.columns]
            parts.append(_subsection("Flagged Triplets",
                _table_html(flagged[cols].sort_values("conflict_score", ascending=False) if "conflict_score" in flagged.columns else flagged[cols])))
        else:
            parts.append(_subsection("Flagged Triplets",
                '<p class="skip-note">No contradiction signals in verified rows.</p>'))
    except Exception as e:
        parts.append(_subsection("Flagged Table", _skip(str(e))))

    return _section("Contradiction Detection", "".join(parts), "contradictions")


def _build_trials(trials_df):
    if not HAS_PLOTLY or trials_df is None or len(trials_df) == 0:
        return _section("ClinicalTrials", _skip("no data"), "trials")
    parts = []

    try:
        if "n_trials" in trials_df.columns:
            td = trials_df.copy()
            drug_col = "drug" if "drug" in td.columns else "drug_primary"
            td["pair"] = td.get(drug_col, pd.Series(["?"]*(len(td)))).astype(str) + " + " + td["cancer_type"].astype(str)
            td = td.sort_values("n_trials", ascending=False).head(15) if "n_trials" in td.columns else td.head(15)
            fig = px.bar(td, x="n_trials", y="pair", orientation="h",
                         color_discrete_sequence=["#1e2d3d"],
                         title="Trial Count by Drug-Cancer Pair")
            fig.update_yaxes(categoryorder="total ascending")
            parts.append(_subsection("Trial Count", _plot_html(fig),
                "Number of ClinicalTrials.gov entries per drug-cancer pair."))
    except Exception as e:
        parts.append(_subsection("Trial Count", _skip(str(e))))

    try:
        if "n_failed_trials" in trials_df.columns:
            td2 = trials_df.copy()
            drug_col = "drug" if "drug" in td2.columns else "drug_primary"
            td2["pair"] = td2.get(drug_col, pd.Series(["?"]*(len(td2)))).astype(str) + " + " + td2["cancer_type"].astype(str)
            td2 = td2.sort_values("n_failed_trials", ascending=False).head(15)
            fig2 = px.bar(td2, x="n_failed_trials", y="pair", orientation="h",
                          color_discrete_sequence=["#f05a28"],
                          title="Negative Clinical Development Signals")
            fig2.update_yaxes(categoryorder="total ascending")
            parts.append(_subsection("Trial Status: Terminated / Withdrawn / Suspended",
                _plot_html(fig2),
                "Terminated/withdrawn/suspended status is NOT equivalent to therapeutic failure or molecular resistance. Cross-check with literature resistance ratio."))
    except Exception as e:
        parts.append(_subsection("Failed Trials", _skip(str(e))))

    try:
        if "highest_trial_phase" in trials_df.columns:
            ph = trials_df["highest_trial_phase"].value_counts().reset_index()
            ph.columns = ["phase","count"]
            fig3 = px.pie(ph, names="phase", values="count",
                          title="Highest Trial Phase Distribution")
            parts.append(_subsection("Trial Phase", _plot_html(fig3, 350),
                "Highest clinical development phase reached."))
    except Exception as e:
        parts.append(_subsection("Phase", _skip(str(e))))

    try:
        # Detect available columns flexibly
        preferred_cols = [
            "drug","cancer_type","top_biomarker_in_query",
            "n_trials_display","result_capped",
            "n_active_trials","n_completed_trials",
            "n_terminated_withdrawn",
            "highest_trial_phase","top_trial_nct",
            "provisional_trial_literature_alignment",
            "concordance_note",
            "literature_resistance_ratio",
        ]
        cols = [c for c in preferred_cols if c in trials_df.columns]
        if not cols:
            cols = list(trials_df.columns)
        # Sort by best available trial count column
        sort_col = _safe_col(trials_df,
            "n_trials","n_trials_display","n_terminated_withdrawn")
        try:
            tbl = trials_df[cols].copy()
            if sort_col and sort_col in tbl.columns:
                tbl = tbl.sort_values(sort_col, ascending=False)
        except Exception:
            tbl = trials_df[cols].copy()
        parts.append(_subsection("Trials Summary Table",
            _table_html(tbl),
            "&ge;200 = result capped at API maximum &mdash; includes off-target trials. "
            "n_terminated_withdrawn: termination &#8800; therapeutic failure or molecular resistance. "
            "Capped/broad results are marked manual_review_required under "
            "provisional_trial_literature_alignment."))
    except Exception as e:
        parts.append(_subsection("Trials Table", _skip(str(e))))

    return _section("ClinicalTrials.gov Evidence", "".join(parts), "trials")


def _build_graph_section(nodes_df, edges_df, rows_verified=None):
    if not HAS_PLOTLY:
        return _section("Knowledge Graph", _skip("plotly not available"), "graph")
    if nodes_df is None or edges_df is None:
        return _section("Knowledge Graph", _skip("graph files not provided"), "graph")
    parts = []

    try:
        if "node_type" in nodes_df.columns:
            nt = nodes_df["node_type"].value_counts().reset_index()
            nt.columns = ["type","count"]
            colors = [NODE_COLORS.get(t, "#7a8799") for t in nt["type"]]
            fig = go.Figure(go.Bar(x=nt["type"], y=nt["count"],
                                   marker_color=colors,
                                   text=nt["count"], textposition="outside"))
            fig.update_layout(title="Graph Node Types")
            _pa4 = _subsection("Node Types", _plot_html(fig, 320),
                "Evidence and Publication are first-class nodes enabling provenance-aware queries.")
    except Exception as e:
        _pa4 = _subsection("Node Types", _skip(str(e)))

    try:
        if "edge_type" in edges_df.columns:
            et = edges_df["edge_type"].value_counts().reset_index()
            et.columns = ["type","count"]
            fig2 = px.bar(et, x="count", y="type", orientation="h",
                          color_discrete_sequence=["#1e2d3d"],
                          title="Graph Edge Types")
            fig2.update_yaxes(categoryorder="total ascending")
            _pb4 = _subsection("Edge Types", _plot_html(fig2, 320),
                "ASSOCIATED_WITH and EVALUATED_IN: conservative labels &mdash; literature mining does not prove expression or treatment.")
        else:
            _pb4 = ""
    except Exception as e:
        _pb4 = _subsection("Edge Types", _skip(str(e)))
    parts.append(_pair_plots(_pa4, _pb4))

    try:
        source_col = _safe_col(edges_df, "source", "Source")
        target_col = _safe_col(edges_df, "target", "Target")
        node_col   = _safe_col(nodes_df, "node_id", "id", "Id")
        label_col  = _safe_col(nodes_df, "label", "name", "Label")

        if source_col and target_col and node_col:
            deg_series = pd.concat([
                edges_df[source_col].astype(str),
                edges_df[target_col].astype(str)
            ]).value_counts()
            deg = deg_series.reset_index()
            deg.columns = [node_col, "degree"]
            meta_cols = [c for c in [node_col, "node_type", label_col] if c and c in nodes_df.columns]
            top = deg.merge(nodes_df[meta_cols], on=node_col, how="left").head(20)
            ylabels = top[label_col] if label_col and label_col in top.columns else top[node_col]
            colors  = [NODE_COLORS.get(t, "#7a8799")
                       for t in top.get("node_type", pd.Series(["Unknown"]*len(top))).fillna("Unknown")]
            fig3 = go.Figure(go.Bar(x=top["degree"], y=ylabels,
                                    orientation="h", marker_color=colors))
            fig3.update_layout(title="Top 20 Nodes by Degree", xaxis_title="Degree")
            fig3.update_yaxes(categoryorder="total ascending")
            parts.append(_subsection("Top Nodes by Degree", _plot_html(fig3, 500),
                "Degree = total incoming + outgoing edges. Color = node type."))
    except Exception as e:
        parts.append(_subsection("Top Nodes", _skip(str(e))))

    # Top genes by degree
    try:
        if "node_type" in nodes_df.columns and "node_id" in nodes_df.columns:
            source_col = _safe_col(edges_df, "source")
            target_col = _safe_col(edges_df, "target")
            if source_col and target_col:
                deg_all = pd.concat([
                    edges_df[source_col].astype(str),
                    edges_df[target_col].astype(str)
                ]).value_counts().reset_index()
                deg_all.columns = ["node_id","degree"]
                merged = deg_all.merge(nodes_df[["node_id","node_type","label"]]
                    if "label" in nodes_df.columns
                    else nodes_df[["node_id","node_type"]], on="node_id", how="left")

                for ntype, color, title_str in [
                    ("Gene",  "#5b4fcf", "Top Genes by Degree"),
                    ("Drug",  "#00a8a8", "Top Drugs by Degree"),
                    ("Cancer","#f05a28", "Top Cancer Nodes by Degree"),
                ]:
                    sub_n = merged[merged["node_type"]==ntype].head(15)
                    if len(sub_n) == 0:
                        continue
                    sub_n = sub_n.sort_values("degree", ascending=True)
                    ylabel = sub_n["label"] if "label" in sub_n.columns else sub_n["node_id"]
                    fig_n = go.Figure(go.Bar(
                        x=sub_n["degree"], y=ylabel,
                        orientation="h",
                        marker_color=color,
                        text=sub_n["degree"], textposition="outside",
                    ))
                    fig_n.update_layout(
                        title=title_str,
                        xaxis_title="Degree (connections)",
                    )
                    fig_n.update_yaxes(categoryorder="total ascending")
                    parts.append(_subsection(title_str, _plot_html(fig_n, 380),
                        f"Top {ntype} nodes by total edge connections in the knowledge graph."))
    except Exception as e:
        parts.append(_subsection("Degree Plots", _skip(str(e))))

    # Source-target node type heatmap
    try:
        if "node_type" in nodes_df.columns:
            source_col = _safe_col(edges_df, "source")
            target_col = _safe_col(edges_df, "target")
            et_col     = _safe_col(edges_df, "edge_type")
            if source_col and target_col:
                etype_map = nodes_df.set_index("node_id")["node_type"].to_dict()
                edges_df2 = edges_df.copy()
                edges_df2["src_type"] = edges_df2[source_col].astype(str).map(etype_map).fillna("Unknown")
                edges_df2["tgt_type"] = edges_df2[target_col].astype(str).map(etype_map).fillna("Unknown")
                matrix = edges_df2.groupby(["src_type","tgt_type"]).size().unstack(fill_value=0)
                fig_m = px.imshow(matrix,
                    color_continuous_scale="Blues",
                    title="Edge Source&rarr;Target Node Type Matrix",
                    labels=dict(x="Target Node Type", y="Source Node Type", color="Edge Count"),
                    text_auto=True,
                )
                parts.append(_subsection("Source&rarr;Target Node Type Heatmap",
                    _plot_html(fig_m, 400),
                    "Number of edges between each pair of node types. "
                    "Shows graph connectivity structure."))
    except Exception as e:
        parts.append(_subsection("Node Type Heatmap", _skip(str(e))))

    # Evidence quality ranking &mdash; meaningful scoring not degree
    try:
        if "node_type" in nodes_df.columns:
            ev_nodes = nodes_df[nodes_df["node_type"]=="Evidence"].copy()
            if len(ev_nodes) > 0:
                # Study design weights
                DESIGN_WEIGHTS = {
                    "rct": 4.0, "phase 3": 4.0, "phase iii": 4.0,
                    "trial": 3.0, "phase 2": 3.0, "phase ii": 3.0,
                    "observational": 2.0, "cohort": 2.0,
                    "case_report": 1.0, "case report": 1.0,
                    "preclinical": 0.5, "in_vitro": 0.3, "in vitro": 0.3,
                }

                def _ev_quality(row):
                    score = 0.0
                    # Verification confidence
                    try:
                        score += float(row.get("llm_confidence",0) or 0) * 1.5
                    except Exception:
                        pass
                    # Study design
                    sd = str(row.get("study_design","") or "").lower()
                    score += DESIGN_WEIGHTS.get(sd, 0.5)
                    # Recency
                    try:
                        yr = int(row.get("year",0) or 0)
                        if yr >= 2022: score += 1.0
                        elif yr >= 2020: score += 0.7
                        elif yr >= 2018: score += 0.4
                    except Exception:
                        pass
                    # Resistance evidence bonus
                    if str(row.get("resistance_evidence","")).lower() == "yes":
                        score += 0.5
                    # Contradiction penalty
                    cf = str(row.get("contradiction_flag","")).lower()
                    if cf == "conflict": score -= 1.0
                    elif cf == "caution": score -= 0.5
                    elif cf == "watch": score -= 0.2
                    return round(score, 3)

                ev_nodes["evidence_quality_score"] = ev_nodes.apply(_ev_quality, axis=1)
                ev_ranked = ev_nodes.sort_values("evidence_quality_score", ascending=False)

                # Clean up numeric columns &mdash; remove .0 suffix
                for num_col in ["pmid","year","pmcid"]:
                    if num_col in ev_ranked.columns:
                        ev_ranked[num_col] = (
                            pd.to_numeric(ev_ranked[num_col], errors="coerce")
                            .astype("Int64")
                            .astype(str)
                            .replace("<NA>","")
                            .replace("nan","")
                        )

                # Add rank column
                ev_ranked = ev_ranked.reset_index(drop=True)
                ev_ranked.insert(0, "rank", ev_ranked.index + 1)

                # Replace pmid with clickable link
                if "pmid" in ev_ranked.columns:
                    ev_ranked["pmid"] = ev_ranked["pmid"].apply(
                        lambda p: (f'<a href="https://pubmed.ncbi.nlm.nih.gov/{p}" '
                                   f'target="_blank">{p}</a>')
                        if str(p).strip() not in ("","nan","<NA>","None") else ""
                    )

                # Evidence nodes now carry full metadata from graph.py
                # No merge needed &mdash; biomarker/drug/cancer/pmid already in nodes CSV
                # If still missing (old graph file), merge from rows_verified
                missing_identity = not any(c in ev_ranked.columns and
                                           ev_ranked[c].fillna("").ne("").any()
                                           for c in ["biomarker","drug_primary"])
                if missing_identity:
                    rv = rows_verified if rows_verified is not None and len(rows_verified) > 0 else pd.DataFrame()
                    if len(rv) > 0:
                        pub_cols = [c for c in ["biomarker","alteration",
                                                 "drug_primary","canonical_cancer_type",
                                                 "pmid","journal","year","study_design",
                                                 "final_evidence_type","resistance_direction",
                                                 "temporal_trend","contradiction_flag"]
                                    if c in rv.columns]
                        rv_tmp = rv[pub_cols].copy() if pub_cols else pd.DataFrame()
                        if len(rv_tmp) > 0 and "pmid" in rv_tmp.columns and "pmid" in ev_ranked.columns:
                            ev_ranked = ev_ranked.merge(
                                rv_tmp.drop_duplicates("pmid"),
                                on="pmid", how="left",
                                suffixes=("","_rv")
                            )
                            # Prefer _rv columns if original blank
                            for col in ["biomarker","drug_primary","canonical_cancer_type",
                                        "alteration","journal","study_design","final_evidence_type"]:
                                if f"{col}_rv" in ev_ranked.columns:
                                    ev_ranked[col] = ev_ranked[col].fillna("").replace("",pd.NA)
                                    ev_ranked[col] = ev_ranked[col].fillna(ev_ranked[f"{col}_rv"])
                                    ev_ranked.drop(columns=[f"{col}_rv"], inplace=True, errors="ignore")

                show_cols = [c for c in [
                    "rank",
                    "biomarker","alteration","drug_primary","canonical_cancer_type",
                    "evidence_quality_score",
                    "final_evidence_type","evidence_type_raw",
                    "resistance_direction",
                    "study_design","year","pmid",
                    "journal","temporal_trend","contradiction_flag",
                ] if c in ev_ranked.columns]

                score_note = (
                    "Evidence quality score = study_design_weight (RCT=4, Trial=3, Obs=2, Case=1, Preclinical=0.5) "
                    "+ verifier_confidence&times;1.5 + recency_bonus (2022+=1.0) "
                    "+ resistance_bonus - contradiction_penalty. "
                    "This is a literature prioritization score &mdash; NOT a clinical actionability score."
                )
                parts.append(_subsection("Evidence Quality Ranking",
                    _table_html(ev_ranked[show_cols].head(50),
                                caption=score_note),
                    "Top 50 evidence rows by quality score. "
                    "Shows gene-drug-cancer identity alongside score components. "
                    "PMID links open PubMed."))
    except Exception as e:
        parts.append(_subsection("Evidence Quality Ranking", _skip(str(e))))

    # Show only key node types in main report &mdash; full data in CSV
    try:
        gene_nodes_t   = nodes_df[nodes_df["node_type"]=="Gene"]   if "node_type" in nodes_df.columns else nodes_df
        drug_nodes_t   = nodes_df[nodes_df["node_type"]=="Drug"]   if "node_type" in nodes_df.columns else pd.DataFrame()
        cancer_nodes_t = nodes_df[nodes_df["node_type"]=="Cancer"] if "node_type" in nodes_df.columns else pd.DataFrame()
        ev_nodes_t     = nodes_df[nodes_df["node_type"]=="Evidence"].head(50) if "node_type" in nodes_df.columns else pd.DataFrame()

        show_gene_cols = [c for c in ["label","node_type","gene_type"] if c in gene_nodes_t.columns]
        show_drug_cols = [c for c in ["label","node_type","therapy_type","accessibility"] if c in drug_nodes_t.columns]
        show_can_cols  = [c for c in ["label","node_type","histology","raw_cancer_type"] if c in cancer_nodes_t.columns]

        parts.append(_subsection("Gene Nodes",
            _table_html(gene_nodes_t[show_gene_cols] if show_gene_cols else gene_nodes_t,
                        caption="Gene nodes in knowledge graph")))
        if len(drug_nodes_t) > 0:
            parts.append(_subsection("Drug Nodes",
                _table_html(drug_nodes_t[show_drug_cols] if show_drug_cols else drug_nodes_t,
                            caption="Drug nodes")))
        if len(cancer_nodes_t) > 0:
            # Shorten raw_cancer_type &mdash; only show first 60 chars
            cn_display = cancer_nodes_t.copy()
            if "raw_cancer_type" in cn_display.columns:
                cn_display["raw_cancer_type"] = (
                    cn_display["raw_cancer_type"].astype(str)
                    .str.split(";").str[0].str.strip().str[:60]
                )
            parts.append(_subsection("Cancer Nodes",
                _table_html(cn_display[show_can_cols] if show_can_cols else cn_display,
                            caption="Cancer nodes (canonical labels; raw_cancer_type truncated)")))
        parts.append(_subsection("Evidence Nodes (Top 50)",
            _table_html(ev_nodes_t, caption="Top 50 evidence nodes"),
            "Full nodes and edges available in CSV files alongside this report."))
        # Edge type summary only &mdash; not full edge table
        if "edge_type" in edges_df.columns:
            et_sum = edges_df["edge_type"].value_counts().reset_index()
            et_sum.columns = ["edge_type","count"]
            parts.append(_subsection("Edge Type Summary",
                _table_html(et_sum, caption="Edge counts by type"),
                "Full edge list available in CSV file."))
    except Exception as e:
        parts.append(_skip(str(e)))

    return _section("Knowledge Graph", "".join(parts), "graph")


def _build_tables(rows_all, rows_ver, trend_ver,
                  contra_ver, trials_df, run_info, max_rows):
    parts = []
    if rows_ver is not None:
        parts.append(_subsection("Verified Rows", _table_html(rows_ver, max_rows)))
    if trend_ver is not None:
        parts.append(_subsection("Temporal Summary", _table_html(trend_ver, max_rows)))
    if contra_ver is not None:
        parts.append(_subsection("Contradictions", _table_html(contra_ver, max_rows)))
    if trials_df is not None:
        parts.append(_subsection("ClinicalTrials", _table_html(trials_df, max_rows)))
    if run_info is not None:
        parts.append(_subsection("Run Metadata", _table_html(run_info, max_rows)))
    return _section("Data Tables", "".join(parts), "tables")


def build_html_report(
    report_path:    str,
    nodes_path:     Optional[str] = None,
    edges_path:     Optional[str] = None,
    trials_path:    Optional[str] = None,
    out_path:       str = "megamine_report.html",
    max_table_rows: int = 300,
    title:          str = "megaMine v2.0 Evidence Report",
) -> str:
    print(f"Building megaMine HTML report...")
    print(f"   Input:  {report_path}")
    print(f"   Output: {out_path}")

    print("   Loading workbook...")
    xl = pd.ExcelFile(report_path)

    def _on(df): return df if (df is not None and len(df) > 0) else None

    rows_all  = _on(_safe_load_sheet(xl, "Rows_All",  "Rows"))
    rows_ver  = _on(_safe_load_sheet(xl, "Rows_Verified"))
    prof_all  = _on(_safe_load_sheet(xl, "Temporal_All",  "TemporalProfiles"))
    prof_ver  = _on(_safe_load_sheet(xl, "Temporal_Verified"))
    trend_all = _on(_safe_load_sheet(xl, "TrendSummary_All",  "TrendSummary"))
    trend_ver = _on(_safe_load_sheet(xl, "TrendSummary_Verified"))
    cont_all  = _on(_safe_load_sheet(xl, "Contradictions_All",  "Contradictions"))
    cont_ver  = _on(_safe_load_sheet(xl, "Contradictions_Verified"))
    trials_xl = _on(_safe_load_sheet(xl, "ClinicalTrials"))
    run_info  = _on(_safe_load_sheet(xl, "RunInfo"))

    nodes_df = edges_df = None
    if nodes_path and os.path.exists(nodes_path):
        try: nodes_df = pd.read_csv(nodes_path)
        except Exception: pass
    if edges_path and os.path.exists(edges_path):
        try: edges_df = pd.read_csv(edges_path)
        except Exception: pass

    trials_df = trials_xl
    if trials_path and os.path.exists(trials_path):
        try: trials_df = pd.read_excel(trials_path)
        except Exception: pass

    print("   Building summary...")
    s1  = _build_summary(rows_all, rows_ver, _choose_df(trend_ver, trend_all),
                         _choose_df(cont_ver, cont_all), trials_df,
                         nodes_df, edges_df, run_info)
    print("   Normalization plots...")
    s3  = _build_normalization(rows_all, max_table_rows)
    print("   Resistance plots...")
    s4  = _build_resistance(rows_all, rows_ver, max_table_rows)
    print("   Evidence plots...")
    s5  = _build_evidence(rows_all, rows_ver, max_table_rows)
    print("   Gene-drug matrix...")
    s6  = _build_matrix(rows_ver)
    print("   Temporal plots...")
    s7  = _build_temporal(_choose_df(trend_ver, trend_all),
                          _choose_df(prof_ver, prof_all))
    print("   Contradiction plots...")
    s8  = _build_contradiction(_choose_df(cont_ver, cont_all))
    print("   ClinicalTrials plots...")
    s9  = _build_trials(trials_df)
    print("   Graph plots...")
    s10 = _build_graph_section(nodes_df, edges_df, rows_verified=rows_ver)
    print("   Data tables...")
    s12 = _build_tables(rows_all, rows_ver, _choose_df(trend_ver, trend_all),
                        _choose_df(cont_ver, cont_all), trials_df,
                        run_info, max_table_rows)

    body    = s1 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s12
    sidebar = _sidebar()
    header  = _header(title, run_info)
    full    = _full_html(title, header, sidebar, body)

    print("   Writing HTML...")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"   Done: {out_path} ({size_mb:.1f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="megaMine HTML Report")
    ap.add_argument("--report",   required=True)
    ap.add_argument("--nodes",    default=None)
    ap.add_argument("--edges",    default=None)
    ap.add_argument("--trials",   default=None)
    ap.add_argument("--out",      default="megamine_report.html")
    ap.add_argument("--title",    default="megaMine v2.0 Evidence Report")
    ap.add_argument("--max-rows", type=int, default=300)
    args = ap.parse_args()
    build_html_report(
        report_path    = args.report,
        nodes_path     = args.nodes,
        edges_path     = args.edges,
        trials_path    = args.trials,
        out_path       = args.out,
        max_table_rows = args.max_rows,
        title          = args.title,
    )

if __name__ == "__main__":
    main()
