"""
maf_pipeline.py — megaMine v2.0
MAF-driven evidence prioritization pipeline.

BETTER THAN IDAP:
  ✅ OncoKB actionability levels (Level 1-4 + R1/R2)
  ✅ ClinVar pathogenicity scores
  ✅ VAF-tiered prioritization (Tier1/2/3)
  ✅ megaMine literature evidence per gene
  ✅ Offline + Claude API verification
  ✅ Interactive HTML report (fully offline)
  ✅ Single CLI command for any MAF + cancer

USAGE:
  megaMine-maf --maf patient.maf --cancer NSCLC --out results/
  megaMine-maf --maf patient.maf --cancer Breast --oncokb-token TOKEN --out results/

AUTHORS: Muhammad Junaid — APML, Ajou University
"""

import os, sys, time, json, argparse, ssl, re
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional

ctx = ssl._create_unverified_context()

# ── VAF thresholds ────────────────────────────────────────────
VAF_TIER1 = 0.20
VAF_TIER2 = 0.05

# ── Actionable variant classes ────────────────────────────────
ACTIONABLE_CLASSES = {
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del",
    "Frame_Shift_Ins","Splice_Site","In_Frame_Del","In_Frame_Ins",
    "Translation_Start_Site","Nonstop_Mutation","Splice_Region",
}

# ── Known oncogenes ───────────────────────────────────────────
ONCOGENES = {
    "EGFR","KRAS","NRAS","HRAS","BRAF","MET","ALK","ROS1","RET",
    "HER2","ERBB2","PIK3CA","PTEN","AKT1","MTOR","CDK4","CDK6",
    "BRCA1","BRCA2","TP53","STK11","KEAP1","SMAD4","RB1","CDKN2A",
    "FGFR1","FGFR2","FGFR3","IDH1","IDH2","FLT3","KIT","PDGFRA",
    "NF1","NF2","TSC1","TSC2","ARID1A","CTNNB1","VHL","POLE",
    "MLH1","MSH2","MSH6","PMS2","DNMT3A","TET2","ASXL1","NPM1",
    "NTRK1","NTRK2","NTRK3","TROP2","CSF1R","AR","ESR1","BCL2",
    "BTK","EZH2","MYC","MYCN","MDM2","CCND1","ATM","PALB2",
    "NOTCH1","FBXW7","SETD2","BAP1","PBRM1","KDR","ABL1",
    "JAK1","JAK2","STAT3","CHEK2","CDK12","ERCC2","POLE2",
}

# ── OncoKB level labels ───────────────────────────────────────
ONCOKB_LEVELS = {
    "LEVEL_1":  ("Level 1",  "FDA-approved biomarker",           "#33A02C", 6),
    "LEVEL_2":  ("Level 2",  "Standard care biomarker",          "#1F78B4", 5),
    "LEVEL_3A": ("Level 3A", "Compelling clinical evidence",     "#FF7F00", 4),
    "LEVEL_3B": ("Level 3B", "Standard care in other tumor",     "#FDBF6F", 3),
    "LEVEL_4":  ("Level 4",  "Compelling biological evidence",   "#CAB2D6", 2),
    "LEVEL_R1": ("Resist R1","Standard resistance biomarker",    "#E31A1C", 1),
    "LEVEL_R2": ("Resist R2","Compelling resistance evidence",   "#FB9A99", 1),
    "NO":       ("No level", "No actionable alteration",         "#999999", 0),
}

def vaf_tier(vaf):
    if vaf >= VAF_TIER1:   return "Tier1-Clonal",    "HIGH",   3
    elif vaf >= VAF_TIER2: return "Tier2-Subclonal", "MEDIUM", 2
    else:                  return "Tier3-LowVAF",    "LOW",    1

def read_maf(path):
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)

def extract_mutations(maf, vaf_min=0.0):
    coding = maf[maf["Variant_Classification"].isin(ACTIONABLE_CLASSES)]
    act    = coding[coding["Hugo_Symbol"].isin(ONCOGENES)].copy()
    if len(act) == 0:
        return pd.DataFrame()
    if "t_depth" in act.columns and "t_alt_count" in act.columns:
        act["VAF"] = (act["t_alt_count"] / act["t_depth"]).round(3)
    elif "tumor_f" in act.columns:
        act["VAF"] = act["tumor_f"].round(3)
    else:
        act["VAF"] = 0.1
    act = act[act["VAF"] >= vaf_min]
    act = (act.sort_values("VAF", ascending=False)
              .drop_duplicates(subset="Hugo_Symbol", keep="first"))
    act[["vaf_tier","vaf_priority","tier_score"]] = (
        act["VAF"].apply(lambda v: pd.Series(vaf_tier(v))))
    cols = ["Hugo_Symbol","Variant_Classification","HGVSp_Short",
            "VAF","vaf_tier","vaf_priority","tier_score"]
    cols = [c for c in cols if c in act.columns]
    return (act[cols]
            .rename(columns={"Hugo_Symbol":"gene","HGVSp_Short":"variant"})
            .sort_values("VAF", ascending=False)
            .reset_index(drop=True))

# ── OncoKB integration ────────────────────────────────────────
def query_oncokb(gene, variant, cancer, token):
    """Query OncoKB for actionability level."""
    if not token:
        return {"level":"NO","label":"No level","desc":"No OncoKB token provided",
                "drugs":[],"color":"#999999","score":0}
    try:
        # Parse variant for OncoKB format
        aa_change = str(variant).replace("p.","") if variant else ""
        url = (f"https://www.oncokb.org/api/v1/annotate/mutations/byProteinChange"
               f"?hugoSymbol={gene}&alteration={aa_change}"
               f"&tumorType={cancer.replace(' ','+')}")
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "User-Agent":    "megaMine/2.0",
            "Accept":        "application/json",
        })
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
            data = json.loads(r.read())

        level = data.get("highestSensitiveLevel") or \
                data.get("highestResistanceLevel") or "NO"
        treatments = data.get("treatments",[])
        drugs = []
        for t in treatments[:3]:
            for d in t.get("drugs",[]):
                drugs.append(d.get("drugName",""))

        level_info = ONCOKB_LEVELS.get(level, ONCOKB_LEVELS["NO"])
        return {
            "level": level,
            "label": level_info[0],
            "desc":  level_info[1],
            "drugs": drugs,
            "color": level_info[2],
            "score": level_info[3],
        }
    except Exception as e:
        return {"level":"NO","label":"No level","desc":str(e)[:50],
                "drugs":[],"color":"#999999","score":0}

# ── ClinVar integration ───────────────────────────────────────
def query_clinvar(gene, variant, email, api_key):
    """Query ClinVar for pathogenicity."""
    try:
        aa = str(variant).replace("p.","") if variant else ""
        query = f"{gene}[gene] AND {aa}[variant name]"
        url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
               f"?db=clinvar&term={urllib.parse.quote(query)}"
               f"&retmode=json&retmax=5&email={email}&api_key={api_key}")
        req = urllib.request.Request(url, headers={"User-Agent":"megaMine/2.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            data = json.loads(r.read())
        ids = data["esearchresult"]["idlist"]
        if not ids:
            return {"pathogenicity":"Unknown","clinvar_id":"","stars":0}

        # Fetch first result
        url2 = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=clinvar&id={ids[0]}&retmode=json"
                f"&email={email}&api_key={api_key}")
        req2 = urllib.request.Request(url2, headers={"User-Agent":"megaMine/2.0"})
        with urllib.request.urlopen(req2, context=ctx, timeout=10) as r:
            data2 = json.loads(r.read())
        result = data2.get("result",{}).get(ids[0],{})
        clinsig = result.get("clinical_significance",{}).get("description","Unknown")
        stars   = result.get("clinical_significance",{}).get("review_status_count",0)
        return {"pathogenicity":clinsig, "clinvar_id":ids[0], "stars":stars}
    except Exception as e:
        return {"pathogenicity":"Unknown","clinvar_id":"","stars":0}

# ── megaMine query ────────────────────────────────────────────
def query_megamine(gene, cancer, email, api_key, hgnc_cache,
                   out_dir, max_records=50, years="2018-2024"):
    for mod in list(sys.modules.keys()):
        if "megaMine" in mod:
            del sys.modules[mod]
    from megaMine.core.extractor import main as run
    query = (f'{gene} AND ("{cancer}" OR cancer OR tumor OR tumour) '
             f'AND ("targeted therapy" OR "drug resistance" OR '
             f'"clinical trial" OR inhibitor OR treatment)')
    out_path = str(out_dir / f"maf_{gene}")
    sys.argv = ["extractor","--q",query,"--years",years,
                "--max-records",str(max_records),"--out",out_path,
                "--email",email,"--ncbi-api-key",api_key,
                "--hgnc-cache",hgnc_cache,
                "--require-gene-and-drug","--require-known-drug"]
    try:
        run()
        df = pd.read_excel(out_path+".xlsx", sheet_name="Rows")
        df["query_gene"] = gene
        return df
    except:
        return pd.DataFrame()

def score_evidence(df):
    EV    = {"efficacy":3,"resistance":2,"background":1,"review":1,"safety":1}
    STUDY = {"RCT":4,"trial":3,"observational":2,"review":1,
             "preclinical":1,"case_report":2,"in_vitro":1}
    df["ev_score"]    = df.get("final_evidence_type",
                               df.get("evidence_type","background")).map(EV).fillna(1)
    df["study_score"] = df.get("study_design",
                               pd.Series(["observational"]*len(df))).map(STUDY).fillna(2)
    df["conf"]        = df.get("llm_confidence",pd.Series([0.5]*len(df))).fillna(0.5)
    df["evidence_quality_score"] = (df["ev_score"]*df["study_score"]*df["conf"]).round(3)
    return df

# ── HTML report generator ─────────────────────────────────────
def build_html_report(summary, all_evidence, ranked,
                      patient_id, cancer, out_path):
    """Generate fully offline interactive HTML report."""

    # Build gene cards
    gene_cards = ""
    for _, r in summary.iterrows():
        gene       = r["gene"]
        variant    = r.get("variant","")
        vaf        = r.get("VAF",0)
        priority   = r.get("vaf_priority","LOW")
        tier       = r.get("vaf_tier","Tier3")
        oncokb_lbl = r.get("oncokb_label","No level")
        oncokb_clr = r.get("oncokb_color","#999999")
        clinvar    = r.get("clinvar_pathogenicity","Unknown")
        top_drug   = r.get("top_drug","N/A")
        top_ev     = r.get("top_evidence_type","N/A")
        top_score  = r.get("top_score",0)
        papers     = r.get("total_papers",0)
        verified   = r.get("verified_rows",0)
        top_pmid   = r.get("top_pmid","")

        priority_color = {"HIGH":"#27ae60","MEDIUM":"#f39c12","LOW":"#e74c3c"}.get(priority,"#999")
        priority_icon  = {"HIGH":"✅","MEDIUM":"⚠️","LOW":"❌"}.get(priority,"❓")

        # Top 3 drugs for this gene
        gene_rows = ranked[ranked["query_gene"]==gene]
        drug_rows_html = ""
        for _, dr in gene_rows.iterrows():
            drug  = dr.get("drug_primary","?")
            ev    = dr.get("final_evidence_type","?")
            score = dr.get("final_score",0)
            pmid  = dr.get("pmid","")
            ev_color = {"efficacy":"#27ae60","resistance":"#e74c3c",
                        "background":"#95a5a6","safety":"#e67e22"}.get(str(ev).lower(),"#95a5a6")
            drug_rows_html += f"""
            <tr>
              <td><strong>{drug}</strong></td>
              <td><span style="color:{ev_color};font-weight:bold">{ev}</span></td>
              <td>{score:.2f}</td>
              <td><a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">{pmid}</a></td>
            </tr>"""

        gene_cards += f"""
        <div class="gene-card">
          <div class="gene-header">
            <div class="gene-title">
              <span class="gene-name">{gene}</span>
              <span class="variant-badge">{variant}</span>
              <span class="vaf-badge" style="background:{priority_color}">
                {priority_icon} VAF={vaf:.3f} ({tier})
              </span>
            </div>
            <div class="gene-scores">
              <span class="oncokb-badge" style="background:{oncokb_clr}">
                OncoKB: {oncokb_lbl}
              </span>
              <span class="clinvar-badge">ClinVar: {clinvar}</span>
            </div>
          </div>
          <div class="gene-stats">
            <div class="stat">
              <div class="stat-value">{papers}</div>
              <div class="stat-label">Papers</div>
            </div>
            <div class="stat">
              <div class="stat-value">{verified}</div>
              <div class="stat-label">Verified</div>
            </div>
            <div class="stat">
              <div class="stat-value">{top_score:.1f}</div>
              <div class="stat-label">Top Score</div>
            </div>
          </div>
          <div class="drug-table-wrap">
            <table class="drug-table">
              <thead><tr>
                <th>Drug</th><th>Evidence</th>
                <th>Score</th><th>PMID</th>
              </tr></thead>
              <tbody>{drug_rows_html}</tbody>
            </table>
          </div>
        </div>"""

    # Tier summary counts
    tier1 = len(summary[summary["vaf_priority"]=="HIGH"])
    tier2 = len(summary[summary["vaf_priority"]=="MEDIUM"])
    tier3 = len(summary[summary["vaf_priority"]=="LOW"])
    total_papers   = int(summary["total_papers"].sum())
    total_verified = int(summary["verified_rows"].sum())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>megaMine MAF Report — {patient_id}</title>
<style>
  :root {{
    --navy:#1e2d3d; --blue:#5b4fcf; --teal:#00a8a8;
    --orange:#f05a28; --amber:#e8a020; --green:#2ea87e;
    --red:#e74c3c; --gray:#7a8799; --light:#f5f0ff;
    --white:#ffffff; --card:#ffffff;
    --radius:12px; --shadow:0 2px 12px rgba(0,0,0,0.08);
    --font:'Segoe UI',system-ui,sans-serif;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:var(--font); background:#f0f4f8; color:#1e293b; }}

  .header {{
    background:linear-gradient(135deg,var(--navy) 0%,var(--blue) 100%);
    color:white; padding:32px 40px;
  }}
  .header h1 {{ font-size:1.8rem; font-weight:800; }}
  .header p  {{ opacity:0.8; margin-top:6px; font-size:0.95rem; }}
  .disclaimer {{
    background:#fff3cd; border-left:4px solid #f39c12;
    padding:12px 20px; margin:20px 40px;
    border-radius:6px; font-size:0.85rem; color:#856404;
  }}

  .metrics {{
    display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr));
    gap:12px; padding:0 40px 20px;
  }}
  .metric {{
    background:var(--card); border-radius:var(--radius);
    padding:16px; text-align:center;
    box-shadow:var(--shadow); border-top:3px solid var(--blue);
  }}
  .metric-value {{ font-size:1.6rem; font-weight:800; color:var(--navy); }}
  .metric-label {{ font-size:0.65rem; color:var(--gray);
                   text-transform:uppercase; margin-top:4px; }}

  .section {{ padding:0 40px 30px; }}
  .section-title {{
    font-size:1.1rem; font-weight:700; color:var(--navy);
    margin-bottom:16px; padding-bottom:8px;
    border-bottom:2px solid var(--light);
  }}

  .gene-grid {{
    display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr));
    gap:16px;
  }}
  .gene-card {{
    background:var(--card); border-radius:var(--radius);
    box-shadow:var(--shadow); overflow:hidden;
  }}
  .gene-header {{
    padding:16px; background:var(--light);
    border-bottom:1px solid #e2e8f0;
  }}
  .gene-title {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
  .gene-name {{
    font-size:1.2rem; font-weight:800; color:var(--navy);
  }}
  .variant-badge {{
    font-size:0.75rem; background:#e2e8f0; padding:2px 8px;
    border-radius:20px; color:#475569; font-family:monospace;
  }}
  .vaf-badge {{
    font-size:0.72rem; padding:3px 10px; border-radius:20px;
    color:white; font-weight:600;
  }}
  .gene-scores {{ display:flex; gap:8px; margin-top:8px; flex-wrap:wrap; }}
  .oncokb-badge {{
    font-size:0.72rem; padding:3px 10px; border-radius:20px;
    color:white; font-weight:600;
  }}
  .clinvar-badge {{
    font-size:0.72rem; padding:3px 10px; border-radius:20px;
    background:#f1f5f9; color:#475569; font-weight:600;
  }}
  .gene-stats {{
    display:flex; gap:0; border-bottom:1px solid #f1f5f9;
  }}
  .stat {{
    flex:1; text-align:center; padding:12px 8px;
    border-right:1px solid #f1f5f9;
  }}
  .stat:last-child {{ border-right:none; }}
  .stat-value {{ font-size:1.3rem; font-weight:700; color:var(--blue); }}
  .stat-label {{ font-size:0.65rem; color:var(--gray);
                 text-transform:uppercase; margin-top:2px; }}
  .drug-table-wrap {{ padding:12px; overflow-x:auto; }}
  .drug-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  .drug-table th {{
    background:#f8fafc; padding:8px 10px; text-align:left;
    font-weight:600; color:var(--gray); font-size:0.72rem;
    text-transform:uppercase; border-bottom:2px solid #e2e8f0;
  }}
  .drug-table td {{
    padding:8px 10px; border-bottom:1px solid #f1f5f9; color:#334155;
  }}
  .drug-table tr:last-child td {{ border-bottom:none; }}
  .drug-table a {{ color:var(--blue); text-decoration:none; }}
  .drug-table a:hover {{ text-decoration:underline; }}

  .tier-legend {{
    display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px;
  }}
  .tier-item {{
    display:flex; align-items:center; gap:6px;
    font-size:0.82rem; color:#475569;
  }}
  .tier-dot {{
    width:12px; height:12px; border-radius:50%;
  }}

  footer {{
    text-align:center; padding:24px; color:var(--gray);
    font-size:0.75rem; border-top:1px solid #e2e8f0;
    margin-top:20px;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>🧬 megaMine v2.0 — MAF Evidence Report</h1>
  <p>Patient: <strong>{patient_id}</strong> &nbsp;|&nbsp;
     Cancer: <strong>{cancer}</strong> &nbsp;|&nbsp;
     Generated by megaMine (APML, Ajou University)</p>
</div>

<div class="disclaimer">
  ⚠️ <strong>DISCLAIMER:</strong> This report contains literature-derived evidence only.
  It is <strong>NOT a clinical treatment recommendation</strong>.
  All outputs require expert oncologist review before any clinical decision.
</div>

<div class="metrics">
  <div class="metric">
    <div class="metric-value">{len(summary)}</div>
    <div class="metric-label">Mutations</div>
  </div>
  <div class="metric" style="border-top-color:#27ae60">
    <div class="metric-value" style="color:#27ae60">{tier1}</div>
    <div class="metric-label">Tier 1 Clonal</div>
  </div>
  <div class="metric" style="border-top-color:#f39c12">
    <div class="metric-value" style="color:#f39c12">{tier2}</div>
    <div class="metric-label">Tier 2 Subclonal</div>
  </div>
  <div class="metric" style="border-top-color:#e74c3c">
    <div class="metric-value" style="color:#e74c3c">{tier3}</div>
    <div class="metric-label">Tier 3 Low VAF</div>
  </div>
  <div class="metric">
    <div class="metric-value">{total_papers}</div>
    <div class="metric-label">Total Papers</div>
  </div>
  <div class="metric" style="border-top-color:#00a8a8">
    <div class="metric-value" style="color:#00a8a8">{total_verified}</div>
    <div class="metric-label">Verified Rows</div>
  </div>
</div>

<div class="section">
  <div class="section-title">Mutation Evidence — Ranked by VAF × Literature Score</div>
  <div class="tier-legend">
    <div class="tier-item">
      <div class="tier-dot" style="background:#27ae60"></div>
      Tier 1 — Clonal (VAF ≥ 20%) — High priority
    </div>
    <div class="tier-item">
      <div class="tier-dot" style="background:#f39c12"></div>
      Tier 2 — Subclonal (VAF 5-20%) — Medium priority
    </div>
    <div class="tier-item">
      <div class="tier-dot" style="background:#e74c3c"></div>
      Tier 3 — Low VAF (&lt;5%) — Interpret with caution
    </div>
  </div>
  <div class="gene-grid">
    {gene_cards}
  </div>
</div>

<footer>
  Generated by megaMine v2.0 — Precision Medicine Lab (APML), Ajou University<br>
  Literature-derived evidence only. Not a clinical treatment recommendation.
</footer>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML report: {out_path}")


def run_maf_pipeline(maf_path, cancer, out_dir, email, api_key,
                     hgnc_cache, oncokb_token="", vaf_min=0.0,
                     max_records=50, top_drugs=3, years="2018-2024"):

    import urllib.parse
    from megaMine.utils.normalizers import (normalize_cancer_column,
                                             add_resistance_context,
                                             reconcile_evidence_type)
    from megaMine.modules.llm_verify import run_llm_verification

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp_queries"; tmp.mkdir(exist_ok=True)
    patient_id = Path(maf_path).stem

    print("=" * 60)
    print("MEGAMINE v2.0 — MAF EVIDENCE PRIORITIZATION")
    print(f"Patient: {patient_id} | Cancer: {cancer}")
    print("=" * 60)

    # Step 1 — Read MAF
    print("\n[1/6] Reading MAF...")
    maf = read_maf(maf_path)
    print(f"  Mutations: {len(maf):,}")

    # Step 2 — Extract mutations
    print("\n[2/6] Extracting actionable mutations...")
    mutations = extract_mutations(maf, vaf_min=vaf_min)
    if len(mutations) == 0:
        print("  No actionable mutations found.")
        return
    for _, r in mutations.iterrows():
        print(f"  {r['gene']:<8} {str(r.get('variant','')):<15} "
              f"VAF={r['VAF']:.3f} ({r['vaf_priority']})")

    # Step 3 — OncoKB + ClinVar
    print("\n[3/6] Querying OncoKB + ClinVar...")
    oncokb_data  = {}
    clinvar_data = {}
    for _, mut in mutations.iterrows():
        gene    = str(mut["gene"])
        variant = str(mut.get("variant",""))
        print(f"  {gene}...", end=" ", flush=True)
        ok = query_oncokb(gene, variant, cancer, oncokb_token)
        cv = query_clinvar(gene, variant, email, api_key)
        oncokb_data[gene]  = ok
        clinvar_data[gene] = cv
        ok_str = ok["label"] if oncokb_token else "No token"
        print(f"OncoKB={ok_str} | ClinVar={cv['pathogenicity']}")
        time.sleep(0.4)

    # Step 4 — megaMine queries
    print(f"\n[4/6] Querying PubMed via megaMine...")
    all_rows = []
    for _, mut in mutations.iterrows():
        gene = str(mut["gene"])
        print(f"  → {gene}...", end=" ", flush=True)
        df = query_megamine(gene, cancer, email, api_key,
                            hgnc_cache, tmp, max_records, years)
        if len(df) > 0:
            df["patient_variant"]  = mut.get("variant","")
            df["patient_VAF"]      = mut["VAF"]
            df["patient_vaf_tier"] = mut["vaf_tier"]
            df["tier_priority"]    = mut["tier_score"]
            # Add OncoKB score to boost
            df["oncokb_score"] = oncokb_data.get(gene,{}).get("score",0)
            all_rows.append(df)
            print(f"{len(df)} rows")
        else:
            print("no results")
        time.sleep(1)

    if not all_rows:
        print("No evidence found.")
        return

    # Step 5 — Normalize + verify + score
    print("\n[5/6] Normalizing, verifying, scoring...")
    combined = pd.concat(all_rows, ignore_index=True)
    combined = normalize_cancer_column(combined)
    combined = add_resistance_context(combined)
    combined = reconcile_evidence_type(combined)
    combined = run_llm_verification(combined, model="offline",
                                    confidence_threshold=0.60)
    combined = score_evidence(combined)
    combined["final_score"] = (
        combined["evidence_quality_score"] *
        combined["tier_priority"] *
        (1 + combined["oncokb_score"] * 0.2)
    ).round(3)

    verified = combined[combined["llm_verified"]=="yes"].copy()
    print(f"  Rows: {len(combined)} | Verified: {len(verified)}")

    ranked = (combined
              .sort_values("final_score", ascending=False)
              .groupby("query_gene")
              .head(top_drugs))

    # Step 6 — Build summary + reports
    print("\n[6/6] Building reports...")
    summary = []
    for _, mut in mutations.iterrows():
        gene = mut["gene"]
        rows = combined[combined["query_gene"]==gene]
        ver  = rows[rows["llm_verified"]=="yes"]
        top  = ranked[ranked["query_gene"]==gene]
        ok   = oncokb_data.get(gene,{})
        cv   = clinvar_data.get(gene,{})
        summary.append({
            "gene":                  gene,
            "variant":               mut.get("variant",""),
            "VAF":                   mut["VAF"],
            "vaf_tier":              mut["vaf_tier"],
            "vaf_priority":          mut["vaf_priority"],
            "oncokb_level":          ok.get("level","NO"),
            "oncokb_label":          ok.get("label","No level"),
            "oncokb_drugs":          "; ".join(ok.get("drugs",[])),
            "oncokb_color":          ok.get("color","#999999"),
            "oncokb_score":          ok.get("score",0),
            "clinvar_pathogenicity": cv.get("pathogenicity","Unknown"),
            "clinvar_id":            cv.get("clinvar_id",""),
            "total_papers":          rows["pmid"].nunique(),
            "verified_rows":         len(ver),
            "top_drug":              top.iloc[0]["drug_primary"] if len(top)>0 else "N/A",
            "top_evidence_type":     top.iloc[0].get("final_evidence_type","N/A") if len(top)>0 else "N/A",
            "top_score":             top.iloc[0]["final_score"] if len(top)>0 else 0,
            "top_pmid":              top.iloc[0]["pmid"] if len(top)>0 else "",
        })

    df_summary = pd.DataFrame(summary)

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'Gene':<8} {'Variant':<14} {'VAF':>6} {'Priority':<8} "
          f"{'OncoKB':<12} {'Top Drug':<18} {'Score':>6}")
    print("-" * 70)
    for _, r in df_summary.iterrows():
        icon={"HIGH":"✅","MEDIUM":"⚠️ ","LOW":"❌"}.get(r["vaf_priority"],"❓")
        print(f"  {r['gene']:<6} {str(r['variant']):<14} {r['VAF']:>6.3f} "
              f"{icon}{r['vaf_priority']:<5} {r['oncokb_label']:<12} "
              f"{r['top_drug']:<18} {r['top_score']:>6.2f}")

    # Save Excel
    out_xlsx = out / f"{patient_id}_megaMine_evidence.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        mutations.to_excel(xl,   sheet_name="Mutations",         index=False)
        df_summary.to_excel(xl,  sheet_name="Summary",           index=False)
        ranked.to_excel(xl,      sheet_name="Top_Evidence",      index=False)
        combined.to_excel(xl,    sheet_name="All_Evidence",      index=False)
        verified.to_excel(xl,    sheet_name="Verified_Evidence", index=False)
        pd.DataFrame([{
            "patient":     patient_id, "cancer":cancer,
            "vaf_min":     vaf_min,    "years":years,
            "max_records": max_records,"top_drugs":top_drugs,
            "oncokb":      bool(oncokb_token),
            "n_mutations": len(mutations),
            "n_rows":      len(combined),
            "n_verified":  len(verified),
            "version":     "megaMine_v2.0.0",
        }]).to_excel(xl, sheet_name="RunInfo", index=False)
    print(f"✅ Excel: {out_xlsx}")

    # HTML report
    out_html = out / f"{patient_id}_megaMine_report.html"
    build_html_report(df_summary, combined, ranked,
                      patient_id, cancer, out_html)

    import subprocess
    subprocess.Popen(["open", str(out_html)])

    print(f"\n⚠️  DISCLAIMER: Literature-derived evidence only.")
    print(f"   Not a clinical treatment recommendation.")
    return df_summary


def main():
    parser = argparse.ArgumentParser(
        description="megaMine MAF Evidence Prioritization — Better than IDAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  megaMine-maf --maf patient.maf --cancer NSCLC --out results/
  megaMine-maf --maf patient.maf --cancer Breast --oncokb-token TOKEN --out results/
  megaMine-maf --maf cohort.maf  --cancer CRC --vaf-min 0.05 --top 5 --out results/
        """)
    parser.add_argument("--maf",           required=True)
    parser.add_argument("--cancer",        required=True)
    parser.add_argument("--out",           required=True)
    parser.add_argument("--email",         default="user@institution.edu")
    parser.add_argument("--ncbi-api-key",  default="")
    parser.add_argument("--hgnc-cache",    default="")
    parser.add_argument("--oncokb-token",  default="",
                        help="Free token from oncokb.org/account/register")
    parser.add_argument("--vaf-min",       type=float, default=0.0)
    parser.add_argument("--max-records",   type=int,   default=50)
    parser.add_argument("--top",           type=int,   default=3)
    parser.add_argument("--years",         default="2018-2024")
    args = parser.parse_args()

    run_maf_pipeline(
        maf_path     = args.maf,
        cancer       = args.cancer,
        out_dir      = args.out,
        email        = args.email,
        api_key      = args.ncbi_api_key,
        hgnc_cache   = args.hgnc_cache,
        oncokb_token = args.oncokb_token,
        vaf_min      = args.vaf_min,
        max_records  = args.max_records,
        top_drugs    = args.top,
        years        = args.years,
    )

if __name__ == "__main__":
    main()
