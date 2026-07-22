"""
maf_pipeline.py — megaMine v2.0
MAF-driven precision oncology evidence synthesis pipeline.

THREE-TIER EVIDENCE SPECIFICITY (v4):
  Tier 1 — Exact alteration evidence   (weight=1.00)
  Tier 2 — Alteration class evidence   (weight=0.75)
  Tier 3 — Gene-level evidence         (weight=0.45)

FIXES APPLIED (v4):
  ✅ (gene,variant) tuple keys — no multi-variant overwrite
  ✅ Variant-specific VAF mapping
  ✅ Explicit efficacy/resistance/background PMID counting
  ✅ Per-gene ranked table uses verified rows only
  ✅ Three-tier framework (matches implementation)
  ✅ Exact OncoKB match labels (gene/variant/allele)
  ✅ Missense query fixed (not assumed activating)
  ✅ Variant-specific output paths (no collisions)
  ✅ Query errors logged, not silently suppressed
  ✅ SSL: create_default_context(), no silent fallback
  ✅ OncoKB token from ONCOKB_TOKEN env var only
  ✅ Co-mutation signals are hypothesis-only
  ✅ evidence_priority_score (not patient_score)

AUTHORS: Muhammad Junaid — APML, Ajou University
"""

import os, sys, time, json, argparse, ssl, re
import urllib.request, urllib.parse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, Dict, List, Tuple

# ── SSL — verified, no silent fallback ───────────────────────
ctx = ssl.create_default_context()

# ── VAF thresholds — confidence modifier only ─────────────────
VAF_CLONAL    = 0.20
VAF_SUBCLONAL = 0.05

# ── Three-tier evidence specificity ───────────────────────────
SPECIFICITY_WEIGHT = {
    "exact_alteration": 1.00,
    "alteration_class": 0.75,
    "gene_level":       0.45,
}

# ── Actionable variant classes ────────────────────────────────
ACTIONABLE_CLASSES = {
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del",
    "Frame_Shift_Ins","Splice_Site","In_Frame_Del","In_Frame_Ins",
    "Translation_Start_Site","Nonstop_Mutation","Splice_Region",
}

ALTERATION_CLASS = {
    "Frame_Shift_Del":       "truncating",
    "Frame_Shift_Ins":       "truncating",
    "Nonsense_Mutation":     "truncating",
    "Splice_Site":           "truncating",
    "Nonstop_Mutation":      "truncating",
    "In_Frame_Del":          "in_frame_indel",
    "In_Frame_Ins":          "in_frame_indel",
    "Missense_Mutation":     "missense",
    "Splice_Region":         "splice",
    "Translation_Start_Site":"truncating",
}

CANCER_RELEVANT_GENES = {
    "EGFR","KRAS","NRAS","HRAS","BRAF","MET","ALK","ROS1","RET",
    "HER2","ERBB2","PIK3CA","PTEN","AKT1","MTOR","CDK4","CDK6",
    "BRCA1","BRCA2","TP53","STK11","KEAP1","SMAD4","RB1","CDKN2A",
    "FGFR1","FGFR2","FGFR3","IDH1","IDH2","FLT3","KIT","PDGFRA",
    "NF1","NF2","TSC1","TSC2","ARID1A","CTNNB1","VHL","POLE",
    "MLH1","MSH2","MSH6","PMS2","DNMT3A","TET2","ASXL1","NPM1",
    "NTRK1","NTRK2","NTRK3","TROP2","CSF1R","AR","ESR1","BCL2",
    "BTK","EZH2","MYC","MDM2","CCND1","ATM","PALB2","NOTCH1",
    "FBXW7","SETD2","BAP1","PBRM1","KDR","ABL1","JAK1","JAK2",
    "STAT3","CHEK2","CDK12","ERCC2","POLE2","CDH1","APC","RNF43",
}

GENE_ROLE = {
    "oncogene": {
        "EGFR","KRAS","NRAS","HRAS","BRAF","MET","ALK","ROS1","RET",
        "HER2","ERBB2","PIK3CA","AKT1","MTOR","CDK4","CDK6","MYC",
        "FLT3","KIT","PDGFRA","IDH1","IDH2","FGFR1","FGFR2","FGFR3",
        "NTRK1","NTRK2","NTRK3","BTK","BCL2","AR","ESR1","CTNNB1",
    },
    "tumor_suppressor": {
        "TP53","PTEN","RB1","CDKN2A","SMAD4","APC","VHL","BRCA1",
        "BRCA2","STK11","KEAP1","NF1","NF2","TSC1","TSC2","ARID1A",
        "SETD2","BAP1","PBRM1","RNF43","MLH1","MSH2","MSH6","PMS2",
        "ATM","PALB2","CHEK2","FBXW7","CDH1",
    },
    "dna_repair": {
        "BRCA1","BRCA2","PALB2","ATM","CHEK2","POLE","POLE2",
        "ERCC2","CDK12","MLH1","MSH2","MSH6","PMS2",
    },
}

ONCOKB_LEVELS = {
    "LEVEL_1":  ("Level 1",  "FDA-approved in this tumor type",     "#33A02C", 6),
    "LEVEL_2":  ("Level 2",  "Standard care biomarker",             "#1F78B4", 5),
    "LEVEL_3A": ("Level 3A", "Compelling clinical evidence",        "#FF7F00", 4),
    "LEVEL_3B": ("Level 3B", "Standard care, different tumor type", "#FDBF6F", 3),
    "LEVEL_4":  ("Level 4",  "Compelling biological evidence",      "#CAB2D6", 2),
    "LEVEL_R1": ("R1",       "Standard resistance biomarker",       "#E31A1C", 1),
    "LEVEL_R2": ("R2",       "Compelling resistance evidence",      "#FB9A99", 1),
    "NO":       ("No level", "No actionable alteration in OncoKB",  "#999999", 0),
}

COMUTATION_HYPOTHESES = [
    {
        "genes":{"KEAP1","ARID1A"}, "min_hits":2,
        "effect":"IO_SENSITIZER_HYPOTHESIS",
        "drug":"Immune checkpoint inhibitor",
        "note":("HYPOTHESIS ONLY: KEAP1+ARID1A co-occurrence has been associated "
                "with SWI/SNF deficiency. Evidence is conflicting — KEAP1 mutation "
                "has also been associated with adverse IO outcomes in NSCLC. "
                "NOT a validated treatment biomarker."),
        "confidence":"Low",
        "refs":"Ricciuti 2022 JCO; Skoulidis 2018 Cancer Cell",
    },
    {
        "genes":{"POLE"}, "min_hits":1,
        "effect":"HYPERMUTATOR_CANDIDATE",
        "drug":"Immune checkpoint inhibitor",
        "note":("HYPOTHESIS ONLY: POLE mutations can cause ultramutator phenotype "
                "only when pathogenic exonuclease-domain variants are confirmed "
                "with high TMB. POLE R1082C has Unknown oncogenicity. "
                "Requires MSI/TMB confirmation before any clinical action."),
        "confidence":"Low — requires TMB/MSI confirmation",
        "refs":"Le 2017 Science; Mehnert 2016 JCO",
    },
    {
        "genes":{"MLH1","MSH2","MSH6","PMS2"}, "min_hits":1,
        "effect":"dMMR_CANDIDATE",
        "drug":"Pembrolizumab (FDA tumor-agnostic if dMMR confirmed)",
        "note":("MODERATE SIGNAL: MMR gene mutation may indicate dMMR/MSI-H. "
                "Requires IHC or PCR/NGS confirmation. If confirmed, pembrolizumab "
                "has FDA tumor-agnostic approval."),
        "confidence":"Medium — requires IHC/MSI confirmation",
        "refs":"Le 2017 Science; FDA 2017",
    },
    {
        "genes":{"BRCA1","BRCA2","PALB2"}, "min_hits":1,
        "effect":"HRD_CANDIDATE",
        "drug":"PARP inhibitor",
        "note":("MODERATE SIGNAL: HRR gene LOF may indicate HRD. "
                "Confirm biallelic inactivation. Germline vs somatic status affects "
                "FDA approval scope."),
        "confidence":"Medium — confirm biallelic LOH",
        "refs":"Robson 2017 NEJM",
    },
    {
        "genes":{"NF1","KRAS","NRAS","HRAS"}, "min_hits":1,
        "effect":"RAS_PATHWAY_HYPOTHESIS",
        "drug":"MEK inhibitor",
        "note":("HYPOTHESIS ONLY: RAS pathway activation — MEK inhibitors have "
                "shown activity in some NF1-mutant NSCLC studies but not validated "
                "as a biomarker."),
        "confidence":"Low — exploratory",
        "refs":"Manchado 2016 Nature",
    },
]

# ─────────────────────────────────────────────────────────────
def vaf_confidence(vaf: float) -> Tuple[str,float,str]:
    if vaf >= VAF_CLONAL:      return "Clonal",    1.00, "#27ae60"
    elif vaf >= VAF_SUBCLONAL: return "Subclonal", 0.70, "#f39c12"
    else:                      return "LowVAF",    0.40, "#e74c3c"

def get_gene_role(gene: str) -> str:
    for role, genes in GENE_ROLE.items():
        if gene in genes: return role
    return "unknown"

def safe_name(s: str) -> str:
    """Sanitize string for use in file names."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))

def read_maf(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)

def extract_mutations(maf: pd.DataFrame, vaf_min: float=0.0) -> pd.DataFrame:
    """
    Extract cancer-relevant coding variants.
    FIX: deduplicate by exact genomic event, preserving multiple variants per gene.
    """
    coding = maf[maf["Variant_Classification"].isin(ACTIONABLE_CLASSES)]
    act    = coding[coding["Hugo_Symbol"].isin(CANCER_RELEVANT_GENES)].copy()
    if len(act) == 0:
        return pd.DataFrame()

    if "t_depth" in act.columns and "t_alt_count" in act.columns:
        act["VAF"] = (act["t_alt_count"] /
                      act["t_depth"].replace(0, np.nan)).round(3)
    elif "tumor_f" in act.columns:
        act["VAF"] = act["tumor_f"].astype(float).round(3)
    else:
        act["VAF"] = 0.1

    act = act[act["VAF"].fillna(0) >= vaf_min].copy()

    # FIX: deduplicate by exact genomic event
    dedup = ["Hugo_Symbol","HGVSp_Short","Variant_Classification"]
    extra = ["Chromosome","Start_Position","Reference_Allele","Tumor_Seq_Allele2"]
    dedup += [c for c in extra if c in act.columns]
    act = act.drop_duplicates(subset=dedup, keep="first")

    act["alteration_class"] = act["Variant_Classification"].map(
        ALTERATION_CLASS).fillna("unknown")
    act["gene_role"] = act["Hugo_Symbol"].apply(get_gene_role)
    act[["vaf_clonality","vaf_conf","vaf_color"]] = (
        act["VAF"].apply(lambda v: pd.Series(vaf_confidence(v))))

    cols = ["Hugo_Symbol","Variant_Classification","HGVSp_Short",
            "VAF","vaf_clonality","vaf_conf","vaf_color",
            "alteration_class","gene_role"]
    cols = [c for c in cols if c in act.columns]
    return (act[cols]
            .rename(columns={"Hugo_Symbol":"gene","HGVSp_Short":"variant"})
            .sort_values("VAF", ascending=False)
            .reset_index(drop=True))

def query_oncokb_exact(gene:str, variant:str, cancer:str, token:str) -> dict:
    """
    FIX: Store gene_exist/variant_exist/allele_exist separately.
    FIX: Detailed match label.
    FIX: drug_levels keyed by drug name.
    """
    base = {
        "sensitive_level":"","resistant_level":"",
        "oncogenicity":"Unknown","mutation_effect":"Unknown",
        "drugs":[],"label":"No token","color":"#999999","score":0,
        "gene_exist":False,"variant_exist":False,"allele_exist":False,
        "match_label":"No OncoKB token","cancer_match":False,
        "alteration_queried":str(variant),"tumor_type_queried":cancer,
        "drug_levels":{},
    }
    if not token:
        return base
    try:
        aa = str(variant).replace("p.","").strip()
        url = (f"https://www.oncokb.org/api/v1/annotate/mutations/byProteinChange"
               f"?hugoSymbol={gene}"
               f"&alteration={urllib.parse.quote(aa)}"
               f"&tumorType={urllib.parse.quote(cancer)}")
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "User-Agent":    "megaMine/2.0",
            "Accept":        "application/json",
        })
        with urllib.request.urlopen(req, ctx, timeout=15) as r:
            raw = r.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)

        ge = bool(data.get("geneExist",    False))
        ve = bool(data.get("variantExist", False))
        ae = bool(data.get("alleleExist",  False))

        # FIX: detailed match label
        if ve and ae:
            match_label = "Exact alteration recognized"
        elif ve:
            match_label = "Variant recognized (not allele-specific)"
        elif ge:
            match_label = "Gene recognized only"
        else:
            match_label = "No OncoKB match"

        sens  = data.get("highestSensitiveLevel","") or ""
        res   = data.get("highestResistanceLevel","") or ""
        onco  = data.get("oncogenic","Unknown") or "Unknown"
        effect= (data.get("mutationEffect",{}) or {}).get("knownEffect","Unknown") or "Unknown"

        drug_levels = {}
        drugs = []
        for t in (data.get("treatments",[]) or []):
            lvl = t.get("level","")
            for d in (t.get("drugs",[]) or []):
                dname = d.get("drugName","")
                if dname:
                    drug_levels[dname.lower()] = lvl
                    drugs.append({"drug":dname,"level":lvl,
                                  "desc":ONCOKB_LEVELS.get(lvl,ONCOKB_LEVELS["NO"])[1]})

        level = sens or res or "NO"
        li = ONCOKB_LEVELS.get(level, ONCOKB_LEVELS["NO"])
        return {
            "sensitive_level":   sens,
            "resistant_level":   res,
            "oncogenicity":      onco,
            "mutation_effect":   effect,
            "drugs":             drugs,
            "label":             li[0],
            "desc":              li[1],
            "color":             li[2],
            "score":             li[3],
            "gene_exist":        ge,
            "variant_exist":     ve,
            "allele_exist":      ae,
            "match_label":       match_label,
            "cancer_match":      bool(sens or res),
            "alteration_queried":aa,
            "tumor_type_queried":cancer,
            "drug_levels":       drug_levels,
        }
    except Exception as e:
        base["label"]       = "Unavailable"
        base["color"]       = "#b0b0b0"
        base["match_label"] = f"OncoKB unavailable: {str(e)[:50]}"
        base["error"]       = str(e)[:100]
        return base

def query_clinvar(gene:str, variant:str, email:str, api_key:str) -> dict:
    try:
        aa = str(variant).replace("p.","").strip()
        q  = f'{gene}[gene] AND "{aa}"[variant name]'
        u  = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
              f"?db=clinvar&term={urllib.parse.quote(q)}"
              f"&retmode=json&retmax=3&email={email}&api_key={api_key}")
        with urllib.request.urlopen(
                urllib.request.Request(u,headers={"User-Agent":"megaMine/2.0"}),
                ctx, timeout=10) as r:
            ids = json.loads(r.read())["esearchresult"]["idlist"]
        if not ids:
            return {"pathogenicity":"Unknown","clinvar_id":""}
        u2 = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
              f"?db=clinvar&id={ids[0]}&retmode=json"
              f"&email={email}&api_key={api_key}")
        with urllib.request.urlopen(
                urllib.request.Request(u2,headers={"User-Agent":"megaMine/2.0"}),
                ctx, timeout=10) as r:
            result = json.loads(r.read()).get("result",{}).get(ids[0],{})
        sig = (result.get("clinical_significance",{}) or {}).get("description","Unknown")
        return {"pathogenicity":sig,"clinvar_id":ids[0]}
    except Exception as e:
        return {"pathogenicity":"Unknown","clinvar_id":"","error":str(e)[:60]}

def build_tiered_queries(gene:str, variant:str, alt_class:str,
                          cancer:str, mutation_effect:str="Unknown") -> List[dict]:
    """
    FIX: Three-tier framework matching implementation.
    FIX: Missense not assumed activating.
    """
    aa = str(variant).replace("p.","").strip()
    queries = []

    # Tier 1 — Exact alteration
    if aa and aa not in ("","nan","None","p."):
        q1 = (f'"{gene}"[tiab] AND "{aa}"[tiab] AND '
              f'("drug" OR "therapy" OR "treatment" OR "inhibitor")')
        queries.append({
            "tier":"exact_alteration","weight":1.00,
            "query":q1,"label":f"Exact: {gene} {aa}",
        })

    # Tier 2 — Alteration class
    class_terms = {
        "truncating":     f'("{gene} truncating mutation"[tiab] OR "{gene} loss of function"[tiab] OR "{gene} nonsense"[tiab])',
        "missense":       f'"{gene} missense mutation"[tiab]',  # FIX: not assumed activating
        "in_frame_indel": f'("{gene} in-frame"[tiab] OR "{gene} deletion"[tiab] OR "{gene} insertion"[tiab])',
        "splice":         f'"{gene} splice"[tiab]',
    }
    # Add gain-of-function tier only when OncoKB confirms it
    if alt_class == "missense" and "gain" in mutation_effect.lower():
        class_terms["missense"] = (
            f'("{gene} missense mutation"[tiab] OR "{gene} activating mutation"[tiab])')

    if alt_class in class_terms:
        q2 = (f'({class_terms[alt_class]}) AND '
              f'("drug" OR "therapy" OR "inhibitor" OR "treatment")')
        queries.append({
            "tier":"alteration_class","weight":0.75,
            "query":q2,"label":f"Class: {gene} {alt_class}",
        })

    # Tier 3 — Gene-level
    q3 = (f'{gene} AND ("{cancer}" OR cancer OR tumor) AND '
          f'("targeted therapy" OR "drug resistance" OR '
          f'"clinical trial" OR inhibitor OR treatment)')
    queries.append({
        "tier":"gene_level","weight":0.45,
        "query":q3,"label":f"Gene: {gene}",
    })

    return queries

def run_tiered_queries(gene:str, variant:str, alt_class:str, cancer:str,
                        mutation_effect:str, email:str, api_key:str,
                        hgnc_cache:str, out_dir:Path,
                        max_records:int, years:str) -> pd.DataFrame:
    """
    FIX: Variant-specific output paths — no collisions.
    FIX: Errors logged, not suppressed.
    """
    queries  = build_tiered_queries(gene, variant, alt_class, cancer, mutation_effect)
    sv       = safe_name(variant)  # FIX: sanitized variant in path
    all_rows = []

    for q_info in queries:
        tier  = q_info["tier"]
        label = q_info["label"]
        # FIX: include variant in output path
        out_p = str(out_dir / f"maf_{gene}_{sv}_{tier}")

        for mod in list(sys.modules.keys()):
            if "megaMine" in mod: del sys.modules[mod]
        from megaMine.core.extractor import main as run

        sys.argv = [
            "extractor",
            "--q",            q_info["query"],
            "--years",        years,
            "--max-records",  str(max_records),
            "--out",          out_p,
            "--email",        email,
            "--ncbi-api-key", api_key,
            "--hgnc-cache",   hgnc_cache,
            "--require-gene-and-drug",
            "--require-known-drug",
        ]
        try:
            run()
            df = pd.read_excel(out_p+".xlsx", sheet_name="Rows")
            df["query_gene"]          = gene
            df["patient_variant"]     = variant
            df["evidence_tier"]       = tier
            df["specificity_weight"]  = q_info["weight"]
            df["evidence_tier_label"] = label
            if len(df) > 0:
                all_rows.append(df)
                print(f"    {label}: {len(df)} rows")
            else:
                print(f"    {label}: 0 rows")
        except Exception as e:
            # FIX: log errors, don't suppress
            print(f"    Query failed [{label}]: {e}")
        time.sleep(0.8)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    # Keep highest specificity per PMID+drug
    if "pmid" in combined.columns and "drug_primary" in combined.columns:
        combined = (combined
                    .sort_values("specificity_weight", ascending=False)
                    .drop_duplicates(
                        subset=["pmid","drug_primary","query_gene","patient_variant"],
                        keep="first"))
    return combined.reset_index(drop=True)

def deduplicate_evidence(df:pd.DataFrame) -> pd.DataFrame:
    key = ["pmid","biomarker","drug_primary",
           "canonical_cancer_type","evidence_tier","patient_variant"]
    key = [c for c in key if c in df.columns]
    if key:
        df = df.drop_duplicates(subset=key, keep="first")
    return df.reset_index(drop=True)

def score_evidence(df:pd.DataFrame, drug_levels:dict) -> pd.DataFrame:
    """
    FIX: resistance from BOTH final_evidence_type AND resistance_evidence.
    FIX: drug-specific OncoKB boost only.
    """
    df = df.copy()
    ev_col    = "final_evidence_type" if "final_evidence_type" in df.columns else "evidence_type"
    study_col = "study_design" if "study_design" in df.columns else None

    EV    = {"efficacy":3,"resistance":1,"background":1,"review":1,"safety":1}
    STUDY = {"RCT":5,"trial":4,"observational":3,
             "preclinical":2,"case_report":3,"review":1,"in_vitro":1}

    df["ev_score"]    = df[ev_col].map(EV).fillna(1)
    df["study_score"] = df[study_col].map(STUDY).fillna(2) if study_col else 2
    df["llm_conf"]    = df.get("llm_confidence",
                               pd.Series([0.5]*len(df),index=df.index)).fillna(0.5)
    df["vaf_conf_col"]= df.get("vaf_confidence",
                               pd.Series([0.7]*len(df),index=df.index)).fillna(0.7)
    df["spec_weight"] = df.get("specificity_weight",
                               pd.Series([0.45]*len(df),index=df.index)).fillna(0.45)

    # FIX: resistance from both columns
    resist_ev   = df[ev_col].astype(str).str.lower().eq("resistance")
    resist_col  = df.get("resistance_evidence",
                         pd.Series(["no"]*len(df),index=df.index))
    resist_flag = resist_col.astype(str).str.lower().eq("yes")
    df["is_resistance"]  = resist_ev | resist_flag
    df["resist_penalty"] = df["is_resistance"].map({True:0.5, False:1.0})

    # FIX: drug-specific OncoKB boost only
    def ok_boost(drug):
        lvl = drug_levels.get(str(drug).lower().strip(),"")
        return 1.0  # OncoKB boost applied once at drug-ranking level only
    df["oncokb_boost"] = 1.0  # removed from row-level score

    # Direction-neutral quality score (used in drug ranking aggregation)
    df["evidence_quality_score"] = (
        df["study_score"] * df["llm_conf"] *
        df["vaf_conf_col"] * df["spec_weight"]
    ).round(4)

    # Direction-aware priority score (used for row-level ranking)
    df["evidence_priority_score"] = (
        df["ev_score"] * df["study_score"] * df["llm_conf"] *
        df["vaf_conf_col"] * df["spec_weight"]
    ).round(4)
    return df

def classify_evidence(tier:str, cancer_match:bool,
                       is_resistance:bool) -> str:
    """
    Classify evidence applicability per reviewer recommendation.
    Direct / Related / Indirect / Cross-cancer / Resistance / Insufficient
    """
    if is_resistance:
        return "Resistance evidence"
    if tier == "exact_alteration" and cancer_match:
        return "Direct evidence"
    if tier == "exact_alteration" and not cancer_match:
        return "Cross-cancer evidence"
    if tier == "alteration_class" and cancer_match:
        return "Related evidence"
    if tier == "alteration_class" and not cancer_match:
        return "Cross-cancer evidence"
    if tier == "gene_level" and cancer_match:
        return "Indirect evidence"
    if tier == "gene_level" and not cancer_match:
        return "Cross-cancer evidence"
    return "Insufficient evidence"

EVIDENCE_CLASS_COLOR = {
    "Direct evidence":       "#27ae60",
    "Related evidence":      "#1F78B4",
    "Indirect evidence":     "#f39c12",
    "Cross-cancer evidence": "#95a5a6",
    "Resistance evidence":   "#e74c3c",
    "Insufficient evidence": "#bdc3c7",
}

def check_comutation_hypotheses(genes_present:set) -> List[dict]:
    hits = []
    for rule in COMUTATION_HYPOTHESES:
        overlap = rule["genes"] & genes_present
        if len(overlap) >= rule["min_hits"]:
            hits.append({
                "pattern":    " + ".join(sorted(overlap)),
                "effect":     rule["effect"],
                "drug":       rule["drug"],
                "note":       rule["note"],
                "confidence": rule["confidence"],
                "refs":       rule["refs"],
            })
    return hits

def build_patient_drug_ranking(verified_df:pd.DataFrame,
                                oncokb_data:dict) -> pd.DataFrame:
    """
    FIX: Rankings from verified evidence only (called with verified_df).
    FIX: Explicit efficacy/resistance/background PMID counts.
    FIX: renamed evidence_priority_score.
    """
    if len(verified_df) == 0:
        return pd.DataFrame()

    ev_col = "final_evidence_type" if "final_evidence_type" in verified_df.columns else "evidence_type"
    results = []

    for drug, grp in verified_df.groupby("drug_primary"):
        genes_supporting = list(grp["query_gene"].unique())

        ev = grp[ev_col].astype(str).str.lower()
        resist_col = grp.get("resistance_evidence",
                             pd.Series(["no"]*len(grp),index=grp.index))

        # FIX: explicit masks
        efficacy_mask   = ev.eq("efficacy")
        resistance_mask = ev.eq("resistance") | resist_col.astype(str).str.lower().eq("yes")
        background_mask = ev.eq("background")

        # FIX: count unique PMIDs per direction
        efficacy_pmids   = grp.loc[efficacy_mask,   "pmid"].nunique()
        resistance_pmids = grp.loc[resistance_mask,  "pmid"].nunique()
        background_pmids = grp.loc[background_mask,  "pmid"].nunique()
        total_pmids      = grp["pmid"].nunique()

        # FIX: quality-weighted top unique PMIDs per direction
        def top_pmid_score(subgrp, n=5):
            if "evidence_quality_score" not in subgrp.columns or len(subgrp)==0:
                return 0.0
            return float(
                subgrp.sort_values("evidence_quality_score", ascending=False)
                      .drop_duplicates("pmid")
                      .head(n)["evidence_quality_score"]
                      .sum()
            )

        eff_quality  = top_pmid_score(grp.loc[efficacy_mask])
        res_quality  = top_pmid_score(grp.loc[resistance_mask])

        # Drug-specific OncoKB match — applied once here only
        oncokb_match      = []
        oncokb_sens_bonus = 0.0
        oncokb_res_pen    = 0.0
        for (gk, vk), ok in oncokb_data.items():
            dl = ok.get("drug_levels",{})
            if str(drug).lower() in dl:
                lvl = dl[str(drug).lower()]
                li  = ONCOKB_LEVELS.get(lvl, ONCOKB_LEVELS["NO"])
                oncokb_match.append(f"{gk} [{lvl}]")
                if "R" in lvl:
                    oncokb_res_pen    += li[3] * 0.5
                else:
                    oncokb_sens_bonus += li[3] * 0.5

        if resistance_pmids > efficacy_pmids:
            direction = "Resistance dominant"; dc = "#e74c3c"
        elif efficacy_pmids > 0 and resistance_pmids == 0:
            direction = "Efficacy only"; dc = "#27ae60"
        elif efficacy_pmids > 0 and resistance_pmids > 0:
            direction = "Mixed evidence"; dc = "#f39c12"
        else:
            direction = "Background/indirect"; dc = "#95a5a6"

        # Net priority = efficacy - resistance + OncoKB signal
        epi_score = round(
            eff_quality + oncokb_sens_bonus
            - res_quality - oncokb_res_pen,
            4)

        results.append({
            "drug":                      drug,
            "supporting_genes":          "; ".join(genes_supporting),
            "efficacy_pmids":            efficacy_pmids,
            "resistance_pmids":          resistance_pmids,
            "background_pmids":          background_pmids,
            "total_pmids":               total_pmids,
            "efficacy_quality_score":    round(eff_quality, 4),
            "resistance_quality_score":  round(res_quality, 4),
            "oncokb_sensitivity_bonus":  round(oncokb_sens_bonus, 4),
            "oncokb_resistance_penalty": round(oncokb_res_pen, 4),
            "direction":                 direction,
            "direction_color":           dc,
            "oncokb_match":              "; ".join(oncokb_match) if oncokb_match else "None",
            "evidence_priority_score":   epi_score,
            "note": (
                "⚠️ Resistance dominant" if resistance_pmids > efficacy_pmids else
                "✅ Efficacy supported"  if efficacy_pmids>0 and resistance_pmids==0 else
                "⚠️ Mixed evidence"      if efficacy_pmids>0 and resistance_pmids>0 else
                "Background evidence only"
            ),
        })

    return (pd.DataFrame(results)
            .sort_values("evidence_priority_score", ascending=False)
            .reset_index(drop=True))

def build_html_report(summary, combined, ranked, patient_drug_ranking,
                       comutations, patient_id, cancer, oncokb_data, out_path):
    """
    Simplified HTML report per reviewer recommendation:
    - Remove drug ranking table
    - Add evidence applicability classification
    - Show evidence sentences
    - Move co-mutation to supplementary
    - Fix OncoKB error display
    """
    ev_col = "final_evidence_type" if "final_evidence_type" in combined.columns else "evidence_type"

    # Metrics
    n_clonal    = int((summary["vaf_clonality"]=="Clonal").sum())
    n_subclonal = int((summary["vaf_clonality"]=="Subclonal").sum())
    n_lowvaf    = int((summary["vaf_clonality"]=="LowVAF").sum())
    total_papers   = int(summary["total_papers"].sum())
    total_verified = int(summary["verified_rows"].sum())

    # ── Per-variant evidence cards ─────────────────────────────
    gene_cards = ""
    for _, r in summary.iterrows():
        gene    = r["gene"]
        variant = r.get("variant","")
        vaf     = r.get("VAF",0)
        clon    = r.get("vaf_clonality","LowVAF")
        vcol    = r.get("vaf_color","#e74c3c")
        ok_lbl  = r.get("oncokb_label","No level")
        ok_col  = r.get("oncokb_color","#b0b0b0")
        ok_onco = r.get("oncokb_oncogenicity","Unknown")
        ok_eff  = r.get("oncokb_mutation_effect","Unknown")
        ok_ml   = r.get("oncokb_match_label","")
        ok_drugs= r.get("oncokb_drugs","")
        clinvar = r.get("clinvar_pathogenicity","Unknown")
        grole   = r.get("gene_role","unknown")
        altcls  = r.get("alteration_class","unknown")
        papers  = r.get("total_papers",0)
        verified= r.get("verified_rows",0)

        # OncoKB status badge
        is_error = "unavailable" in ok_ml.lower() or "error" in ok_ml.lower()
        ok_status_html = (
            f'<span style="background:#b0b0b0;color:#fff;font-size:.7rem;padding:2px 8px;border-radius:20px">'
            f'⚠️ OncoKB Unavailable</span>'
            if is_error else
            f'<span style="background:{ok_col};color:#fff;font-size:.7rem;padding:2px 8px;border-radius:20px">'
            f'{ok_lbl} · {ok_ml}</span>'
        )

        # Evidence rows with classification
        gene_ev = ranked[
            (ranked["query_gene"]==gene) &
            (ranked["patient_variant"].astype(str)==str(variant))
        ] if len(ranked)>0 else pd.DataFrame()

        ev_rows_html = ""
        for _, dr in gene_ev.iterrows():
            drug     = dr.get("drug_primary","?")
            ev       = dr.get(ev_col,"?")
            tier     = dr.get("evidence_tier","gene_level")
            spec     = dr.get("specificity_weight",0.45)
            score    = dr.get("evidence_priority_score",0)
            pmid     = dr.get("pmid","")
            is_res   = bool(dr.get("is_resistance",False))
            cancer_m = bool(dr.get("patient_cancer_match",False))
            ev_class = dr.get("evidence_class",
                              classify_evidence(tier, cancer_m, is_res))
            sent     = str(dr.get("summary_sentence",""))[:200]
            ec_col   = EVIDENCE_CLASS_COLOR.get(ev_class,"#95a5a6")
            ev_col2  = {"efficacy":"#27ae60","resistance":"#e74c3c",
                        "background":"#95a5a6","safety":"#e67e22"}.get(ev,"#95a5a6")

            # Applicability statement
            if ev_class == "Direct evidence":
                applicability = "Exact variant + same cancer: strongest evidence"
            elif ev_class == "Related evidence":
                applicability = "Same alteration class + same cancer: moderate evidence"
            elif ev_class == "Indirect evidence":
                applicability = "Gene-level only: applicability to this variant not established"
            elif ev_class == "Cross-cancer evidence":
                applicability = "Different tumour type: cancer-context transfer required"
            elif ev_class == "Resistance evidence":
                applicability = "Resistance signal: drug may be ineffective"
            else:
                applicability = "Insufficient evidence for this variant"

            ev_rows_html += f"""
            <tr>
              <td><strong>{drug}</strong></td>
              <td><span style="color:{ev_col2};font-weight:600">{ev}</span></td>
              <td><span style="background:{ec_col};color:#fff;font-size:.68rem;
                               padding:2px 7px;border-radius:20px">{ev_class}</span></td>
              <td style="font-size:.72rem">{tier.replace("_"," ")} ({spec:.2f})</td>
              <td>{score:.3f}</td>
              <td><a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                     target="_blank" style="color:#5b4fcf">{pmid}</a></td>
              <td style="font-size:.7rem;color:#475569;max-width:280px">{sent}</td>
              <td style="font-size:.7rem;color:#64748b;font-style:italic">{applicability}</td>
            </tr>"""

        if not ev_rows_html:
            ev_rows_html = '<tr><td colspan="8" style="color:#94a3b8;text-align:center;padding:12px">No verified evidence found for this variant</td></tr>'

        gene_cards += f"""
        <div style="background:#fff;border-radius:10px;
                    box-shadow:0 2px 8px rgba(0,0,0,.06);
                    overflow:hidden;margin-bottom:16px">
          <div style="padding:14px;background:#f8fafc;border-bottom:1px solid #e2e8f0">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px">
              <strong style="font-size:1.15rem;color:#1e2d3d">{gene}</strong>
              <code style="font-size:.75rem;background:#e2e8f0;padding:2px 8px;
                           border-radius:20px;color:#475569">{variant}</code>
              <span style="font-size:.72rem;padding:3px 9px;border-radius:20px;
                           color:#fff;background:{vcol}">VAF {vaf:.3f} · {clon}</span>
              {ok_status_html}
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:5px">
              <span style="font-size:.72rem;background:#fff;padding:2px 7px;
                           border:1px solid #e2e8f0;border-radius:4px;color:#475569">
                🧬 Oncogenicity: {ok_onco}</span>
              <span style="font-size:.72rem;background:#fff;padding:2px 7px;
                           border:1px solid #e2e8f0;border-radius:4px;color:#475569">
                ⚙️ Mutation effect: {ok_eff}</span>
              <span style="font-size:.72rem;background:#fff;padding:2px 7px;
                           border:1px solid #e2e8f0;border-radius:4px;color:#475569">
                🔬 ClinVar: {clinvar}</span>
              <span style="font-size:.72rem;background:#fff;padding:2px 7px;
                           border:1px solid #e2e8f0;border-radius:4px;color:#475569">
                📌 {grole} · {altcls}</span>
              {f'<span style="font-size:.72rem;background:#dbeafe;padding:2px 7px;border-radius:4px;color:#1e40af">💊 OncoKB drugs: {ok_drugs}</span>' if ok_drugs and not is_error else ''}
            </div>
            <div style="margin-top:8px;font-size:.75rem;color:#64748b;
                        background:#f1f5f9;padding:6px 10px;border-radius:6px">
              <strong>Applicability note:</strong>
              {"No therapeutic literature evidence was retrieved for this gene under the current query settings." if papers==0 else
               "Literature records were retrieved, but none passed relation and context verification." if verified==0 else
               "Literature evidence retrieved and graded by alteration specificity and cancer context."}
              {" FGFR2 truncating variants (loss-of-function in an oncogene) may not respond like activating mutations or fusions — applicability not established." if gene=="FGFR2" and altcls=="truncating" else ""}
              {" ARID1A loss-of-function has been investigated as an immunotherapy-associated biomarker, but no patient-specific evidence was verified in this analysis." if gene=="ARID1A" else ""}
              {" KEAP1 loss-of-function has conflicting immunotherapy evidence in NSCLC. No patient-specific evidence verified here." if gene=="KEAP1" else ""}
              {" POLE mutation oncogenicity is Unknown in this report. Hypermutator relevance requires independent pathogenicity and TMB assessment." if gene=="POLE" else ""}
              {" MET P325T is not a classical exon-14 skipping event; MET inhibitor applicability is uncertain." if gene=="MET" else ""}
            </div>
          </div>
          <div style="display:flex;border-bottom:1px solid #f1f5f9">
            <div style="flex:1;text-align:center;padding:9px;border-right:1px solid #f1f5f9">
              <div style="font-size:1.1rem;font-weight:700;color:#5b4fcf">{papers}</div>
              <div style="font-size:.6rem;color:#64748b;text-transform:uppercase">Papers</div>
            </div>
            <div style="flex:1;text-align:center;padding:9px">
              <div style="font-size:1.1rem;font-weight:700;color:#5b4fcf">{verified}</div>
              <div style="font-size:.6rem;color:#64748b;text-transform:uppercase">Verified</div>
            </div>
          </div>
          <div style="padding:10px;overflow-x:auto">
            <table style="width:100%;border-collapse:collapse;font-size:.76rem">
              <thead><tr style="background:#f8fafc">
                <th style="padding:5px 7px;border-bottom:2px solid #e2e8f0;text-align:left;font-size:.66rem;color:#64748b">Drug</th>
                <th style="padding:5px 7px;border-bottom:2px solid #e2e8f0;text-align:left;font-size:.66rem;color:#64748b">Direction</th>
                <th style="padding:5px 7px;border-bottom:2px solid #e2e8f0;text-align:left;font-size:.66rem;color:#64748b">Evidence Class</th>
                <th style="padding:5px 7px;border-bottom:2px solid #e2e8f0;text-align:left;font-size:.66rem;color:#64748b">Specificity</th>
                <th style="padding:5px 7px;border-bottom:2px solid #e2e8f0;text-align:left;font-size:.66rem;color:#64748b">Score</th>
                <th style="padding:5px 7px;border-bottom:2px solid #e2e8f0;text-align:left;font-size:.66rem;color:#64748b">PMID</th>
                <th style="padding:5px 7px;border-bottom:2px solid #e2e8f0;text-align:left;font-size:.66rem;color:#64748b">Evidence Sentence</th>
                <th style="padding:5px 7px;border-bottom:2px solid #e2e8f0;text-align:left;font-size:.66rem;color:#64748b">Applicability</th>
              </tr></thead>
              <tbody>{ev_rows_html}</tbody>
            </table>
          </div>
        </div>"""

    # ── Co-mutation hypotheses (supplementary) ─────────────────
    comut_html = ""
    for c in comutations:
        conf  = c.get("confidence","Low")
        cc    = {"Low":"#e74c3c","Medium":"#f39c12"}.get(conf.split("—")[0].strip(),"#e74c3c")
        comut_html += f"""
        <div style="background:#fff;border-radius:8px;padding:12px;
                    margin-bottom:8px;box-shadow:0 1px 4px rgba(0,0,0,.05)">
          <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px">
            <strong style="font-size:.9rem">{c["pattern"]}</strong>
            <span style="background:{cc};color:#fff;font-size:.68rem;
                         padding:2px 7px;border-radius:20px">{c["effect"]}</span>
            <span style="background:#f1f5f9;color:#475569;font-size:.68rem;
                         padding:2px 7px;border-radius:20px">Confidence: {conf}</span>
          </div>
          <div style="font-size:.78rem;color:#475569;line-height:1.5;
                      background:#fff3cd;padding:7px 10px;border-radius:5px;
                      border-left:3px solid #f39c12">{c["note"]}</div>
          <div style="font-size:.7rem;color:#94a3b8;margin-top:4px">Refs: {c["refs"]}</div>
        </div>"""
    if not comut_html:
        comut_html = "<p style='color:#94a3b8;font-size:.82rem'>No patterns detected.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>megaMine v2.0 — {patient_id}</title>
<style>
  :root{{--navy:#1e2d3d;--blue:#5b4fcf;--font:'Segoe UI',system-ui,sans-serif}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:var(--font);background:#f0f4f8;color:#1e293b;font-size:14px}}
  a{{color:var(--blue)}}
  .hdr{{background:linear-gradient(135deg,#1e2d3d,#5b4fcf);color:#fff;padding:24px 32px}}
  .hdr h1{{font-size:1.4rem;font-weight:800}}
  .hdr p{{opacity:.8;margin-top:4px;font-size:.84rem}}
  .disc{{background:#fff3cd;border-left:4px solid #f39c12;padding:10px 18px;
         margin:12px 32px;border-radius:6px;font-size:.78rem;color:#856404}}
  .metrics{{display:flex;flex-wrap:wrap;gap:8px;padding:0 32px 16px}}
  .m{{background:#fff;border-radius:8px;padding:10px 14px;text-align:center;
      box-shadow:0 2px 6px rgba(0,0,0,.05);border-top:3px solid var(--blue);min-width:90px}}
  .mv{{font-size:1.25rem;font-weight:800;color:var(--navy)}}
  .ml{{font-size:.58rem;color:#64748b;text-transform:uppercase;margin-top:2px}}
  .sec{{padding:0 32px 22px}}
  .st{{font-size:.9rem;font-weight:700;color:var(--navy);margin-bottom:10px;
       padding-bottom:5px;border-bottom:2px solid #e2e8f0}}
  .legend{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}}
  .leg{{font-size:.72rem;padding:3px 9px;border-radius:20px;color:#fff;font-weight:600}}
  footer{{text-align:center;padding:14px;color:#64748b;font-size:.67rem;
          border-top:1px solid #e2e8f0;margin-top:14px}}
</style>
</head><body>
<div class="hdr">
  <h1>🧬 megaMine v2.0 — Patient-Guided Evidence Report</h1>
  <p>Patient: <strong>{patient_id}</strong> &nbsp;|&nbsp;
     Cancer: <strong>{cancer}</strong> &nbsp;|&nbsp;
     APML · Ajou University</p>
</div>
<div class="disc">
  ⚠️ <strong>DISCLAIMER:</strong>
  Literature evidence retrieval only.
  <strong>NOT a clinical treatment recommendation.</strong>
  Evidence is retrieved at gene-level and graded by specificity.
  Exact applicability to patient variants requires expert oncologist review.
  Scores are for literature prioritization only.
</div>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
            padding:8px 18px;margin:0 32px 12px;font-size:.76rem;color:#475569">
  <strong>OncoKB annotation:</strong>
  {f"Available — queried {len(summary)} variants"
   if any(r.get("oncokb_label","") not in ("Unavailable","No token","API error","")
          for _,r in summary.iterrows())
   else "Unavailable — OncoKB API could not be reached. Variant-level OncoKB interpretation not included in this report."}
  &nbsp;|&nbsp;
  <strong>Same-cancer verified evidence:</strong>
  {int(summary["same_cancer_verified_rows"].sum()) if "same_cancer_verified_rows" in summary.columns else 0} rows
  &nbsp;|&nbsp;
  <strong>Cross-cancer evidence:</strong>
  {int(summary["cross_cancer_verified_rows"].sum()) if "cross_cancer_verified_rows" in summary.columns else 0} rows
</div>
<div class="metrics">
  <div class="m"><div class="mv">{len(summary)}</div><div class="ml">Variants</div></div>
  <div class="m" style="border-color:#27ae60">
    <div class="mv" style="color:#27ae60">{n_clonal}</div><div class="ml">Clonal</div></div>
  <div class="m" style="border-color:#f39c12">
    <div class="mv" style="color:#f39c12">{n_subclonal}</div><div class="ml">Subclonal</div></div>
  <div class="m" style="border-color:#e74c3c">
    <div class="mv" style="color:#e74c3c">{n_lowvaf}</div><div class="ml">Low VAF</div></div>
  <div class="m"><div class="mv">{total_papers}</div><div class="ml">Papers</div></div>
  <div class="m" style="border-color:#1F78B4">
    <div class="mv" style="color:#1F78B4">{total_verified}</div>
    <div class="ml">Relation Verified</div></div>
  <div class="m" style="border-color:#27ae60">
    <div class="mv" style="color:#27ae60">{{summary["same_cancer_verified_rows"].sum() if "same_cancer_verified_rows" in summary.columns else 0}}</div>
    <div class="ml">Same-Cancer</div></div>
</div>

<div class="sec">
  <div class="st">🧬 Per-Variant Evidence with Specificity Grading</div>
  <div class="legend">
    <span class="leg" style="background:#27ae60">Direct — exact variant, same cancer</span>
    <span class="leg" style="background:#1F78B4">Related — alteration class, same cancer</span>
    <span class="leg" style="background:#f39c12">Indirect — gene-level, same cancer</span>
    <span class="leg" style="background:#95a5a6">Cross-cancer — different tumour type</span>
    <span class="leg" style="background:#e74c3c">Resistance evidence</span>
  </div>
  {gene_cards}
</div>

<div class="sec">
  <div class="st">🔗 Exploratory Co-mutation Hypothesis Signals
    <span style="font-size:.72rem;font-weight:400;color:#e74c3c;margin-left:8px">
    ⚠️ Hypothesis-generating only — NOT validated biomarkers — require confirmation</span>
  </div>
  {comut_html}
</div>

<div class="sec">
  <div class="st">✅ Verification Ladder</div>
  <div style="display:flex;flex-direction:column;gap:4px">
    <div style="padding:6px 10px;background:#d4edda;color:#155724;border-radius:5px;font-size:.77rem">✅ Entity verified — gene in HGNC, drug in curated whitelist</div>
    <div style="padding:6px 10px;background:#d4edda;color:#155724;border-radius:5px;font-size:.77rem">✅ Sentence relation verified — gene–drug co-mention</div>
    <div style="padding:6px 10px;background:#d4edda;color:#155724;border-radius:5px;font-size:.77rem">✅ Deduplication — PMID+gene+drug+cancer+tier+variant</div>
    <div style="padding:6px 10px;background:#fff3cd;color:#856404;border-radius:5px;font-size:.77rem">⚡ Three-tier specificity — exact (1.00) · class (0.75) · gene (0.45)</div>
    <div style="padding:6px 10px;background:#fff3cd;color:#856404;border-radius:5px;font-size:.77rem">⚡ Evidence direction — efficacy/resistance/background by offline LLM</div>
    <div style="padding:6px 10px;background:#d1ecf1;color:#0c5460;border-radius:5px;font-size:.77rem">ℹ️ VAF as confidence modifier — not primary actionability tier</div>
    <div style="padding:6px 10px;background:#d1ecf1;color:#0c5460;border-radius:5px;font-size:.77rem">ℹ️ OncoKB exact alteration annotation (when available)</div>
  </div>
</div>

<footer>
  megaMine v2.0 — APML, Ajou University · Literature evidence retrieval only.<br>
  Not a clinical recommendation. Scores for expert literature review only.
</footer>
</body></html>"""

    with open(out_path,"w",encoding="utf-8") as f:
        f.write(html)
    sz = Path(out_path).stat().st_size//1024
    print(f"✅ HTML: {out_path} ({sz} KB)")


def run_maf_pipeline(maf_path, cancer, out_dir, email, api_key,
                     hgnc_cache, vaf_min=0.0, max_records=50,
                     top_drugs=3, years="2018-2024"):

    # FIX: token from env var only
    oncokb_token = os.environ.get("ONCOKB_TOKEN","")

    from megaMine.utils.normalizers import (normalize_cancer_column,
                                             add_resistance_context,
                                             reconcile_evidence_type)
    from megaMine.modules.llm_verify import run_llm_verification

    out = Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    tmp = out/"tmp_queries"; tmp.mkdir(exist_ok=True)
    patient_id = Path(maf_path).stem

    print("="*65)
    print("MEGAMINE v2.0 — MAF EVIDENCE SYNTHESIS PIPELINE v4")
    print(f"Patient: {patient_id} | Cancer: {cancer}")
    print(f"OncoKB: {'enabled' if oncokb_token else 'disabled (export ONCOKB_TOKEN=...)'}")
    print("="*65)

    print("\n[1/7] Reading MAF...")
    maf = read_maf(maf_path)
    print(f"  Mutations: {len(maf):,}")

    print("\n[2/7] Extracting cancer-relevant variants...")
    mutations = extract_mutations(maf, vaf_min=vaf_min)
    if len(mutations)==0:
        print("  No actionable mutations found."); return
    for _, r in mutations.iterrows():
        print(f"  {r['gene']:<8} {str(r.get('variant','')):<15} "
              f"VAF={r['VAF']:.3f} ({r['vaf_clonality']}) "
              f"class={r['alteration_class']} role={r['gene_role']}")

    genes_present = set(mutations["gene"].tolist())

    print("\n[3/7] OncoKB exact alteration + ClinVar...")
    # FIX: (gene,variant) tuple keys
    oncokb_data  = {}
    clinvar_data = {}
    all_drug_levels = {}  # {(gene,variant): {drug_lower: level}}

    for _, mut in mutations.iterrows():
        gene    = str(mut["gene"])
        variant = str(mut.get("variant",""))
        vk      = (gene, variant)  # FIX: variant-specific key
        ok = query_oncokb_exact(gene, variant, cancer, oncokb_token)
        cv = query_clinvar(gene, variant, email, api_key)
        oncokb_data[vk]   = ok
        clinvar_data[vk]  = cv
        all_drug_levels[vk] = ok.get("drug_levels",{})
        print(f"  {gene} {variant}: {ok['match_label']} | "
              f"OncoKB={ok['label']} | ClinVar={cv['pathogenicity']}")
        time.sleep(0.4)

    print("\n[4/7] Co-mutation hypothesis check...")
    comutations = check_comutation_hypotheses(genes_present)
    for c in comutations:
        print(f"  {c['pattern']} → {c['effect']} [{c['confidence']}]")
    if not comutations:
        print("  None detected.")

    print(f"\n[5/7] Three-tier literature queries ({years})...")
    all_rows = []
    for _, mut in mutations.iterrows():
        gene     = str(mut["gene"])
        variant  = str(mut.get("variant",""))
        altclass = str(mut.get("alteration_class","unknown"))
        vk       = (gene, variant)
        mut_eff  = oncokb_data.get(vk,{}).get("mutation_effect","Unknown")
        print(f"  → {gene} {variant} ({altclass}) effect={mut_eff}:")
        df = run_tiered_queries(gene, variant, altclass, cancer,
                                mut_eff, email, api_key, hgnc_cache,
                                tmp, max_records//3, years)
        if len(df)>0:
            df["patient_variant"]  = variant
            df["patient_VAF"]      = mut["VAF"]
            df["vaf_clonality"]    = mut["vaf_clonality"]
            df["vaf_confidence"]   = mut["vaf_conf"]
            df["alteration_class"] = altclass
            df["gene_role"]        = mut["gene_role"]
            all_rows.append(df)
        time.sleep(0.5)

    if not all_rows:
        print("No evidence found."); return

    print("\n[6/7] Processing evidence...")
    combined = pd.concat(all_rows, ignore_index=True)
    combined = normalize_cancer_column(combined)
    combined = add_resistance_context(combined)
    combined = reconcile_evidence_type(combined)
    combined = deduplicate_evidence(combined)
    print(f"  After deduplication: {len(combined)} rows")

    combined = run_llm_verification(combined, model="offline",
                                    confidence_threshold=0.60)

    # FIX: variant-specific VAF mapping
    vaf_map = {
        (str(r["gene"]), str(r["variant"])): float(r["vaf_conf"])
        for _, r in mutations.iterrows()
    }
    combined["vaf_confidence"] = combined.apply(
        lambda r: vaf_map.get(
            (str(r["query_gene"]), str(r["patient_variant"])), 0.7),
        axis=1)

    # Score with drug-specific OncoKB boost per (gene,variant)
    scored = []
    for _, mut in mutations.iterrows():
        gene    = str(mut["gene"])
        variant = str(mut.get("variant",""))
        vk      = (gene, variant)
        mask    = ((combined["query_gene"]==gene) &
                   (combined["patient_variant"].astype(str)==variant))
        part    = score_evidence(combined[mask], all_drug_levels.get(vk,{}))
        scored.append(part)
    if scored:
        combined = pd.concat(scored, ignore_index=True)

    # Apply evidence classification
    ev_col_tmp = "final_evidence_type" if "final_evidence_type" in combined.columns else "evidence_type"
    combined["evidence_class"] = combined.apply(lambda r: classify_evidence(
        str(r.get("evidence_tier","gene_level")),
        bool(r.get("patient_cancer_match", False)),
        bool(r.get("is_resistance", False))
    ), axis=1)

    # FIX: query-gene consistency gate
    # Reject rows where query gene absent from biomarker AND evidence sentence
    def query_gene_relation_valid(row) -> bool:
        qg   = str(row.get("query_gene","")).upper().strip()
        bm   = str(row.get("biomarker","")).upper().strip()
        sent = " ".join(str(row.get(c,"")) for c in
                        ["summary_sentence","conclusion","evidence_sentence"]
                        ).upper()
        if not qg: return False
        return (bm == qg or
                re.search(rf"\\b{re.escape(qg)}\\b", sent) is not None)

    combined["query_gene_relation_valid"] = combined.apply(
        query_gene_relation_valid, axis=1)

    n_invalid = (~combined["query_gene_relation_valid"]).sum()
    if n_invalid > 0:
        print(f"  ⚠️  Rejecting {n_invalid} rows: query gene absent from relation")
        for _, r in combined[~combined["query_gene_relation_valid"]].iterrows():
            print(f"    Rejected: query={r['query_gene']} "
                  f"biomarker={r.get('biomarker','')} "
                  f"drug={r.get('drug_primary','')}")
        combined.loc[~combined["query_gene_relation_valid"],
                     "llm_verified"] = "no"
        combined.loc[~combined["query_gene_relation_valid"],
                     "llm_reason"]   = "Rejected: query gene absent from relation"

    verified = combined[combined["llm_verified"]=="yes"].copy()
    print(f"  Relation-verified: {len(verified)}/{len(combined)}")

    # FIX: normalized cancer alias matching — no substring fallback
    CANCER_ALIASES = {
        "nsclc":{"non-small cell lung cancer","nsclc","lung adenocarcinoma",
                 "lung squamous cell carcinoma","non-small-cell lung cancer",
                 "non small cell lung cancer","lung cancer"},
        "non-small cell lung cancer":{"non-small cell lung cancer","nsclc",
                 "lung adenocarcinoma","lung squamous cell carcinoma",
                 "non-small-cell lung cancer","lung cancer"},
        "breast cancer":{"breast cancer","breast carcinoma",
                 "triple-negative breast cancer","her2-positive breast cancer",
                 "breast neoplasm"},
        "colorectal cancer":{"colorectal cancer","colon cancer","rectal cancer",
                 "colorectal carcinoma","crc"},
        "acute myeloid leukemia":{"acute myeloid leukemia","aml",
                 "acute myelogenous leukemia"},
        "cholangiocarcinoma":{"cholangiocarcinoma","bile duct cancer",
                 "intrahepatic cholangiocarcinoma","biliary tract cancer"},
        "gastrointestinal stromal tumor":{"gastrointestinal stromal tumor","gist"},
        "thyroid cancer":{"thyroid cancer","papillary thyroid carcinoma",
                 "medullary thyroid carcinoma","thyroid carcinoma"},
        "urothelial carcinoma":{"urothelial carcinoma","bladder cancer",
                 "bladder carcinoma","urothelial cancer"},
        "lymphoma":{"lymphoma","diffuse large b-cell lymphoma","dlbcl",
                 "mantle cell lymphoma","follicular lymphoma"},
        "glioma":{"glioma","glioblastoma","gbm","astrocytoma","brain tumor"},
    }

    def get_cancer_aliases(cancer_str:str) -> set:
        cl = cancer_str.lower().strip()
        if cl in CANCER_ALIASES: return CANCER_ALIASES[cl]
        for canonical, aliases in CANCER_ALIASES.items():
            if cl in aliases: return aliases | {canonical}
        return {cl}  # exact match only, no fallback

    patient_aliases = get_cancer_aliases(cancer)
    print(f"  Cancer aliases: {sorted(patient_aliases)[:3]}...")

    def is_cancer_match(ct:str) -> bool:
        return str(ct).lower().strip() in patient_aliases

    for df_name, df_obj in [("combined", combined), ("verified", verified)]:
        ct_col = "canonical_cancer_type"
        if ct_col in df_obj.columns:
            df_obj["patient_cancer_match"] = df_obj[ct_col].apply(is_cancer_match)
        else:
            df_obj["patient_cancer_match"] = False

    cancer_matched = verified[verified["patient_cancer_match"]==True].copy()
    cross_cancer   = verified[verified["patient_cancer_match"]==False].copy()
    print(f"  Cancer-matched: {len(cancer_matched)} | Cross-cancer: {len(cross_cancer)}")
    if len(cancer_matched)==0:
        print("  ⚠️  No same-cancer verified evidence. "
              "Drug ranking empty — cross-cancer in Excel.")

    # FIX: ranked from verified only, grouped by gene+variant
    if len(verified)>0:
        ranked = (verified
                  .sort_values("evidence_priority_score", ascending=False)
                  .groupby(["query_gene","patient_variant"], dropna=False)
                  .head(top_drugs))
    else:
        ranked = pd.DataFrame()

    print("\n[7/7] Building reports...")
    summary = []
    for _, mut in mutations.iterrows():
        gene    = str(mut["gene"])
        variant = str(mut.get("variant",""))
        vk      = (gene, variant)
        rows    = combined[(combined["query_gene"]==gene) &
                           (combined["patient_variant"].astype(str)==variant)]
        ver     = verified[(verified["query_gene"]==gene) &
                           (verified["patient_variant"].astype(str)==variant)]
        top     = ranked[(ranked["query_gene"]==gene) &
                         (ranked["patient_variant"].astype(str)==variant)] if len(ranked)>0 else pd.DataFrame()
        ok      = oncokb_data.get(vk,{})
        cv      = clinvar_data.get(vk,{})
        ok_drugs_str = "; ".join([d["drug"] for d in ok.get("drugs",[])[:3]])

        # FIX: top drug from same-cancer evidence only
        same_cancer_ver = ver[ver.get("patient_cancer_match", pd.Series([False]*len(ver), index=ver.index))==True] if "patient_cancer_match" in ver.columns else pd.DataFrame()
        cross_cancer_ver= ver[ver.get("patient_cancer_match", pd.Series([True]*len(ver), index=ver.index))==False] if "patient_cancer_match" in ver.columns else pd.DataFrame()

        if len(same_cancer_ver)>0 and "evidence_priority_score" in same_cancer_ver.columns:
            top_same = same_cancer_ver.sort_values("evidence_priority_score",ascending=False).iloc[0]
            top_drug_sc    = top_same.get("drug_primary","N/A")
            top_ev_sc      = top_same.get("final_evidence_type","N/A")
            top_score_sc   = top_same.get("evidence_priority_score",0)
            top_pmid_sc    = top_same.get("pmid","")
        else:
            top_drug_sc  = "N/A"
            top_ev_sc    = "No same-cancer evidence"
            top_score_sc = 0.0
            top_pmid_sc  = ""

        top_drug_cc = cross_cancer_ver.iloc[0].get("drug_primary","N/A") if len(cross_cancer_ver)>0 else "N/A"

        summary.append({
            "gene":                   gene,
            "variant":                variant,
            "VAF":                    mut["VAF"],
            "vaf_clonality":          mut["vaf_clonality"],
            "vaf_conf":               mut["vaf_conf"],
            "vaf_color":              mut["vaf_color"],
            "alteration_class":       mut["alteration_class"],
            "gene_role":              mut["gene_role"],
            "oncokb_label":           ok.get("label","No level"),
            "oncokb_color":           ok.get("color","#999999"),
            "oncokb_oncogenicity":    ok.get("oncogenicity","Unknown"),
            "oncokb_mutation_effect": ok.get("mutation_effect","Unknown"),
            "oncokb_match_label":     ok.get("match_label",""),
            "oncokb_sensitive_level": ok.get("sensitive_level",""),
            "oncokb_resistant_level": ok.get("resistant_level",""),
            "oncokb_drugs":              ok_drugs_str,
            "oncokb_score":              ok.get("score",0),
            "clinvar_pathogenicity":     cv.get("pathogenicity","Unknown"),
            "total_papers":              rows["pmid"].nunique(),
            "relation_verified_rows":    int(len(ver)),
            "same_cancer_verified_rows": int(len(same_cancer_ver)),
            "cross_cancer_verified_rows":int(len(cross_cancer_ver)),
            "top_drug":                  top_drug_sc,
            "top_evidence_type":         top_ev_sc,
            "top_score":                 top_score_sc,
            "top_pmid":                  top_pmid_sc,
            "top_cross_cancer_drug":     top_drug_cc,
        })

    df_summary = pd.DataFrame(summary)
    # FIX: cancer-matched only for patient ranking — no fallback
    patient_drug_ranking = build_patient_drug_ranking(cancer_matched, oncokb_data)

    print(f"\n{'='*65}")
    print("PATIENT-LEVEL DRUG RANKING (verified evidence only):")
    for i,(_, r) in enumerate(patient_drug_ranking.head(8).iterrows()):
        print(f"  {i+1}. {r['drug']:<22} "
              f"score={r['evidence_priority_score']:.4f}  "
              f"{r['direction']}  OncoKB={r['oncokb_match']}")

    print(f"\nPER-VARIANT SUMMARY:")
    for _, r in df_summary.iterrows():
        print(f"  {r['gene']:<8} {str(r['variant']):<15} "
              f"VAF={r['VAF']:.3f} OncoKB={r['oncokb_label']:<10} "
              f"[{r['oncokb_match_label']}] top={r['top_drug']}")

    out_xlsx = out/f"{patient_id}_megaMine_evidence.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        mutations.to_excel(xl,             sheet_name="Mutations",             index=False)
        df_summary.to_excel(xl,            sheet_name="Summary",               index=False)
        patient_drug_ranking.to_excel(xl,  sheet_name="Patient_Drug_Ranking",  index=False)
        (ranked.to_excel(xl,               sheet_name="Top_Evidence",          index=False)
         if len(ranked)>0 else None)
        combined.to_excel(xl,              sheet_name="All_Evidence",          index=False)
        verified.to_excel(xl,              sheet_name="Verified_Evidence",     index=False)
        pd.DataFrame(comutations).to_excel(xl, sheet_name="CoMutation_Hypotheses", index=False)
        pd.DataFrame([{
            "patient":    patient_id,"cancer":cancer,"vaf_min":vaf_min,
            "years":years,"oncokb":bool(oncokb_token),
            "n_variants": len(mutations),"n_rows":len(combined),
            "n_verified": len(verified),"version":"megaMine_v2.0.0",
        }]).to_excel(xl, sheet_name="RunInfo", index=False)
    print(f"✅ Excel: {out_xlsx}")

    out_html = out/f"{patient_id}_megaMine_report.html"
    build_html_report(df_summary, combined, ranked,
                      patient_drug_ranking, comutations,
                      patient_id, cancer, oncokb_data, out_html)

    import subprocess
    subprocess.Popen(["open", str(out_html)])
    print(f"\n⚠️  Evidence priority scores for literature review only.")
    print(f"   Not a clinical treatment recommendation.")
    return df_summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "megaMine MAF-driven precision-oncology "
            "evidence synthesis pipeline"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment:
  export ONCOKB_TOKEN=your_token   # free from oncokb.org/account/register

Examples:
  megaMine-maf --maf patient.maf --cancer NSCLC --out results/
  megaMine-maf --maf patient.maf --cancer Breast --vaf-min 0.05 --out results/
        """)
    parser.add_argument("--maf",          required=True)
    parser.add_argument("--cancer",       required=True)
    parser.add_argument("--out",          required=True)
    parser.add_argument("--email",        default="user@institution.edu")
    parser.add_argument("--ncbi-api-key", default="")
    parser.add_argument("--hgnc-cache",   default="")
    parser.add_argument("--vaf-min",      type=float, default=0.0)
    parser.add_argument("--max-records",  type=int,   default=50)
    parser.add_argument("--top",          type=int,   default=3)
    parser.add_argument("--years",        default="2018-2024")
    args = parser.parse_args()
    run_maf_pipeline(
        maf_path=args.maf, cancer=args.cancer, out_dir=args.out,
        email=args.email, api_key=args.ncbi_api_key,
        hgnc_cache=args.hgnc_cache, vaf_min=args.vaf_min,
        max_records=args.max_records, top_drugs=args.top,
        years=args.years,
    )

if __name__ == "__main__":
    main()
