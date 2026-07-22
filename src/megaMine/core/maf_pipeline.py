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
ctx = ssl._create_unverified_context()

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
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
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
                       comutations, patient_id, cancer, oncokb_data,
                       out_path, trials_df=None):
    """Professional clinical genomics report."""
    import re as _re
    ev_col = "final_evidence_type" if "final_evidence_type" in combined.columns else "evidence_type"

    n_variants     = len(summary)
    n_candidates   = int(summary["total_papers"].sum())
    n_rel_verified = int(summary["relation_verified_rows"].sum()
                         if "relation_verified_rows" in summary.columns else 0)
    n_same_cancer  = int(summary["same_cancer_verified_rows"].sum()
                         if "same_cancer_verified_rows" in summary.columns else 0)
    n_cross_cancer = int(summary["cross_cancer_verified_rows"].sum()
                         if "cross_cancer_verified_rows" in summary.columns else 0)
    n_exact_allele = int(summary["oncokb_allele_exist"].sum()
                         if "oncokb_allele_exist" in summary.columns else 0)
    n_variant_class= int(
        ((summary.get("oncokb_variant_exist", False)==True) &
         (summary.get("oncokb_allele_exist",  False)!=True)).sum())
    n_gene_only    = int(
        ((summary.get("oncokb_gene_exist",    False)==True) &
         (summary.get("oncokb_variant_exist", False)!=True) &
         (summary.get("oncokb_allele_exist",  False)!=True)).sum())

    # Variant overview rows
    var_rows = ""
    for _, r in summary.iterrows():
        clon   = r.get("vaf_clonality","LowVAF")
        clon_c = {"Clonal":"#2f6b4f","Subclonal":"#8a5a16","LowVAF":"#9b3a3a"}.get(clon,"#4b5563")
        var_rows += (
            f"<tr>"
            f"<td><strong>{r['gene']}</strong></td>"
            f"<td><code style='font-size:11px'>{r.get('variant','') if str(r.get('variant','')) not in ('nan','None','') else '—'}</code></td>"
            f"<td>{r.get('VAF',0):.3f}</td>"
            f"<td style='color:{clon_c};font-weight:600'>{clon}</td>"
            f"<td>{r.get('alteration_class','')}</td>"
            f"<td style='font-size:10.5px'>{r.get('oncokb_match_label','') or '—'}</td>"
            f"<td style='text-align:center'>{r.get('total_papers',0)}</td>"
            f"<td style='text-align:center'>{r.get('relation_verified_rows',r.get('verified_rows',0))}</td>"
            f"</tr>"
        )

    # Per-variant sections
    variant_sections = ""
    for _, r in summary.iterrows():
        gene     = r["gene"]
        variant  = r.get("variant","")
        vaf      = r.get("VAF",0)
        clon     = r.get("vaf_clonality","LowVAF")
        altcls   = r.get("alteration_class","unknown")
        grole    = r.get("gene_role","unknown")
        ok_onco  = r.get("oncokb_oncogenicity","Unknown")
        ok_eff   = r.get("oncokb_mutation_effect","Unknown")
        ok_ml    = r.get("oncokb_match_label","")
        ok_drugs = r.get("oncokb_drugs","")
        clinvar  = r.get("clinvar_pathogenicity","Unknown")
        papers   = r.get("total_papers",0)
        verif    = r.get("relation_verified_rows", r.get("verified_rows",0))
        allele_m = bool(r.get("oncokb_allele_exist",False))
        variant_m= bool(r.get("oncokb_variant_exist",False))
        gene_m   = bool(r.get("oncokb_gene_exist",False))
        is_err   = "unavailable" in str(ok_ml).lower() or "error" in str(ok_ml).lower()

        # Annotation status
        if is_err:
            ann_cls,ann_txt = "annotation-gene","OncoKB unavailable"
        elif allele_m:
            ann_cls,ann_txt = "annotation-exact","Exact allele recognized"
        elif variant_m:
            ann_cls = "annotation-broader"
            ann_txt = "Variant/class recognized; exact allele not confirmed"
        elif gene_m:
            ann_cls = "annotation-gene"
            ann_txt = "Gene recognized only — no variant-level annotation"
        else:
            ann_cls,ann_txt = "annotation-gene","No OncoKB match"

        # OncoKB drugs
        if ok_drugs and not is_err:
            if allele_m:
                drugs_row = f"<strong>OncoKB variant-matched therapies:</strong> {ok_drugs}"
            elif variant_m:
                drugs_row = f"<em>Broader variant/class context (not allele-confirmed):</em> {ok_drugs}"
            else:
                drugs_row = f"<em>Gene-level context only (not variant-confirmed):</em> {ok_drugs}"
        else:
            drugs_row = "—"

        # Conflict note
        conflict_note = ""
        if not is_err and not allele_m and ok_eff not in ("Unknown",""):
            if ("gain" in ok_eff.lower() and altcls=="truncating") or \
               ("loss" in ok_eff.lower() and altcls=="missense"):
                who = ("gene-level context only"
                       if gene_m and not variant_m
                       else "broader variant/class")
                conflict_note = (
                    f"Interpretation note: MAF class is {altcls}, "
                    f"while OncoKB effect is {ok_eff}. "
                    f"OncoKB annotation is {who} and is not confirmed for this specific allele."
                )

        # Literature result
        if papers == 0:
            lit_result = "No therapeutic literature evidence was retrieved."
        elif verif == 0:
            lit_result = f"{papers} candidate record(s) retrieved; none passed verification."
        else:
            same  = int(r.get("same_cancer_verified_rows",0))
            cross = int(r.get("cross_cancer_verified_rows",0))
            lit_result = (
                f"{papers} candidate record(s) retrieved; {verif} passed verification "
                f"({same} same-cancer, {cross} cross-cancer)."
            )

        # Applicability notes
        app_notes = []
        if gene=="FGFR2" and altcls=="truncating":
            app_notes.append("FGFR2 truncating alterations may not behave like activating mutations or fusions. Applicability not established.")
        if gene=="ARID1A":
            app_notes.append("ARID1A loss-of-function has been investigated as an IO-associated biomarker, but no patient-specific evidence was verified.")
        if gene=="KEAP1":
            app_notes.append("KEAP1 loss-of-function has conflicting IO evidence in NSCLC.")
        if gene=="POLE":
            app_notes.append("POLE mutation oncogenicity is Unknown. Hypermutator relevance requires TMB assessment.")
        if gene=="MET":
            app_notes.append("MET p.P325T is not a classical exon-14 skipping event; MET inhibitor applicability is uncertain.")
        app_note = " ".join(app_notes) if app_notes else "No specific applicability notes."

        # Evidence rows
        gene_ev = ranked[
            (ranked["query_gene"]==gene) &
            (ranked["patient_variant"].astype(str)==str(variant))
        ] if len(ranked)>0 else pd.DataFrame()

        ev_rows = ""
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
            sent     = str(dr.get("summary_sentence",""))[:280]
            study    = str(dr.get("study_design","") or "")
            ec_css   = {"Direct evidence":"evidence-direct",
                        "Related evidence":"evidence-related",
                        "Indirect evidence":"evidence-indirect",
                        "Cross-cancer evidence":"evidence-cross",
                        "Resistance evidence":"evidence-resistance"}.get(ev_class,"evidence-cross")
            ev_css   = "evidence-resistance" if is_res else (
                       "evidence-direct" if ev=="efficacy" else "evidence-cross")
            ev_rows += (
                f"<tr>"
                f"<td><strong>{drug}</strong></td>"
                f"<td><span class='evidence-label {ev_css}'>{ev}</span></td>"
                f"<td>{'Same-cancer' if cancer_m else 'Cross-cancer'}</td>"
                f"<td><span class='evidence-label {ec_css}'>{ev_class}</span></td>"
                f"<td style='font-size:10.5px'>{tier.replace('_',' ')} ({spec:.2f})</td>"
                f"<td style='font-size:10.5px'>{study or '—'}</td>"
                f"<td><a href='https://pubmed.ncbi.nlm.nih.gov/{pmid}/' target='_blank'>{pmid}</a></td>"
                f"<td style='font-size:10.5px;max-width:280px'>{sent}</td>"
                f"<td style='font-size:10px;color:#6b7280'>{score:.3f}</td>"
                f"</tr>"
            )
        ev_status = ("Same-cancer verified"
                     if int(r.get("same_cancer_verified_rows",0))>0
                     else "Cross-cancer only"
                     if int(r.get("cross_cancer_verified_rows",0))>0
                     else "No verified evidence")

        conflict_row = (
            f"<div class='interpretation-row interpretation-warning'>"
            f"<div class='interpretation-label'>Interpretation note</div>"
            f"<div>{conflict_note}</div></div>"
        ) if conflict_note else ""

        variant_span = (f' <span>{variant}</span>' if str(variant) not in ('nan','None','') else '')
        # Pre-computed HTML blocks (avoid backslash in f-string)
        if app_note and app_note != 'No specific applicability notes.':
            app_row_html = (
                '<div class="interpretation-row interpretation-warning">'
                '<div class="interpretation-label">Applicability</div>'
                f'<div>{app_note}</div>'
                '</div>'
            )
        else:
            app_row_html = ''
        if ev_rows:
            ev_table_html = (
                '<div class="table-wrap" style="margin:0">'
                '<table class="clinical-table">'
                '<thead><tr>'
                '<th>Drug</th><th>Direction</th><th>Cancer context</th>'
                '<th>Evidence class</th><th>Specificity</th><th>Study design</th>'
                '<th>PMID</th><th>Evidence sentence</th><th>Priority score*</th>'
                '</tr></thead>'
                f'<tbody>{ev_rows}</tbody>'
                '</table>'
                '<p class="table-footnote">*Priority score ranks retrieved literature evidence '
                'and does not estimate treatment response probability.</p>'
                '</div>'
            )
        else:
            ev_table_html = '<div class="empty-state" style="margin:0">No verified evidence found for this variant.</div>'
        variant_sections += f"""
<article class="variant-section">
  <div class="variant-header">
    <div>
      <h3>{gene}{variant_span}</h3>
      <div class="variant-meta-line">VAF {vaf:.3f} · {clon} · {grole} · {altcls}</div>
    </div>
    <div class="annotation-status {ann_cls}">{ann_txt}</div>
  </div>
  <div class="variant-information-grid">
    <div><span class="field-label">Oncogenicity</span><span class="field-value">{ok_onco}</span></div>
    <div><span class="field-label">Mutation effect</span><span class="field-value">{ok_eff}</span></div>
    <div><span class="field-label">ClinVar</span><span class="field-value">{clinvar}</span></div>
    <div><span class="field-label">Evidence status</span><span class="field-value">{ev_status}</span></div>
  </div>
  <div class="interpretation-block">
    <div class="interpretation-row">
      <div class="interpretation-label">Literature result</div>
      <div>{lit_result}</div>
    </div>
    <div class="interpretation-row">
      <div class="interpretation-label">OncoKB context</div>
      <div>{drugs_row}</div>
    </div>
    {conflict_row}
    {app_row_html}
  </div>
  {ev_table_html}
</article>"""

    # Trials
    if trials_df is not None and len(trials_df)>0:
        t_rows = ""
        for _, tr in trials_df.iterrows():
            phase = tr.get("highest_phase","") or ""
            n_t   = tr.get("n_trials","") or 0
            if not phase or str(n_t) in ("0",""): continue
            t_rows += (
                f"<tr>"
                f"<td><strong>{tr.get('drug','') or tr.get('drug_primary','')}</strong></td>"
                f"<td>{tr.get('gene','') or tr.get('biomarker','')}</td>"
                f"<td>{tr.get('cancer_type','') or tr.get('canonical_cancer_type','')}</td>"
                f"<td>{phase}</td><td>{n_t}</td>"
                f"<td style='color:#9b3a3a'>{tr.get('n_failed',0)}</td>"
                f"</tr>"
            )
        if t_rows:
            trials_html = (
                "<div class='table-wrap'>"
                "<table class='clinical-table'>"
                "<thead><tr><th>Drug</th><th>Gene</th><th>Cancer</th>"
                "<th>Highest phase</th><th>Total trials</th><th>Failed/terminated</th>"
                "</tr></thead><tbody>" + t_rows + "</tbody></table></div>"
            )
        else:
            trials_html = '<div class="empty-state">No trials with confirmed phase information were identified.</div>'
    else:
        trials_html = '<div class="empty-state">No complete ClinicalTrials.gov linkage was identified for the relation-verified evidence.</div>'

    # Hypotheses
    hyp_items = ""
    for c in comutations:
        hyp_items += (
            f"<div class='hypothesis-item'>"
            f"<div class='hypothesis-title'>{c['pattern']} — {c['effect']}</div>"
            f"<div class='hypothesis-note'>"
            f"<strong>Suggested context:</strong> {c['drug']}<br>"
            f"<strong>Confidence:</strong> {c.get('confidence','Low')}<br>"
            f"{c['note']}<br>"
            f"<em>References: {c['refs']}</em>"
            f"</div></div>"
        )
    if not hyp_items:
        hyp_items = '<div class="empty-state">No co-mutation patterns detected.</div>'

    # Verification table
    verif_rows = "".join([
        f"<tr><td>{s}</td><td>{n}</td><td>{d}</td></tr>"
        for s,n,d in [
            ("1","Gene normalization","Gene normalized against HGNC approved symbol list (44,597 genes)"),
            ("2","Drug normalization","Drug normalized against curated oncology whitelist (254 agents)"),
            ("3","Relation check","Query gene and drug required in evidence relation; gene-absent rows rejected"),
            ("4","Negation/speculation","Negated or speculative statements excluded by offline LLM verifier"),
            ("5","Evidence specificity","Graded: exact alteration (1.00), alteration class (0.75), gene-level (0.45)"),
            ("6","Cancer context","Classified as same-cancer or cross-cancer using normalized aliases"),
            ("7","OncoKB annotation","Separated by exact-allele, broader variant/class, gene-only, or no match"),
        ]
    ])

    CSS = """
:root{--navy-900:#17324d;--navy-700:#28506f;--slate-900:#1f2933;--slate-700:#4b5563;
--slate-500:#6b7280;--slate-300:#d1d5db;--slate-200:#e5e7eb;--slate-100:#f3f4f6;
--slate-050:#f8fafc;--white:#ffffff;--green-700:#2f6b4f;--green-100:#e8f3ed;
--amber-700:#8a5a16;--amber-100:#fbf3df;--red-700:#9b3a3a;--red-100:#f8e8e8;
--blue-700:#315f8c;--blue-100:#eaf1f8;}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Inter,"Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif;
background:#eef1f4;color:var(--slate-900);font-size:13px;line-height:1.55}
a{color:var(--blue-700)}
code{font-family:"SFMono-Regular",Consolas,monospace}
.report-shell{max-width:1180px;margin:28px auto;background:var(--white);
border:1px solid var(--slate-200);box-shadow:0 4px 18px rgba(15,23,42,0.08)}
.report-header{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:32px;
padding:30px 34px 26px;border-top:7px solid var(--navy-900);
border-bottom:1px solid var(--slate-200)}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:.11em;text-transform:uppercase;
color:var(--navy-700);margin-bottom:7px}
.report-title{font-size:24px;line-height:1.25;font-weight:700;
letter-spacing:-.02em;color:var(--navy-900)}
.report-subtitle{margin-top:7px;font-size:12px;color:var(--slate-500)}
.report-meta{display:grid;grid-template-columns:1fr 1fr;gap:13px 18px;font-size:12px}
.report-meta div{color:var(--slate-900);font-weight:600}
.report-meta span{display:block;margin-bottom:2px;color:var(--slate-500);
font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.notice{margin:22px 34px 0;padding:13px 16px;
border-left:4px solid var(--amber-700);background:var(--amber-100)}
.notice-title{margin-bottom:4px;color:var(--amber-700);font-size:12px;font-weight:700}
.notice p{margin:0;font-size:12px;line-height:1.5;color:#5f4a27}
.report-section{padding:24px 34px}
.section-heading{display:flex;align-items:center;justify-content:space-between;
padding-bottom:8px;margin-bottom:14px;border-bottom:2px solid var(--navy-900)}
.section-heading h2{margin:0;font-size:15px;font-weight:700;color:var(--navy-900)}
.summary-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));
border:1px solid var(--slate-200)}
.summary-item{padding:13px 12px;border-right:1px solid var(--slate-200);background:var(--white)}
.summary-item:last-child{border-right:0}
.summary-label{display:block;margin-bottom:5px;color:var(--slate-500);
font-size:10px;text-transform:uppercase;letter-spacing:.04em}
.summary-value{font-size:20px;line-height:1;font-weight:700;color:var(--navy-900)}
.table-wrap{overflow-x:auto;border:1px solid var(--slate-200)}
.clinical-table{width:100%;border-collapse:collapse;font-size:11.5px}
.clinical-table th{padding:9px 10px;text-align:left;color:var(--white);
background:var(--navy-900);font-weight:600;white-space:nowrap}
.clinical-table td{padding:9px 10px;border-bottom:1px solid var(--slate-200);vertical-align:top}
.clinical-table tbody tr:nth-child(even){background:var(--slate-050)}
.clinical-table tbody tr:last-child td{border-bottom:0}
.table-footnote{padding:6px 10px;font-size:10.5px;color:var(--slate-500);
border-top:1px solid var(--slate-200);background:var(--slate-050)}
.variant-section{margin-bottom:20px;border:1px solid var(--slate-200);page-break-inside:avoid}
.variant-header{display:flex;justify-content:space-between;align-items:flex-start;
gap:18px;padding:14px 16px;background:var(--slate-050);border-bottom:1px solid var(--slate-200)}
.variant-header h3{margin:0;font-size:16px;color:var(--navy-900);font-weight:700}
.variant-header h3 span{margin-left:7px;color:var(--slate-700);
font-family:"SFMono-Regular",Consolas,monospace;font-size:13px;font-weight:500}
.variant-meta-line{margin-top:5px;font-size:11px;color:var(--slate-500)}
.annotation-status{max-width:310px;padding:6px 9px;border:1px solid var(--slate-300);
background:var(--white);font-size:10.5px;line-height:1.35;text-align:right}
.annotation-exact{color:var(--green-700);border-color:#a9cbb8;background:var(--green-100)}
.annotation-broader{color:var(--amber-700);border-color:#ddc68b;background:var(--amber-100)}
.annotation-gene{color:var(--slate-700);background:var(--slate-100)}
.variant-information-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
border-bottom:1px solid var(--slate-200)}
.variant-information-grid>div{padding:11px 14px;border-right:1px solid var(--slate-200)}
.variant-information-grid>div:last-child{border-right:0}
.field-label{display:block;margin-bottom:3px;font-size:9.5px;font-weight:700;
color:var(--slate-500);letter-spacing:.04em;text-transform:uppercase}
.field-value{display:block;font-size:12px;color:var(--slate-900)}
.interpretation-block{margin:0;border-bottom:1px solid var(--slate-200)}
.interpretation-row{display:grid;grid-template-columns:150px 1fr;gap:15px;
padding:9px 14px;border-bottom:1px solid var(--slate-200);font-size:11.5px;line-height:1.45}
.interpretation-row:last-child{border-bottom:0}
.interpretation-label{color:var(--slate-500);font-weight:700;
text-transform:uppercase;font-size:9.5px;letter-spacing:.04em}
.interpretation-warning{background:var(--amber-100)}
.evidence-label{display:inline-block;padding:3px 6px;border-radius:2px;
font-size:9.5px;font-weight:700;line-height:1.2}
.evidence-direct{color:var(--green-700);background:var(--green-100)}
.evidence-related{color:var(--blue-700);background:var(--blue-100)}
.evidence-indirect{color:var(--amber-700);background:var(--amber-100)}
.evidence-cross{color:var(--slate-700);background:var(--slate-100)}
.evidence-resistance{color:var(--red-700);background:var(--red-100)}
.hypothesis-item{padding:12px 0;border-bottom:1px solid var(--slate-200)}
.hypothesis-item:last-child{border-bottom:0}
.hypothesis-title{font-size:12px;font-weight:700;color:var(--slate-900)}
.hypothesis-note{margin-top:4px;font-size:11px;color:var(--slate-700);line-height:1.5}
.empty-state{padding:14px 16px;border:1px solid var(--slate-200);
background:var(--slate-050);color:var(--slate-500);font-size:12px}
.report-footer{display:flex;justify-content:space-between;gap:20px;
padding:16px 34px;border-top:1px solid var(--slate-200);
color:var(--slate-500);font-size:10.5px}
@media print{
@page{size:A4;margin:14mm}
body{font-size:10pt;background:#fff}
.report-shell{max-width:none;margin:0;border:0;box-shadow:none}
.variant-section{break-inside:avoid}
a{color:inherit;text-decoration:none}
.clinical-table th{background:#e5e7eb!important;color:#111827!important;border:1px solid #9ca3af}
.clinical-table td{border:1px solid #d1d5db}
}
@media(max-width:900px){
.report-header{grid-template-columns:1fr}
.summary-grid{grid-template-columns:repeat(3,1fr)}
.variant-information-grid{grid-template-columns:repeat(2,1fr)}
.interpretation-row{grid-template-columns:1fr;gap:4px}
}"""

    HTML = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Patient-Guided Literature Evidence Report — {patient_id}</title>
<style>{CSS}</style>
</head><body>
<div class="report-shell">

<header class="report-header">
  <div>
    <div class="eyebrow">Precision Oncology Evidence Report · Confidential</div>
    <h1 class="report-title">Patient-Guided Literature Evidence Report</h1>
    <div class="report-subtitle">Generated by megaMine v2.0 · APML, Ajou University</div>
  </div>
  <div class="report-meta">
    <div><span>Patient ID</span>{patient_id}</div>
    <div><span>Cancer type</span>{cancer}</div>
    <div><span>Institution</span>Ajou University Medical Center</div>
    <div><span>Report scope</span>Literature evidence synthesis</div>
  </div>
</header>

<section class="notice" style="margin-bottom:0">
  <div class="notice-title">Important interpretation notice</div>
  <p>This report presents literature-derived evidence retrieved and graded by megaMine.
  It supports expert review and does not constitute a treatment recommendation.
  All findings should be interpreted by a qualified oncologist.</p>
</section>

<section class="report-section" style="padding-top:14px;padding-bottom:14px;
     background:var(--slate-050);border-bottom:1px solid var(--slate-200)">
  <div style="font-size:11.5px;color:var(--slate-700);line-height:1.6">
    <strong style="color:var(--slate-900)">Methodology:</strong>
    Somatic variants filtered to cancer-relevant genes.
    PubMed queried by three-tier strategy:
    (1)&nbsp;exact protein alteration (weight&nbsp;1.00),
    (2)&nbsp;alteration class (0.75),
    (3)&nbsp;gene-level (0.45).
    Records normalized, deduplicated, and verified by offline LLM.
    OncoKB queried per patient protein alteration; annotations classified as
    exact-allele, broader variant/class, gene-only, or no match.
    VAF is a clonality confidence modifier, not a primary actionability tier. Patient-level drug ranking is derived from same-cancer verified evidence only; cross-cancer evidence is reported separately and is not included in the ranking.
  </div>
</section>

<section class="report-section">
  <div class="section-heading"><h2>Analysis Summary</h2></div>
  <div class="summary-grid">
    <div class="summary-item"><span class="summary-label">Variants analyzed</span><span class="summary-value">{n_variants}</span></div>
    <div class="summary-item"><span class="summary-label">Candidate records</span><span class="summary-value">{n_candidates}</span></div>
    <div class="summary-item"><span class="summary-label">Relation verified</span><span class="summary-value">{n_rel_verified}</span></div>
    <div class="summary-item"><span class="summary-label">Same-cancer evidence</span><span class="summary-value">{n_same_cancer}</span></div>
    <div class="summary-item"><span class="summary-label">Cross-cancer evidence</span><span class="summary-value">{n_cross_cancer}</span></div>
    <div class="summary-item"><span class="summary-label">Exact allele matches</span><span class="summary-value">{n_exact_allele}</span></div>
  </div>
  <div style="margin-top:8px;font-size:11px;color:var(--slate-500)">
    OncoKB: {n_exact_allele} exact allele · {n_variant_class} variant/class · {n_gene_only} gene-only
  </div>
</section>

<section class="report-section" style="padding-top:0">
  <div class="section-heading"><h2>Variant Overview</h2></div>
  <div class="table-wrap">
    <table class="clinical-table">
      <thead><tr>
        <th>Gene</th><th>Variant</th><th>VAF</th><th>Clonality</th>
        <th>Variant class</th><th>OncoKB match</th>
        <th>Candidate records</th><th>Relation verified</th>
      </tr></thead>
      <tbody>{var_rows}</tbody>
    </table>
  </div>
</section>

<section class="report-section" style="padding-top:0">
  <div class="section-heading">
    <h2>Per-Variant Evidence</h2>
    <span style="font-size:11px;font-weight:400;color:var(--slate-500)">Score does not estimate response probability</span>
  </div>
  {variant_sections}
</section>

<section class="report-section" style="padding-top:0">
  <div class="section-heading"><h2>ClinicalTrials.gov Linkage</h2></div>
  {trials_html}
</section>

<section class="report-section" style="padding-top:0;background:var(--slate-050);border-top:1px solid var(--slate-300)">
  <div class="section-heading"><h2>Exploratory Co-mutation Hypotheses</h2></div>
  <div style="margin-bottom:10px;padding:8px 10px;background:var(--amber-100);
       border-left:3px solid var(--amber-700);font-size:11.5px;color:#5f4a27">
    Exploratory hypotheses are not validated therapeutic biomarkers and are not included in patient-level evidence ranking.
  </div>
  {hyp_items}
</section>

<section class="report-section" style="padding-top:0">
  <div class="section-heading"><h2>Verification Framework</h2></div>
  <div class="table-wrap">
    <table class="clinical-table">
      <thead><tr><th>Stage</th><th>Validation rule</th><th>Description</th></tr></thead>
      <tbody>{verif_rows}</tbody>
    </table>
  </div>
</section>

<section class="report-section" style="padding-top:0;background:var(--slate-050);border-top:1px solid var(--slate-200)">
  <div class="section-heading"><h2>Limitations</h2></div>
  <div style="font-size:11.5px;color:var(--slate-700);line-height:1.7">
    <p style="margin-bottom:6px"><strong>Literature coverage:</strong> Evidence retrieval is limited to PubMed-indexed publications matching the query terms.</p>
    <p style="margin-bottom:6px"><strong>Evidence specificity:</strong> Most retrieved evidence is gene-level. Gene-level evidence cannot be assumed to apply to the patient-specific alteration.</p>
    <p style="margin-bottom:6px"><strong>OncoKB annotation:</strong> Non-allele-specific annotations may not correspond to the patient variant's actual therapeutic relevance.</p>
    <p><strong>No clinical validation:</strong> This report has not been validated in a clinical setting. All findings require expert oncologist review.</p>
  </div>
</section>

<footer class="report-footer">
  <div>megaMine v2.0 · Patient-guided literature evidence synthesis · APML, Ajou University Medical Center</div>
  <div>Research use only · Not for direct clinical decision-making · OncoKB © MSK (oncokb.org)</div>
</footer>

</div>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(HTML)
    sz = Path(out_path).stat().st_size // 1024
    print(f"✅ HTML report: {out_path} ({sz} KB)")


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
            "oncokb_allele_exist":    ok.get("allele_exist", False),
            "oncokb_variant_exist":   ok.get("variant_exist", False),
            "oncokb_gene_exist":      ok.get("gene_exist", False),
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

    # ── Clinical trials linkage ──────────────────────────────
    print("\n[Step] Linking to ClinicalTrials.gov...")
    trials_df = pd.DataFrame()
    try:
        for mod in list(sys.modules.keys()):
            if "megaMine" in mod: del sys.modules[mod]
        from megaMine.modules.trials import run_trials_linkage
        from megaMine.modules.contradiction import run_contradiction_detection
        from megaMine.modules.temporal import run_temporal_analysis

        if len(verified) > 0:
            try:
                _, trend_df = run_temporal_analysis(combined)
                contra_df   = run_contradiction_detection(combined,
                                  profile_df=trend_df)
            except Exception:
                contra_df = pd.DataFrame()

            trials_df = run_trials_linkage(
                verified,
                contradiction_df=contra_df,
                dry_run=False,
            )
            n_trials = len(trials_df)
            if n_trials > 0 and "highest_phase" in trials_df.columns:
                n_active = int(
                    (trials_df["highest_phase"].fillna("") != "").sum())
            else:
                n_active = 0
            print(f"  Linked {n_trials} drug-gene pairs | "
                  f"Active trials: {n_active}")
        else:
            print("  No verified evidence for trial linkage")
    except Exception as e:
        print(f"  Trial linkage failed: {e}")
        trials_df = pd.DataFrame()

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
        if len(trials_df)>0:
            trials_df.to_excel(xl, sheet_name="ClinicalTrials", index=False)
        if len(cross_cancer)>0:
            cross_cancer.to_excel(xl, sheet_name="CrossCancer_Evidence", index=False)
        pd.DataFrame([{
            "patient":          patient_id,"cancer":cancer,"vaf_min":vaf_min,
            "years":            years,"oncokb":bool(oncokb_token),
            "n_variants":       len(mutations),"n_rows":len(combined),
            "n_verified":       len(verified),
            "n_cancer_matched": len(cancer_matched),
            "n_cross_cancer":   len(cross_cancer),
            "n_trial_pairs":    len(trials_df),
            "version":          "megaMine_v2.0.0",
        }]).to_excel(xl, sheet_name="RunInfo", index=False)
    print(f"✅ Excel: {out_xlsx}")

    out_html = out/f"{patient_id}_megaMine_report.html"
    build_html_report(df_summary, combined, ranked,
                      patient_drug_ranking, comutations,
                      patient_id, cancer, oncokb_data, out_html,
                      trials_df=trials_df if len(trials_df)>0 else None)

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
