"""
maf_pipeline.py — megaMine v2.0
MAF-driven evidence prioritization pipeline.

USAGE:
    megaMine-maf --maf patient.maf --cancer NSCLC --out results/
    megaMine-maf --maf cohort.maf  --cancer Breast --out results/ --top 5

WHAT IT DOES:
    1. Reads MAF file
    2. Extracts actionable mutations (coding, oncogenes)
    3. Assigns VAF tiers (Tier1/2/3)
    4. Queries PubMed per gene using megaMine
    5. Ranks drug evidence per mutation
    6. Saves Excel report + HTML summary

AUTHORS: Muhammad Junaid — APML, Ajou University
"""

import os, sys, time, argparse
import pandas as pd
from pathlib import Path
from typing import Optional

# ── VAF thresholds ────────────────────────────────────────────
VAF_TIER1 = 0.20   # Clonal    — high priority
VAF_TIER2 = 0.05   # Subclonal — medium priority
# < VAF_TIER2 = Tier3 — low priority

# ── Actionable variant classes ────────────────────────────────
ACTIONABLE_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Translation_Start_Site", "Nonstop_Mutation", "Splice_Region",
}

# ── Known oncogenes / tumor suppressors ──────────────────────
ONCOGENES = {
    "EGFR","KRAS","NRAS","HRAS","BRAF","MET","ALK","ROS1","RET",
    "HER2","ERBB2","PIK3CA","PTEN","AKT1","MTOR","CDK4","CDK6",
    "BRCA1","BRCA2","TP53","STK11","KEAP1","SMAD4","RB1","CDKN2A",
    "FGFR1","FGFR2","FGFR3","IDH1","IDH2","FLT3","KIT","PDGFRA",
    "NF1","NF2","TSC1","TSC2","ARID1A","CTNNB1","VHL","POLE",
    "MLH1","MSH2","MSH6","PMS2","DNMT3A","TET2","ASXL1","NPM1",
    "NTRK1","NTRK2","NTRK3","TROP2","CSF1R","AR","ESR1","BCL2",
    "BTK","EZH2","MYC","MYCN","MDM2","CCND1","CDKN1B","RET",
    "KDR","VEGFR","ABL1","JAK1","JAK2","STAT3","NOTCH1","FBXW7",
    "ARIDIA","SETD2","BAP1","PBRM1","ATM","ATR","CHEK2","PALB2",
}

def vaf_tier(vaf: float) -> tuple:
    if vaf >= VAF_TIER1:
        return "Tier1-Clonal",   "HIGH",   3
    elif vaf >= VAF_TIER2:
        return "Tier2-Subclonal","MEDIUM", 2
    else:
        return "Tier3-LowVAF",  "LOW",    1


def read_maf(maf_path: str) -> pd.DataFrame:
    """Read MAF file — handles comments and tab-separated format."""
    return pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)


def extract_mutations(maf: pd.DataFrame,
                      vaf_min: float = 0.0) -> pd.DataFrame:
    """
    Extract actionable mutations from MAF.
    Returns one row per gene (highest VAF mutation per gene).
    """
    # Filter to coding mutations in oncogenes
    coding = maf[maf["Variant_Classification"].isin(ACTIONABLE_CLASSES)]
    actionable = coding[coding["Hugo_Symbol"].isin(ONCOGENES)].copy()

    if len(actionable) == 0:
        return pd.DataFrame()

    # Compute VAF
    if "t_depth" in actionable.columns and "t_alt_count" in actionable.columns:
        actionable["VAF"] = (
            actionable["t_alt_count"] / actionable["t_depth"]
        ).round(3)
    elif "tumor_f" in actionable.columns:
        actionable["VAF"] = actionable["tumor_f"].round(3)
    else:
        actionable["VAF"] = 0.1  # default if not available

    # Filter by minimum VAF
    actionable = actionable[actionable["VAF"] >= vaf_min]

    # One row per gene — highest VAF
    actionable = (actionable
                  .sort_values("VAF", ascending=False)
                  .drop_duplicates(subset="Hugo_Symbol", keep="first"))

    # Add VAF tier
    actionable[["vaf_tier","vaf_priority","tier_score"]] = (
        actionable["VAF"].apply(lambda v: pd.Series(vaf_tier(v))))

    # Select key columns
    cols = ["Hugo_Symbol","Variant_Classification","HGVSp_Short",
            "VAF","vaf_tier","vaf_priority","tier_score"]
    cols = [c for c in cols if c in actionable.columns]
    result = actionable[cols].rename(columns={"Hugo_Symbol":"gene",
                                               "HGVSp_Short":"variant"})
    return result.sort_values("VAF", ascending=False).reset_index(drop=True)


def query_megamine(gene: str, cancer: str,
                   email: str, api_key: str,
                   hgnc_cache: str, out_dir: Path,
                   max_records: int = 50,
                   years: str = "2018-2024") -> pd.DataFrame:
    """Run megaMine extractor for one gene+cancer pair."""
    import sys
    for mod in list(sys.modules.keys()):
        if "megaMine" in mod:
            del sys.modules[mod]

    from megaMine.core.extractor import main as run

    query = (f'{gene} AND ("{cancer}" OR cancer OR tumor OR tumour) '
             f'AND ("targeted therapy" OR "drug resistance" OR '
             f'"clinical trial" OR inhibitor OR treatment)')

    out_path = str(out_dir / f"maf_{gene}")
    sys.argv = [
        "extractor",
        "--q",            query,
        "--years",        years,
        "--max-records",  str(max_records),
        "--out",          out_path,
        "--email",        email,
        "--ncbi-api-key", api_key,
        "--hgnc-cache",   hgnc_cache,
        "--require-gene-and-drug",
        "--require-known-drug",
    ]

    try:
        run()
        df = pd.read_excel(out_path + ".xlsx", sheet_name="Rows")
        df["query_gene"] = gene
        return df
    except Exception as e:
        print(f"    Warning: {gene} query failed — {e}")
        return pd.DataFrame()


def score_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Add evidence quality score to rows."""
    EV_SCORE    = {"efficacy":3,"resistance":2,"background":1,
                   "review":1,"safety":1}
    STUDY_SCORE = {"RCT":4,"trial":3,"observational":2,
                   "review":1,"preclinical":1,"case_report":2,"in_vitro":1}

    df["ev_score"]    = df.get("final_evidence_type",
                               df.get("evidence_type","background")
                               ).map(EV_SCORE).fillna(1)
    df["study_score"] = df.get("study_design",
                               pd.Series(["observational"]*len(df))
                               ).map(STUDY_SCORE).fillna(2)
    df["conf"]        = df.get("llm_confidence", pd.Series([0.5]*len(df))).fillna(0.5)
    df["evidence_quality_score"] = (
        df["ev_score"] * df["study_score"] * df["conf"]
    ).round(3)
    return df


def run_maf_pipeline(
    maf_path:   str,
    cancer:     str,
    out_dir:    str,
    email:      str,
    api_key:    str,
    hgnc_cache: str,
    vaf_min:    float = 0.0,
    max_records: int  = 50,
    top_drugs:   int  = 3,
    years:       str  = "2018-2024",
):
    """Main MAF evidence prioritization pipeline."""
    from megaMine.utils.normalizers import (normalize_cancer_column,
                                             add_resistance_context,
                                             reconcile_evidence_type)
    from megaMine.modules.llm_verify import run_llm_verification

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp_queries"
    tmp.mkdir(exist_ok=True)

    print("=" * 60)
    print("MEGAMINE v2.0 — MAF EVIDENCE PRIORITIZATION")
    print(f"MAF:    {maf_path}")
    print(f"Cancer: {cancer}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # Step 1 — Read MAF
    print("\n[1/5] Reading MAF...")
    maf = read_maf(maf_path)
    print(f"  Total mutations: {len(maf):,}")

    # Step 2 — Extract mutations
    print("\n[2/5] Extracting actionable mutations...")
    mutations = extract_mutations(maf, vaf_min=vaf_min)
    if len(mutations) == 0:
        print("  No actionable mutations found. Exiting.")
        return
    print(f"  Found {len(mutations)} actionable mutations:")
    for _, r in mutations.iterrows():
        print(f"    {r['gene']:<8} {str(r.get('variant','')):<15} "
              f"VAF={r['VAF']:.3f} ({r['vaf_priority']})")

    # Step 3 — Query megaMine per gene
    print(f"\n[3/5] Querying PubMed ({years}, max {max_records}/gene)...")
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
            all_rows.append(df)
            print(f"{len(df)} rows | PMIDs: {df['pmid'].nunique()}")
        else:
            print("no results")
        time.sleep(1)

    if not all_rows:
        print("No evidence found for any gene.")
        return

    # Step 4 — Normalize + verify
    print("\n[4/5] Normalizing and verifying...")
    combined = pd.concat(all_rows, ignore_index=True)
    combined = normalize_cancer_column(combined)
    combined = add_resistance_context(combined)
    combined = reconcile_evidence_type(combined)
    combined = run_llm_verification(combined, model="offline",
                                    confidence_threshold=0.60)
    combined = score_evidence(combined)

    # Multiply score by VAF tier priority
    combined["final_score"] = (
        combined["evidence_quality_score"] * combined["tier_priority"]
    ).round(3)

    verified = combined[combined["llm_verified"]=="yes"].copy()
    print(f"  Total rows: {len(combined)} | Verified: {len(verified)}")

    # Step 5 — Rank and report
    print("\n[5/5] Ranking evidence...")
    ranked = (combined
              .sort_values("final_score", ascending=False)
              .groupby("query_gene")
              .head(top_drugs))

    # Summary table
    summary = []
    for _, mut in mutations.iterrows():
        gene = mut["gene"]
        rows = combined[combined["query_gene"]==gene]
        ver  = rows[rows["llm_verified"]=="yes"]
        top  = ranked[ranked["query_gene"]==gene]

        summary.append({
            "gene":              gene,
            "variant":           mut.get("variant",""),
            "VAF":               mut["VAF"],
            "vaf_tier":          mut["vaf_tier"],
            "vaf_priority":      mut["vaf_priority"],
            "total_papers":      rows["pmid"].nunique(),
            "verified_rows":     len(ver),
            "top_drug":          top.iloc[0]["drug_primary"] if len(top)>0 else "N/A",
            "top_evidence_type": top.iloc[0].get("final_evidence_type","N/A") if len(top)>0 else "N/A",
            "top_score":         top.iloc[0]["final_score"] if len(top)>0 else 0,
            "top_pmid":          top.iloc[0]["pmid"] if len(top)>0 else "N/A",
        })

    df_summary = pd.DataFrame(summary)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS — Patient Evidence Summary")
    print(f"{'='*60}")
    print(f"\n{'Gene':<8} {'Variant':<14} {'VAF':>6} {'Priority':<8} "
          f"{'Top Drug':<20} {'Evidence':<12} {'Score':>7}")
    print("-" * 80)
    for _, r in df_summary.iterrows():
        icon = "✅" if r["vaf_priority"]=="HIGH" else "⚠️ " if r["vaf_priority"]=="MEDIUM" else "❌"
        print(f"  {r['gene']:<6} {str(r['variant']):<14} {r['VAF']:>6.3f} "
              f"{icon}{r['vaf_priority']:<6} {r['top_drug']:<20} "
              f"{r['top_evidence_type']:<12} {r['top_score']:>7.2f}")

    # Save Excel
    sample = Path(maf_path).stem
    out_xlsx = out / f"{sample}_megaMine_evidence.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        mutations.to_excel(xl,   sheet_name="Mutations",         index=False)
        df_summary.to_excel(xl,  sheet_name="Summary",           index=False)
        ranked.to_excel(xl,      sheet_name="Top_Evidence",      index=False)
        combined.to_excel(xl,    sheet_name="All_Evidence",      index=False)
        verified.to_excel(xl,    sheet_name="Verified_Evidence", index=False)
        pd.DataFrame([{
            "maf":         maf_path,
            "cancer":      cancer,
            "vaf_min":     vaf_min,
            "years":       years,
            "max_records": max_records,
            "top_drugs":   top_drugs,
            "n_mutations": len(mutations),
            "n_rows":      len(combined),
            "n_verified":  len(verified),
            "version":     "megaMine_v2.0.0",
        }]).to_excel(xl, sheet_name="RunInfo", index=False)

    print(f"\n✅ Report: {out_xlsx}")
    print(f"\n⚠️  DISCLAIMER: Literature-derived evidence only.")
    print(f"   Not a clinical treatment recommendation.")
    print(f"   All outputs require expert oncologist review.")
    return df_summary


def main():
    parser = argparse.ArgumentParser(
        description="megaMine MAF Evidence Prioritization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  megaMine-maf --maf patient.maf --cancer NSCLC --out results/
  megaMine-maf --maf cohort.maf  --cancer Breast --out results/ --vaf-min 0.05
  megaMine-maf --maf tumor.maf   --cancer CRC    --out results/ --top 5 --years 2020-2024
        """
    )
    parser.add_argument("--maf",        required=True,  help="Path to MAF file")
    parser.add_argument("--cancer",     required=True,  help="Cancer type (e.g. NSCLC, Breast, CRC)")
    parser.add_argument("--out",        required=True,  help="Output directory")
    parser.add_argument("--email",      default="user@institution.edu", help="NCBI email")
    parser.add_argument("--ncbi-api-key", default="",   help="NCBI API key")
    parser.add_argument("--hgnc-cache", default="",     help="HGNC cache JSON path")
    parser.add_argument("--vaf-min",    type=float, default=0.0,
                        help="Minimum VAF threshold (default: 0.0 = all mutations)")
    parser.add_argument("--max-records", type=int, default=50,
                        help="Max PubMed records per gene (default: 50)")
    parser.add_argument("--top",        type=int, default=3,
                        help="Top N drugs per gene (default: 3)")
    parser.add_argument("--years",      default="2018-2024",
                        help="Publication year range (default: 2018-2024)")

    args = parser.parse_args()

    run_maf_pipeline(
        maf_path    = args.maf,
        cancer      = args.cancer,
        out_dir     = args.out,
        email       = args.email,
        api_key     = args.ncbi_api_key,
        hgnc_cache  = args.hgnc_cache,
        vaf_min     = args.vaf_min,
        max_records = args.max_records,
        top_drugs   = args.top,
        years       = args.years,
    )

if __name__ == "__main__":
    main()
