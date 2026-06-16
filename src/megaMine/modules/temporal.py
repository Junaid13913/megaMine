"""
temporal.py — megaMine v2.0
Temporal Evidence Tracking Module

PURPOSE:
    Shows how the evidence balance for a gene-drug-cancer triplet
    shifts over time. Transforms megaMine from a static extractor
    into a dynamic evidence tracker.

    No existing literature mining tool does this.
    This is a novel contribution of megaMine v2.0.

INPUT:
    pandas DataFrame — megaMine output (extractor.py rows)

OUTPUT:
    pandas DataFrame — one row per (gene, drug, cancer, year_bin)
    with evidence balance metrics and trend classification

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Optional

# ─── Year bin definitions ──────────────────────────────────────
# Each bin represents a meaningful era in precision oncology
YEAR_BINS = [
    (0,    2009, "pre-2010"),
    (2010, 2014, "2010-2014"),
    (2015, 2017, "2015-2017"),
    (2018, 2020, "2018-2020"),
    (2021, 2023, "2021-2023"),
    (2024, 9999, "2024+"),
]

# ─── Trend classification thresholds ──────────────────────────
# How much resistance_ratio must change to call a trend
RISING_RESISTANCE_THRESHOLD  = 0.10  # resistance growing by 10%+
DECLINING_EFFICACY_THRESHOLD = 0.15  # efficacy dropping by 15%+
MIN_PAPERS_FOR_TREND         = 3     # lowered from 5 — more trends with smaller datasets
MIN_PAPERS_EMERGING          = 3     # emerging evidence threshold


def assign_year_bin(year: int) -> str:
    """
    Assign a publication year to its era bin.

    Example:
        2019 → "2018-2020"
        2023 → "2021-2023"
        0    → "pre-2010"  (unknown year)
    """
    for start, end, label in YEAR_BINS:
        if start <= year <= end:
            return label
    return "unknown"


def compute_temporal_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-year-bin evidence profiles for every
    (biomarker, drug_primary, cancer_type) triplet.

    For each triplet + year bin calculates:
        n_papers          : total papers in this bin
        efficacy_papers   : papers with evidence_type == efficacy
        resistance_papers : papers with resistance_observed == yes
        efficacy_ratio    : efficacy_papers / n_papers
        resistance_ratio  : resistance_papers / n_papers
        dominant_evidence : which type dominates this bin

    Parameters
    ----------
    df : pd.DataFrame
        Raw megaMine output dataframe

    Returns
    -------
    pd.DataFrame
        One row per (biomarker, drug_primary, cancer_type, year_bin)
    """
    # Validate required columns exist
    required = ["biomarker", "drug_primary", "cancer_type",
                "year", "evidence_type", "resistance_observed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Assign year bins
    df = df.copy()
    df["year_bin"] = df["year"].apply(
        lambda y: assign_year_bin(int(y) if pd.notna(y) else 0)
    )

    # Group by triplet + year bin
    rows = []
    # Use canonical_cancer_type if available — falls back to cancer_type
    cancer_col = "canonical_cancer_type" if "canonical_cancer_type" in df.columns else "cancer_type"
    groups = df.groupby(
        ["biomarker", "drug_primary", cancer_col, "year_bin"],
        as_index=False
    )

    for (gene, drug, cancer, ybin), grp in groups:
        n_papers = len(grp)

        # Count evidence types using final_evidence_type (mutually exclusive)
        # If final_evidence_type exists use it — avoids double-counting
        # (a row cannot be both efficacy and resistance simultaneously)
        et_col = "final_evidence_type" if "final_evidence_type" in grp.columns else "evidence_type"
        efficacy_papers          = int((grp[et_col] == "efficacy").sum())
        resistance_papers        = int((grp[et_col] == "resistance").sum())
        post_resistance_papers   = int((grp[et_col] == "post_resistance_efficacy").sum())
        review_papers            = int((grp[et_col] == "review").sum())
        background_papers        = int((grp[et_col] == "background").sum())

        # Calculate ratios using DIRECTIONAL TOTAL as denominator
        # This makes efficacy_share + resistance_share = 1.0
        # Both are interpretable as competing evidence fractions
        # Also compute prevalence (papers / total papers in bin)
        directional_total = efficacy_papers + resistance_papers
        efficacy_share   = round(efficacy_papers   / directional_total, 3) if directional_total else 0.0
        resistance_share = round(resistance_papers / directional_total, 3) if directional_total else 0.0
        # Keep prevalence for reference
        efficacy_ratio   = round(efficacy_papers   / n_papers, 3) if n_papers else 0.0
        resistance_ratio = round(resistance_papers / n_papers, 3) if n_papers else 0.0

        # Dominant evidence type in this bin
        counts = {
            "efficacy":   efficacy_papers,
            "resistance": resistance_papers,
            "review":     review_papers,
            "background": background_papers,
        }
        dominant_evidence = max(counts, key=counts.get)

        rows.append({
            "biomarker":             gene,
            "drug_primary":          drug,
            "cancer_type":           cancer,
            "canonical_cancer_type": cancer,
            "year_bin":              ybin,
            "n_papers":                n_papers,
            "efficacy_papers":         efficacy_papers,
            "resistance_papers":       resistance_papers,
            "post_resistance_papers":  post_resistance_papers,
            "review_papers":           review_papers,
            "background_papers":       background_papers,
            "efficacy_ratio":     efficacy_ratio,      # prevalence: efficacy / all papers
            "resistance_ratio":   resistance_ratio,    # prevalence: resistance / all papers
            "efficacy_share":     efficacy_share,      # share: efficacy / (efficacy+resistance)
            "resistance_share":   resistance_share,    # share: resistance / (efficacy+resistance)
            "directional_total":  directional_total,   # efficacy + resistance papers
            "dominant_evidence":  dominant_evidence,
        })

    profile_df = pd.DataFrame(rows)

    # Sort by triplet then by year bin order
    bin_order = [b[2] for b in YEAR_BINS] + ["unknown"]
    profile_df["year_bin_order"] = profile_df["year_bin"].apply(
        lambda x: bin_order.index(x) if x in bin_order else 99
    )
    profile_df = profile_df.sort_values(
        ["biomarker", "drug_primary", "cancer_type", "year_bin_order"]
    ).drop(columns=["year_bin_order"]).reset_index(drop=True)

    return profile_df


def classify_trend(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (biomarker, drug_primary, cancer_type) triplet,
    classify the overall temporal trend across all year bins.

    Trend categories:
        stable              : resistance_ratio change < threshold
        rising_resistance   : resistance growing significantly
        declining_efficacy  : efficacy dropping significantly
        emerging            : few papers but growing
        conflicted          : both efficacy and resistance high
        insufficient_data   : not enough papers to classify

    Parameters
    ----------
    profile_df : pd.DataFrame
        Output from compute_temporal_profile()

    Returns
    -------
    pd.DataFrame
        One row per triplet with trend classification
    """
    results = []

    # Always group by canonical_cancer_type when available
    cancer_col_t = "canonical_cancer_type" if "canonical_cancer_type" in profile_df.columns else "cancer_type"
    triplets = profile_df.groupby(
        ["biomarker", "drug_primary", cancer_col_t]
    )

    for (gene, drug, cancer), grp in triplets:
        # cancer here is the canonical_cancer_type value
        # Only use bins with enough papers
        valid = grp[grp["n_papers"] >= MIN_PAPERS_FOR_TREND].copy()
        total_papers = grp["n_papers"].sum()

        # Not enough data
        if len(valid) < 2:
            if total_papers >= MIN_PAPERS_EMERGING:
                trend = "emerging"
            else:
                trend = "insufficient_data"

            results.append({
                "biomarker":                  gene,
                "drug_primary":               drug,
                "cancer_type":                cancer,
                "canonical_cancer_type":      cancer,  # ← KEY FIX
                "temporal_trend":             trend,
                "total_papers":               int(total_papers),
                "peak_evidence_bin":          grp.loc[grp["n_papers"].idxmax(), "year_bin"],
                "resistance_emergence_bin":   _find_resistance_emergence(grp),
                "first_bin_with_data":        grp[grp["n_papers"] > 0]["year_bin"].iloc[0] if len(grp) > 0 else "",
                "latest_bin_with_data":       grp[grp["n_papers"] > 0]["year_bin"].iloc[-1] if len(grp) > 0 else "",
                "early_efficacy_share":       float(valid["efficacy_share"].iloc[0]) if len(valid) > 0 else 0.0,
                "latest_efficacy_share":      float(valid["efficacy_share"].iloc[-1]) if len(valid) > 0 else 0.0,
                "early_resistance_share":     float(valid["resistance_share"].iloc[0]) if len(valid) > 0 else 0.0,
                "latest_resistance_share":    float(valid["resistance_share"].iloc[-1]) if len(valid) > 0 else 0.0,
                "early_efficacy_ratio":       float(valid["efficacy_ratio"].iloc[0]) if len(valid) > 0 else 0.0,
                "latest_efficacy_ratio":      float(valid["efficacy_ratio"].iloc[-1]) if len(valid) > 0 else 0.0,
                "early_resistance_ratio":     float(valid["resistance_ratio"].iloc[0]) if len(valid) > 0 else 0.0,
                "latest_resistance_ratio":    float(valid["resistance_ratio"].iloc[-1]) if len(valid) > 0 else 0.0,
            })
            continue

        # Calculate changes from earliest to latest valid bin
        # Use SHARES (directional denominator) for trend classification
        # This prevents both ratios approaching 1 simultaneously
        share_col_e = "efficacy_share"   if "efficacy_share"   in valid.columns else "efficacy_ratio"
        share_col_r = "resistance_share" if "resistance_share" in valid.columns else "resistance_ratio"

        early_efficacy   = float(valid[share_col_e].iloc[0])
        latest_efficacy  = float(valid[share_col_e].iloc[-1])
        early_resistance = float(valid[share_col_r].iloc[0])
        latest_resistance= float(valid[share_col_r].iloc[-1])

        efficacy_change   = latest_efficacy   - early_efficacy
        resistance_change = latest_resistance - early_resistance

        # Classify trend using shares — mutually exclusive denominators
        # stable_mixed = balanced evidence, no clear direction
        # stable = efficacy clearly dominant and stable
        if latest_resistance > 0.55 and latest_efficacy > 0.45:
            trend = "conflicted"
        elif resistance_change >= RISING_RESISTANCE_THRESHOLD:
            trend = "rising_resistance"
        elif efficacy_change >= RISING_RESISTANCE_THRESHOLD:
            trend = "rising_efficacy"
        elif efficacy_change <= -DECLINING_EFFICACY_THRESHOLD:
            trend = "declining_efficacy"
        elif abs(resistance_change) < RISING_RESISTANCE_THRESHOLD:
            # Stable — but distinguish mixed from efficacy-dominant
            if latest_efficacy >= 0.65:
                trend = "stable_efficacy_dominant"
            elif latest_resistance >= 0.40:
                trend = "stable_mixed"   # balanced — biologically important distinction
            else:
                trend = "stable"
        else:
            trend = "stable"

        results.append({
            "biomarker":                  gene,
            "drug_primary":               drug,
            "cancer_type":                cancer,
            "canonical_cancer_type":      cancer,  # ← KEY FIX
            "temporal_trend":             trend,
            "total_papers":               int(total_papers),
            "peak_evidence_bin":          grp.loc[grp["n_papers"].idxmax(), "year_bin"],
            "resistance_emergence_bin":   _find_resistance_emergence(grp),
            "first_bin_with_data":        valid["year_bin"].iloc[0],
            "latest_bin_with_data":       valid["year_bin"].iloc[-1],
            "early_efficacy_share":       round(early_efficacy, 3),
            "latest_efficacy_share":      round(latest_efficacy, 3),
            "early_resistance_share":     round(early_resistance, 3),
            "latest_resistance_share":    round(latest_resistance, 3),
            "early_efficacy_ratio":       round(float(valid["efficacy_ratio"].iloc[0]), 3),
            "latest_efficacy_ratio":      round(float(valid["efficacy_ratio"].iloc[-1]), 3),
            "early_resistance_ratio":     round(float(valid["resistance_ratio"].iloc[0]), 3),
            "latest_resistance_ratio":    round(float(valid["resistance_ratio"].iloc[-1]), 3),
        })

    return pd.DataFrame(results)


def _find_resistance_emergence(grp: pd.DataFrame) -> str:
    """
    Find the year bin where resistance evidence NEWLY EMERGED.
    Only reports emergence when resistance was ABSENT in first bin
    then appeared later — avoids misleading label for stable_mixed.

    Returns empty string if:
      - resistance was present from the start (stable_mixed pattern)
      - resistance never appeared
    """
    if len(grp) == 0:
        return ""

    bin_order = ["pre-2010","2010-2014","2015-2017",
                  "2018-2020","2021-2023","2024+"]
    grp_sorted = grp.copy()
    grp_sorted["_ord"] = grp_sorted["year_bin"].apply(
        lambda x: bin_order.index(x) if x in bin_order else 99
    )
    grp_sorted = grp_sorted.sort_values("_ord")

    # Use mutually exclusive resistance count
    res_col = "resistance_papers"
    if res_col not in grp_sorted.columns:
        return ""

    first_resistance = grp_sorted[res_col].iloc[0] if len(grp_sorted) > 0 else 0

    # If resistance present from start — not emergence
    if first_resistance > 0:
        return ""

    # Find first bin where resistance newly appears
    emerged = grp_sorted[grp_sorted[res_col] > 0]["year_bin"]
    return emerged.iloc[0] if len(emerged) > 0 else ""


def run_temporal_analysis(
    df: pd.DataFrame,
    output_path: Optional[str] = None
) -> tuple:
    """
    Main entry point for temporal analysis.
    Runs both compute_temporal_profile and classify_trend.

    Parameters
    ----------
    df : pd.DataFrame
        Raw megaMine output
    output_path : str, optional
        If provided, saves results to Excel at this path

    Returns
    -------
    tuple: (profile_df, trend_df)
        profile_df : per year-bin evidence profiles
        trend_df   : per triplet trend classifications
    """
    print("🕐 Running temporal evidence analysis...")

    profile_df = compute_temporal_profile(df)
    print(f"   ✅ Year-bin profiles: {len(profile_df)} rows")

    trend_df = classify_trend(profile_df)
    print(f"   ✅ Trend classifications: {len(trend_df)} triplets")

    # Print summary of trends found
    if len(trend_df) > 0:
        trend_counts = trend_df["temporal_trend"].value_counts()
        print("\n   📊 Trend summary:")
        for trend, count in trend_counts.items():
            print(f"      {trend}: {count}")

    # Save if path provided
    if output_path:
        with pd.ExcelWriter(output_path, engine="openpyxl") as xl:
            profile_df.to_excel(xl, sheet_name="YearBinProfiles", index=False)
            trend_df.to_excel(xl, sheet_name="TrendSummary", index=False)
        print(f"\n   💾 Saved to {output_path}")

    return profile_df, trend_df


def get_triplet_timeline(
    profile_df: pd.DataFrame,
    gene: str,
    drug: str,
    cancer: str
) -> pd.DataFrame:
    """
    Get the year-bin timeline for one specific triplet.
    Useful for generating example outputs for the paper.

    Example:
        timeline = get_triplet_timeline(profile_df,
                       "EGFR", "erlotinib", "Non-Small Cell Lung Cancer")
        print(timeline[["year_bin","n_papers",
                         "efficacy_ratio","resistance_ratio"]])
    """
    mask = (
        (profile_df["biomarker"].str.upper() == gene.upper()) &
        (profile_df["drug_primary"].str.lower() == drug.lower()) &
        (profile_df["cancer_type"].str.lower().str.contains(
            cancer.lower().split(";")[0].strip(), na=False
        ))
    )
    result = profile_df[mask].copy()
    if len(result) == 0:
        print(f"No data found for {gene} + {drug} + {cancer}")
    return result
