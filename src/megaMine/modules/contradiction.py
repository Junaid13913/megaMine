"""
contradiction.py — megaMine v2.0
Contradiction Detection Module

PURPOSE:
    Flags gene-drug-cancer triplets where the literature
    shows BOTH strong efficacy AND strong resistance signals.

    Clinically important — doctors need to know when evidence
    is conflicted, not just how much evidence exists.

    No existing pipeline does this systematically.
    Novel contribution of megaMine v2.0.

INPUT:
    pandas DataFrame — megaMine output (extractor.py rows)
    optional: temporal profile from temporal.py

OUTPUT:
    pandas DataFrame — one row per triplet with:
        contradiction_flag   : none/watch/caution/conflict
        conflict_score       : 0.0 to 1.0
        temporal_conflict    : yes/no (recent shift detected)

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Optional

# ─── Conflict score thresholds ────────────────────────────────
# conflict_score = resistance_papers / (efficacy + resistance)
WATCH_THRESHOLD    = 0.20  # conflict_score >= 0.20 → WATCH
CAUTION_THRESHOLD  = 0.40  # conflict_score >= 0.40 → CAUTION
CONFLICT_THRESHOLD = 0.60  # conflict_score >= 0.60 → CONFLICT

# Minimum papers needed to flag a contradiction
MIN_EFFICACY_FOR_FLAG    = 5
MIN_RESISTANCE_FOR_FLAG  = 3


def compute_conflict_score(
    efficacy_papers: int,
    resistance_papers: int
) -> float:
    """
    Calculate conflict score for a triplet.

    Formula:
        conflict_score = resistance / (efficacy + resistance)

    Range: 0.0 (all efficacy) to 1.0 (all resistance)
    0.5 means equal evidence on both sides — maximum conflict

    Examples:
        100 efficacy, 10 resistance → score 0.09 → none
        100 efficacy, 50 resistance → score 0.33 → watch
        50  efficacy, 50 resistance → score 0.50 → conflict
        10  efficacy, 90 resistance → score 0.90 → conflict
    """
    total = efficacy_papers + resistance_papers
    if total == 0:
        return 0.0
    return round(resistance_papers / total, 3)


def classify_conflict(score: float) -> str:
    """
    Classify a conflict score into a flag level.

    none     → score < 0.20  : evidence is consistent
    watch    → score 0.20-0.39: emerging conflict signal
    caution  → score 0.40-0.59: moderate conflict
    conflict → score >= 0.60  : strong conflicting evidence
    """
    if score >= CONFLICT_THRESHOLD:
        return "conflict"
    elif score >= CAUTION_THRESHOLD:
        return "caution"
    elif score >= WATCH_THRESHOLD:
        return "watch"
    else:
        return "none"


def detect_temporal_conflict(
    profile_df: pd.DataFrame,
    gene: str,
    drug: str,
    cancer: str
) -> bool:
    """
    Detect if a triplet shows temporal conflict:
    efficacy dominated EARLY but resistance dominates RECENTLY.

    This is the most clinically dangerous pattern —
    the drug worked before but evidence is shifting.

    Parameters
    ----------
    profile_df : pd.DataFrame
        Output from temporal.compute_temporal_profile()
        Pass None if temporal analysis not available

    Returns
    -------
    bool : True if temporal conflict detected
    """
    if profile_df is None or len(profile_df) == 0:
        return False

    # Use canonical_cancer_type if available for matching
    cancer_col = ("canonical_cancer_type"
                  if "canonical_cancer_type" in profile_df.columns
                  else "cancer_type")
    cancer_key = cancer.lower().split(";")[0].strip()
    mask = (
        (profile_df["biomarker"].str.upper() == gene.upper()) &
        (profile_df["drug_primary"].str.lower() == drug.lower()) &
        (profile_df[cancer_col].str.lower().str.contains(
            cancer_key, na=False
        ))
    )
    triplet = profile_df[mask].copy()

    if len(triplet) < 2:
        return False

    # Need at least 5 papers per bin to trust the ratio
    valid = triplet[triplet["n_papers"] >= 5]
    if len(valid) < 2:
        return False

    early_resistance  = float(valid["resistance_ratio"].iloc[0])
    latest_resistance = float(valid["resistance_ratio"].iloc[-1])

    # Temporal conflict: resistance was low early, high recently
    return early_resistance < 0.25 and latest_resistance > 0.50


def run_contradiction_detection(
    df: pd.DataFrame,
    profile_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Main entry point for contradiction detection.
    Analyzes every (biomarker, drug_primary, cancer_type) triplet.

    Parameters
    ----------
    df : pd.DataFrame
        Raw megaMine output
    profile_df : pd.DataFrame, optional
        Temporal profile from temporal.run_temporal_analysis()
        If provided, enables temporal conflict detection
    output_path : str, optional
        Save results to Excel if provided

    Returns
    -------
    pd.DataFrame
        One row per triplet with contradiction assessment
    """
    print("🔍 Running contradiction detection...")

    required = ["biomarker", "drug_primary", "cancer_type",
                "evidence_type", "resistance_observed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results = []

    # Use canonical_cancer_type if available
    cancer_col = "canonical_cancer_type" if "canonical_cancer_type" in df.columns else "cancer_type"
    triplets = df.groupby(
        ["biomarker", "drug_primary", cancer_col]
    )

    for (gene, drug, cancer), grp in triplets:
        n_total = len(grp)

        # Count evidence
        efficacy_papers   = int((grp["evidence_type"] == "efficacy").sum())
        # Use resistance_evidence (precise) not resistance_observed (too broad)
        # resistance_observed=yes for 152/153 rows in real run — not useful
        # resistance_evidence=yes only for sentences directly reporting resistance
        resist_col        = "resistance_evidence" if "resistance_evidence" in grp.columns else "resistance_observed"
        resistance_papers = int((grp[resist_col] == "yes").sum())
        review_papers     = int((grp["evidence_type"] == "review").sum())
        background_papers = int((grp["evidence_type"] == "background").sum())

        # Only flag if minimum paper counts met
        can_flag = (
            efficacy_papers   >= MIN_EFFICACY_FOR_FLAG and
            resistance_papers >= MIN_RESISTANCE_FOR_FLAG
        )

        if can_flag:
            conflict_score = compute_conflict_score(
                efficacy_papers, resistance_papers
            )
            contradiction_flag = classify_conflict(conflict_score)
        else:
            conflict_score     = compute_conflict_score(
                efficacy_papers, resistance_papers
            )
            contradiction_flag = "insufficient_data"

        # Temporal conflict detection
        temp_conflict = detect_temporal_conflict(
            profile_df, gene, drug, cancer
        ) if profile_df is not None else False

        # Upgrade flag if temporal conflict detected
        if temp_conflict and contradiction_flag == "watch":
            contradiction_flag = "caution"
        elif temp_conflict and contradiction_flag == "none":
            contradiction_flag = "watch"

        results.append({
            "biomarker":            gene,
            "drug_primary":         drug,
            "cancer_type":          cancer,
            "canonical_cancer_type": cancer,
            "contradiction_flag":   contradiction_flag,
            "conflict_score":       conflict_score,
            "temporal_conflict":    "yes" if temp_conflict else "no",
            "n_total_papers":       n_total,
            "efficacy_papers":      efficacy_papers,
            "resistance_papers":    resistance_papers,
            "review_papers":        review_papers,
            "background_papers":    background_papers,
            "can_flag":             can_flag,
        })

    result_df = pd.DataFrame(results)

    # Sort by conflict score descending — highest conflict first
    result_df = result_df.sort_values(
        "conflict_score", ascending=False
    ).reset_index(drop=True)

    # Print summary
    if len(result_df) > 0:
        flag_counts = result_df["contradiction_flag"].value_counts()
        print(f"   ✅ Analyzed {len(result_df)} triplets")
        print("\n   📊 Contradiction summary:")
        for flag, count in flag_counts.items():
            emoji = {"conflict":"🔴","caution":"🟠",
                     "watch":"🟡","none":"🟢",
                     "insufficient_data":"⚪"}.get(flag,"•")
            print(f"      {emoji} {flag}: {count}")

    # Save if path provided
    if output_path:
        result_df.to_excel(output_path, index=False)
        print(f"\n   💾 Saved to {output_path}")

    return result_df
