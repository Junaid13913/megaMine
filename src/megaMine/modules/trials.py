"""
trials.py — megaMine v2.0
ClinicalTrials.gov Linkage Module

PURPOSE:
    For every gene-drug-cancer triplet extracted by megaMine,
    query ClinicalTrials.gov and attach real clinical trial
    evidence. This bridges literature mining and clinical reality.

    Critically — this module looks for FAILED and TERMINATED
    trials, not just successful ones. A drug with multiple
    failed Phase II trials in a cancer type is a negative
    clinical development signal. This is interpreted as
    resistance ONLY when it agrees with the literature-derived
    resistance ratio from contradiction.py. Trial failure alone
    does not imply resistance — trials terminate for many reasons
    including toxicity, funding, or recruitment failure.

KEY SIGNALS:
    has_failed_trial     → negative clinical development signal
                           (interpreted as resistance only when
                            literature resistance ratio also high)
    highest_phase        → how far did this drug get clinically
    trial_literature_concordance → does trial evidence match
                                   literature evidence?

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import time
import requests
import pandas as pd
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

# ─── ClinicalTrials.gov v2 REST API ───────────────────────────
CT_BASE_URL  = "https://clinicaltrials.gov/api/v2/studies"
CT_SLEEP     = 0.5   # seconds between requests — be polite
CT_TIMEOUT   = 30    # seconds per request
CT_MAX_PAGES = 5     # max pages per query (200 results/page)

# ─── Phase priority for ranking ───────────────────────────────
# Explicit phase ranking — prevents Phase2 beating Phase3
# ClinicalTrials API returns varied phase strings — map all variants
PHASE_PRIORITY = {
    "Phase 4":           5,
    "Phase 3":           4,
    "Phase 2/Phase 3":   3.5,
    "Phase 2":           3,
    "Phase 1/Phase 2":   2,
    "Phase 1":           1,
    "Early Phase 1":     0.5,
    "N/A":               0,
    "":                  0,
}

# ─── Status categories ────────────────────────────────────────
FAILED_STATUSES = {
    "TERMINATED",
    "WITHDRAWN",
    "SUSPENDED",
}
ACTIVE_STATUSES = {
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
}
COMPLETED_STATUSES = {
    "COMPLETED",
}


def _phase_score(phase: str) -> float:
    """
    Convert phase string to numeric priority score.
    Normalizes phase strings before lookup to handle
    API variations like "PHASE2" vs "Phase 2".
    """
    if not phase:
        return 0
    # Normalize: title case, replace underscores
    normalized = phase.strip().replace("_", " ").title()
    # Try direct lookup first
    if normalized in PHASE_PRIORITY:
        return PHASE_PRIORITY[normalized]
    # Try partial matches for edge cases
    if "4" in normalized: return 5
    if "3" in normalized and "2" not in normalized: return 4
    if "2" in normalized and "3" in normalized: return 3.5
    if "2" in normalized: return 3
    if "1" in normalized and "2" in normalized: return 2
    if "1" in normalized: return 1
    return 0


def _normalize_status(status: str) -> str:
    """Normalize status string to uppercase."""
    return (status or "").upper().strip()


def _build_query(drug: str, cancer: str, biomarker: str = "") -> str:
    """
    Build a ClinicalTrials.gov search query.
    Adding biomarker improves precision — prevents unrelated trials.
    e.g. erlotinib + NSCLC + EGFR >> erlotinib + NSCLC alone.
    """
    drug_clean   = drug.split(";")[0].strip()
    cancer_clean = cancer.split(";")[0].strip()
    if biomarker and len(biomarker) > 1:
        return f"{drug_clean} {cancer_clean} {biomarker}"
    return f"{drug_clean} {cancer_clean}"


def fetch_trials_for_drug_cancer(
    drug: str,
    cancer: str,
    biomarker: str = "",
    max_results: int = 200,
) -> List[Dict]:
    """
    Query ClinicalTrials.gov for trials involving a
    specific drug and cancer type combination.
    Optional biomarker narrows results for precision.

    Parameters
    ----------
    drug      : Drug name (e.g. erlotinib)
    cancer    : Cancer type (e.g. NSCLC)
    biomarker : Gene/biomarker (e.g. EGFR) — improves precision
    max_results : Maximum trials to retrieve
    """
    query  = _build_query(drug, cancer, biomarker)
    trials = []
    token  = None   # next page token

    for page in range(CT_MAX_PAGES):
        params = {
            "query.term":  query,
            "pageSize":    min(max_results, 200),
            "format":      "json",
        }
        if token:
            params["pageToken"] = token

        try:
            time.sleep(CT_SLEEP)
            resp = requests.get(
                CT_BASE_URL,
                params=params,
                timeout=CT_TIMEOUT,
                headers={"User-Agent": "megaMine/2.0.0"}
            )
            if resp.status_code != 200:
                break

            data = resp.json()
            studies = data.get("studies", []) or []

            for study in studies:
                trial = _parse_trial(study, drug, cancer)
                if trial:
                    trials.append(trial)

            # Check for next page
            token = data.get("nextPageToken")
            if not token or len(trials) >= max_results:
                break

        except requests.RequestException:
            break
        except Exception:
            break

    return trials[:max_results]


def _parse_trial(study: dict, drug: str, cancer: str) -> Optional[Dict]:
    """
    Parse a single ClinicalTrials.gov study response
    into a clean structured dict.
    """
    try:
        proto   = study.get("protocolSection", {}) or {}
        id_mod  = proto.get("identificationModule", {}) or {}
        stat_mod= proto.get("statusModule", {}) or {}
        desc_mod= proto.get("descriptionModule", {}) or {}
        design  = proto.get("designModule", {}) or {}
        cond_mod= proto.get("conditionsModule", {}) or {}
        inter_mod=proto.get("armsInterventionsModule", {}) or {}

        nct_id  = id_mod.get("nctId", "")
        title   = id_mod.get("briefTitle", "")
        status  = _normalize_status(
            stat_mod.get("overallStatus", "")
        )
        phase_list = design.get("phases", []) or []
        phase   = phase_list[0] if phase_list else ""
        # Clean phase string
        phase   = phase.replace("_", " ").title() if phase else ""

        conditions = cond_mod.get("conditions", []) or []
        condition  = "; ".join(conditions[:3])

        if not nct_id:
            return None

        return {
            "nct_id":    nct_id,
            "title":     title,
            "status":    status,
            "phase":     phase,
            "condition": condition,
            "drug":      drug,
            "cancer":    cancer.split(";")[0].strip(),
        }

    except Exception:
        return None


def summarize_trials(trials: List[Dict]) -> Dict:
    """
    Summarize a list of trials for one drug-cancer pair.

    Returns
    -------
    dict with:
        n_trials           : total trials found
        n_failed           : terminated/withdrawn/suspended
        n_active           : currently recruiting
        n_completed        : completed trials
        highest_phase      : best phase reached
        has_failed_trial   : yes/no
        top_trial_nct      : NCT ID of highest phase trial
        failed_trial_ids   : list of failed NCT IDs
        phase_distribution : count per phase
    """
    if not trials:
        return {
            "n_trials":         0,
            "n_failed":         0,
            "n_active":         0,
            "n_completed":      0,
            "highest_phase":    "",
            "has_failed_trial": "no",
            "top_trial_nct":    "",
            "failed_trial_ids": "",
            "phase_distribution": "",
        }

    n_failed    = sum(1 for t in trials if t["status"] in FAILED_STATUSES)
    n_active    = sum(1 for t in trials if t["status"] in ACTIVE_STATUSES)
    n_completed = sum(1 for t in trials if t["status"] in COMPLETED_STATUSES)

    # Find highest phase trial
    best_trial  = max(trials, key=lambda t: _phase_score(t["phase"]))
    highest_phase = best_trial["phase"]
    top_nct       = best_trial["nct_id"]

    # Negative clinical development signal trial IDs
    # Note: termination alone does not imply resistance
    # concordance with literature resistance ratio needed
    failed_ids = [t["nct_id"] for t in trials
                  if t["status"] in FAILED_STATUSES]

    # Phase distribution
    phase_counts = defaultdict(int)
    for t in trials:
        phase_counts[t["phase"] or "Unknown"] += 1
    phase_dist = "; ".join(
        f"{p}:{c}" for p, c in sorted(
            phase_counts.items(),
            key=lambda x: _phase_score(x[0]),
            reverse=True
        )
    )

    # Flag if result was capped at API maximum
    result_capped = len(trials) >= 200

    return {
        "n_trials":                    len(trials),
        "n_trials_display":            f"≥{len(trials)}" if result_capped else str(len(trials)),
        "result_capped":               "yes" if result_capped else "no",
        # Renamed from n_failed — more defensible terminology
        "n_terminated_withdrawn":      n_failed,
        "n_failed":                    n_failed,  # kept for backward compat
        "n_active":                    n_active,
        "n_completed":                 n_completed,
        "highest_phase":               highest_phase,
        "has_failed_trial":            "yes" if n_failed > 0 else "no",
        "negative_clinical_signal":    "yes" if n_failed > 0 else "no",
        "top_trial_nct":               top_nct,
        "failed_trial_ids":            "; ".join(failed_ids[:5]),
        "phase_distribution":          phase_dist,
        # concordance is unknown unless exact biomarker match verified
        "concordance_note": ("result_capped_verify_manually"
                             if result_capped else "ok_to_interpret"),
    }


def compute_concordance(
    summary: Dict,
    literature_resistance: float,
) -> str:
    """
    Assess concordance between trial evidence
    and literature resistance signal.

    If literature says high resistance AND
    trials are mostly failed/terminated →
    HIGH concordance (both agree drug fails)

    If literature says high efficacy AND
    trials are Phase 3/4 completed →
    HIGH concordance (both agree drug works)

    Parameters
    ----------
    summary : dict from summarize_trials()
    literature_resistance : float
        Resistance ratio from megaMine output (0.0 to 1.0)

    Returns
    -------
    str : high / medium / low / unknown
    """
    if summary["n_trials"] == 0:
        return "unknown"

    trial_failure_rate = (
        summary["n_failed"] / summary["n_trials"]
        if summary["n_trials"] > 0 else 0.0
    )
    high_phase = _phase_score(summary["highest_phase"]) >= 3

    # Both agree drug works
    if literature_resistance < 0.25 and high_phase:
        return "high"

    # Both agree drug fails/resists
    if literature_resistance > 0.50 and trial_failure_rate > 0.30:
        return "high"

    # Mixed signals
    if literature_resistance > 0.50 and high_phase:
        return "low"

    return "medium"


def run_trials_linkage(
    df: pd.DataFrame,
    contradiction_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Main entry point for ClinicalTrials linkage.

    For each unique (drug, cancer) pair in megaMine output,
    queries ClinicalTrials.gov and attaches trial evidence.

    Parameters
    ----------
    df : pd.DataFrame
        Raw megaMine output
    contradiction_df : pd.DataFrame, optional
        Contradiction flags — used to compute resistance ratio
    output_path : str, optional
        Save results to Excel if provided
    dry_run : bool
        If True — skip API calls, return dummy data
        Useful for testing without internet

    Returns
    -------
    pd.DataFrame
        One row per (drug, cancer) with trial summary
    """
    print("🏥 Running ClinicalTrials.gov linkage...")

    # Use canonical_cancer_type if available — prevents noisy queries
    cancer_col = "canonical_cancer_type" if "canonical_cancer_type" in df.columns else "cancer_type"
    pairs = df[["drug_primary", cancer_col]].drop_duplicates()
    pairs = pairs.rename(columns={cancer_col: "cancer_type"})
    pairs = pairs[
        pairs["drug_primary"].notna() &
        pairs["cancer_type"].notna() &
        (pairs["drug_primary"] != "") &
        (pairs["cancer_type"] != "")
    ]
    # Drop duplicates after normalization
    pairs = pairs.drop_duplicates(["drug_primary", "cancer_type"])

    print(f"   Found {len(pairs)} unique drug-cancer pairs")

    # Build resistance lookup from contradiction_df
    resistance_lookup = {}
    if contradiction_df is not None:
        for _, row in contradiction_df.iterrows():
            key = (
                row["drug_primary"].lower(),
                row["cancer_type"].split(";")[0].strip().lower()
            )
            total = (row.get("efficacy_papers", 0) +
                     row.get("resistance_papers", 0))
            if total > 0:
                resistance_lookup[key] = (
                    row.get("resistance_papers", 0) / total
                )

    results = []

    for _, pair_row in pairs.iterrows():
        drug   = str(pair_row["drug_primary"]).strip()
        cancer = str(pair_row["cancer_type"]).strip()
        cancer_short = cancer.split(";")[0].strip()

        # Get top biomarker for this drug-cancer pair for precision
        top_gene = ""
        if "biomarker" in df.columns:
            mask_dc = (
                (df["drug_primary"].str.lower() == drug.lower()) &
                (df["canonical_cancer_type"].str.lower().str.contains(
                    cancer_short.lower(), na=False
                ) if "canonical_cancer_type" in df.columns else
                 df["cancer_type"].str.lower().str.contains(
                    cancer_short.lower(), na=False
                ))
            )
            top_genes = df[mask_dc]["biomarker"].value_counts()
            if len(top_genes) > 0:
                top_gene = top_genes.index[0]

        label = f"{drug} + {cancer_short}"
        if top_gene:
            label += f" + {top_gene}"
        print(f"   Querying: {label}...", end=" ")

        if dry_run:
            trials = _get_dummy_trials(drug, cancer_short)
        else:
            trials = fetch_trials_for_drug_cancer(
                drug, cancer_short, biomarker=top_gene
            )

        summary = summarize_trials(trials)

        # Get literature resistance ratio
        rkey = (drug.lower(), cancer_short.lower())
        lit_resistance = resistance_lookup.get(rkey, 0.0)

        # Compute concordance
        concordance = compute_concordance(summary, lit_resistance)

        print(f"{summary['n_trials']} trials "
              f"({summary['n_failed']} failed, "
              f"highest: {summary['highest_phase'] or 'N/A'})")

        results.append({
            "drug":                      drug,
            "cancer_type":               cancer_short,
            "top_biomarker_in_query":    top_gene,
            # n_trials shown as ≥200 when capped
            "n_trials":                  summary["n_trials"],
            "n_trials_display":          summary.get("n_trials_display", str(summary["n_trials"])),
            "result_capped":             summary.get("result_capped","no"),
            # Renamed from n_failed — terminaton ≠ failure
            "n_terminated_withdrawn":    summary["n_failed"],
            "n_failed_trials":           summary["n_failed"],  # backward compat
            "n_active_trials":           summary["n_active"],
            "n_completed_trials":        summary["n_completed"],
            "highest_trial_phase":       summary["highest_phase"],
            "has_terminated_trial":      summary["has_failed_trial"],
            "negative_clinical_signal":  summary.get("negative_clinical_signal","no"),
            "top_trial_nct":             summary["top_trial_nct"],
            "terminated_trial_ids":      summary["failed_trial_ids"],
            "phase_distribution":        summary["phase_distribution"],
            "literature_resistance_ratio": round(lit_resistance, 3),
            # concordance is unknown when results are capped or broad
            "trial_literature_concordance": (
                "unknown_verify_manually"
                if summary.get("result_capped","no") == "yes"
                else concordance
            ),
            # More defensible terminology
            "provisional_trial_literature_alignment": (
                "manual_review_required"
                if summary.get("result_capped","no") == "yes"
                else ("potentially_aligned" if concordance == "high"
                      else ("insufficient" if concordance == "low"
                            else "review_recommended"))
            ),
            "concordance_note":          summary.get("concordance_note",""),
        })

    result_df = pd.DataFrame(results)

    # Sort by failed trials descending — most failed first
    result_df = result_df.sort_values(
        "n_failed_trials", ascending=False
    ).reset_index(drop=True)

    # Print summary
    print(f"\n   ✅ Linked {len(result_df)} drug-cancer pairs")
    # Use has_terminated_trial (renamed) or fall back to has_failed_trial
    flag_col = "has_terminated_trial" if "has_terminated_trial" in result_df.columns else "has_failed_trial"
    if flag_col in result_df.columns:
        has_failed = result_df[result_df[flag_col] == "yes"]
        print(f"   ⚠️  {len(has_failed)} pairs have negative clinical development signals (terminated/withdrawn trials)")
    else:
        has_failed = pd.DataFrame()
        print(f"   ⚠️  Terminal trial column not found")
    print(f"   ℹ️  Cross-check with literature resistance ratio for resistance interpretation")

    # Save if path provided
    if output_path:
        result_df.to_excel(output_path, index=False)
        print(f"   💾 Saved to {output_path}")

    return result_df


def _get_dummy_trials(drug: str, cancer: str) -> List[Dict]:
    """
    Return realistic dummy trial data for testing
    without making real API calls.
    """
    dummy = {
        "erlotinib": [
            {"nct_id":"NCT00010101","title":"Erlotinib in NSCLC Phase III","status":"COMPLETED","phase":"Phase 3","condition":"NSCLC","drug":drug,"cancer":cancer},
            {"nct_id":"NCT00010102","title":"Erlotinib KRAS-mutant NSCLC","status":"TERMINATED","phase":"Phase 2","condition":"NSCLC","drug":drug,"cancer":cancer},
            {"nct_id":"NCT00010103","title":"Erlotinib maintenance NSCLC","status":"TERMINATED","phase":"Phase 2","condition":"NSCLC","drug":drug,"cancer":cancer},
            {"nct_id":"NCT00010104","title":"Erlotinib first line","status":"COMPLETED","phase":"Phase 3","condition":"NSCLC","drug":drug,"cancer":cancer},
            {"nct_id":"NCT00010105","title":"Erlotinib combination study","status":"RECRUITING","phase":"Phase 2","condition":"NSCLC","drug":drug,"cancer":cancer},
        ],
        "sotorasib": [
            {"nct_id":"NCT00020101","title":"Sotorasib KRAS G12C NSCLC","status":"COMPLETED","phase":"Phase 3","condition":"NSCLC","drug":drug,"cancer":cancer},
            {"nct_id":"NCT00020102","title":"Sotorasib combination","status":"RECRUITING","phase":"Phase 2","condition":"NSCLC","drug":drug,"cancer":cancer},
        ],
        "olaparib": [
            {"nct_id":"NCT00030101","title":"Olaparib BRCA breast cancer","status":"COMPLETED","phase":"Phase 3","condition":"Breast Cancer","drug":drug,"cancer":cancer},
            {"nct_id":"NCT00030102","title":"Olaparib maintenance","status":"COMPLETED","phase":"Phase 3","condition":"Breast Cancer","drug":drug,"cancer":cancer},
            {"nct_id":"NCT00030103","title":"Olaparib combination study","status":"RECRUITING","phase":"Phase 2","condition":"Breast Cancer","drug":drug,"cancer":cancer},
        ],
    }
    return dummy.get(drug.lower(), [
        {"nct_id":"NCT99999999","title":f"{drug} in {cancer}","status":"COMPLETED","phase":"Phase 2","condition":cancer,"drug":drug,"cancer":cancer},
    ])
