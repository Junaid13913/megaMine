"""
normalizers.py — megaMine v2.0
Cancer Type and Evidence Normalization Module

PURPOSE:
    Fixes the most critical real-output quality problem:
    cancer_type column contains sentence fragments mixed
    with real cancer names.

    Bad examples from real run:
        "Comprehending The Tumor"
        "Has Been Shown To Possess Anticancer"
        "The Tumor"
        "In Non-Small Cell Lung Cancer"  ← close but needs cleaning

    Good examples:
        "Non-Small Cell Lung Cancer"
        "NSCLC"
        "Breast Cancer"

    All downstream modules (temporal, contradiction, graph,
    trials) should use canonical_cancer_type not raw cancer_type.

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import re
import pandas as pd
from typing import Optional, Dict, Tuple
from collections import OrderedDict

# ─── Known cancer canonical forms ────────────────────────────
# Maps variations → canonical long name + short code
CANCER_CANONICAL = OrderedDict([
    # Lung
    ("non-small cell lung cancer",      ("Non-Small Cell Lung Cancer", "NSCLC")),
    ("nsclc",                           ("Non-Small Cell Lung Cancer", "NSCLC")),
    ("lung adenocarcinoma",             ("Lung Adenocarcinoma",        "LUAD")),
    ("luad",                            ("Lung Adenocarcinoma",        "LUAD")),
    ("lung squamous cell carcinoma",    ("Lung Squamous Cell Carcinoma","LUSC")),
    ("lusc",                            ("Lung Squamous Cell Carcinoma","LUSC")),
    ("small cell lung cancer",          ("Small Cell Lung Cancer",     "SCLC")),
    ("sclc",                            ("Small Cell Lung Cancer",     "SCLC")),
    ("lung cancer",                     ("Lung Cancer",                "LC")),
    # Breast
    ("triple-negative breast cancer",   ("Triple-Negative Breast Cancer","TNBC")),
    ("tnbc",                            ("Triple-Negative Breast Cancer","TNBC")),
    ("breast cancer",                   ("Breast Cancer",              "BC")),
    ("bc",                              ("Breast Cancer",              "BC")),
    # Colorectal
    ("colorectal cancer",               ("Colorectal Cancer",          "CRC")),
    ("colon cancer",                    ("Colorectal Cancer",          "CRC")),
    ("rectal cancer",                   ("Colorectal Cancer",          "CRC")),
    ("crc",                             ("Colorectal Cancer",          "CRC")),
    # Brain
    ("glioblastoma",                    ("Glioblastoma",               "GBM")),
    ("gbm",                             ("Glioblastoma",               "GBM")),
    ("glioma",                          ("Glioma",                     "Glioma")),
    # Pancreatic
    ("pancreatic ductal adenocarcinoma",("Pancreatic Ductal Adenocarcinoma","PDAC")),
    ("pancreatic cancer",               ("Pancreatic Cancer",          "PC")),
    ("pdac",                            ("Pancreatic Ductal Adenocarcinoma","PDAC")),
    # Liver
    ("hepatocellular carcinoma",        ("Hepatocellular Carcinoma",   "HCC")),
    ("hcc",                             ("Hepatocellular Carcinoma",   "HCC")),
    # Prostate
    ("prostate cancer",                 ("Prostate Cancer",            "PCa")),
    ("pca",                             ("Prostate Cancer",            "PCa")),
    # Ovarian
    ("ovarian cancer",                  ("Ovarian Cancer",             "OC")),
    ("oc",                              ("Ovarian Cancer",             "OC")),
    # Melanoma
    ("melanoma",                        ("Melanoma",                   "Melanoma")),
    # Gastric
    ("gastric cancer",                  ("Gastric Cancer",             "GC")),
    ("gc",                              ("Gastric Cancer",             "GC")),
    ("stomach cancer",                  ("Gastric Cancer",             "GC")),
    # Kidney
    ("renal cell carcinoma",            ("Renal Cell Carcinoma",       "RCC")),
    ("rcc",                             ("Renal Cell Carcinoma",       "RCC")),
    # Bladder
    ("urothelial carcinoma",            ("Urothelial Carcinoma",       "UC")),
    ("bladder cancer",                  ("Bladder Cancer",             "BC")),
    ("uc",                              ("Urothelial Carcinoma",       "UC")),
    # Head and neck
    ("head and neck squamous cell carcinoma",("Head And Neck Squamous Cell Carcinoma","HNSCC")),
    ("hnscc",                           ("Head And Neck Squamous Cell Carcinoma","HNSCC")),
    # Thyroid
    ("thyroid cancer",                  ("Thyroid Cancer",             "TC")),
    # Leukemia / lymphoma
    ("acute myeloid leukemia",          ("Acute Myeloid Leukemia",     "AML")),
    ("aml",                             ("Acute Myeloid Leukemia",     "AML")),
    ("chronic lymphocytic leukemia",    ("Chronic Lymphocytic Leukemia","CLL")),
    ("cll",                             ("Chronic Lymphocytic Leukemia","CLL")),
    ("diffuse large b-cell lymphoma",   ("Diffuse Large B-Cell Lymphoma","DLBCL")),
    # Lymphoma
    ("anaplastic large cell lymphoma",  ("Anaplastic Large Cell Lymphoma","ALCL")),
    ("alcl",                            ("Anaplastic Large Cell Lymphoma","ALCL")),
    ("alk-positive anaplastic large cell lymphoma",("Anaplastic Large Cell Lymphoma","ALCL")),
    ("diffuse large b cell lymphoma",   ("Diffuse Large B-Cell Lymphoma","DLBCL")),
    ("dlbcl",                           ("Diffuse Large B-Cell Lymphoma","DLBCL")),
    ("hodgkin lymphoma",                ("Hodgkin Lymphoma",            "HL")),
    ("non-hodgkin lymphoma",            ("Non-Hodgkin Lymphoma",        "NHL")),
    ("mantle cell lymphoma",            ("Mantle Cell Lymphoma",        "MCL")),
    ("follicular lymphoma",             ("Follicular Lymphoma",         "FL")),
    # Cholangiocarcinoma
    ("cholangiocarcinoma",              ("Cholangiocarcinoma",         "CCA")),
    ("cca",                             ("Cholangiocarcinoma",         "CCA")),
    ("biliary tract cancer",            ("Biliary Tract Cancer",       "BTC")),
    # Endometrial
    ("endometrial cancer",              ("Endometrial Cancer",         "EC")),
    ("endometrial carcinoma",           ("Endometrial Cancer",         "EC")),
    # Esophageal
    ("esophageal cancer",               ("Esophageal Cancer",          "EsC")),
    ("esophageal squamous cell carcinoma",("Esophageal Squamous Cell Carcinoma","ESCC")),
    ("escc",                            ("Esophageal Squamous Cell Carcinoma","ESCC")),
])

# ─── Rejection patterns — these are NOT cancer names ──────────
# Sentence fragments, paper title words, method terms
REJECTION_PATTERNS = [
    r"^the\b",
    r"^in\s+the\b",
    r"^a\b",
    r"^an\b",
    r"^this\b",
    r"^these\b",
    r"^our\b",
    r"^we\b",
    r"^has\s+been",
    r"^have\s+been",
    r"^could\b",
    r"^may\b",
    r"^comprehend",
    r"^possess",
    r"^suppress",
    r"^anticancer\b",
    r"^anti-cancer\b",
    r"\bshown\s+to\b",
    r"\bpossess\s+anticancer\b",
    r"\bdose.depend",
    r"\binhibit\b",
    r"\binduce\b",
    r"\bsignaling\b",
    r"\bpathway\b",
    r"\bexpression\b",
    r"\bactivation\b",
    r"\bphosphorylation\b",
    r"\bmutation\b",
    r"\btreatment\b",
    r"\btherapy\b",
    r"\bpatients\b",
    r"\bstudies\b",
    r"\banalysis\b",
    r"\bresults\b",
    r"\bmethod",
    r"\bfigure\b",
    r"\btable\b",
    r"\bsupplement",
]

# Extra rejection patterns for sentence fragments
# that look like cancer names but are not
REJECTION_PATTERNS.extend([
    r"^and\s+the\s+",
    r"^and\s+[a-z]+-",          # "and c-met in these..."
    r"^but\s+also\s+",          # "but also tki-resistant..."
    r"^pathological\s+transformation",
    r"^transformation\s+to\s+",
    r"^histological\s+transformation",
    r"^squamous\s+transformation",
    r"^small\s+cell\s+transformation",
    r"^malignant\s+transformation",
    r"^oncogenic\s+transformation",
    r"^epithelial\s+mesenchymal",
    r"^phenotypic\s+",
    r"^acquired\s+",
    r"^resistance\s+to\s+",
    r"^progression\s+",
    r"^treatment\s+with\s+",
    r"^inhibition\s+of\s+",
    r"^tki.resistant\s+",       # "tki-resistant cancer"
    r"^drug.resistant\s+",
    r"^chemo.resistant\s+",
    r"^these\s+cancer",
    r"these\s+cancer",
    r"^in\s+these\s+",
    r"^also\s+",
    r"^including\s+",
    r"^or\s+other\s+",
    r"^with\s+",
    r"^of\s+",
    r"^the\s+",
    r"^a\s+",
])

REJECTION_RE = re.compile(
    "|".join(REJECTION_PATTERNS), re.IGNORECASE
)

# ─── Title keyword filter (from extractor fix) ────────────────
TITLE_KEYWORDS = {
    "study","trial","assess","efficacy","safety",
    "patients","treatment","therapy","analysis",
    "randomized","double-blind","placebo","versus",
    "combination","phase","clinical","novel","approach",
    "method","results","outcome","response","testing",
    "reading","scoring","predict","established","means",
    "selecting","optimal","wild-type","comprehend",
    "suppress","anticancer","possess","inhibit","induce",
}


def normalize_cancer_type(raw_cancer: str) -> Dict:
    """
    Normalize a raw cancer type string to canonical form.

    Safer logic — searches for known cancer names BEFORE
    rejecting sentence fragments. This prevents discarding
    rows like:
        "Cddo-Me Could Suppress Tumor; Non-Small Cell Lung Cancer"
    where the valid cancer is in a later semicolon segment.

    Order:
        1. Search full raw string for known canonical cancers
        2. Search each semicolon-separated segment
        3. Reject obvious sentence/title fragments
        4. Keep unknown but cancer-like names as low confidence

    Returns
    -------
    dict with:
        canonical_cancer_type : str — clean canonical name
        cancer_code           : str — short code (NSCLC etc)
        cancer_confidence     : str — high / medium / low / rejected
        raw_cancer_type       : str — original input
    """
    raw = str(raw_cancer or "").strip()

    if not raw or raw.lower() in ("", "unknown", "nan", "none"):
        return {
            "canonical_cancer_type": "",
            "cancer_code":           "",
            "cancer_confidence":     "rejected",
            "raw_cancer_type":       raw,
        }

    raw_lower = raw.lower()

    # ── Step 1: Search FULL raw string for known cancers ──────
    # Do this BEFORE any rejection — a long noisy string may
    # still contain a valid cancer name somewhere
    for key, (canonical, code) in CANCER_CANONICAL.items():
        if key in raw_lower:
            confidence = "high" if raw_lower.strip() == key else "medium"
            return {
                "canonical_cancer_type": canonical,
                "cancer_code":           code,
                "cancer_confidence":     confidence,
                "raw_cancer_type":       raw,
            }

    # ── Step 2: Search each semicolon-separated segment ───────
    # Handles cases like:
    #   "Bad fragment; Non-Small Cell Lung Cancer; NSCLC"
    #   → finds NSCLC in second or third segment
    segments = [
        re.sub(r"^(in|of|for|with|from|the|a|an)\s+",
               "", s.strip(), flags=re.IGNORECASE)
        for s in re.split(r";|\||,", raw)
        if s and s.strip()
    ]

    for seg in segments:
        seg_lower = seg.lower().strip()

        # Exact segment match
        if seg_lower in CANCER_CANONICAL:
            canonical, code = CANCER_CANONICAL[seg_lower]
            return {
                "canonical_cancer_type": canonical,
                "cancer_code":           code,
                "cancer_confidence":     "high",
                "raw_cancer_type":       raw,
            }

        # Partial match within segment
        for key, (canonical, code) in CANCER_CANONICAL.items():
            if len(key) > 3 and key in seg_lower:
                return {
                    "canonical_cancer_type": canonical,
                    "cancer_code":           code,
                    "cancer_confidence":     "medium",
                    "raw_cancer_type":       raw,
                }

    # ── Step 3: Reject obvious non-cancer sentence fragments ──
    # Only reject AFTER confirming no known cancer was found
    clean = re.sub(r"^(in|of|for|with|from|the|a|an)\s+",
                   "", raw, flags=re.IGNORECASE).strip()
    clean_lower = clean.lower().strip()
    word_count  = len(clean.split())

    if word_count > 7:
        return {
            "canonical_cancer_type": "",
            "cancer_code":           "",
            "cancer_confidence":     "rejected",
            "raw_cancer_type":       raw,
        }

    if REJECTION_RE.search(clean_lower):
        return {
            "canonical_cancer_type": "",
            "cancer_code":           "",
            "cancer_confidence":     "rejected",
            "raw_cancer_type":       raw,
        }

    if any(kw in clean_lower for kw in TITLE_KEYWORDS):
        return {
            "canonical_cancer_type": "",
            "cancer_code":           "",
            "cancer_confidence":     "rejected",
            "raw_cancer_type":       raw,
        }

    if sum(c.isdigit() for c in clean) > 2:
        return {
            "canonical_cancer_type": "",
            "cancer_code":           "",
            "cancer_confidence":     "rejected",
            "raw_cancer_type":       raw,
        }

    # ── Step 4: Keep unknown cancer-like names as low conf ────
    cancer_suffixes = [
        "carcinoma", "sarcoma", "leukemia", "lymphoma",
        "melanoma", "glioma", "cancer", "neoplasm",
        "adenocarcinoma", "blastoma",
    ]
    if any(clean_lower.endswith(sfx) for sfx in cancer_suffixes):
        return {
            "canonical_cancer_type": clean.title(),
            "cancer_code":           "",
            "cancer_confidence":     "low",
            "raw_cancer_type":       raw,
        }

    # ── Step 5: Reject everything else ────────────────────────
    return {
        "canonical_cancer_type": "",
        "cancer_code":           "",
        "cancer_confidence":     "rejected",
        "raw_cancer_type":       raw,
    }


def normalize_cancer_column(df: pd.DataFrame,
                              cancer_col: str = "cancer_type") -> pd.DataFrame:
    """
    Apply cancer normalization to an entire DataFrame.

    Adds three new columns:
        canonical_cancer_type
        cancer_code
        cancer_confidence

    All downstream modules should use canonical_cancer_type
    instead of the raw cancer_type column.

    Parameters
    ----------
    df         : pd.DataFrame — megaMine output
    cancer_col : str          — name of raw cancer column

    Returns
    -------
    pd.DataFrame with 3 new columns added
    """
    df = df.copy()

    print("🏷️  Normalizing cancer types...")

    results = df[cancer_col].apply(normalize_cancer_type)
    df["canonical_cancer_type"] = results.apply(
        lambda x: x["canonical_cancer_type"]
    )
    df["cancer_code"]           = results.apply(
        lambda x: x["cancer_code"]
    )
    df["cancer_confidence"]     = results.apply(
        lambda x: x["cancer_confidence"]
    )

    # Report stats
    total    = len(df)
    high     = (df["cancer_confidence"] == "high").sum()
    medium   = (df["cancer_confidence"] == "medium").sum()
    low      = (df["cancer_confidence"] == "low").sum()
    rejected = (df["cancer_confidence"] == "rejected").sum()
    has_canonical = (df["canonical_cancer_type"] != "").sum()

    print(f"   Total rows:          {total:,}")
    print(f"   High confidence:     {high:,}")
    print(f"   Medium confidence:   {medium:,}")
    print(f"   Low confidence:      {low:,}")
    print(f"   Rejected (non-cancer): {rejected:,}")
    print(f"   Has canonical type:  {has_canonical:,}")
    print()
    print("   Canonical types found:")
    for ct, count in df["canonical_cancer_type"].value_counts().head(10).items():
        if ct:
            print(f"     {ct}: {count}")

    return df


def add_resistance_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split resistance_observed into three precise fields.

    Addresses Priority 2: resistance_observed was too broad —
    it flagged any resistance mention even in efficacy sentences.

    New fields:
        resistance_context    : sentence mentions resistance anywhere
        resistance_evidence   : sentence directly reports resistance
        resistance_direction  : sensitivity / resistance /
                                post_resistance_efficacy / unclear

    contradiction.py should use resistance_evidence not
    resistance_observed.

    Parameters
    ----------
    df : pd.DataFrame — megaMine output

    Returns
    -------
    pd.DataFrame with 3 new columns
    """
    import re

    df = df.copy()

    # Patterns for direct resistance evidence
    DIRECT_RESISTANCE = re.compile(
        r"(conferred?\s+resistance|"
        r"acquired\s+resistance|"
        r"primary\s+resistance|"
        r"resistant\s+to\s+\w+|"
        r"resistance\s+to\s+\w+|"
        r"refractory\s+to|"
        r"progressed\s+on|"
        r"treatment\s+failure|"
        r"did\s+not\s+respond|"
        r"no\s+response\s+to|"
        r"failed\s+to\s+respond)",
        re.IGNORECASE
    )

    # Patterns for post-resistance efficacy
    POST_RESISTANCE = re.compile(
        r"(after\s+\w+\s+resistance|"
        r"following\s+\w+\s+resistance|"
        r"overcome\s+resistance|"
        r"bypass\s+resistance|"
        r"after\s+progression\s+on|"
        r"second.line\s+after|"
        r"salvage\s+therapy)",
        re.IGNORECASE
    )

    # Context: any mention of resistance
    RESISTANCE_CONTEXT = re.compile(
        r"\bresist\w*\b|\brefractory\b|\bprogression\b",
        re.IGNORECASE
    )

    resistance_context   = []
    resistance_evidence  = []
    resistance_direction = []

    for _, row in df.iterrows():
        sent = str(row.get("summary_sentence", "") or "")
        ev   = str(row.get("evidence_type", "") or "").lower()
        res  = str(row.get("resistance_observed", "") or "").lower()

        has_context  = bool(RESISTANCE_CONTEXT.search(sent))
        has_direct   = bool(DIRECT_RESISTANCE.search(sent))
        has_post_res = bool(POST_RESISTANCE.search(sent))

        resistance_context.append("yes" if has_context else "no")
        resistance_evidence.append("yes" if has_direct else "no")

        # has_direct = sentence directly states resistance
        # Do NOT require ev == "resistance" — extractor may
        # misclassify a negated sentence as "efficacy"
        # The direct pattern is more reliable than evidence_type
        if has_direct:
            direction = "resistance"
        elif has_post_res and ev == "efficacy":
            direction = "post_resistance_efficacy"
        elif ev == "efficacy" and not has_direct:
            direction = "sensitivity"
        elif res == "yes" and ev == "resistance":
            direction = "possible_resistance"
        elif has_context and not has_direct:
            direction = "unclear"
        else:
            direction = "unclear"

        resistance_direction.append(direction)

    df["resistance_context"]   = resistance_context
    df["resistance_evidence"]  = resistance_evidence
    df["resistance_direction"] = resistance_direction

    print("🔬 Resistance context split:")
    print(f"   resistance_context=yes:  "
          f"{df['resistance_context'].eq('yes').sum()}")
    print(f"   resistance_evidence=yes: "
          f"{df['resistance_evidence'].eq('yes').sum()}")
    print("   Direction distribution:")
    for d, c in df["resistance_direction"].value_counts().items():
        print(f"     {d}: {c}")

    return df


def reconcile_evidence_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconcile evidence_type with resistance_direction.

    Problem: extractor may label a sentence as "efficacy"
    even when resistance_direction says "resistance"
    (e.g. negated efficacy sentences).

    This creates final_evidence_type which is authoritative:
        resistance_direction=resistance     → final=resistance
        resistance_direction=post_resist... → final=post_resistance_efficacy
        evidence_type=efficacy              → final=efficacy
        else                                → final=evidence_type

    Keep evidence_type_raw for auditability.

    Parameters
    ----------
    df : pd.DataFrame — megaMine output with both fields

    Returns
    -------
    pd.DataFrame with final_evidence_type added
    """
    df = df.copy()

    # Preserve original
    if "evidence_type_raw" not in df.columns:
        df["evidence_type_raw"] = df["evidence_type"].copy()

    def _reconcile(row):
        rd = str(row.get("resistance_direction","") or "").lower()
        et = str(row.get("evidence_type","") or "").lower()
        if rd == "resistance":
            return "resistance"
        elif rd == "post_resistance_efficacy":
            return "post_resistance_efficacy"
        elif et in ("efficacy","resistance","toxicity","background","review"):
            return et
        return et

    df["final_evidence_type"] = df.apply(_reconcile, axis=1)

    print("🔄 Evidence type reconciliation:")
    print(f"   evidence_type_raw distribution:")
    print(df["evidence_type_raw"].value_counts().to_string())
    print(f"   final_evidence_type distribution:")
    print(df["final_evidence_type"].value_counts().to_string())

    return df
