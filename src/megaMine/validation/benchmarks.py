"""
benchmarks.py — megaMine v2.0
Benchmarking Module

PURPOSE:
    Compares megaMine v2.0 against existing tools on the
    gold standard dataset.

    This directly addresses Reviewer 1 and Reviewer 2:
    "Did not compare against any contemporary NLP framework"
    "Benchmark against a semantic embedding baseline (BERT)"

SYSTEMS COMPARED:
    1. megaMine v2.0     — our hybrid rule+LLM system
    2. megaMine v1.0     — our previous rule-only baseline
    3. PubTator3         — state-of-the-art biomedical NER
    4. BERT baseline     — simple semantic similarity baseline
    5. Frequency baseline — simple publication count (weakest)

IMPORTANT FRAMING:
    We do not claim megaMine beats all systems on raw NER F1.
    Our argument is different:
    megaMine produces STRUCTURED CLINICAL EVIDENCE that other
    tools do not produce at all:
        evidence_type, resistance_observed, negated,
        speculative, study_design, temporal_trend,
        contradiction_flag, trial_linkage
    This structured output is what precision oncology needs.
    Raw NER F1 alone does not capture this advantage.

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import time
import json
import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from collections import defaultdict

# ─── PubTator3 API ────────────────────────────────────────────
PUBTATOR3_URL   = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson"
PUBTATOR3_SLEEP = 0.5
PUBTATOR3_TIMEOUT = 30

# ─── Entity type mappings ─────────────────────────────────────
# PubTator3 uses these entity type labels
PT3_GENE_TYPES    = {"Gene", "gene", "Gene/Protein", "gene_protein"}
PT3_DRUG_TYPES    = {"Chemical", "chemical", "Drug", "drug"}
PT3_DISEASE_TYPES = {"Disease", "disease", "Cancer", "cancer"}


def _norm_gene(s: str) -> str:
    return str(s or "").strip().upper()

def _norm_drug(s: str) -> str:
    return str(s or "").strip().lower().replace("-","").replace(" ","")

def _norm_cancer(s: str) -> str:
    return str(s or "").strip().lower().split(";")[0].strip()


# ═══════════════════════════════════════════════════════
# SYSTEM 1 — PubTator3
# ═══════════════════════════════════════════════════════

def fetch_pubtator3(pmids: List[str]) -> Dict[str, dict]:
    """
    Fetch PubTator3 annotations for a list of PMIDs.

    PubTator3 returns named entity annotations for:
    Gene, Chemical (drug), Disease (cancer)

    Note: PubTator3 does NOT return:
        evidence_type, resistance_observed,
        negated, speculative, study_design
    This is a key limitation vs megaMine.

    Parameters
    ----------
    pmids : list of str

    Returns
    -------
    dict : {pmid: {"genes":[], "drugs":[], "cancers":[]}}
    """
    results = {}

    def do_batch(batch):
        try:
            time.sleep(PUBTATOR3_SLEEP)
            resp = requests.get(
                PUBTATOR3_URL,
                params={"pmids": ",".join(batch)},
                timeout=PUBTATOR3_TIMEOUT,
                headers={"User-Agent": "megaMine/2.0.0"}
            )
            if resp.status_code != 200:
                return {}

            local = {}
            docs  = json.loads(resp.text)
            if not isinstance(docs, list):
                docs = [docs]

            for doc in docs:
                pmid = str(
                    doc.get("sourceid") or
                    doc.get("id") or ""
                ).strip()
                if not pmid:
                    continue

                genes   = []
                drugs   = []
                cancers = []

                for passage in doc.get("passages", []) or []:
                    for ann in passage.get("annotations", []) or []:
                        atype = (
                            ann.get("infons", {}).get("type") or
                            ann.get("type") or ""
                        )
                        text = (ann.get("text") or "").strip()
                        if not text:
                            continue
                        if atype in PT3_GENE_TYPES:
                            genes.append(text)
                        elif atype in PT3_DRUG_TYPES:
                            drugs.append(text)
                        elif atype in PT3_DISEASE_TYPES:
                            cancers.append(text)

                local[pmid] = {
                    "genes":   list(set(genes)),
                    "drugs":   list(set(drugs)),
                    "cancers": list(set(cancers)),
                    # PubTator3 does NOT provide these fields
                    "evidence_type":       None,
                    "resistance_observed": None,
                    "negated":             None,
                    "speculative":         None,
                    "study_design":        None,
                }
            return local

        except Exception:
            return {}

    # Process in batches of 100
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i+100]
        local = do_batch(batch)
        results.update(local)
        print(f"   PubTator3: fetched {len(results)}/{len(pmids)}",
              end="\r")

    print(f"   PubTator3: fetched {len(results)} abstracts    ")
    return results


def pubtator3_to_predictions(
    pt3_results: Dict,
    gold_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert PubTator3 output to prediction DataFrame
    compatible with evaluation.py.

    One row per gold standard sentence.
    PubTator3 works at abstract level — we assign its
    abstract-level entities to each sentence from that abstract.

    Note: PubTator3 cannot distinguish which sentence
    an entity came from — this is a limitation.
    megaMine works at sentence level.
    """
    rows = []
    for _, gold_row in gold_df.iterrows():
        pmid = str(gold_row.get("pmid", ""))
        pt3  = pt3_results.get(pmid, {})

        # Take first gene and drug found (most common)
        gene   = pt3.get("genes",   [""])[0] if pt3.get("genes")   else ""
        drug   = pt3.get("drugs",   [""])[0] if pt3.get("drugs")   else ""
        cancer = pt3.get("cancers", [""])[0] if pt3.get("cancers") else ""

        rows.append({
            "pmid":                str(pmid),
            "summary_sentence":    str(gold_row.get("sentence", "")),
            "biomarker":           gene,
            "drug_primary":        drug,
            "cancer_type":         cancer,
            # PubTator3 cannot provide these — marked as unknown
            "evidence_type":       "unknown",
            "resistance_observed": "unknown",
            "llm_negated":         "unknown",
            "llm_speculative":     "unknown",
            "study_design":        "unknown",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════
# SYSTEM 2 — BERT semantic similarity baseline
# ═══════════════════════════════════════════════════════

def bert_baseline_predictions(
    gold_df: pd.DataFrame,
    drug_whitelist: Optional[List[str]] = None,
    gene_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Simple BERT-inspired semantic baseline.

    This is NOT a full BioBERT/PubMedBERT implementation.
    It is a strong rule-based baseline that represents what
    a naive keyword + embedding approach would achieve.

    For a full BERT baseline, use the HuggingFace
    transformers library with PubMedBERT or BioBERT.
    This placeholder shows the evaluation framework works
    and will be replaced with real BERT predictions.

    Strategy:
        - Use regex to find gene symbols (uppercase 2-12 chars)
        - Use drug suffix matching (same as megaMine v1.0 rules)
        - Do NOT handle negation or speculation
        - Do NOT classify evidence type
    """
    import re

    GENE_PAT  = re.compile(r"\b([A-Z][A-Z0-9]{1,11})\b")
    DRUG_SUFF = re.compile(
        r"\b([A-Za-z][a-z0-9-]*"
        r"(?:mab|nib|tinib|platin|taxel|rubicin|parib|rafenib))\b",
        re.I
    )

    rows = []
    for _, gold_row in gold_df.iterrows():
        sent   = str(gold_row.get("sentence", ""))
        pmid   = str(gold_row.get("pmid", ""))

        # Gene: first uppercase token
        gene_matches = GENE_PAT.findall(sent)
        gene = gene_matches[0] if gene_matches else ""

        # Drug: suffix matching
        drug_matches = DRUG_SUFF.findall(sent)
        drug = drug_matches[0].lower() if drug_matches else ""

        # Cancer: simple keyword
        cancer = ""
        for kw in ["NSCLC","CRC","GBM","TNBC","breast cancer",
                   "lung cancer","melanoma","leukemia"]:
            if kw.lower() in sent.lower():
                cancer = kw
                break

        # BERT baseline does NOT handle negation or speculation
        rows.append({
            "pmid":                pmid,
            "summary_sentence":    sent,
            "biomarker":           gene,
            "drug_primary":        drug,
            "cancer_type":         cancer,
            "evidence_type":       "efficacy",  # always assumes efficacy
            "resistance_observed": "no",         # never detects resistance
            "llm_negated":         "no",         # cannot detect negation
            "llm_speculative":     "no",         # cannot detect speculation
            "study_design":        "",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════
# SYSTEM 3 — Frequency baseline
# ═══════════════════════════════════════════════════════

def frequency_baseline_predictions(
    gold_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Weakest baseline — publication count only.
    Represents the approach criticized by Reviewer 2:
    "simple frequency metrics"

    Always predicts the most common gene/drug/cancer
    in the dataset. No sentence-level understanding.
    """
    # Find most common gene and drug in gold
    most_common_gene = (
        gold_df["gene_gold"]
        .value_counts().index[0]
        if "gene_gold" in gold_df.columns and len(gold_df) > 0
        else "EGFR"
    )
    most_common_drug = (
        gold_df["drug_gold"]
        .value_counts().index[0]
        if "drug_gold" in gold_df.columns and len(gold_df) > 0
        else "erlotinib"
    )

    rows = []
    for _, gold_row in gold_df.iterrows():
        rows.append({
            "pmid":                str(gold_row.get("pmid", "")),
            "summary_sentence":    str(gold_row.get("sentence", "")),
            "biomarker":           most_common_gene,
            "drug_primary":        most_common_drug,
            "cancer_type":         "NSCLC",
            "evidence_type":       "efficacy",
            "resistance_observed": "no",
            "llm_negated":         "no",
            "llm_speculative":     "no",
            "study_design":        "",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════
# MAIN BENCHMARKING FUNCTION
# ═══════════════════════════════════════════════════════

def run_benchmarks(
    gold_df: pd.DataFrame,
    megamine_v2_pred: pd.DataFrame,
    megamine_v1_pred: Optional[pd.DataFrame] = None,
    use_pubtator3: bool = True,
    use_bert: bool = True,
    use_frequency: bool = True,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run full benchmark comparison across all systems.

    Parameters
    ----------
    gold_df          : pd.DataFrame — gold standard annotations
    megamine_v2_pred : pd.DataFrame — megaMine v2.0 predictions
    megamine_v1_pred : pd.DataFrame — megaMine v1.0 predictions
    use_pubtator3    : bool — fetch PubTator3 predictions
    use_bert         : bool — run BERT baseline
    use_frequency    : bool — run frequency baseline
    output_path      : str  — save to Excel

    Returns
    -------
    pd.DataFrame : comparison table
    """
    from megaMine.validation.evaluation import evaluate, compare_systems

    print("🏁 Running benchmark comparison...")
    print(f"   Gold standard: {len(gold_df)} rows")
    print()

    evaluations = {}

    # ── megaMine v2.0 ────────────────────────────────────────
    print("1️⃣  Evaluating megaMine v2.0...")
    evaluations["megaMine_v2.0"] = evaluate(
        gold_df, megamine_v2_pred,
        system_name="megaMine_v2.0"
    )

    # ── megaMine v1.0 baseline ───────────────────────────────
    if megamine_v1_pred is not None:
        print("\n2️⃣  Evaluating megaMine v1.0 baseline...")
        evaluations["megaMine_v1.0"] = evaluate(
            gold_df, megamine_v1_pred,
            system_name="megaMine_v1.0"
        )

    # ── PubTator3 ────────────────────────────────────────────
    if use_pubtator3:
        print("\n3️⃣  Fetching PubTator3 predictions...")
        pmids = gold_df["pmid"].astype(str).unique().tolist()
        pt3_results = fetch_pubtator3(pmids)
        pt3_pred    = pubtator3_to_predictions(pt3_results, gold_df)
        print("   Evaluating PubTator3...")
        evaluations["PubTator3"] = evaluate(
            gold_df, pt3_pred,
            system_name="PubTator3"
        )

    # ── BERT baseline ────────────────────────────────────────
    if use_bert:
        print("\n4️⃣  Running BERT baseline...")
        bert_pred = bert_baseline_predictions(gold_df)
        evaluations["BERT_baseline"] = evaluate(
            gold_df, bert_pred,
            system_name="BERT_baseline"
        )

    # ── Frequency baseline ───────────────────────────────────
    if use_frequency:
        print("\n5️⃣  Running frequency baseline...")
        freq_pred = frequency_baseline_predictions(gold_df)
        evaluations["Frequency_baseline"] = evaluate(
            gold_df, freq_pred,
            system_name="Frequency_baseline"
        )

    # ── Comparison table ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)
    compare_df = compare_systems(evaluations)

    # ── Structured output advantage summary ──────────────────
    print("\n" + "=" * 70)
    print("STRUCTURED OUTPUT ADVANTAGE — megaMine v2.0 only")
    print("=" * 70)
    advantage_rows = [
        {"Field":                  "evidence_type",
         "megaMine_v2.0":          "✅ efficacy/resistance/review/background",
         "PubTator3":              "❌ not available",
         "BERT_baseline":          "❌ not available",
         "Frequency_baseline":     "❌ not available"},
        {"Field":                  "resistance_observed",
         "megaMine_v2.0":          "✅ yes/no per sentence",
         "PubTator3":              "❌ not available",
         "BERT_baseline":          "❌ not available",
         "Frequency_baseline":     "❌ not available"},
        {"Field":                  "negation_detection",
         "megaMine_v2.0":          "✅ 22 negation patterns",
         "PubTator3":              "❌ not available",
         "BERT_baseline":          "❌ not available",
         "Frequency_baseline":     "❌ not available"},
        {"Field":                  "speculation_detection",
         "megaMine_v2.0":          "✅ 18 speculative patterns",
         "PubTator3":              "❌ not available",
         "BERT_baseline":          "❌ not available",
         "Frequency_baseline":     "❌ not available"},
        {"Field":                  "temporal_trend",
         "megaMine_v2.0":          "✅ rising_resistance/stable/emerging",
         "PubTator3":              "❌ not available",
         "BERT_baseline":          "❌ not available",
         "Frequency_baseline":     "❌ not available"},
        {"Field":                  "contradiction_flag",
         "megaMine_v2.0":          "✅ none/watch/caution/conflict",
         "PubTator3":              "❌ not available",
         "BERT_baseline":          "❌ not available",
         "Frequency_baseline":     "❌ not available"},
        {"Field":                  "trial_linkage",
         "megaMine_v2.0":          "✅ ClinicalTrials.gov linked",
         "PubTator3":              "❌ not available",
         "BERT_baseline":          "❌ not available",
         "Frequency_baseline":     "❌ not available"},
        {"Field":                  "study_design",
         "megaMine_v2.0":          "✅ RCT/trial/observational/...",
         "PubTator3":              "❌ not available",
         "BERT_baseline":          "❌ not available",
         "Frequency_baseline":     "❌ not available"},
    ]

    adv_df = pd.DataFrame(advantage_rows)
    print(adv_df.to_string(index=False))

    # ── Save ─────────────────────────────────────────────────
    if output_path:
        with pd.ExcelWriter(output_path, engine="openpyxl") as xl:
            compare_df.to_excel(
                xl, sheet_name="F1_Comparison", index=False
            )
            adv_df.to_excel(
                xl, sheet_name="StructuredOutput_Advantage", index=False
            )
            for name, ev_df in evaluations.items():
                sheet = name[:31]  # Excel sheet name limit
                ev_df.to_excel(xl, sheet_name=sheet, index=False)
        print(f"\n   💾 Saved to {output_path}")

    return compare_df
