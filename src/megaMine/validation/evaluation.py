"""
evaluation.py — megaMine v2.0
Evaluation Metrics Module

PURPOSE:
    Computes Precision, Recall, F1 scores for megaMine
    extraction against the manually curated gold standard.

    This directly addresses all three reviewers who asked for:
    "True Precision / Recall / F1 against human-annotated text"

METRICS COMPUTED:
    Entity-level:
        gene_F1         — gene extraction accuracy
        drug_F1         — drug extraction accuracy
        cancer_F1       — cancer type extraction accuracy
        alteration_F1   — mutation extraction accuracy

    Relation-level:
        triplet_F1      — full gene+drug+cancer triplet accuracy
        relation_F1     — binary relation detection (yes/no)

    Evidence classification:
        evidence_type_F1    — macro F1 across evidence types
        resistance_F1       — resistance detection F1
        negation_F1         — negation detection F1
        speculation_F1      — speculation detection F1
        study_design_F1     — study design classification F1

    Overall:
        weighted_F1     — weighted average across all metrics

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import pandas as pd
import numpy as np
# numpy required for np.where in relation_pred derivation
from typing import Optional, List, Dict, Tuple
from collections import defaultdict


# ─── Column selection helper ─────────────────────────────────
def _get_col(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """
    Safely retrieve a column from a merged DataFrame.

    After pd.merge() with suffixes, column names can become:
        negated_gold → negated_gold_gold  (if both sides had it)
        llm_negated  → llm_negated_pred   (if only pred side had it)

    This helper tries multiple candidate names in order
    and returns the first one found. Returns empty Series
    if none found — prevents silent column mismatch errors.

    Parameters
    ----------
    df         : pd.DataFrame — merged evaluation DataFrame
    candidates : list of str  — column names to try in order

    Returns
    -------
    pd.Series : normalized lowercase stripped string values
    """
    for c in candidates:
        if c in df.columns:
            return (
                df[c].fillna("").astype(str)
                .str.strip().str.lower()
            )
    return pd.Series([""] * len(df))


# ─── Normalization helpers ────────────────────────────────────
def _norm_gene(s: str) -> str:
    """Normalize gene symbol for comparison."""
    return str(s or "").strip().upper()


def _norm_drug(s: str) -> str:
    """Normalize drug name for comparison."""
    return str(s or "").strip().lower().replace("-", "").replace(" ", "")


def _norm_cancer(s: str) -> str:
    """Normalize cancer type — use first token before semicolon."""
    s = str(s or "").strip().lower()
    return s.split(";")[0].strip().replace(" ", "_")


def _norm_binary(s: str) -> str:
    """Normalize yes/no fields."""
    return str(s or "").strip().lower()


def _norm_evtype(s: str) -> str:
    """Normalize evidence type."""
    return str(s or "").strip().lower()


# ─── Core metric computation ──────────────────────────────────
def precision_recall_f1(
    tp: int, fp: int, fn: int
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, F1 from counts.

    Returns
    -------
    tuple: (precision, recall, f1)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def entity_f1(
    gold_series: pd.Series,
    pred_series: pd.Series,
    norm_fn=None,
) -> Dict:
    """
    Compute entity-level Precision, Recall, F1.

    Treats each row independently:
        TP: both gold and pred are non-empty and match
        FP: pred is non-empty but does not match gold
        FN: gold is non-empty but pred is empty or wrong

    Parameters
    ----------
    gold_series : pd.Series — gold standard values
    pred_series : pd.Series — megaMine predicted values
    norm_fn     : function  — normalization function

    Returns
    -------
    dict with tp, fp, fn, precision, recall, f1
    """
    norm_fn = norm_fn or (lambda x: str(x or "").strip().lower())
    tp = fp = fn = 0

    for gold, pred in zip(gold_series, pred_series):
        gold_norm = norm_fn(gold)
        pred_norm = norm_fn(pred)

        gold_empty = gold_norm == ""
        pred_empty = pred_norm == ""

        if not gold_empty and not pred_empty:
            if gold_norm == pred_norm:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif not gold_empty and pred_empty:
            fn += 1
        elif gold_empty and not pred_empty:
            fp += 1
        # both empty = true negative, not counted

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def binary_f1(
    gold_series: pd.Series,
    pred_series: pd.Series,
    positive_label: str = "yes",
) -> Dict:
    """
    Compute binary classification F1.
    Used for: resistance_observed, negated,
              speculative, relation detection.

    Parameters
    ----------
    gold_series     : pd.Series — gold labels
    pred_series     : pd.Series — predicted labels
    positive_label  : str       — what counts as positive

    Returns
    -------
    dict with tp, fp, fn, tn, precision, recall, f1, accuracy
    """
    tp = fp = fn = tn = 0

    for gold, pred in zip(gold_series, pred_series):
        g = _norm_binary(gold) == positive_label
        p = _norm_binary(pred) == positive_label

        if g and p:     tp += 1
        elif not g and p: fp += 1
        elif g and not p: fn += 1
        else:             tn += 1

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    total    = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  round(accuracy, 4),
    }


def macro_f1_multiclass(
    gold_series: pd.Series,
    pred_series: pd.Series,
) -> Dict:
    """
    Compute macro-averaged F1 for multi-class classification.
    Used for: evidence_type, study_design.

    Macro F1 = unweighted average of per-class F1 scores.
    This treats all classes equally regardless of frequency.

    Parameters
    ----------
    gold_series : pd.Series — gold labels
    pred_series : pd.Series — predicted labels

    Returns
    -------
    dict with per_class_f1, macro_f1, weighted_f1
    """
    classes = set(
        list(gold_series.dropna().unique()) +
        list(pred_series.dropna().unique())
    )
    classes = {c for c in classes if str(c).strip()}

    per_class = {}
    for cls in sorted(classes):
        gold_bin = gold_series.apply(
            lambda x: _norm_evtype(x) == _norm_evtype(cls)
        )
        pred_bin = pred_series.apply(
            lambda x: _norm_evtype(x) == _norm_evtype(cls)
        )
        tp = int((gold_bin & pred_bin).sum())
        fp = int((~gold_bin & pred_bin).sum())
        fn = int((gold_bin & ~pred_bin).sum())
        _, _, f1 = precision_recall_f1(tp, fp, fn)
        support  = int(gold_bin.sum())
        per_class[cls] = {"f1": f1, "support": support}

    # Macro F1 — unweighted average
    f1_scores = [v["f1"] for v in per_class.values()]
    macro     = round(np.mean(f1_scores), 4) if f1_scores else 0.0

    # Weighted F1 — weighted by support
    total_support = sum(v["support"] for v in per_class.values())
    if total_support > 0:
        weighted = round(
            sum(v["f1"] * v["support"] for v in per_class.values())
            / total_support, 4
        )
    else:
        weighted = 0.0

    return {
        "per_class_f1": per_class,
        "macro_f1":     macro,
        "weighted_f1":  weighted,
        "n_classes":    len(classes),
    }


def triplet_f1(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    gene_col_g:   str = "gene_gold",
    drug_col_g:   str = "drug_gold",
    cancer_col_g: str = "cancer_gold",
    gene_col_p:   str = "biomarker",
    drug_col_p:   str = "drug_primary",
    cancer_col_p: str = "cancer_type",
) -> Dict:
    """
    Compute full triplet extraction F1.
    A triplet (gene, drug, cancer) is correct only if
    ALL THREE entities match the gold standard.

    This is the strictest and most meaningful metric —
    a triplet is only useful clinically if all three
    components are correctly extracted.

    Parameters
    ----------
    gold_df : pd.DataFrame — gold standard annotations
    pred_df : pd.DataFrame — megaMine predictions

    Returns
    -------
    dict with tp, fp, fn, precision, recall, f1
    """
    # Build sets of normalized triplets
    def make_triplet_set(df, gc, dc, cc):
        """
        Strict triplet: ALL THREE of gene + drug + cancer required.
        A triplet missing any component is not counted.
        This is the clinically meaningful unit — a drug
        recommendation requires knowing gene, drug AND cancer.
        """
        triplets = set()
        for _, row in df.iterrows():
            g = _norm_gene(row.get(gc, ""))
            d = _norm_drug(row.get(dc, ""))
            c = _norm_cancer(row.get(cc, ""))
            # Strict: all three must be non-empty
            if g and d and c:
                triplets.add((g, d, c))
        return triplets

    gold_triplets = make_triplet_set(
        gold_df[gold_df.get("relation_gold", pd.Series("yes")) == "yes"]
        if "relation_gold" in gold_df.columns else gold_df,
        gene_col_g, drug_col_g, cancer_col_g
    )
    pred_triplets = make_triplet_set(
        pred_df, gene_col_p, drug_col_p, cancer_col_p
    )

    tp = len(gold_triplets & pred_triplets)
    fp = len(pred_triplets - gold_triplets)
    fn = len(gold_triplets - pred_triplets)

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "gold_triplets":  len(gold_triplets),
        "pred_triplets":  len(pred_triplets),
    }


# ─── Main evaluation function ─────────────────────────────────
def evaluate(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_path: Optional[str] = None,
    system_name: str = "megaMine_v2",
) -> pd.DataFrame:
    """
    Run full evaluation of megaMine against gold standard.

    Computes all metrics and returns a summary DataFrame.

    Parameters
    ----------
    gold_df : pd.DataFrame
        Gold standard annotations from gold_standard.py
    pred_df : pd.DataFrame
        megaMine extraction output
    output_path : str, optional
        Save results to Excel if provided
    system_name : str
        Label for the system being evaluated

    Returns
    -------
    pd.DataFrame : metric summary table
    """
    print(f"📐 Running evaluation for: {system_name}")
    print(f"   Gold standard rows: {len(gold_df)}")
    print(f"   Predictions rows:   {len(pred_df)}")

    # ── Derive relation_pred from extracted entities ───────────
    # Using therapeutic_active is scientifically wrong because
    # resistance relations have therapeutic_active=no but are
    # still real gene-drug-cancer relationships.
    #
    # Correct definition:
    #   relation = yes if gene + drug + cancer all present
    #              AND evidence_type is not background
    pred_df = pred_df.copy()
    pred_df["relation_pred"] = np.where(
        (pred_df.get("biomarker", pd.Series()).astype(str).str.strip() != "") &
        (pred_df.get("drug_primary", pd.Series()).astype(str).str.strip() != "") &
        (pred_df.get("cancer_type", pd.Series()).astype(str).str.strip() != "") &
        (~pred_df.get("evidence_type", pd.Series())
         .astype(str).str.lower().isin(["background", ""])),
        "yes", "no"
    )

    # ── Row alignment strategy ──────────────────────────────────
    # Priority 1: annotation_id — stable unique identifier
    #             assigned during gold standard creation
    # Priority 2: PMID + full normalized sentence
    #             (not truncated — truncation loses uniqueness)
    # Priority 3: full normalized sentence only
    #             (fallback when PMID not available)
    #
    # We normalize sentences: lowercase + strip whitespace
    # to avoid alignment failures from minor formatting diffs
    gold_df = gold_df.copy()
    pred_df = pred_df.copy()

    def _norm_sentence(s):
        return str(s or "").strip().lower()

    if "annotation_id" in gold_df.columns and        "annotation_id" in pred_df.columns:
        # Best case — stable annotation ID exists
        merge_key = "annotation_id"
        merged = pd.merge(
            gold_df, pred_df,
            on=merge_key,
            suffixes=("_gold", "_pred"),
            how="left"
        )
    elif "pmid" in gold_df.columns and "pmid" in pred_df.columns:
        # Use PMID + full normalized sentence (not truncated)
        gold_sent_col = ("sentence" if "sentence" in gold_df.columns
                         else "summary_sentence")
        pred_sent_col = ("summary_sentence" if "summary_sentence"
                         in pred_df.columns else "sentence")

        gold_df["_key"] = (
            gold_df["pmid"].astype(str) + "||" +
            gold_df[gold_sent_col].apply(_norm_sentence)
        )
        pred_df["_key"] = (
            pred_df["pmid"].astype(str) + "||" +
            pred_df[pred_sent_col].apply(_norm_sentence)
        )
        merged = pd.merge(
            gold_df, pred_df,
            on="_key",
            suffixes=("_gold", "_pred"),
            how="left"
        )
    else:
        # Fallback — normalized sentence only
        gold_sent_col = ("sentence" if "sentence" in gold_df.columns
                         else "summary_sentence")
        pred_sent_col = ("summary_sentence" if "summary_sentence"
                         in pred_df.columns else "sentence")

        gold_df["_key"] = gold_df[gold_sent_col].apply(_norm_sentence)
        pred_df["_key"] = pred_df[pred_sent_col].apply(_norm_sentence)
        merged = pd.merge(
            gold_df, pred_df,
            on="_key",
            suffixes=("_gold", "_pred"),
            how="left"
        )

    results = []

    # ── 1. Entity-level metrics ──────────────────────────────
    print("   Computing entity metrics...")

    # gene_gold unique to gold side, biomarker unique to pred side
    gene_metrics = entity_f1(
        _get_col(merged, ["gene_gold"]),
        _get_col(merged, ["biomarker"]),
        norm_fn=_norm_gene
    )
    results.append({
        "metric":    "gene_extraction",
        "system":    system_name,
        "precision": gene_metrics["precision"],
        "recall":    gene_metrics["recall"],
        "f1":        gene_metrics["f1"],
        "tp":        gene_metrics["tp"],
        "fp":        gene_metrics["fp"],
        "fn":        gene_metrics["fn"],
        "notes":     "HGNC-validated gene symbol extraction",
    })

    drug_metrics = entity_f1(
        _get_col(merged, ["drug_gold"]),
        _get_col(merged, ["drug_primary"]),
        norm_fn=_norm_drug
    )
    results.append({
        "metric":    "drug_extraction",
        "system":    system_name,
        "precision": drug_metrics["precision"],
        "recall":    drug_metrics["recall"],
        "f1":        drug_metrics["f1"],
        "tp":        drug_metrics["tp"],
        "fp":        drug_metrics["fp"],
        "fn":        drug_metrics["fn"],
        "notes":     "Drug name extraction from whitelist",
    })

    cancer_metrics = entity_f1(
        _get_col(merged, ["cancer_gold"]),
        _get_col(merged, ["cancer_type"]),
        norm_fn=_norm_cancer
    )
    results.append({
        "metric":    "cancer_extraction",
        "system":    system_name,
        "precision": cancer_metrics["precision"],
        "recall":    cancer_metrics["recall"],
        "f1":        cancer_metrics["f1"],
        "tp":        cancer_metrics["tp"],
        "fp":        cancer_metrics["fp"],
        "fn":        cancer_metrics["fn"],
        "notes":     "Cancer type normalization",
    })

    # ── 2. Relation detection ────────────────────────────────
    print("   Computing relation metrics...")

    # Use derived relation_pred — not therapeutic_active
    # Resistance relations are real relations too
    # relation_gold unique to gold side, relation_pred derived on pred side
    rel_gold = _get_col(merged, ["relation_gold"])
    rel_pred = _get_col(merged, ["relation_pred"])
    rel_metrics = binary_f1(rel_gold, rel_pred, positive_label="yes")
    results.append({
        "metric":    "relation_detection",
        "system":    system_name,
        "precision": rel_metrics["precision"],
        "recall":    rel_metrics["recall"],
        "f1":        rel_metrics["f1"],
        "tp":        rel_metrics["tp"],
        "fp":        rel_metrics["fp"],
        "fn":        rel_metrics["fn"],
        "notes":     f"Accuracy={rel_metrics['accuracy']}",
    })

    # ── 3. Triplet F1 ────────────────────────────────────────
    print("   Computing triplet metrics...")

    trip_metrics = triplet_f1(gold_df, pred_df)
    results.append({
        "metric":    "triplet_extraction",
        "system":    system_name,
        "precision": trip_metrics["precision"],
        "recall":    trip_metrics["recall"],
        "f1":        trip_metrics["f1"],
        "tp":        trip_metrics["tp"],
        "fp":        trip_metrics["fp"],
        "fn":        trip_metrics["fn"],
        "notes":     (f"Gold={trip_metrics['gold_triplets']} "
                      f"Pred={trip_metrics['pred_triplets']}"),
    })

    # ── 4. Evidence type classification ──────────────────────
    print("   Computing evidence type metrics...")

    # evidence_type_gold unique to gold, evidence_type unique to pred
    ev_gold = _get_col(merged, ["evidence_type_gold"])
    ev_pred = _get_col(merged, ["evidence_type"])
    ev_metrics = macro_f1_multiclass(ev_gold, ev_pred)
    results.append({
        "metric":    "evidence_type_macro_F1",
        "system":    system_name,
        "precision": None,
        "recall":    None,
        "f1":        ev_metrics["macro_f1"],
        "tp":        None,
        "fp":        None,
        "fn":        None,
        "notes":     (f"weighted_F1={ev_metrics['weighted_f1']} "
                      f"classes={ev_metrics['n_classes']}"),
    })

    # ── 5. Resistance detection ──────────────────────────────
    print("   Computing resistance detection metrics...")

    # resistance_observed_gold unique to gold side — no suffix
    # resistance_observed unique to pred side — no suffix
    res_gold = _get_col(merged, ["resistance_observed_gold"])
    res_pred = _get_col(merged, ["resistance_observed"])
    res_metrics = binary_f1(res_gold, res_pred, positive_label="yes")
    results.append({
        "metric":    "resistance_detection",
        "system":    system_name,
        "precision": res_metrics["precision"],
        "recall":    res_metrics["recall"],
        "f1":        res_metrics["f1"],
        "tp":        res_metrics["tp"],
        "fp":        res_metrics["fp"],
        "fn":        res_metrics["fn"],
        "notes":     f"Accuracy={res_metrics['accuracy']}",
    })

    # ── 6. Negation detection ────────────────────────────────
    print("   Computing negation detection metrics...")

    # After merge: negated_gold stays as negated_gold (not renamed)
    # llm_negated stays as llm_negated (unique to pred side)
    # pmid splits into pmid_gold and pmid_pred after merge
    neg_gold = _get_col(merged, ["negated_gold"])
    neg_pred = _get_col(merged, ["llm_negated"])
    if (neg_gold != "").sum() > 0 and (neg_pred != "").sum() > 0:
        neg_metrics = binary_f1(neg_gold, neg_pred, positive_label="yes")
        results.append({
            "metric":    "negation_detection",
            "system":    system_name,
            "precision": neg_metrics["precision"],
            "recall":    neg_metrics["recall"],
            "f1":        neg_metrics["f1"],
            "tp":        neg_metrics["tp"],
            "fp":        neg_metrics["fp"],
            "fn":        neg_metrics["fn"],
            "notes":     f"Accuracy={neg_metrics['accuracy']}",
        })

    # ── 7. Speculation detection ─────────────────────────────
    print("   Computing speculation detection metrics...")

    # speculative_gold and llm_speculative both unique — no suffix added
    spec_gold = _get_col(merged, ["speculative_gold"])
    spec_pred = _get_col(merged, ["llm_speculative"])
    if (spec_gold != "").sum() > 0 and (spec_pred != "").sum() > 0:
        spec_metrics = binary_f1(
            spec_gold, spec_pred, positive_label="yes"
        )
        results.append({
            "metric":    "speculation_detection",
            "system":    system_name,
            "precision": spec_metrics["precision"],
            "recall":    spec_metrics["recall"],
            "f1":        spec_metrics["f1"],
            "tp":        spec_metrics["tp"],
            "fp":        spec_metrics["fp"],
            "fn":        spec_metrics["fn"],
            "notes":     f"Accuracy={spec_metrics['accuracy']}",
        })

    # ── 8. Study design classification ───────────────────────
    print("   Computing study design metrics...")

    # study_design_gold unique to gold, study_design unique to pred
    sd_gold = _get_col(merged, ["study_design_gold"])
    sd_pred = _get_col(merged, ["study_design"])
    if (sd_gold != "").sum() > 0:
        sd_metrics = macro_f1_multiclass(sd_gold, sd_pred)
        results.append({
            "metric":    "study_design_macro_F1",
            "system":    system_name,
            "precision": None,
            "recall":    None,
            "f1":        sd_metrics["macro_f1"],
            "tp":        None,
            "fp":        None,
            "fn":        None,
            "notes":     f"weighted_F1={sd_metrics['weighted_f1']}",
        })

    # ── 9. Overall weighted F1 ───────────────────────────────
    f1_scores = [r["f1"] for r in results if r["f1"] is not None]
    overall_f1 = round(np.mean(f1_scores), 4) if f1_scores else 0.0
    # Named mean_F1 not weighted_F1 because this is a simple
    # unweighted mean across metric F1 scores — not weighted
    # by support or sample count. A true weighted F1 would
    # require weighting by number of gold instances per metric.
    results.append({
        "metric":    "OVERALL_mean_F1",
        "system":    system_name,
        "precision": None,
        "recall":    None,
        "f1":        overall_f1,
        "tp":        None,
        "fp":        None,
        "fn":        None,
        "notes":     (f"Unweighted mean of {len(f1_scores)} "
                      f"metric F1 scores. Not weighted by support."),
    })

    result_df = pd.DataFrame(results)

    # ── Print summary table ──────────────────────────────────
    print(f"\n{'='*65}")
    print(f"EVALUATION RESULTS — {system_name}")
    print(f"{'='*65}")
    print(f"{'Metric':<30s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print(f"{'-'*65}")
    for _, row in result_df.iterrows():
        p = f"{row['precision']:.4f}" if row['precision'] is not None else "   N/A  "
        r = f"{row['recall']:.4f}"    if row['recall']    is not None else "   N/A  "
        f = f"{row['f1']:.4f}"        if row['f1']        is not None else "   N/A  "
        bold = "★ " if row["metric"].startswith("OVERALL_mean") else "  "
        print(f"{bold}{row['metric']:<28s} {p:>10s} {r:>10s} {f:>10s}")

    # ── Save if path provided ────────────────────────────────
    if output_path:
        with pd.ExcelWriter(output_path, engine="openpyxl") as xl:
            result_df.to_excel(
                xl, sheet_name="EvaluationSummary", index=False
            )
            # Also save per-class evidence type breakdown
            if "per_class_f1" in ev_metrics:
                pc_rows = [
                    {"class": cls,
                     "f1": vals["f1"],
                     "support": vals["support"]}
                    for cls, vals in ev_metrics["per_class_f1"].items()
                ]
                pd.DataFrame(pc_rows).to_excel(
                    xl, sheet_name="EvidenceTypeBreakdown", index=False
                )
        print(f"\n   💾 Saved to {output_path}")

    return result_df


def compare_systems(
    evaluations: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare multiple systems side by side.
    Used for megaMine vs PubTator3 vs BERT baseline comparison.

    Parameters
    ----------
    evaluations : dict
        {system_name: evaluation_df from evaluate()}
    output_path : str, optional
        Save comparison table to Excel

    Returns
    -------
    pd.DataFrame : wide-format comparison table
    """
    all_metrics = set()
    for df in evaluations.values():
        all_metrics.update(df["metric"].tolist())

    rows = []
    for metric in sorted(all_metrics):
        row = {"metric": metric}
        for system, df in evaluations.items():
            match = df[df["metric"] == metric]
            if len(match) > 0:
                row[f"{system}_F1"] = match.iloc[0]["f1"]
                row[f"{system}_P"]  = match.iloc[0]["precision"]
                row[f"{system}_R"]  = match.iloc[0]["recall"]
            else:
                row[f"{system}_F1"] = None
                row[f"{system}_P"]  = None
                row[f"{system}_R"]  = None
        rows.append(row)

    compare_df = pd.DataFrame(rows)

    print(f"\n{'='*70}")
    print("SYSTEM COMPARISON")
    print(f"{'='*70}")
    systems = list(evaluations.keys())
    header  = f"{'Metric':<30s}"
    for sys in systems:
        header += f" {sys:>12s}_F1"
    print(header)
    print("-" * 70)

    for _, row in compare_df.iterrows():
        line = f"{row['metric']:<30s}"
        for sys in systems:
            val = row.get(f"{sys}_F1")
            line += f" {val:>12.4f}" if val is not None else f" {'N/A':>12s}"
        print(line)

    if output_path:
        compare_df.to_excel(output_path, index=False)
        print(f"\n   💾 Comparison saved to {output_path}")

    return compare_df
