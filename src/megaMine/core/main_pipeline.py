"""
main_pipeline.py — megaMine v2.0
Main Pipeline Orchestrator

PURPOSE:
    Wires all megaMine v2.0 modules together into a single
    end-to-end pipeline. Takes a PubMed query and produces
    a fully structured evidence report.

PIPELINE FLOW:
    1. Query PubMed via extractor.py
    2. Run LLM verification on extracted rows
    3. Compute temporal evidence tracking
    4. Detect contradictions
    5. Build knowledge graph
    6. Link ClinicalTrials.gov evidence
    7. Generate final Excel report

USAGE:
    python main_pipeline.py \\
        --q "EGFR AND erlotinib AND resistance AND NSCLC" \\
        --years 2018-2024 \\
        --max-records 500 \\
        --cancer-type NSCLC \\
        --out results/egfr_erlotinib \\
        --llm-verify \\
        --output-graph \\
        --link-trials

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime
from typing import Optional

# ─── Add src to path ──────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def run_pipeline(
    query:           str,
    years:           str,
    max_records:     int  = 500,
    cancer_type:     str  = "",
    out_prefix:      str  = "megamine_output",
    email:           str  = "noreply@example.com",
    ncbi_api_key:    Optional[str] = None,
    llm_verify:      bool = False,
    llm_model:       str  = "offline",
    llm_api_key:     Optional[str] = None,
    llm_threshold:   float = 0.70,
    output_graph:    bool = False,
    link_trials:     bool = False,
    dry_run_trials:  bool = True,
    year_binned:     bool = False,
    hgnc_cache:      Optional[str] = None,
    drug_whitelist:  Optional[str] = None,
) -> dict:
    """
    Run the complete megaMine v2.0 pipeline.

    Parameters
    ----------
    query        : PubMed query string
    years        : year range e.g. "2018-2024" or "2022"
    max_records  : maximum PubMed records to retrieve
    cancer_type  : cancer type label for filtering
    out_prefix   : output file path prefix
    email        : NCBI email
    ncbi_api_key : NCBI API key
    llm_verify   : run LLM verification layer
    llm_model    : "claude" / "gpt4mini" / "offline"
    llm_api_key  : API key for LLM
    llm_threshold: confidence threshold for LLM verification
    output_graph : export knowledge graph
    link_trials  : link ClinicalTrials.gov evidence
    dry_run_trials: use dummy trial data (no API call)
    year_binned  : use year-binned PubMed search
    hgnc_cache   : path to HGNC cache JSON
    drug_whitelist: path to drug whitelist file

    Returns
    -------
    dict : paths to all output files generated
    """

    start_time = time.time()
    outputs    = {}

    print("=" * 65)
    print("megaMine v2.0 — Full Pipeline")
    print("=" * 65)
    print(f"Query:       {query}")
    print(f"Years:       {years}")
    print(f"Max records: {max_records}")
    print(f"LLM verify:  {llm_verify} ({llm_model})")
    print(f"Graph:       {output_graph}")
    print(f"Trials:      {link_trials}")
    print(f"Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Create output directory ───────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Extract from PubMed ───────────────────────────
    print("\n📚 STEP 1: Extracting from PubMed...")
    print(f"   Module: extractor.py")

    try:
        # Build extractor arguments
        extractor_args = [
            "--q",            query,
            "--years",        years,
            "--max-records",  str(max_records),
            "--out",          out_prefix,
            "--email",        email,
        ]
        if ncbi_api_key:
            extractor_args += ["--ncbi-api-key", ncbi_api_key]
        if hgnc_cache:
            extractor_args += ["--hgnc-cache", hgnc_cache]
        if drug_whitelist:
            extractor_args += ["--drug-whitelist", drug_whitelist]
        if year_binned:
            extractor_args += ["--year-binned"]
        if cancer_type:
            extractor_args += ["--all-cancers"]

        # Run extractor
        from megaMine.core.extractor import main as extractor_main
        import unittest.mock as mock

        with mock.patch("sys.argv", ["extractor"] + extractor_args):
            extractor_main()

        # Load extracted output
        xlsx_path = f"{out_prefix}.xlsx"
        if os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path, sheet_name="Rows")
            print(f"   ✅ Extracted {len(df):,} rows from PubMed")
            outputs["extraction"] = xlsx_path
        else:
            print(f"   ⚠️  No output file found at {xlsx_path}")
            print(f"   Creating empty DataFrame for pipeline testing")
            df = pd.DataFrame()

    except Exception as e:
        print(f"   ⚠️  Extractor error: {e}")
        print(f"   Continuing with empty DataFrame")
        df = pd.DataFrame()

    if len(df) == 0:
        print("\n   No rows extracted — pipeline complete with no data")
        return outputs

    # ── Step 2: LLM Verification ──────────────────────────────
    if llm_verify:
        print(f"\n🤖 STEP 2: LLM Verification ({llm_model})...")
        print(f"   Module: llm_verify.py")
        try:
            from megaMine.modules.llm_verify import run_llm_verification
            df = run_llm_verification(
                df,
                model              = llm_model,
                api_key            = llm_api_key,
                confidence_threshold = llm_threshold,
            )
            verified = (df.get("llm_verified", pd.Series()) == "yes").sum()
            print(f"   ✅ LLM verified: {verified:,} / {len(df):,} rows")
        except Exception as e:
            print(f"   ⚠️  LLM verification error: {e}")
    else:
        print(f"\n⏭️  STEP 2: LLM Verification skipped (--llm-verify not set)")

    # ── Step 3: Temporal Analysis ─────────────────────────────
    print(f"\n🕐 STEP 3: Temporal Evidence Tracking...")
    print(f"   Module: temporal.py")
    profile_df = None
    trend_df   = None
    try:
        from megaMine.modules.temporal import run_temporal_analysis
        profile_df, trend_df = run_temporal_analysis(df)
        print(f"   ✅ Temporal profiles: {len(profile_df):,} year-bin rows")
        print(f"   ✅ Trend classifications: {len(trend_df):,} triplets")
    except Exception as e:
        print(f"   ⚠️  Temporal analysis error: {e}")

    # ── Step 4: Contradiction Detection ──────────────────────
    print(f"\n🔍 STEP 4: Contradiction Detection...")
    print(f"   Module: contradiction.py")
    contradiction_df = None
    try:
        from megaMine.modules.contradiction import run_contradiction_detection
        contradiction_df = run_contradiction_detection(
            df, profile_df=profile_df
        )
        flagged = (
            contradiction_df["contradiction_flag"]
            .isin(["watch","caution","conflict"])
        ).sum()
        print(f"   ✅ {flagged} triplets flagged for contradiction")
    except Exception as e:
        print(f"   ⚠️  Contradiction detection error: {e}")

    # ── Step 5: Knowledge Graph ───────────────────────────────
    if output_graph:
        print(f"\n🕸️  STEP 5: Building Knowledge Graph...")
        print(f"   Module: graph.py")
        try:
            from megaMine.modules.graph import build_graph, export_graph
            G = build_graph(
                df,
                temporal_df     = trend_df,
                contradiction_df= contradiction_df,
            )
            graph_prefix = f"{out_prefix}_graph"
            graph_outputs = export_graph(
                G,
                output_prefix = graph_prefix,
                formats       = ["graphml", "csv"],
            )
            outputs.update(graph_outputs)
            print(f"   ✅ Graph: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges")
        except Exception as e:
            print(f"   ⚠️  Graph build error: {e}")
    else:
        print(f"\n⏭️  STEP 5: Knowledge Graph skipped (--output-graph not set)")

    # ── Step 6: ClinicalTrials Linkage ────────────────────────
    trials_df = None
    if link_trials:
        print(f"\n🏥 STEP 6: ClinicalTrials.gov Linkage...")
        print(f"   Module: trials.py")
        try:
            from megaMine.modules.trials import run_trials_linkage
            trials_df = run_trials_linkage(
                df,
                contradiction_df = contradiction_df,
                dry_run          = dry_run_trials,
            )
            failed = (trials_df["has_failed_trial"] == "yes").sum()
            print(f"   ✅ Linked {len(trials_df)} drug-cancer pairs")
            print(f"   ⚠️  {failed} pairs have negative clinical "
                  f"development signals")
        except Exception as e:
            print(f"   ⚠️  Trials linkage error: {e}")
    else:
        print(f"\n⏭️  STEP 6: ClinicalTrials skipped (--link-trials not set)")

    # ── Step 7: Generate Final Report ────────────────────────
    print(f"\n📊 STEP 7: Generating Final Report...")

    report_path = f"{out_prefix}_report.xlsx"
    try:
        with pd.ExcelWriter(report_path, engine="openpyxl") as xl:

            # Sheet 1: Main extraction results
            df.to_excel(xl, sheet_name="Rows", index=False)

            # Sheet 2: Temporal profiles
            if profile_df is not None and len(profile_df) > 0:
                profile_df.to_excel(
                    xl, sheet_name="TemporalProfiles", index=False
                )

            # Sheet 3: Trend classifications
            if trend_df is not None and len(trend_df) > 0:
                trend_df.to_excel(
                    xl, sheet_name="TrendSummary", index=False
                )

            # Sheet 4: Contradiction flags
            if contradiction_df is not None and len(contradiction_df) > 0:
                contradiction_df.to_excel(
                    xl, sheet_name="ContradictionFlags", index=False
                )

            # Sheet 5: Clinical trials
            if trials_df is not None and len(trials_df) > 0:
                trials_df.to_excel(
                    xl, sheet_name="ClinicalTrials", index=False
                )

            # Sheet 6: Run metadata
            elapsed = round(time.time() - start_time, 2)
            meta = pd.DataFrame([{
                "query":           query,
                "years":           years,
                "max_records":     max_records,
                "cancer_type":     cancer_type,
                "n_rows":          len(df),
                "n_triplets":      len(trend_df) if trend_df is not None else 0,
                "n_contradictions":len(contradiction_df) if contradiction_df is not None else 0,
                "n_trial_pairs":   len(trials_df) if trials_df is not None else 0,
                "llm_verify":      llm_verify,
                "llm_model":       llm_model if llm_verify else "N/A",
                "output_graph":    output_graph,
                "link_trials":     link_trials,
                "runtime_seconds": elapsed,
                "generated_at":    datetime.now().isoformat(
                    timespec="seconds"
                ),
                "version":         "megaMine_v2.0.0",
                "schema":          "normalized_v2",
            }])
            meta.to_excel(xl, sheet_name="RunInfo", index=False)

        outputs["report"] = report_path
        print(f"   ✅ Report saved: {report_path}")

    except Exception as e:
        print(f"   ⚠️  Report generation error: {e}")

    # ── Summary ───────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 2)
    print(f"\n{'='*65}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*65}")
    print(f"  Runtime:      {elapsed}s")
    print(f"  Rows:         {len(df):,}")
    if trend_df is not None:
        print(f"  Triplets:     {len(trend_df):,}")
    if contradiction_df is not None:
        flagged = (contradiction_df["contradiction_flag"]
                   .isin(["watch","caution","conflict"])).sum()
        print(f"  Contradictions flagged: {flagged}")
    print(f"  Output files:")
    for k, v in outputs.items():
        print(f"    {k}: {v}")
    print(f"{'='*65}")

    return outputs


# ─── CLI entry point ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="megaMine v2.0 — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python main_pipeline.py \\
      --q "EGFR AND erlotinib AND NSCLC" \\
      --years 2018-2024 --max-records 200 \\
      --out results/egfr_erlotinib

  # Full run with all modules
  python main_pipeline.py \\
      --q "KRAS AND sotorasib AND colorectal cancer" \\
      --years 2020-2024 --max-records 500 \\
      --llm-verify --llm-model offline \\
      --output-graph --link-trials \\
      --out results/kras_sotorasib
        """
    )

    # Required
    ap.add_argument("--q",           required=True,
                    help="PubMed query string")
    ap.add_argument("--years",       required=True,
                    help="Year range: YYYY or YYYY-YYYY")
    ap.add_argument("--out",         required=True,
                    help="Output prefix (no extension)")

    # Optional
    ap.add_argument("--max-records", type=int, default=500)
    ap.add_argument("--cancer-type", default="",
                    help="Cancer type label for context")
    ap.add_argument("--email",       default="noreply@example.com")
    ap.add_argument("--ncbi-api-key",default=None)
    ap.add_argument("--hgnc-cache",  default=None)
    ap.add_argument("--drug-whitelist", default=None)
    ap.add_argument("--year-binned", action="store_true")

    # LLM verification
    ap.add_argument("--llm-verify",  action="store_true",
                    help="Enable LLM verification layer")
    ap.add_argument("--llm-model",   default="offline",
                    choices=["claude","gpt4mini","offline"])
    ap.add_argument("--llm-api-key", default=None)
    ap.add_argument("--llm-threshold", type=float, default=0.70)

    # New modules
    ap.add_argument("--output-graph", action="store_true",
                    help="Export knowledge graph")
    ap.add_argument("--link-trials",  action="store_true",
                    help="Link ClinicalTrials.gov evidence")
    ap.add_argument("--dry-run-trials", action="store_true",
                    default=True,
                    help="Use dummy trial data (no API call)")

    args = ap.parse_args()

    run_pipeline(
        query          = args.q,
        years          = args.years,
        max_records    = args.max_records,
        cancer_type    = args.cancer_type,
        out_prefix     = args.out,
        email          = args.email,
        ncbi_api_key   = args.ncbi_api_key,
        llm_verify     = args.llm_verify,
        llm_model      = args.llm_model,
        llm_api_key    = args.llm_api_key,
        llm_threshold  = args.llm_threshold,
        output_graph   = args.output_graph,
        link_trials    = args.link_trials,
        dry_run_trials = args.dry_run_trials,
        year_binned    = args.year_binned,
        hgnc_cache     = args.hgnc_cache,
        drug_whitelist = args.drug_whitelist,
    )


if __name__ == "__main__":
    main()
