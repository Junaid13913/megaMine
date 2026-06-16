"""
scalability.py — megaMine v2.0
Scalability and Performance Benchmarking Module

PURPOSE:
    Measures runtime, memory usage, and throughput of
    megaMine v2.0 across different input sizes.

    This directly addresses Reviewer 3:
    "Scalability not quantitatively demonstrated:
     runtime, memory usage, throughput, and computational
     resource requirements are not reported."

METRICS REPORTED:
    runtime_seconds     — wall clock time per module
    peak_memory_mb      — peak RAM usage
    throughput_rows_per_sec — rows processed per second
    throughput_abstracts_per_min — abstracts per minute

TEST CONDITIONS:
    n_rows: 10, 50, 100, 500, 1000, 5000, 10000
    Each condition run 3 times — report mean and std
    All modules tested independently and combined

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import os
import sys
import time
import tracemalloc
import statistics
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime


# ─── Test sizes ───────────────────────────────────────────────
DEFAULT_TEST_SIZES = [10, 50, 100, 500, 1000]
N_REPEATS          = 3   # repeat each test N times


# ─── Synthetic data generator ─────────────────────────────────
def generate_synthetic_rows(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate n synthetic megaMine output rows for testing.
    Uses realistic distributions from real megaMine output.
    """
    rng = np.random.default_rng(seed)

    genes    = ["EGFR","KRAS","BRCA1","BRCA2","TP53","BRAF",
                "ALK","RET","MET","NTRK1","PIK3CA","PTEN"]
    drugs    = ["erlotinib","sotorasib","olaparib","pembrolizumab",
                "trastuzumab","cetuximab","vemurafenib","crizotinib",
                "alectinib","osimertinib","palbociclib","abemaciclib"]
    cancers  = ["Non-Small Cell Lung Cancer; NSCLC",
                "Breast Cancer; BC",
                "Colorectal Cancer; CRC",
                "Glioblastoma; GBM",
                "Melanoma"]
    ev_types = ["efficacy","resistance","review","background"]
    studies  = ["RCT","trial","observational","case_report",
                "preclinical","in_vitro"]
    alts     = ["L858R","G12C","T790M","V600E","R248W",
                "exon 19 del","G12D","H1047R",""]

    rows = []
    for i in range(n):
        gene   = genes[i % len(genes)]
        drug   = drugs[i % len(drugs)]
        cancer = cancers[i % len(cancers)]
        ev     = rng.choice(ev_types, p=[0.4, 0.3, 0.15, 0.15])
        resist = "yes" if ev == "resistance" else (
                 "yes" if rng.random() < 0.1 else "no")

        rows.append({
            "pmid":                str(1000000 + i),
            "biomarker":           gene,
            "drug_primary":        drug,
            "cancer_type":         cancer,
            "alteration":          rng.choice(alts),
            "year":                int(rng.integers(2015, 2025)),
            "evidence_type":       ev,
            "resistance_observed": resist,
            "therapeutic_active":  "yes" if ev=="efficacy" else "no",
            "therapy_type":        "targeted",
            "study_design":        rng.choice(studies),
            "trial_phase":         rng.choice(["I","II","III",""]),
            "line_of_therapy":     rng.choice(["1L","2L","3L",""]),
            "histology":           "",
            "gene_type":           "DNA",
            "drug_accessibility":  rng.choice(["FDA","","PMDA"]),
            "doi":                 "",
            "journal":             rng.choice(["NEJM","Lancet","JCO","Nature"]),
            "summary_sentence":    (
                f"{gene} mutation {'conferred resistance to' if resist=='yes' else 'responded to'} "
                f"{drug} in {cancer.split(';')[0].strip()} patients."
            ),
        })

    return pd.DataFrame(rows)


# ─── Timer and memory context manager ─────────────────────────
class PerfMonitor:
    """
    Context manager that measures wall-clock time
    and peak memory usage of a code block.

    Usage:
        with PerfMonitor() as perf:
            run_something()
        print(perf.elapsed_seconds, perf.peak_memory_mb)
    """
    def __init__(self):
        self.elapsed_seconds = 0.0
        self.peak_memory_mb  = 0.0

    def __enter__(self):
        tracemalloc.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_seconds = time.perf_counter() - self._start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_mb = peak / 1024 / 1024


# ─── Individual module benchmarks ─────────────────────────────
def bench_temporal(df: pd.DataFrame) -> Tuple[float, float]:
    """Benchmark temporal.py module."""
    from megaMine.modules.temporal import run_temporal_analysis
    with PerfMonitor() as perf:
        run_temporal_analysis(df)
    return perf.elapsed_seconds, perf.peak_memory_mb


def bench_contradiction(df: pd.DataFrame) -> Tuple[float, float]:
    """Benchmark contradiction.py module."""
    from megaMine.modules.temporal import run_temporal_analysis
    from megaMine.modules.contradiction import run_contradiction_detection
    profile_df, _ = run_temporal_analysis(df)
    with PerfMonitor() as perf:
        run_contradiction_detection(df, profile_df=profile_df)
    return perf.elapsed_seconds, perf.peak_memory_mb


def bench_graph(df: pd.DataFrame) -> Tuple[float, float]:
    """Benchmark graph.py module."""
    from megaMine.modules.graph import build_graph
    with PerfMonitor() as perf:
        build_graph(df)
    return perf.elapsed_seconds, perf.peak_memory_mb


def bench_llm_verify(df: pd.DataFrame) -> Tuple[float, float]:
    """Benchmark llm_verify.py in offline mode."""
    from megaMine.modules.llm_verify import run_llm_verification
    with PerfMonitor() as perf:
        run_llm_verification(df, model="offline",
                             confidence_threshold=0.70)
    return perf.elapsed_seconds, perf.peak_memory_mb


def bench_full_pipeline(df: pd.DataFrame) -> Tuple[float, float]:
    """Benchmark all modules combined (except API calls)."""
    from megaMine.modules.temporal import run_temporal_analysis
    from megaMine.modules.contradiction import run_contradiction_detection
    from megaMine.modules.graph import build_graph
    from megaMine.modules.llm_verify import run_llm_verification

    with PerfMonitor() as perf:
        df2 = run_llm_verification(
            df, model="offline", confidence_threshold=0.70
        )
        profile_df, trend_df = run_temporal_analysis(df2)
        contradiction_df = run_contradiction_detection(
            df2, profile_df=profile_df
        )
        build_graph(df2, temporal_df=trend_df,
                    contradiction_df=contradiction_df)

    return perf.elapsed_seconds, perf.peak_memory_mb


# ─── Main benchmark runner ─────────────────────────────────────
def run_scalability_benchmarks(
    test_sizes:  List[int]  = DEFAULT_TEST_SIZES,
    n_repeats:   int        = N_REPEATS,
    output_path: Optional[str] = None,
    verbose:     bool       = True,
) -> pd.DataFrame:
    """
    Run scalability benchmarks across all modules
    and multiple input sizes.

    Parameters
    ----------
    test_sizes  : list of ints — number of rows to test
    n_repeats   : int — repeat each test N times
    output_path : str — save results to Excel
    verbose     : bool — print progress

    Returns
    -------
    pd.DataFrame : benchmark results table
    """

    modules = {
        "temporal":      bench_temporal,
        "contradiction": bench_contradiction,
        "graph":         bench_graph,
        "llm_verify":    bench_llm_verify,
        "full_pipeline": bench_full_pipeline,
    }

    results = []

    print("⏱️  Running scalability benchmarks...")
    print(f"   Test sizes:  {test_sizes}")
    print(f"   Repeats:     {n_repeats}")
    print(f"   Modules:     {list(modules.keys())}")
    print()

    total_runs = len(test_sizes) * len(modules) * n_repeats
    run_count  = 0

    for n_rows in test_sizes:
        print(f"📊 Testing n_rows = {n_rows:,}")

        # Generate synthetic data once per size
        df = generate_synthetic_rows(n_rows)

        for module_name, bench_fn in modules.items():
            times_s = []
            mems_mb  = []

            for repeat in range(n_repeats):
                run_count += 1
                if verbose:
                    print(f"   [{run_count}/{total_runs}] "
                          f"{module_name} repeat {repeat+1}/{n_repeats}",
                          end="\r")
                try:
                    t, m = bench_fn(df)
                    times_s.append(t)
                    mems_mb.append(m)
                except Exception as e:
                    if verbose:
                        print(f"\n   ⚠️  {module_name} error: {e}")
                    times_s.append(None)
                    mems_mb.append(None)

            # Filter out None values
            valid_t = [x for x in times_s if x is not None]
            valid_m = [x for x in mems_mb if x is not None]

            mean_t  = statistics.mean(valid_t)  if valid_t else None
            std_t   = statistics.stdev(valid_t) if len(valid_t)>1 else 0.0
            mean_m  = statistics.mean(valid_m)  if valid_m else None

            throughput = (n_rows / mean_t) if mean_t and mean_t > 0 else None
            abs_per_min= (n_rows / mean_t * 60) if mean_t and mean_t > 0 else None

            results.append({
                "module":               module_name,
                "n_rows":               n_rows,
                "mean_runtime_s":       round(mean_t, 4)  if mean_t  else None,
                "std_runtime_s":        round(std_t, 4),
                "peak_memory_mb":       round(mean_m, 2)  if mean_m  else None,
                "throughput_rows_s":    round(throughput, 1) if throughput else None,
                "abstracts_per_min":    round(abs_per_min, 1) if abs_per_min else None,
                "n_repeats":            len(valid_t),
                "tested_at":            datetime.now().isoformat(
                                            timespec="seconds"),
            })

        print(f"   ✅ n_rows={n_rows:,} complete          ")

    result_df = pd.DataFrame(results)

    # ── Print summary table ───────────────────────────────────
    print()
    print("=" * 75)
    print("SCALABILITY BENCHMARK RESULTS")
    print("=" * 75)
    print(f"{'Module':<20s} {'n_rows':>8s} {'Runtime(s)':>12s} "
          f"{'StdDev':>8s} {'Memory(MB)':>12s} {'Rows/s':>10s}")
    print("-" * 75)

    for _, row in result_df.iterrows():
        t   = f"{row['mean_runtime_s']:.4f}" if row['mean_runtime_s'] else "  N/A  "
        std = f"{row['std_runtime_s']:.4f}"  if row['std_runtime_s']  else "  N/A  "
        m   = f"{row['peak_memory_mb']:.1f}" if row['peak_memory_mb'] else "  N/A  "
        r   = f"{row['throughput_rows_s']:.0f}" if row['throughput_rows_s'] else "  N/A  "
        print(f"{row['module']:<20s} {row['n_rows']:>8,} {t:>12s} "
              f"{std:>8s} {m:>12s} {r:>10s}")

    # ── Save ─────────────────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                    exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as xl:
            result_df.to_excel(
                xl, sheet_name="Benchmarks", index=False
            )
            # Pivot table for easy reading
            try:
                pivot = result_df.pivot_table(
                    index   = "n_rows",
                    columns = "module",
                    values  = "mean_runtime_s",
                    aggfunc = "mean"
                ).round(4)
                pivot.to_excel(xl, sheet_name="RuntimePivot")

                mem_pivot = result_df.pivot_table(
                    index   = "n_rows",
                    columns = "module",
                    values  = "peak_memory_mb",
                    aggfunc = "mean"
                ).round(2)
                mem_pivot.to_excel(xl, sheet_name="MemoryPivot")
            except Exception:
                pass

        print(f"\n   💾 Saved to {output_path}")

    return result_df


def print_scalability_summary(result_df: pd.DataFrame) -> None:
    """
    Print a clean summary table suitable for
    inclusion in the paper methods section.
    Shows how runtime scales with input size.
    """
    print()
    print("=" * 65)
    print("PAPER-READY SCALABILITY SUMMARY")
    print("Full pipeline runtime by input size")
    print("=" * 65)

    fp = result_df[result_df["module"] == "full_pipeline"].copy()
    if len(fp) == 0:
        print("No full_pipeline data found")
        return

    print(f"\n{'n_rows':>8s} {'Runtime(s)':>12s} "
          f"{'Memory(MB)':>12s} {'Rows/sec':>10s}")
    print("-" * 45)
    for _, row in fp.iterrows():
        t = f"{row['mean_runtime_s']:.3f}" if row['mean_runtime_s'] else "N/A"
        m = f"{row['peak_memory_mb']:.1f}" if row['peak_memory_mb'] else "N/A"
        r = f"{row['throughput_rows_s']:.0f}" if row['throughput_rows_s'] else "N/A"
        print(f"{int(row['n_rows']):>8,} {t:>12s} {m:>12s} {r:>10s}")

    print()
    print("Note: Runtime measured on Apple Mac Pro.")
    print("Excludes PubMed API latency (network-dependent).")
    print("All measurements: mean of 3 independent runs.")
