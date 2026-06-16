# megaMine v2.0 — Complete Methods Notes

## Overview
megaMine v2.0 is a hybrid literature mining framework combining
rule-based extraction with LLM verification, temporal evidence
tracking, contradiction detection, and knowledge graph export.
Target journal: Briefings in Bioinformatics

---

## Module 1 — Core Extraction Engine (extractor.py)
### What it does
- Queries PubMed via Entrez API
- Extracts gene-drug-cancer triplets from abstracts
- Uses HGNC-validated gene recognition
- Extracts clinical context: TMB, MSI, study_design, trial_phase
- Two modes: therapy and driver

### Bug fixes needed FIRST before new features
1. year-binned cap
   - File: extractor.py
   - Function: esearch_year_with_pagination
   - Problem: per_year_max not enforced — broad queries overfetch
   - Fix: add hard slice after each year fetch

2. Error suppression
   - File: extractor.py
   - Function: ensure_parent_dir_for
   - Problem: bare except pass silently swallows all errors
   - Fix: replace with specific exception handling and logging

3. API return value mismatch
   - File: extractor.py
   - Function: main()
   - Problem: always returns .xlsx path even when .csv.gz written
   - Fix: return actual output path based on which file was written

4. Cancer normalization capturing paper titles
   - File: extractor.py
   - Function: extract_cancers_from_all_sources
   - Problem: article titles being extracted as cancer type names
   - Fix: add length filter + title pattern detection

### Improvements needed
- Add --list-cancers flag to argparse
- Rename conclusion column to auto_conclusion
- Fix rate limiting: increase SLEEP_BETWEEN from 0.02 to 0.12
- Reduce MAX_WORKERS from 8 to 4 for safer API usage
- Expand DEFAULT_ONCO_DRUGS from 50 to 500+ drugs

---

## Module 2 — LLM Verification Layer (modules/llm_verify.py)
### What it does
Takes rule-extracted candidate triplets and passes each one
through an LLM to confirm the relationship is real.

### Design
Input:
  - sentence (the extracted evidence sentence)
  - gene (e.g. EGFR)
  - drug (e.g. erlotinib)
  - cancer (e.g. NSCLC)
  - evidence_type (e.g. efficacy)

Prompt template:
  "You are a biomedical expert. Read this sentence carefully:
   [{sentence}]
   Does this sentence provide direct evidence that [{drug}]
   is used for [{gene}] alteration in [{cancer}]?
   Answer with:
   verified: yes or no
   confidence: a number between 0.0 and 1.0
   reason: one sentence explanation"

Output added to schema:
  - llm_verified: yes / no
  - llm_confidence: 0.0 to 1.0
  - llm_reason: short explanation
  - llm_model: which model was used

Model options (in priority order):
  Option 1: Claude Haiku API — fastest and cheapest
  Option 2: GPT-4o-mini — good alternative
  Option 3: Local BioGPT — offline, no API cost

Flag: --llm-verify (off by default, opt-in)
Flag: --llm-model {claude, gpt4, biogpt} (default: claude)
Flag: --llm-confidence-threshold 0.7 (filter below this)

### Why this answers Reviewer 1
Reviewer 1 said pure rule-based is antiquated in 2026.
Our response: megaMine is hybrid. Rules provide speed,
interpretability, and traceability. LLM layer adds semantic
verification. Users control the balance via flags.
This is more powerful than pure LLM (which is a black box)
and more accurate than pure rules.

---

## Module 3 — Temporal Evidence Tracking (modules/temporal.py)
### What it does
Shows how the evidence balance for a gene-drug-cancer triplet
has shifted over time. Transforms megaMine from a static
extractor into a dynamic evidence tracker.

### Design
Input: megaMine output dataframe
Grouping: (gene, drug, cancer_type, year_bin)

Year bins:
  2010-2014: early evidence
  2015-2017: pre-immunotherapy era
  2018-2020: immunotherapy expansion
  2021-2023: resistance emergence
  2024+:     current evidence

Per bin calculate:
  - n_papers: total papers in this bin
  - efficacy_ratio: efficacy papers / total papers
  - resistance_ratio: resistance papers / total papers
  - dominant_study: most common study_design in bin

Trend classification:
  - stable: resistance_ratio change < 0.10 across bins
  - rising_resistance: resistance_ratio increasing > 0.10
  - declining_efficacy: efficacy_ratio decreasing > 0.15
  - emerging: n_papers < 10 but growing fast
  - conflicted: see contradiction module

Output fields added to schema:
  - temporal_trend: stable / rising_resistance /
                    declining_efficacy / emerging / conflicted
  - peak_evidence_year: year bin with most papers
  - resistance_emergence_year: first year bin where
    resistance_ratio > 0.20

### Example output
  EGFR + erlotinib + NSCLC:
  2015-2017: efficacy 0.89, resistance 0.11 → stable
  2018-2020: efficacy 0.74, resistance 0.26 → rising_resistance
  2021-2023: efficacy 0.51, resistance 0.49 → rising_resistance
  2024+:     efficacy 0.38, resistance 0.62 → declining_efficacy
  temporal_trend: declining_efficacy
  resistance_emergence_year: 2018-2020

### Why this is novel
No existing literature mining tool tracks evidence temporally.
This transforms megaMine into a living evidence system that
shows how scientific consensus evolves — clinically valuable
for treatment planning.

---

## Module 4 — Contradiction Detection (modules/contradiction.py)
### What it does
Flags gene-drug-cancer triplets where the literature shows
both strong efficacy and strong resistance signals.
Clinically important — doctors need to know when evidence
is conflicted, not just how much evidence exists.

### Design
Input: megaMine output + temporal tracking output
Grouping: (gene, drug, cancer_type)

Conflict score formula:
  conflict_score = resistance_papers / (efficacy_papers +
                   resistance_papers)

Thresholds:
  conflict_score < 0.20: no conflict
  conflict_score 0.20-0.40: emerging conflict (WATCH)
  conflict_score 0.40-0.60: moderate conflict (CAUTION)
  conflict_score > 0.60: strong conflict (CONFLICT)

Additional flags:
  temporal_conflict: efficacy dominated early, resistance
                     dominates recently
  subgroup_conflict: efficacy in one patient subgroup,
                     resistance in another

Output fields:
  - contradiction_flag: none / watch / caution / conflict
  - conflict_score: 0.0 to 1.0
  - efficacy_papers: count
  - resistance_papers: count
  - temporal_conflict: yes / no

### Example
  EGFR + gefitinib + NSCLC:
  efficacy_papers: 234
  resistance_papers: 67
  conflict_score: 0.22 → WATCH
  temporal_conflict: yes (resistance rising post 2019)

### Why this is novel
Contradiction detection in oncology literature has never
been done systematically in a mining pipeline. Direct
clinical value — prevents overconfident treatment decisions.

---

## Module 5 — Knowledge Graph Export (modules/graph.py)
### What it does
Converts megaMine output into a knowledge graph where
nodes are genes, drugs, and cancers, and edges are
the evidence-weighted relationships between them.

### Design
Node types:
  - Gene: EGFR, KRAS, BRCA1 etc
  - Drug: erlotinib, pembrolizumab etc
  - Cancer: NSCLC, BRCA, GBM etc
  - Mutation: L858R, G12C etc

Edge types:
  - treats: Drug → Cancer (with evidence weight)
  - targets: Drug → Gene (with evidence weight)
  - mutated_in: Gene → Cancer
  - causes_resistance: Mutation → Drug
  - associated_with: Gene → Drug (literature-derived)

Edge properties (from megaMine):
  - evidence_weight: n_papers normalized 0-1
  - efficacy_score: efficacy_ratio
  - resistance_score: resistance_ratio
  - temporal_trend: from temporal module
  - contradiction_flag: from contradiction module
  - top_pmids: list of top 5 supporting PMIDs

Export formats:
  - NetworkX .graphml (for Python analysis)
  - Edge list .csv (for Gephi / Cytoscape visualization)
  - Node list .csv

Flag: --output-graph (off by default)
Flag: --graph-format {graphml, csv, both} (default: both)

### Why this is novel
Combines literature-derived evidence weighting with
temporal tracking and contradiction detection in one
graph. No existing biomedical KG does all three.
Positions megaMine as foundation for future KG paper.

---

## Module 6 — ClinicalTrials Linkage (modules/trials.py)
### What it does
For each gene-drug-cancer triplet, queries ClinicalTrials.gov
and links the literature evidence to real clinical trials.
Bridges literature mining and clinical reality.

### Design
Input: unique (drug, cancer_type) pairs from megaMine output
API: ClinicalTrials.gov v2 REST API (free)
Query: drug AND cancer_type

Per trial extract:
  - nct_id: trial identifier
  - phase: Phase I / II / III / IV
  - status: recruiting / completed / terminated
  - title: brief title

Trial summary per drug:
  - n_trials: total trials found
  - highest_phase: best phase reached
  - has_failed_trial: yes if terminated trials exist
  - failed_trial_ids: list of terminated NCT IDs

Output fields added:
  - n_clinical_trials: count
  - highest_trial_phase: Phase I/II/III/IV
  - has_failed_trial: yes / no
  - top_trial_nct: NCT ID of highest phase trial
  - trial_literature_concordance: high/medium/low
    (does trial evidence match literature evidence?)

Flag: --link-trials (off by default)

---

## Module 7 — Gold Standard Validation (validation/gold_standard.py)
### What it does
Evaluates megaMine extraction accuracy against a manually
curated gold standard dataset. Produces Precision, Recall,
F1 scores for entity and relation extraction.

### Gold standard construction plan
Step 1: Select 200 PubMed abstracts manually
  - 50 NSCLC papers (well-studied, high confidence)
  - 50 breast cancer papers
  - 50 colorectal cancer papers
  - 50 rare cancer papers (low evidence, harder cases)
  - Mix of years: 2015-2024
  - Mix of study types: RCT, observational, case report

Step 2: Manual annotation schema
  For each abstract annotate:
  - genes: list of HGNC gene symbols mentioned
  - drugs: list of drug names mentioned
  - cancer_type: normalized cancer name
  - evidence_type: efficacy/resistance/review/background
  - resistance_observed: yes/no
  - key_triplets: list of (gene, drug, cancer) confirmed pairs

Step 3: Annotation guidelines
  - Two independent annotators per abstract
  - Inter-annotator agreement: Cohen kappa > 0.80 required
  - Disagreements resolved by discussion
  - Final annotations stored in:
    validation/gold_standard/annotations.json

Step 4: Evaluation metrics
  Per field calculate:
  - Precision = true positives / (true positives + false positives)
  - Recall = true positives / (true positives + false negatives)
  - F1 = 2 * (Precision * Recall) / (Precision + Recall)

  Report separately for:
  - Gene extraction
  - Drug extraction
  - Cancer type extraction
  - Evidence type classification
  - Resistance detection
  - Full triplet extraction (all three correct)

---

## Module 8 — PubTator3 Benchmark (validation/benchmarks.py)
### What it does
Runs the same gold standard abstracts through PubTator3
and compares extraction results against megaMine.

### Design
For each abstract in gold standard:
  1. Run megaMine extraction
  2. Query PubTator3 API for same PMID
  3. Compare both against manual annotations
  4. Calculate F1 for both systems

Comparison table columns:
  - System: megaMine vs PubTator3
  - Gene F1
  - Drug F1
  - Cancer F1
  - Triplet F1
  - Speed (abstracts per minute)
  - Structured output fields: megaMine has 40+, PubTator3 has 5

### Our argument
megaMine may not beat PubTator3 on raw NER F1.
But megaMine produces structured clinical metadata
(TMB, MSI, study_design, trial_phase, resistance_observed,
temporal_trend, contradiction_flag) that PubTator3
does not produce at all.
The comparison is not just accuracy — it is richness
of structured output for precision oncology use cases.

---

## Paper Structure Plan (Briefings in Bioinformatics)

Section 1: Introduction
  - Problem: literature evidence for precision oncology
    is unstructured and scattered
  - Gap: existing tools do NER but not structured clinical
    evidence extraction with temporal and contradiction awareness
  - Our contribution: megaMine v2.0 hybrid framework

Section 2: Methods
  2.1 Core extraction engine
  2.2 LLM verification layer
  2.3 Temporal evidence tracking
  2.4 Contradiction detection
  2.5 Knowledge graph export
  2.6 ClinicalTrials linkage
  2.7 Gold standard construction
  2.8 Benchmarking against PubTator3

Section 3: Results
  3.1 Gold standard validation (F1 scores — lead with this)
  3.2 LLM verification improves precision
  3.3 Temporal tracking — EGFR/erlotinib example
  3.4 Contradiction detection — case studies
  3.5 Knowledge graph statistics
  3.6 Runtime and memory benchmarks
  3.7 Comparison with PubTator3

Section 4: Discussion
  - Interpretability vs LLM black box argument
  - Limitations (PubMed bias, rule coverage gaps)
  - Future directions (REMAP integration)

Section 5: Conclusion

---

## Timeline — One Month

Week 1: Bug fixes + gold standard construction
  Day 1-2:   Fix 4 bugs in extractor.py
  Day 3-5:   Select and annotate 200 abstracts
  Day 6-7:   Run megaMine on gold standard, calculate F1

Week 2: Temporal + Contradiction modules
  Day 8-10:  Build temporal.py
  Day 11-12: Build contradiction.py
  Day 13-14: Test both on 100k dataset

Week 3: LLM + Graph + Trials modules
  Day 15-17: Build llm_verify.py
  Day 18-19: Build graph.py
  Day 20-21: Build trials.py

Week 4: Paper + Benchmarking + Polish
  Day 22-23: Run PubTator3 benchmark
  Day 24-26: Write new paper draft
  Day 27-28: Polish code + update README + CHANGELOG
  Day 29-30: Final review + GitHub release tag v2.0.0
