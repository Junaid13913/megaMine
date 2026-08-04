# megaMine v2.0

**Hybrid literature mining for precision oncology.**

megaMine extracts gene–drug–cancer therapeutic evidence from PubMed,
verifies it with an LLM gate, normalizes cancer labels, tracks temporal
resistance trends, detects contradictions, links to ClinicalTrials.gov,
and exports a knowledge graph with an interactive HTML report.

> Literature-derived evidence only. Not a clinical treatment recommendation.

---

## What it does

Given a PubMed query or patient MAF file, megaMine produces:

- Normalized gene–drug–cancer evidence rows with LLM verification
- ChEMBL drug vocabulary (2,305 drugs + 20,823 brand name synonyms)
- 3-tier resistance refinement (observed → context → direct evidence)
- Temporal trend classification per gene–drug–cancer triplet
- Contradiction detection with conflict scores
- ClinicalTrials.gov linkage
- Knowledge graph (GraphML + CSV) — 417 nodes, 1,562 edges
- Patient MAF analysis with per-variant drug prioritization
- Standalone interactive HTML report (no internet required)

---

## Installation

```bash
conda create -n megamine python=3.9 -y
conda activate megamine
pip install "git+https://github.com/Junaid13913/megaMine.git"
```

---

## Literature mining — Quick start

```bash
megaMine \
  --q "EGFR AND erlotinib AND resistance AND NSCLC" \
  --years 2020-2024 \
  --max-records 500 \
  --email "your@email.com" \
  --ncbi-api-key "YOUR_KEY" \
  --require-gene-and-drug \
  --require-known-drug \
  --year-binned \
  --out my_run
```

### Output files

| File | Contents |
|------|----------|
| `my_run.xlsx` | Evidence rows, temporal trends, contradictions, trials |
| `my_run_graph_nodes.csv` | Knowledge graph nodes |
| `my_run_graph_edges.csv` | Knowledge graph edges |
| `my_run_graph.graphml` | Graph for Cytoscape / Neo4j |
| `my_run_HTML_REPORT.html` | Interactive report — open in any browser |

---

## Patient MAF analysis — Quick start

```bash
export ONCOKB_TOKEN=your_token_here
# Free token: https://oncokb.org/account/register

megaMine-maf \
    --maf patient.maf \
    --cancer NSCLC \
    --out results/patient_001/ \
    --years 2018-2024 \
    --max-records 50 \
    --vaf-min 0.05 \
    --top 3 \
    --email your@institution.edu \
    --ncbi-api-key YOUR_NCBI_KEY
```

### MAF output files

| File | Contents |
|------|----------|
| `MAF_EVIDENCE_REPORT.xlsx` | Per-variant literature evidence |
| `MAF_PRIORITIZED_REPORT.xlsx` | Drug priority ranking by evidence score |
| `maf_report.html` | Interactive clinical report |
| `maf_GENE.xlsx` | Per-gene evidence detail |

> **Privacy:** Never commit real patient MAF files to git.
> All `*.maf` files are protected by `.gitignore`.
> See `examples/` for a synthetic MAF demo.

---

## MAF arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--maf` | Patient MAF file path | required |
| `--cancer` | Cancer type (NSCLC, Breast, CRC...) | required |
| `--out` | Output directory | required |
| `--years` | Year range e.g. 2018-2024 | 2018-2024 |
| `--max-records` | Max PubMed papers per gene | 50 |
| `--vaf-min` | Minimum variant allele frequency | 0.0 |
| `--top` | Top drugs to report per variant | 3 |
| `--email` | NCBI email | required |
| `--ncbi-api-key` | NCBI API key | optional |

---

## Requirements

- Python >= 3.9
- NCBI email + API key (free at https://www.ncbi.nlm.nih.gov/account/)
- OncoKB token for MAF analysis (free at https://oncokb.org/account/register)
- Internet access for PubMed, Europe PMC, PubTator3, HGNC, ChEMBL

---

## Case study — EGFR erlotinib NSCLC

Query: `EGFR AND erlotinib AND resistance AND NSCLC` · 648 PMIDs · 2015–2024

| Metric | Value |
|--------|-------|
| Rows extracted | 1,145 |
| Rows after LLM verification | 208 |
| Unique PMIDs verified | 164 |
| Cancer types normalized | 13 |
| Unique drugs detected | 14 |
| Temporal triplets | 36 |
| Knowledge graph nodes | 417 |
| Knowledge graph edges | 1,562 |
| ClinicalTrials pairs | 23 |
| Overall F1 vs gold standard | 0.736 |
| vs PubTator3 overall F1 | +24% |
| Cancer type F1 | 0.908 (+113% vs PubTator3) |

---

## Validation

Evaluated on a 72-PMID manually annotated gold standard
across 4 cancer groups (NSCLC, Breast, CRC, Rare).

| Task | F1 | TP | FP | FN |
|------|----|----|----|----|
| Gene extraction | 0.735 | 50 | 15 | 21 |
| Drug extraction | 0.653 | 47 | 25 | 25 |
| Cancer type | 0.908 | 64 | 6 | 7 |
| Relation detection | 0.941 | 64 | 0 | 8 |
| Resistance detection | 0.881 | 52 | 13 | 1 |
| Triplet extraction | 0.431 | 31 | 41 | 41 |
| **Overall** | **0.736** | | | |

---

## Getting help

```bash
megaMine --help
megaMine-maf --help
```

---

## Author

Muhammad Junaid
Ajou Precision Medicine Lab (APML)
Ajou University, School of Medicine, South Korea
junaidm@ajou.ac.kr
