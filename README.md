# megaMine

**Hybrid literature mining for precision oncology.**

megaMine extracts gene–drug–cancer therapeutic evidence from PubMed, normalizes cancer labels, refines resistance evidence, tracks temporal trends, detects contradictions, links to ClinicalTrials.gov, and exports a provenance-aware knowledge graph with an interactive HTML report.

> ⚠️ Literature-derived evidence only. Not a clinical treatment recommendation.

---

## What it does

Given a PubMed query, megaMine produces:

- Normalized gene–drug–cancer evidence rows
- 3-tier resistance refinement (observed → context → direct evidence)
- Temporal trend classification per gene-drug-cancer triplet
- Contradiction detection with conflict scores
- ClinicalTrials.gov linkage
- Knowledge graph (GraphML + CSV)
- Standalone interactive HTML report (no internet required)

---

## Installation

```bash
conda create -n megamine python=3.9 -y
conda activate megamine
pip install "git+https://github.com/Junaid13913/megaMine.git"
```

---

## Quick start

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

Output files:

| File | Contents |
|---|---|
| `my_run.xlsx` | Evidence rows, temporal trends, contradictions, trials |
| `my_run_graph_nodes.csv` | Knowledge graph nodes |
| `my_run_graph_edges.csv` | Knowledge graph edges |
| `my_run_graph.graphml` | Graph for Cytoscape / Neo4j |
| `my_run_HTML_REPORT.html` | Interactive report — open in any browser |

---

## Requirements

- Python ≥ 3.9
- NCBI email + API key (free at https://www.ncbi.nlm.nih.gov/account/)
- Internet access for PubMed, Europe PMC, PubTator, HGNC

---

## Case study

Query: `EGFR AND erlotinib AND resistance AND NSCLC` · 1,000 papers · 2015–2024

- 1,150 extracted rows → 210 verified evidence rows from 166 unique PMIDs
- 13 canonical cancer types after normalization
- 14 unique drugs detected
- 45 temporal triplets — 1 rising resistance signal (EGFR + erlotinib + NSCLC)
- 436 graph nodes · 1,587 edges
- 23 ClinicalTrials pairs

---

## Getting help

```bash
megaMine --help
megaMine --list-cancers
```

---

## Author

Muhammad Junaid · Ajou University · junaidm@ajou.ac.kr
