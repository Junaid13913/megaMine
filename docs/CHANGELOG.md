# megaMine Changelog

## v2.0.0 (in development)

### New Features
- LLM verification layer — rule-based candidates verified by 
  lightweight LLM (BioGPT / API-based) with confidence score
- Temporal evidence tracking — shows how efficacy/resistance 
  balance shifts over time per gene-drug-cancer triplet
- Contradiction detection — flags triplets with conflicting 
  efficacy and resistance signals
- Knowledge graph export — outputs as NetworkX graph 
  with --output-graph flag
- ClinicalTrials.gov linkage — links extracted pairs to trials
- Gold standard validation — F1/Precision/Recall against 
  manually curated dataset
- PubTator3 benchmark comparison table

### Bug Fixes
- Fixed --year-binned not honoring per-year cap
- Fixed export errors being silently suppressed
- Fixed Python API returning .xlsx when .csv.gz was written
- Fixed cancer normalization capturing paper titles as cancer names
- Fixed rate limiting to respect NCBI 10 req/sec limit

### Improvements  
- Expanded default drug whitelist (500+ drugs)
- Added --list-cancers flag
- Renamed conclusion to auto_conclusion (clarity)
- Added runtime and memory benchmark table
- Added --llm-verify flag to enable verification layer
- Added --output-graph flag for knowledge graph export

## v1.0.0 (submitted to CSJB 2025)
- Initial public release
- Rule-based gene-drug-cancer extraction from PubMed
- HGNC-validated gene recognition
- Context-aware evidence typing
- TMB/MSI/immune feature extraction
- Driver mode and therapy mode
