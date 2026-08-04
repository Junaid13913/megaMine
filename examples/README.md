# megaMine Examples

## MAF Analysis — Patient Mutation Evidence Synthesis

Analyze patient tumor mutations against published oncology literature.

### Setup

    export ONCOKB_TOKEN=your_token_here
    # Free token: https://oncokb.org/account/register

### Quick start

    megaMine-maf \
        --maf examples/example_synthetic.maf \
        --cancer NSCLC \
        --out results/patient_001/ \
        --years 2018-2024 \
        --max-records 50 \
        --vaf-min 0.05 \
        --top 3 \
        --email your@institution.edu

### Full example — Breast cancer

    megaMine-maf \
        --maf patient.maf \
        --cancer Breast \
        --out results/breast_patient/ \
        --years 2015-2024 \
        --max-records 100 \
        --vaf-min 0.10 \
        --top 5 \
        --email your@institution.edu \
        --ncbi-api-key YOUR_NCBI_KEY

### Arguments

    --maf           Patient MAF file path (required)
    --cancer        Cancer type (required): NSCLC, Breast, CRC, Gastric...
    --out           Output directory (required)
    --years         Year range e.g. 2018-2024 (default: 2018-2024)
    --max-records   Max PubMed papers per gene (default: 50)
    --vaf-min       Minimum variant allele frequency (default: 0.0)
    --top           Top drugs to report per variant (default: 3)
    --email         NCBI email (required for API)
    --ncbi-api-key  NCBI API key (optional, increases rate limit)
    --hgnc-cache    Path to HGNC cache JSON (optional, speeds up startup)

### Output files

    results/patient_001/
        MAF_EVIDENCE_REPORT.xlsx     per-variant literature evidence
        MAF_PRIORITIZED_REPORT.xlsx  drug priority ranking by evidence score
        maf_report.html              interactive clinical report (open in browser)
        maf_GENE.xlsx                per-gene evidence detail

### Python API

    import sys
    sys.path.insert(0, "src")
    from megaMine.core.maf_pipeline import run_maf_pipeline

    run_maf_pipeline(
        maf_path    = "patient.maf",
        cancer      = "NSCLC",
        out_dir     = "results/",
        email       = "your@institution.edu",
        max_records = 50,
        top_drugs   = 3,
        years       = "2018-2024",
        vaf_min     = 0.05,
    )

### Supported cancer types

NSCLC, Breast, CRC, Gastric, Ovarian, Melanoma,
Leukemia, Lymphoma, Glioma, Pancreatic, Bladder,
Prostate, Colorectal, Head and Neck, and more.

### Privacy

**Never commit real patient MAF files to git.**
All *.maf files are protected by .gitignore.
Use example_synthetic.maf for testing only.
Real patient data must remain on secure local storage.
