
import sys
sys.path.insert(0, "src")
from megaMine.core.maf_pipeline import run_maf_pipeline
run_maf_pipeline(
    maf_path="examples/example_synthetic.maf",
    cancer="NSCLC",
    out_dir="results/maf_example",
    email="your@institution.edu",
    max_records=50, top_drugs=3,
    years="2018-2024", vaf_min=0.05,
)
print("Done. Check results/maf_example/")
