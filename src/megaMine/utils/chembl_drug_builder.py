"""
ChEMBL drug vocabulary builder for megaMine v2.0
Builds comprehensive anticancer drug list from ChEMBL
via four endpoints:
  1. Drug indications (neoplasm/cancer)
  2. Mechanism of action (anticancer targets)
  3. ATC classification (L = antineoplastic)
  4. Approval status (approved drugs only)

Based on: Zdrazil et al. 2024 (ChEMBL database)
"""

import requests
import json
import time
import pandas as pd
from pathlib import Path

CHEMBL_API  = "https://www.ebi.ac.uk/chembl/api/data"
OUT_DIR     = Path("/Volumes/LaCie/megamine_v2/data/chembl")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE  = OUT_DIR / "chembl_raw_cache.json"

def _get(endpoint, params={}, retries=3):
    """GET with retry and rate limiting"""
    url = f"{CHEMBL_API}/{endpoint}.json"
    for attempt in range(retries):
        try:
            r = requests.get(url, params={**params, "limit":1000,
                                          "format":"json"},
                             timeout=30)
            if r.status_code == 200:
                time.sleep(0.3)
                return r.json()
            print(f"  HTTP {r.status_code} on {endpoint}")
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

def _paginate(endpoint, params={}, max_records=5000):
    """Paginate through all results"""
    results = []
    offset  = 0
    limit   = 200

    while True:
        data = _get(endpoint, {**params,
                               "limit": limit,
                               "offset": offset})
        if not data:
            break

        # Get the actual results key
        key = [k for k in data.keys()
               if k not in ("page_meta",)][0]
        batch = data.get(key, [])
        results.extend(batch)

        meta   = data.get("page_meta", {})
        total  = meta.get("total_count", 0)
        offset += limit

        print(f"  Fetched {len(results)}/{total}...",
              end="\r")

        if offset >= total or offset >= max_records:
            break

    print(f"  Total fetched: {len(results)}")
    return results

# ══════════════════════════════════════════════════════════════
# ENDPOINT 1: Drug indications — neoplasm/cancer
# ══════════════════════════════════════════════════════════════
def fetch_by_indication():
    print("\n[1] Fetching by drug indication (neoplasm/cancer)...")
    results = _paginate(
        "drug_indication",
        params={"mesh_heading__icontains": "neoplasm",
                "max_phase_for_ind__gte": 3}
    )

    drugs = []
    for r in results:
        chembl_id = r.get("molecule_chembl_id")
        name      = r.get("molecule_name", "")
        mesh      = r.get("mesh_heading", "")
        phase     = r.get("max_phase_for_ind", 0)
        if chembl_id and name:
            drugs.append({
                "chembl_id"  : chembl_id,
                "name"       : name.lower().strip(),
                "source"     : "indication",
                "indication" : mesh,
                "max_phase"  : phase
            })

    print(f"  Found {len(drugs)} drugs from indications")
    return drugs

# ══════════════════════════════════════════════════════════════
# ENDPOINT 2: Mechanism of action — anticancer targets
# ══════════════════════════════════════════════════════════════
def fetch_by_mechanism():
    print("\n[2] Fetching by mechanism of action (anticancer)...")

    anticancer_targets = [
        "EGFR", "KRAS", "BRAF", "ALK", "MET", "ROS1",
        "HER2", "ERBB2", "PIK3CA", "PTEN", "AKT",
        "MTOR", "CDK4", "CDK6", "VEGFR", "FGFR",
        "BCR-ABL", "JAK", "PD-1", "PD-L1", "CTLA4",
        "BRCA1", "BRCA2", "PARP", "BCL2", "MDM2",
        "RET", "NTRK", "IDH1", "IDH2", "FLT3"
    ]

    drugs = []
    for target in anticancer_targets:
        print(f"  Target: {target}...", end=" ")
        results = _paginate(
            "mechanism",
            params={"target_name__icontains": target},
            max_records=500
        )
        for r in results:
            chembl_id = r.get("molecule_chembl_id")
            name      = r.get("molecule_name", "")
            moa       = r.get("mechanism_of_action", "")
            if chembl_id and name:
                drugs.append({
                    "chembl_id": chembl_id,
                    "name"     : name.lower().strip(),
                    "source"   : "mechanism",
                    "target"   : target,
                    "moa"      : moa
                })

    print(f"\n  Found {len(drugs)} drugs from mechanisms")
    return drugs

# ══════════════════════════════════════════════════════════════
# ENDPOINT 3: ATC classification — L = antineoplastic
# ══════════════════════════════════════════════════════════════
def fetch_by_atc():
    print("\n[3] Fetching by ATC classification (L = antineoplastic)...")
    results = _paginate(
        "atc_classification",
        params={"level1": "L"}
    )

    # Get ChEMBL IDs from ATC codes
    atc_codes = [r.get("level5") for r in results
                 if r.get("level5")]
    print(f"  Found {len(atc_codes)} ATC-L codes")

    drugs = []
    for atc in atc_codes[:200]:  # limit for speed
        data = _get("molecule",
                    params={"atc_classifications": atc,
                            "molecule_type": "Small molecule"})
        if data:
            mols = data.get("molecules", [])
            for mol in mols:
                chembl_id = mol.get("molecule_chembl_id")
                name      = mol.get("pref_name", "")
                if chembl_id and name:
                    drugs.append({
                        "chembl_id": chembl_id,
                        "name"     : name.lower().strip(),
                        "source"   : "atc_L",
                        "atc_code" : atc
                    })

    print(f"  Found {len(drugs)} drugs from ATC-L")
    return drugs

# ══════════════════════════════════════════════════════════════
# ENDPOINT 4: Approved drugs — filter to anticancer
# ══════════════════════════════════════════════════════════════
def fetch_approved():
    print("\n[4] Fetching approved anticancer drugs...")
    results = _paginate(
        "molecule",
        params={
            "max_phase": 4,
            "molecule_type": "Small molecule",
            "therapeutic_flag": True
        },
        max_records=3000
    )

    # Filter to those with cancer-related indications
    cancer_keywords = [
        "cancer","tumor","carcinoma","leukemia",
        "lymphoma","melanoma","sarcoma","myeloma",
        "glioma","adenocarcinoma","neoplasm","oncology"
    ]

    drugs = []
    for mol in results:
        chembl_id  = mol.get("molecule_chembl_id")
        name       = mol.get("pref_name", "")
        indication = mol.get("indication_class", "") or ""

        is_cancer = any(k in indication.lower()
                       for k in cancer_keywords)
        if chembl_id and name and is_cancer:
            drugs.append({
                "chembl_id" : chembl_id,
                "name"      : name.lower().strip(),
                "source"    : "approved",
                "indication": indication,
                "max_phase" : 4
            })

    print(f"  Found {len(drugs)} approved anticancer drugs")
    return drugs

# ══════════════════════════════════════════════════════════════
# GET ALL SYNONYMS for final drug list
# ══════════════════════════════════════════════════════════════
def fetch_synonyms_for_list(chembl_ids: list) -> dict:
    """
    For each ChEMBL ID fetch all synonyms
    Returns: {synonym → generic_name}
    """
    print(f"\n[5] Fetching synonyms for {len(chembl_ids)} drugs...")
    expansion = {}

    for i, chembl_id in enumerate(chembl_ids):
        print(f"  [{i+1}/{len(chembl_ids)}] {chembl_id}...",
              end="\r")
        data = _get(f"molecule/{chembl_id}")
        if not data:
            continue

        pref_name = (data.get("pref_name") or "").lower().strip()
        syns      = data.get("molecule_synonyms", [])

        for s in syns:
            syn      = (s.get("molecule_synonym") or "").lower().strip()
            syn_type = s.get("syn_type", "")
            if syn:
                expansion[syn] = pref_name
                # also map chembl_id
                expansion[chembl_id.lower()] = pref_name

        # map preferred name to itself
        if pref_name:
            expansion[pref_name] = pref_name

        time.sleep(0.2)

    print(f"\n  Total synonym mappings: {len(expansion)}")
    return expansion

# ══════════════════════════════════════════════════════════════
# MAIN — build complete drug vocabulary
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("ChEMBL anticancer drug vocabulary builder")
    print("megaMine v2.0 · Zdrazil et al. 2024")
    print("=" * 60)

    # Collect from all four endpoints
    all_drugs = []
    all_drugs.extend(fetch_by_indication())
    all_drugs.extend(fetch_by_mechanism())
    all_drugs.extend(fetch_approved())
    # ATC takes long — optional
    # all_drugs.extend(fetch_by_atc())

    # Build DataFrame and deduplicate
    df = pd.DataFrame(all_drugs)
    print(f"\nTotal before dedup: {len(df)}")

    df = df.dropna(subset=["chembl_id","name"])
    df = df.drop_duplicates(subset=["chembl_id"])
    print(f"After dedup by ChEMBL ID: {len(df)}")

    # Save raw drug list
    df.to_csv(OUT_DIR / "chembl_drugs_raw.csv",
              index=False)
    print(f"✅ Raw list saved: {len(df)} drugs")

    # Get all synonyms
    chembl_ids = df["chembl_id"].tolist()
    expansion  = fetch_synonyms_for_list(chembl_ids[:500])

    # Save expansion dictionary
    with open(OUT_DIR / "chembl_expansion.json", "w") as f:
        json.dump(expansion, f, indent=2)
    print(f"✅ Expansion saved: {len(expansion)} synonym mappings")

    # Save final drug list
    final_drugs = sorted(df["name"].dropna().unique().tolist())
    with open(OUT_DIR / "chembl_drug_whitelist.json", "w") as f:
        json.dump(final_drugs, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL DRUG VOCABULARY:")
    print(f"  Unique drugs:    {len(final_drugs)}")
    print(f"  Synonym mappings:{len(expansion)}")
    print(f"  Sources:         indication + mechanism + approved")
    print(f"  Reference:       Zdrazil et al. 2024 (ChEMBL)")
    print(f"  Output dir:      {OUT_DIR}")
    print(f"{'='*60}")

    # Show sample
    print(f"\nSample drugs:")
    for d in final_drugs[:20]:
        print(f"  {d}")

    # Compare with our old whitelist
    try:
        import sys
        sys.path.insert(0,
            "/Volumes/LaCie/megamine_v2/src")
        from megaMine.utils.drug_whitelist import DRUG_WHITELIST
        old = set(d.lower() for d in DRUG_WHITELIST)
        new = set(final_drugs)
        print(f"\nComparison with old whitelist:")
        print(f"  Old whitelist:  {len(old)} drugs")
        print(f"  ChEMBL list:    {len(new)} drugs")
        print(f"  New drugs added:{len(new-old)}")
        print(f"  Sample new:     {list(new-old)[:10]}")
    except Exception as e:
        print(f"\nCould not compare with old whitelist: {e}")
