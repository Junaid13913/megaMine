"""
ChEMBL drug synonym lookup
Queries ChEMBL REST API to resolve brand names,
abbreviations, and synonyms to generic drug names
"""

import requests
import json
import time
import os
from pathlib import Path

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
CACHE_PATH = Path(__file__).parent / "chembl_cache.json"

# ── Load/save cache ───────────────────────────────────────────
def _load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

# ── Query ChEMBL for drug synonyms ───────────────────────────
def get_synonyms(drug_name: str, cache: dict) -> dict:
    """
    Returns:
      generic_name: str
      synonyms: list
      brand_names: list
      chembl_id: str
      found: bool
    """
    key = drug_name.lower().strip()
    if key in cache:
        return cache[key]

    result = {
        "query": drug_name,
        "generic_name": drug_name,
        "synonyms": [],
        "brand_names": [],
        "chembl_id": None,
        "found": False
    }

    try:
        # Search by name
        url = f"{CHEMBL_API}/molecule/search.json"
        params = {
            "q": drug_name,
            "format": "json",
            "limit": 1
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            cache[key] = result
            return result

        data = r.json()
        molecules = data.get("molecules", [])
        if not molecules:
            cache[key] = result
            return result

        mol = molecules[0]
        chembl_id = mol.get("molecule_chembl_id")

        # Get preferred name
        pref_name = mol.get("pref_name", drug_name)
        if pref_name:
            result["generic_name"] = pref_name.lower()

        # Get all synonyms
        syns = mol.get("molecule_synonyms", [])
        synonyms = []
        brand_names = []
        for s in syns:
            syn = s.get("molecule_synonym", "")
            syn_type = s.get("syn_type", "")
            if syn:
                synonyms.append(syn.lower())
                if syn_type in ("TRADE_NAME", "BRAND"):
                    brand_names.append(syn.lower())

        result.update({
            "generic_name": pref_name.lower() if pref_name else drug_name,
            "synonyms"    : list(set(synonyms)),
            "brand_names" : list(set(brand_names)),
            "chembl_id"   : chembl_id,
            "found"       : True
        })

        time.sleep(0.3)  # rate limit

    except Exception as e:
        print(f"ChEMBL lookup failed for {drug_name}: {e}")

    cache[key] = result
    return result

# ── Build expanded drug whitelist ─────────────────────────────
def expand_drug_whitelist(whitelist: list) -> dict:
    """
    Takes our 254 drug whitelist
    Returns expanded dict:
      synonym/brand_name → generic_name
    """
    cache = _load_cache()
    expansion = {}
    failed = []

    print(f"Expanding {len(whitelist)} drugs via ChEMBL...")

    for i, drug in enumerate(whitelist):
        print(f"  [{i+1}/{len(whitelist)}] {drug}...", end=" ")
        result = get_synonyms(drug, cache)

        if result["found"]:
            # Map all synonyms back to generic name
            generic = result["generic_name"]
            for syn in result["synonyms"]:
                expansion[syn] = generic
            for brand in result["brand_names"]:
                expansion[brand] = generic
            # Map original name too
            expansion[drug.lower()] = generic
            print(f"✅ {len(result['synonyms'])} synonyms")
        else:
            failed.append(drug)
            print("❌ not found")

    _save_cache(cache)
    print(f"\n✅ Expanded to {len(expansion)} drug terms")
    print(f"❌ Failed: {len(failed)} drugs")
    if failed:
        print(f"   {failed[:10]}")

    return expansion

# ── Normalize drug name using expansion ──────────────────────
def normalize_drug(drug_name: str, expansion: dict) -> str:
    """
    Given any drug mention (brand/abbreviation/synonym)
    returns the generic name from our whitelist
    """
    key = drug_name.lower().strip()
    return expansion.get(key, drug_name)

# ── Test with our known problem cases ────────────────────────
if __name__ == "__main__":
    cache = _load_cache()

    test_cases = [
        "Tarceva",      # brand name for erlotinib
        "OSI-774",      # abbreviation for erlotinib
        "Iressa",       # brand name for gefitinib
        "Tagrisso",     # brand name for osimertinib
        "Herceptin",    # brand name for trastuzumab
        "Gleevec",      # brand name for imatinib
        "erlotinib",    # should find itself
        "gefitinib",    # should find itself
    ]

    print("=== ChEMBL drug synonym lookup test ===\n")
    for drug in test_cases:
        result = get_synonyms(drug, cache)
        print(f"Query:   {drug}")
        print(f"Generic: {result['generic_name']}")
        print(f"ChEMBL:  {result['chembl_id']}")
        print(f"Brands:  {result['brand_names'][:3]}")
        print(f"Synonyms:{result['synonyms'][:3]}")
        print()

    _save_cache(cache)
    print("✅ Cache saved")
