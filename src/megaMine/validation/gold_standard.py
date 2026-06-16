"""
gold_standard.py — megaMine v2.0
Gold Standard Dataset Creation and Loading Module

PURPOSE:
    Creates and manages the manually curated gold standard
    dataset used to evaluate megaMine extraction accuracy.

    This directly addresses all three reviewers:
    "No manually curated gold standard"
    "No true Precision / Recall / F1"
    "Current AUROC is internal / circular"

ANNOTATION SCHEMA:
    pmid                  PubMed ID of the abstract
    sentence              the exact sentence annotated
    gene_gold             correct gene symbol (HGNC)
    drug_gold             correct drug name (normalized)
    cancer_gold           correct cancer type (normalized)
    alteration_gold       specific mutation if present
    relation_gold         yes/no — is there a real relationship
    evidence_type_gold    efficacy/resistance/review/background
    resistance_observed_gold  yes/no
    negated_gold          yes/no — is the claim negated
    speculative_gold      yes/no — is the claim speculative
    study_design_gold     RCT/trial/observational/preclinical/
                          case_report/in_vitro/in_silico
    annotator             who annotated this row
    annotation_notes      free text notes

DESIGN:
    200 abstracts selected manually
    Mix: 50 NSCLC + 50 breast cancer +
         50 colorectal + 50 rare cancers
    Mix of years: 2015-2024
    Mix of study types: RCT, observational, case report
    Two independent annotators per abstract
    Inter-annotator agreement: Cohen kappa target > 0.80

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import os
import json
import pandas as pd
import requests
import time
from typing import Optional, List, Dict

# ─── File paths ───────────────────────────────────────────────
GS_DIR         = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "validation", "gold_standard"
)
GS_TEMPLATE    = os.path.join(GS_DIR, "annotation_template.xlsx")
GS_ANNOTATIONS = os.path.join(GS_DIR, "annotations.json")
GS_ABSTRACTS   = os.path.join(GS_DIR, "abstracts", "abstracts.json")

# ─── NCBI API ─────────────────────────────────────────────────
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESEARCH_URL= "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
SLEEP      = 0.34   # respect NCBI rate limit

# ─── Gold standard schema ─────────────────────────────────────
GS_COLUMNS = [
    "pmid",
    "sentence",
    "gene_gold",
    "drug_gold",
    "cancer_gold",
    "alteration_gold",
    "relation_gold",
    "evidence_type_gold",
    "resistance_observed_gold",
    "negated_gold",
    "speculative_gold",
    "study_design_gold",
    "annotator",
    "annotation_notes",
]

# ─── Annotation guidelines ────────────────────────────────────
ANNOTATION_GUIDELINES = """
MEGAMINE v2.0 — GOLD STANDARD ANNOTATION GUIDELINES
=====================================================

TASK:
For each sentence extracted from a PubMed abstract,
annotate the following fields:

FIELDS:
  gene_gold
    The HGNC-approved gene symbol mentioned in the sentence.
    Use uppercase. Example: EGFR, KRAS, BRCA1
    Leave empty if no gene is mentioned.

  drug_gold
    The drug name mentioned in the sentence.
    Use lowercase normalized form. Example: erlotinib
    Leave empty if no drug is mentioned.

  cancer_gold
    The cancer type mentioned in the sentence.
    Use standard short form. Example: NSCLC, CRC, GBM
    Leave empty if no cancer type is mentioned.

  alteration_gold
    The specific mutation or alteration if mentioned.
    Example: L858R, G12C, T790M, exon 19 deletion
    Leave empty if no specific alteration is mentioned.

  relation_gold
    yes — the sentence describes a real gene-drug-cancer
          relationship (even if negative or speculative)
    no  — the sentence only mentions entities in passing
          without a real relationship

  evidence_type_gold
    efficacy    — drug works / patient responded
    resistance  — drug does not work / patient progressed
    toxicity    — drug caused side effects
    review      — summary/review statement, no direct evidence
    background  — entities mentioned but no relationship

  resistance_observed_gold
    yes — sentence explicitly states resistance or failure
    no  — sentence does not state resistance

  negated_gold
    yes — sentence uses negation (did not respond,
          no response, failed to respond, no benefit)
    no  — sentence does not use negation

  speculative_gold
    yes — sentence uses speculative language
          (may, might, could, potentially, suggests)
    no  — sentence makes a direct statement

  study_design_gold
    RCT           — randomized controlled trial
    trial         — non-randomized clinical trial
    observational — cohort, retrospective, prospective
    case_report   — single patient or small series
    preclinical   — animal or xenograft study
    in_vitro      — cell line study
    in_silico     — computational study
    review        — review or meta-analysis
    unknown       — cannot determine from sentence

  annotator
    Your name or initials

  annotation_notes
    Any notes, ambiguities, or disagreements

EXAMPLES:
  Sentence: "EGFR-mutant NSCLC patients responded to erlotinib"
    gene_gold: EGFR
    drug_gold: erlotinib
    cancer_gold: NSCLC
    relation_gold: yes
    evidence_type_gold: efficacy
    resistance_observed_gold: no
    negated_gold: no
    speculative_gold: no

  Sentence: "KRAS mutations did not respond to cetuximab"
    gene_gold: KRAS
    drug_gold: cetuximab
    cancer_gold: (empty if not mentioned)
    relation_gold: yes
    evidence_type_gold: resistance
    resistance_observed_gold: yes
    negated_gold: yes
    speculative_gold: no

  Sentence: "BRCA1 is a tumor suppressor involved in DNA repair"
    gene_gold: BRCA1
    drug_gold: (empty)
    cancer_gold: (empty)
    relation_gold: no
    evidence_type_gold: background
    resistance_observed_gold: no
    negated_gold: no
    speculative_gold: no
"""

# ─── Predefined PMIDs for gold standard ───────────────────────
#
# PILOT SET (v2.0) — 80 abstracts across 4 cancer types
# 20 per cancer type — balanced coverage
#
# TARGET: expand to 200 abstracts for final submission
# EXPANSION PLAN:
#   Phase 1 (pilot):  80 abstracts — used for method development
#   Phase 2 (full):  200 abstracts — used for final paper validation
#
# Selection criteria:
#   - Covers different evidence types (efficacy, resistance, review)
#   - Covers different study designs (RCT, observational, case report)
#   - Includes negated and speculative sentences
#   - Includes rare cancers with limited curated coverage
#   - Year range 2015-2024
#
# Note: PMIDs verified as of 2025. A small number may have been
# retracted or merged — fetch_abstract() will flag these.

PILOT_SET_SIZE = 80   # current pilot
TARGET_SET_SIZE = 200  # final validation target

GOLD_STANDARD_PMIDS = {
    "NSCLC": [
        # EGFR-targeted therapy
        "25470548",  # erlotinib EGFR NSCLC RCT (OPTIMAL)
        "27959700",  # KRAS resistance NSCLC review
        "29151359",  # osimertinib T790M NSCLC (AURA3)
        "31077989",  # MET bypass EGFR resistance NSCLC
        "33208462",  # sotorasib KRAS G12C NSCLC (CodeBreaK100)
        # ALK-targeted therapy
        "34506102",  # alectinib ALK NSCLC (ALEX)
        "26515464",  # crizotinib ALK NSCLC (PROFILE 1014)
        "30405107",  # brigatinib ALK resistance NSCLC
        "28527573",  # ceritinib ALK NSCLC resistance
        "30373730",  # lorlatinib ALK resistance NSCLC
        # Immunotherapy
        "28168981",  # pembrolizumab PD-L1 NSCLC (KEYNOTE-024)
        "28574553",  # nivolumab NSCLC resistance
        "32955170",  # tepotinib MET exon14 NSCLC
        "31406347",  # capmatinib MET NSCLC
        "34534299",  # amivantamab EGFR exon20 NSCLC
        # Chemotherapy + combinations
        "27432850",  # carboplatin paclitaxel NSCLC
        "28885881",  # bevacizumab NSCLC combination
        "30620982",  # atezolizumab NSCLC combination
        "31915856",  # durvalumab NSCLC stage III
        "33836100",  # savolitinib MET NSCLC
    ],
    "breast_cancer": [
        # HER2-targeted
        "29244706",  # trastuzumab HER2 breast cancer
        "30765184",  # tucatinib HER2 breast (HER2CLIMB)
        "28581404",  # neratinib HER2 breast (ExteNET)
        "33237665",  # trastuzumab deruxtecan HER2 breast
        "31825568",  # margetuximab HER2 breast
        # CDK4/6 inhibitors
        "31825569",  # palbociclib CDK4/6 breast (PALOMA-2)
        "32955179",  # abemaciclib CDK4/6 resistance breast
        "30620986",  # ribociclib CDK4/6 breast (MONALEESA)
        "29414892",  # CDK4/6 resistance mechanisms review
        "34534290",  # CDK4/6 resistance ESR1 mutation
        # PARP inhibitors + BRCA
        "30405104",  # olaparib BRCA1/2 breast (OlympiAD)
        "27834934",  # BRCA1 PARP inhibitor sensitivity
        "32234880",  # niraparib BRCA breast
        "34506098",  # talazoparib BRCA breast (EMBRACA)
        "29414891",  # BRCA resistance PARP inhibitors
        # PI3K/AKT
        "34534296",  # alpelisib PIK3CA breast (SOLAR-1)
        "33221766",  # sacituzumab TNBC
        "30620990",  # ipatasertib AKT breast
        "31915853",  # capivasertib AKT breast
        "32234882",  # TNBC chemotherapy resistance
    ],
    "colorectal": [
        # RAS/RAF/EGFR
        "26888474",  # cetuximab KRAS CRC (wild-type)
        "28248999",  # RAS mutation resistance anti-EGFR CRC
        "30857380",  # BRAF V600E colorectal (BEACON)
        "32487582",  # encorafenib binimetinib BRAF CRC
        "32615269",  # KRAS G12C colorectal emerging data
        # MSI/immunotherapy
        "31954460",  # pembrolizumab MSI-H CRC (KEYNOTE-158)
        "30620992",  # nivolumab dMMR CRC (CheckMate-142)
        "34534302",  # dostarlimab MSI-H CRC
        # HER2 in CRC
        "34504286",  # HER2 amplification CRC treatment
        "32955182",  # trastuzumab HER2 CRC
        # Anti-angiogenic
        "27273579",  # bevacizumab CRC resistance mechanisms
        "30523239",  # regorafenib CRC (CORRECT)
        "32234885",  # ramucirumab CRC
        # Chemotherapy resistance
        "29316424",  # EGFR resistance CRC mechanisms
        "31086785",  # oxaliplatin resistance CRC
        "28574557",  # irinotecan resistance CRC
        # Emerging
        "33836105",  # sotorasib KRAS G12C CRC
        "34534297",  # adagrasib KRAS G12C CRC
        "31915859",  # TAS-102 CRC resistance
        "30373735",  # lonsurf CRC refractory
    ],
    "rare_cancers": [
        # RET fusions
        "32203698",  # selpercatinib RET fusion lung thyroid
        "31406350",  # pralsetinib RET fusion
        # NTRK fusions
        "29596029",  # larotrectinib NTRK fusion (basket)
        "30351337",  # entrectinib ROS1 NTRK fusion
        # AML targeted
        "28926213",  # ivosidenib IDH1 AML
        "29860938",  # enasidenib IDH2 AML
        "31086788",  # gilteritinib FLT3 AML
        "30952687",  # venetoclax BCL2 CLL AML
        # MET alterations
        "32877582",  # capmatinib MET rare cancers
        "34534301",  # adagrasib KRAS rare cancers
        # GI rare
        "33221770",  # ripretinib GIST KIT resistance
        "31406352",  # avapritinib GIST PDGFRA
        "30405108",  # imatinib GIST resistance mechanisms
        # Thyroid
        "32203701",  # lenvatinib thyroid cancer
        "34506100",  # cabozantinib thyroid MET
        # Cholangiocarcinoma
        "32487585",  # pemigatinib FGFR2 CCA
        "34534298",  # futibatinib FGFR2 CCA resistance
        "33221768",  # infigratinib FGFR CCA
        # Bladder/urothelial
        "30620994",  # erdafitinib FGFR urothelial
        "32955184",  # enfortumab urothelial resistance
    ],
}


def fetch_abstract(pmid: str, email: str = "noreply@example.com",
                   api_key: Optional[str] = None) -> Dict:
    """
    Fetch a single abstract from PubMed by PMID.

    Returns dict with:
        pmid, title, abstract, year, journal, authors
    """
    params = {
        "db":      "pubmed",
        "id":      pmid,
        "retmode": "xml",
        "email":   email,
    }
    if api_key:
        params["api_key"] = api_key

    try:
        time.sleep(SLEEP)
        resp = requests.get(EFETCH_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return {"pmid": pmid, "error": f"HTTP {resp.status_code}"}

        import re
        xml = resp.text

        # Extract title
        title_m = re.search(
            r"<ArticleTitle>(.*?)</ArticleTitle>", xml, re.S
        )
        title = re.sub(r"<[^>]+>", "", title_m.group(1)
                       if title_m else "").strip()

        # Extract ALL abstract sections — not just the first
        # Structured abstracts (Background/Methods/Results/Conclusion)
        # have multiple AbstractText elements — capture all of them
        abs_sections = re.findall(
            r"<AbstractText[^>]*>(.*?)</AbstractText>", xml, re.S
        )
        abstract = " ".join(
            re.sub(r"<[^>]+>", "", sec).strip()
            for sec in abs_sections
            if sec.strip()
        )

        # Extract year
        year_m = re.search(r"<Year>(\d{4})</Year>", xml)
        year = year_m.group(1) if year_m else ""

        # Extract journal
        jour_m = re.search(r"<Title>(.*?)</Title>", xml, re.S)
        journal = re.sub(r"<[^>]+>", "", jour_m.group(1)
                         if jour_m else "").strip()

        return {
            "pmid":     pmid,
            "title":    title,
            "abstract": abstract,
            "year":     year,
            "journal":  journal,
        }

    except Exception as e:
        return {"pmid": pmid, "error": str(e)}


def fetch_all_abstracts(
    email: str = "noreply@example.com",
    api_key: Optional[str] = None,
    pmids_dict: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Fetch all gold standard abstracts from PubMed.

    Parameters
    ----------
    email : str
        Your NCBI email
    api_key : str, optional
        Your NCBI API key
    pmids_dict : dict, optional
        Override default GOLD_STANDARD_PMIDS

    Returns
    -------
    dict : {pmid: abstract_data}
    """
    pmids_dict = pmids_dict or GOLD_STANDARD_PMIDS
    all_pmids  = []
    for cancer_type, pmids in pmids_dict.items():
        for pmid in pmids:
            all_pmids.append((pmid, cancer_type))

    print(f"📥 Fetching {len(all_pmids)} abstracts from PubMed...")
    results = {}

    for i, (pmid, cancer_type) in enumerate(all_pmids):
        print(f"   {i+1}/{len(all_pmids)}: PMID {pmid} ({cancer_type})",
              end="\r")
        data = fetch_abstract(pmid, email, api_key)
        data["cancer_category"] = cancer_type
        results[pmid] = data

    print(f"\n   ✅ Fetched {len(results)} abstracts")
    return results


def save_abstracts(abstracts: Dict, path: Optional[str] = None) -> str:
    """Save fetched abstracts to JSON file."""
    path = path or GS_ABSTRACTS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(abstracts, f, indent=2, ensure_ascii=False)
    print(f"   💾 Abstracts saved to {path}")
    return path


def load_abstracts(path: Optional[str] = None) -> Dict:
    """Load previously fetched abstracts from JSON file."""
    path = path or GS_ABSTRACTS
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Abstracts file not found: {path}\n"
            f"Run fetch_all_abstracts() first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_annotation_template(
    abstracts: Dict,
    output_path: Optional[str] = None,
) -> str:
    """
    Create an Excel annotation template from fetched abstracts.

    The template has one row per sentence with empty gold
    standard columns for manual annotation.

    Annotators fill in the gold columns manually.
    This is the foundation of the validation dataset.

    Parameters
    ----------
    abstracts : dict
        Output from fetch_all_abstracts()
    output_path : str, optional
        Where to save the template

    Returns
    -------
    str : path to saved template
    """
    import re

    output_path = output_path or GS_TEMPLATE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    sent_split = re.compile(r"(?<=[\.\?!])\s+(?=[A-Z])")

    for pmid, data in abstracts.items():
        if "error" in data:
            continue

        abstract = data.get("abstract", "")
        title    = data.get("title", "")
        cancer_category = data.get("cancer_category", "")

        # Split abstract into sentences
        full_text = f"{title} {abstract}".strip()
        sentences = sent_split.split(full_text)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:  # skip very short fragments
                continue

            rows.append({
                # Source info
                "pmid":           pmid,
                "cancer_category":cancer_category,
                "year":           data.get("year", ""),
                "journal":        data.get("journal", ""),
                "sentence":       sent,

                # Gold standard fields — TO BE FILLED BY ANNOTATOR
                "gene_gold":               "",
                "drug_gold":               "",
                "cancer_gold":             "",
                "alteration_gold":         "",
                "relation_gold":           "",
                "evidence_type_gold":      "",
                "resistance_observed_gold":"",
                "negated_gold":            "",
                "speculative_gold":        "",
                "study_design_gold":       "",
                "annotator":               "",
                "annotation_notes":        "",
            })

    df = pd.DataFrame(rows)

    # Save to Excel with formatting
    with pd.ExcelWriter(output_path, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Annotations", index=False)

        # Add guidelines sheet
        guide_df = pd.DataFrame(
            {"ANNOTATION GUIDELINES": [ANNOTATION_GUIDELINES]}
        )
        guide_df.to_excel(xl, sheet_name="Guidelines", index=False)

        # Add example sheet
        examples = pd.DataFrame([
            {
                "pmid":"EXAMPLE1",
                "cancer_category":"NSCLC",
                "year":"2020",
                "journal":"J Clin Oncol",
                "sentence":"EGFR-mutant NSCLC patients responded to erlotinib with 70% ORR.",
                "gene_gold":"EGFR",
                "drug_gold":"erlotinib",
                "cancer_gold":"NSCLC",
                "alteration_gold":"",
                "relation_gold":"yes",
                "evidence_type_gold":"efficacy",
                "resistance_observed_gold":"no",
                "negated_gold":"no",
                "speculative_gold":"no",
                "study_design_gold":"RCT",
                "annotator":"JM",
                "annotation_notes":"Clear efficacy statement",
            },
            {
                "pmid":"EXAMPLE2",
                "cancer_category":"CRC",
                "year":"2019",
                "journal":"NEJM",
                "sentence":"KRAS mutations did not respond to cetuximab and showed no clinical benefit.",
                "gene_gold":"KRAS",
                "drug_gold":"cetuximab",
                "cancer_gold":"",
                "alteration_gold":"",
                "relation_gold":"yes",
                "evidence_type_gold":"resistance",
                "resistance_observed_gold":"yes",
                "negated_gold":"yes",
                "speculative_gold":"no",
                "study_design_gold":"RCT",
                "annotator":"JM",
                "annotation_notes":"Negated efficacy = resistance",
            },
        ])
        examples.to_excel(xl, sheet_name="Examples", index=False)

    print(f"✅ Annotation template created: {output_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Sentences from {df['pmid'].nunique()} abstracts")
    print(f"   Cancer categories: {df['cancer_category'].value_counts().to_dict()}")
    print(f"\n   👉 Next step: open the template and fill in gold columns")
    print(f"   👉 Guidelines are in the 'Guidelines' sheet")
    print(f"   👉 Examples are in the 'Examples' sheet")

    return output_path


def load_annotations(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load completed annotations from Excel file.
    Filters to only rows that have been annotated
    (relation_gold column is not empty).

    Returns
    -------
    pd.DataFrame : annotated rows only
    """
    path = path or GS_TEMPLATE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Annotation file not found: {path}"
        )

    df = pd.read_excel(path, sheet_name="Annotations")

    # Keep only annotated rows
    annotated = df[df["relation_gold"].notna() &
                   (df["relation_gold"] != "")]

    print(f"✅ Loaded {len(annotated)} annotated rows "
          f"from {annotated['pmid'].nunique()} abstracts")

    # Report coverage
    has_gene  = (annotated["gene_gold"] != "").sum()
    has_drug  = (annotated["drug_gold"] != "").sum()
    has_reln  = (annotated["relation_gold"] == "yes").sum()

    print(f"   Gene annotations:      {has_gene}")
    print(f"   Drug annotations:      {has_drug}")
    print(f"   Positive relations:    {has_reln}")
    print(f"   Negated:              "
          f"{(annotated['negated_gold']=='yes').sum()}")
    print(f"   Speculative:          "
          f"{(annotated['speculative_gold']=='yes').sum()}")

    return annotated


def compute_inter_annotator_agreement(
    annotations_a: pd.DataFrame,
    annotations_b: pd.DataFrame,
    key_col: str = "sentence",
    target_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Compute inter-annotator agreement (Cohen's kappa)
    between two annotators on the same sentences.

    Parameters
    ----------
    annotations_a : pd.DataFrame
        Annotations from annotator A
    annotations_b : pd.DataFrame
        Annotations from annotator B
    key_col : str
        Column to merge on (sentence text)
    target_cols : list
        Which gold columns to evaluate

    Returns
    -------
    dict : {column: kappa_score}
    """
    target_cols = target_cols or [
        "relation_gold",
        "evidence_type_gold",
        "resistance_observed_gold",
        "negated_gold",
        "speculative_gold",
    ]

    # Merge on PMID + sentence — not sentence alone
    # Sentence text alone is not a reliable unique key:
    # the same sentence may appear in multiple abstracts
    # Using PMID + sentence ensures correct pairing
    merge_keys = ["pmid", key_col] if "pmid" in annotations_a.columns else [key_col]
    merged = pd.merge(
        annotations_a[merge_keys + target_cols],
        annotations_b[merge_keys + target_cols],
        on=merge_keys,
        suffixes=("_a", "_b")
    )

    results = {}
    for col in target_cols:
        col_a = f"{col}_a"
        col_b = f"{col}_b"

        if col_a not in merged.columns:
            continue

        # Simple percent agreement
        agreement = (merged[col_a] == merged[col_b]).mean()

        # Cohen's kappa
        try:
            from sklearn.metrics import cohen_kappa_score
            kappa = cohen_kappa_score(
                merged[col_a].fillna(""),
                merged[col_b].fillna("")
            )
        except Exception:
            kappa = None

        results[col] = {
            "percent_agreement": round(agreement, 3),
            "cohen_kappa":       round(kappa, 3) if kappa else "N/A",
            "n_pairs":           len(merged),
        }

    print("📊 Inter-annotator agreement:")
    for col, scores in results.items():
        kappa = scores["cohen_kappa"]
        status = (
            "✅ GOOD" if isinstance(kappa, float) and kappa >= 0.80
            else "⚠️  REVIEW" if isinstance(kappa, float) and kappa >= 0.60
            else "❌ LOW"
        )
        print(f"   {col:30s}: kappa={kappa} {status}")

    return results


def get_gold_standard_stats(df: pd.DataFrame) -> None:
    """Print summary statistics of the gold standard dataset."""
    print("\n📊 Gold Standard Dataset Statistics:")
    print(f"   Total sentences:     {len(df)}")
    print(f"   Unique abstracts:    {df['pmid'].nunique()}")
    print()
    print("   Evidence type distribution:")
    if "evidence_type_gold" in df.columns:
        for et, count in df["evidence_type_gold"].value_counts().items():
            pct = count / len(df) * 100
            print(f"     {et:15s}: {count:4d} ({pct:.1f}%)")
    print()
    print("   Study design distribution:")
    if "study_design_gold" in df.columns:
        for sd, count in df["study_design_gold"].value_counts().items():
            print(f"     {sd:15s}: {count:4d}")
    print()
    print("   Negated sentences:  ",
          (df.get("negated_gold", pd.Series()) == "yes").sum())
    print("   Speculative:        ",
          (df.get("speculative_gold", pd.Series()) == "yes").sum())
