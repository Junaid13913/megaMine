#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mega_mine_free_plus_v7_3_11.py
(v7.3.11: Robust I/O — ensure parent directory exists before saving outputs; write an Excel file even when no rows are extracted.
All other behavior, flags, regex, outputs, and schemas remain unchanged vs v7.3.10.)

Includes earlier updates:
- Publisher-based predatory filter
- Fusion regex extended (::, /, -, tolerates alpha/beta)
- Journal backfill via ESummary
- Reference genome (hg19/hg38/GRCh37/GRCh38.p13) → reference_genome column
- TMB value/unit/state → TMB_value, TMB_unit, TMB_state columns
- MSI state → MSI_state column
- CNA/Fusion synonym normalization columns: cna_normalized, fusion_normalized
- Regulatory authority mentions: FDA / PMDA / MFDS → drug_accessibility column
- Year-binned ESearch pagination in 10k chunks with clipped-count handling (9999/10000) and no double date-filter injection
"""

from __future__ import annotations
import argparse, concurrent.futures as cf, csv, gzip, json, os, re, time
from collections import OrderedDict, Counter
from datetime import datetime
from math import ceil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

# -------------------- tqdm (progress bars) --------------------
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    class tqdm:  # type: ignore
        def __init__(self, it=None, total=None, desc=None, unit=None, leave=True): self.it=it
        def __iter__(self):
            for x in (self.it or []): yield x
        def update(self, n=1): pass
        def close(self): pass

UA = "mega-mine-free/7.3.11 (+https://example.org)"
HEADERS_DEFAULT = {"User-Agent": UA}
REQUEST_TIMEOUT = 45
# v2.0 BUG 3 FIX: rate limiting was too aggressive (Reviewer 1)
# NCBI allows max 10 req/sec with API key, 3 req/sec without
# OLD: 0.02s sleep + 8 workers = ~400 req/sec (way too fast)
# NEW: 0.12s sleep + 4 workers = ~8 req/sec (safe with API key)
SLEEP_BETWEEN = 0.12
MAX_WORKERS = int(os.environ.get("MM_MAX_WORKERS", "4"))
EN_DASH = "–"

# -------------------- Endpoints --------------------
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_ESEARCH = f"{NCBI_EUTILS}/esearch.fcgi"
PUBMED_ESUMMARY = f"{NCBI_EUTILS}/esummary.fcgi"
PUBMED_EFETCH = f"{NCBI_EUTILS}/efetch.fcgi"
PTC_EXPORT = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson"
EPMC_ANN = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"
EPMC_FTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
HGNC_APPROVED = "https://rest.genenames.org/fetch/status/Approved"

# -------------------- Regex & dictionaries --------------------
GENE_TOKEN = re.compile(r"\b[A-Z0-9]{2,12}\b")

# Protein-level AA substitutions (kept as-is)
AA_SNV     = re.compile(r"\b[A-Z][0-9]{1,4}[A-Z]\b")
AA_SNV_3L  = re.compile(r"\bp\.\(?([A-Z][a-z]{2})(\d{1,4})([A-Z][a-z]{2})\)?")

# Existing nucleotide change (kept)
NUC_CHANGE = re.compile(r"\bc\.\d+[+-]?\d*[ACGT]>[ACGT]\b", re.I)

# Exon-level patterns (kept)
EXON_EVENT = re.compile(r"\bex(?:on)?\s*(\d{1,2})\s*(del(?:etion)?|ins(?:ertion)?)\b", re.I)
E_DEL      = re.compile(r"\bE(\d{1,2})del\b")

# Fusion delimiters & suffix handling (extended earlier)
FUSION_PAIR = re.compile(
    r"\b([A-Z0-9]{2,12})(?:\s+(alpha|beta))?\s*(::|[-–/:])\s*([A-Z0-9]{2,12})(?:\s+(alpha|beta))?\b"
)

SENT_SPLIT = re.compile(r"(?<=[\.\?!;])\s+(?=[A-Z])")

# -------------------- NEW: appended nucleotide-level rules --------------------
NUC_CDNA_SIMPLE = re.compile(r"\bc\.\d+[ACGT]>[ACGT]\b", re.I)
SPLICE_IVS      = re.compile(r"\bIVS\d+\+\d+[ACGT]>[ACGT]\b", re.I)
NUC_DELINS      = re.compile(r"\bc\.\d+(?:_\d+)?delins[ACGT]+\b", re.I)
NUC_DUP         = re.compile(r"\bc\.\d+(?:_\d+)?dup[ACGT]*\b", re.I)
NUC_DEL         = re.compile(r"\bc\.\d+(?:_\d+)?del[ACGT]*\b", re.I)

# -------------------- Drugs --------------------
DRUG_WORD  = re.compile(
    r"\b([A-Za-z][a-z0-9-]*(?:mab|limab|zumab|umab|nib|tinib|ciclib|parib|platin|tecan|taxel|poside|zomib|trexate|rubicin|rafenib|lisib|degib|cetinib|metinib))\b",
    re.I
)
DRUG_INHIB = re.compile(r"\b([A-Za-z][a-z0-9-]{2,})\s+inhibitor(s)?\b", re.I)
DRUG_STOPWORDS = {"kinase","specific","molecule","small-molecule","antibody","antibodies",
                  "therapy","therapies","agent","agents","drug","drugs","treatment"}

# Context words
EFFICACY_VERBS = re.compile(
    r"(treated with|received|responded to|response to|sensitive to|sensitivity to|benefit(ed)? from|"
    r"efficac(y|ious)|activity against|showed (?:efficacy|activity)|achieved|ORR|PFS|OS|survival benefit|"
    r"partial response|complete response|disease control|DCR)", re.I
)
SAFETY_TERMS = re.compile(r"(toxicit(y|ies)|adverse event|ae[s]?|safety|tolerability|bleeding|thrombosis|hepatotoxicity|rash|neutropenia)", re.I)
REVIEW_TERMS = re.compile(r"\b(review|meta[- ]analysis|systematic review|pooled analysis)\b", re.I)

TRIAL_PAT = re.compile(r"\btrial\b|\bphase\s+[I1]{1,4}\b|\bphase\s+[1-4]\b", re.I)
RCT_PAT   = re.compile(r"randomi[sz]ed|double[- ]blind|placebo[- ]controlled", re.I)
OBS_PAT   = re.compile(r"(observational|prospective|retrospective|cohort|case[- ]control|registry)", re.I)
CASE_REPORT_PAT = re.compile(r"\bcase report(s)?\b", re.I)
IN_VITRO  = re.compile(r"\b(in\s+vitro|cell[- ]line|organoid)\b", re.I)
PRECLIN   = re.compile(r"\b(in\s+vivo|xenograft|murine|mouse|mice|rat|hamster|zebrafish)\b", re.I)
IN_SILICO = re.compile(r"\b(in[- ]silico|computational|bioinformatic[s]?)\b", re.I)
PHASE_PAT = re.compile(r"\bphase\s*(?:(I{1,4})|([1-4]))\b", re.I)

SETTING_PAT = re.compile(r"\b(adjuvant|neoadjuvant|maintenance|perioperative|metastatic)\b", re.I)
LINE_PAT    = re.compile(r"\b(1L|2L|3L|4L|first[- ]line|second[- ]line|third[- ]line|fourth[- ]line)\b", re.I)
STAGE_TXT   = re.compile(r"\bstage\s+([ivx]{1,4}|\d{1,2})[abcd]?\b", re.I)
TNM_PAT     = re.compile(r"\bT(\d)[abc]?\s*N(\d)[abc]?\s*M([01])[abc]?\b", re.I)
RESECT_PAT  = re.compile(r"\b(resectable|unresectable)\b", re.I)
META_PAT    = re.compile(r"\b(metastatic|advanced|locally advanced|recurrent)\b", re.I)

OMICS_PAT   = re.compile(r"(RNA[- ]?seq|WES|WGS|NGS|scRNA|ATAC[- ]?seq|IHC|proteomic|proteome|single[- ]cell|MS/MS|mass[- ]spectrometry)", re.I)

IMMUNE_PAT  = re.compile(
    r"(PD[- ]?L1|CD274|TMB|tumou?r\s+mutation\s+burden|MSI|microsatellite\s+instability|MSS|dMMR|mismatch\s+repair|CTLA-4|CD8\+\s*T)",
    re.I
)

RESIST_PAT  = re.compile(r"(resistan(t|ce)|non[- ]?respond(er|ing)|refractory|progression on|progressed on)", re.I)
RESPONSE_PAT= re.compile(r"(response|responded|PR|CR|partial response|complete response|ORR|DCR|benefit|shrinkage)", re.I)

# Cancer normalization
from collections import OrderedDict as OD
ACRONYM_MAP = OD([
    ("non-small cell lung cancer", "NSCLC"),
    ("small cell lung cancer", "SCLC"),
    ("lung adenocarcinoma", "LUAD"),
    ("lung squamous cell carcinoma", "LUSC"),
    ("colorectal cancer", "CRC"),
    ("rectal cancer", "CRC"),
    ("colon cancer", "CRC"),
    ("hepatocellular carcinoma", "HCC"),
    ("pancreatic ductal adenocarcinoma", "PDAC"),
    ("pancreatic cancer", "PC"),
    ("cholangiocarcinoma", "CCA"),
    ("biliary tract cancer", "BTC"),
    ("triple-negative breast cancer", "TNBC"),
    ("breast cancer", "BC"),
    ("ovarian cancer", "OC"),
    ("endometrial cancer", "EC"),
    ("prostate cancer", "PCa"),
    ("renal cell carcinoma", "RCC"),
    ("urothelial carcinoma", "UC"),
    ("gastric cancer", "GC"),
    ("esophageal squamous cell carcinoma", "ESCC"),
    ("esophageal adenocarcinoma", "EAC"),
    ("glioblastoma", "GBM"),
    ("glioma", "glioma"),
    ("melanoma", "melanoma"),
    ("head and neck squamous cell carcinoma", "HNSCC"),
])
ACRONYM_SHORT = re.compile(r"\b(NSCLC|SCLC|LUAD|LUSC|CRC|TNBC|HCC|GBM|RCC|UC|PDAC|CCA|BTC|GC|EAC|ESCC|PCa|BC|OC|EC|HNSCC|PC)\b", re.I)
SYNONYMS = {
    "hnscc": "head and neck squamous cell carcinoma", "nsclc": "non-small cell lung cancer",
    "sclc": "small cell lung cancer", "luad": "lung adenocarcinoma", "lusc": "lung squamous cell carcinoma",
    "crc": "colorectal cancer", "hcc": "hepatocellular carcinoma", "pdac": "pancreatic ductal adenocarcinoma",
    "tnbc": "triple-negative breast cancer", "rcc": "renal cell carcinoma", "uc": "urothelial carcinoma",
    "gbm": "glioblastoma", "cca": "cholangiocarcinoma", "btc": "biliary tract cancer", "gc": "gastric cancer",
    "eac": "esophageal adenocarcinoma", "escc": "esophageal squamous cell carcinoma", "pc": "pancreatic cancer",
    "pca": "prostate cancer", "bc": "breast cancer", "oc": "ovarian cancer", "ec": "endometrial cancer",
}
CANCER_SUFFIX = re.compile(
    r"\b([A-Za-z][A-Za-z -]*?(?:carcinoma|sarcoma|leukemia|lymphoma|melanoma|glioma|tumou?r|neoplasm|cancer))\b",
    re.I
)
CANCER_CLAUSE = '(neoplasms[MeSH Terms] OR cancer[tiab] OR neoplasm*[tiab] OR tumor*[tiab] OR tumour*[tiab] OR carcinoma*[tiab] OR sarcoma*[tiab] OR leukemia*[tiab] OR lymphoma*[tiab] OR melanoma*[tiab])'

# -------------------- Reference genome / TMB / MSI regexes --------------------
REF_GENOME_PAT = re.compile(
    r"""
    \b(
        hg[\s-]?19
      | hg[\s-]?38
      | grch[\s-]?37
      | grch[\s-]?38
        (?:\s*\(?p\.?\s*\d+\)?|p\d+)?      # optional patch: p13, (p13), p.13
      | build[\s-]?(37|38)
      | b(37|38)
    )\b
    """,
    re.I | re.X
)

TMB_PAT = re.compile(
    r"\bTMB\b\s*[:=]?\s*~?\s*(\d+(?:\.\d+)?)\s*(mut(?:ations)?\/?\s*Mb|per\s*Mb)\b"
    r"|"
    r"\b(\d+(?:\.\d+)?)\s*(mut(?:ations)?\/?\s*Mb|per\s*Mb)\b",
    re.I
)
TMB_STATE_PAT = re.compile(r"\bTMB[-\s]?(high|low)\b|\b(high|low)\s+TMB\b", re.I)
MSI_PAT = re.compile(r"\bMSI[-\s]?(high|low|stable)\b|\bMSS\b", re.I)

# -------------------- Regulatory authority detection (FDA/PMDA/MFDS) --------------------
AUTHORITY_PAT = re.compile(
    r"""
    \bFDA\b|\bFood\s+and\s+Drug\s+Administration\b
    |
    \bPMDA\b|\bPharmaceuticals?\s+and\s+Medical\s+Devices\s+Agency\b
    |
    \bMFDS\b|\bMinistry\s+of\s+Food\s+and\s+Drug\s+Safety\b|\bKorean\s+MFDS\b
    """,
    re.I | re.X
)

def normalize_authority_token(tok: str) -> Optional[str]:
    s = (tok or "").strip().lower()
    if not s: return None
    if s == "fda" or "food and drug administration" in s:
        return "FDA"
    if s == "pmda" or "pharmaceuticals" in s:
        return "PMDA"
    if s == "mfds" or "ministry of food and drug safety" in s or "korean mfds" in s:
        return "MFDS"
    return None

def extract_regulatory_authorities(text: str) -> str:
    if not text: return ""
    vals = []
    for m in AUTHORITY_PAT.finditer(text or ""):
        lab = normalize_authority_token(m.group(0))
        if lab: vals.append(lab)
    if not vals: return ""
    out = list(OrderedDict.fromkeys(vals).keys())
    return "; ".join(out)

# -------------------- CNA / Fusion synonym lexicon --------------------
CNA_SYNONYMS: Dict[str, str] = {
    "amplification": "Amplification",
    "gain": "Amplification",
    "duplication": "Amplification",
    "loss": "Loss",
    "deletion": "Loss",
    "homozygous deletion": "Loss",
    "fusion": "Fusion/Rearrangement",
    "gene fusion": "Fusion/Rearrangement",
    "frameshift variant": "Fusion/Rearrangement",
    "translocation": "Fusion/Rearrangement",
    "directional gene fusion": "Fusion/Rearrangement",
}

def normalize_cna_term(term: str) -> str:
    if not term:
        return term
    t = re.sub(r"\s+", " ", term.strip().lower())
    return CNA_SYNONYMS.get(t, term)

# -------------------- Utilities --------------------
def http_get(url, params=None, headers=None, timeout=REQUEST_TIMEOUT):
    time.sleep(SLEEP_BETWEEN)
    return requests.get(url, params=params, headers=headers or HEADERS_DEFAULT, timeout=timeout)

def batched(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i:i+size])

def first_year_in(*vals) -> int:
    for v in vals:
        if not v: continue
        m = re.search(r"(\d{4})", str(v))
        if m: return int(m.group(1))
    return 0

def norm_section(lbl: str) -> str:
    u = (lbl or "").upper()
    if "ABSTRACT" in u or "TITLE" in u: return "Abstract"
    if "INTRO" in u or "BACKGROUND" in u: return "Introduction"
    if "METHOD" in u or "MATERIAL" in u: return "Methods"
    if "RESULT" in u or "FINDING" in u: return "Results"
    if "DISCUSSION" in u or "CONCLUSION" in u: return "Discussion"
    if "FIG" in u: return "Figure"
    if "TABLE" in u: return "Table"
    if "SUPPLEMENT" in u or "SUPP" in u: return "Supplementary"
    return "Abstract"

def map_to_primary_section(lbl: str) -> str:
    s = norm_section(lbl)
    if s in {"Abstract", "Methods", "Results", "Discussion"}: return s
    if s in {"Figure", "Table", "Supplementary"}: return "Results"
    if s == "Introduction": return "Abstract"
    return "Results"

def therapy_class(drug: str) -> str:
    if not drug: return ""
    d = drug.lower()
    if d.endswith(("mab","limab","zumab","umab")): return "immunotherapy"
    if d.endswith(("platin","taxel","rubicin","tecan","poside","trexate")): return "chemotherapy"
    if d.endswith(("nib","tinib","ciclib","parib","lisib","degib","cetinib","metinib","rafenib")) or "inhibitor" in d:
        return "targeted"
    return "other"

def aa3_to1(a3: str) -> str:
    conv = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
            "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V",
            "Ter":"*","Stop":"*"}
    return conv.get(a3, "")

def resolve_journal(esum_meta: dict, ef_fb: dict) -> str:
    j = (esum_meta.get("fulljournalname") or
         esum_meta.get("source") or
         ef_fb.get("journal") or "").strip()
    return j if j else "Unknown Journal"

def resolve_publisher(esum_meta: dict, ef_fb: dict) -> str:
    p = (esum_meta.get("publisher") or ef_fb.get("publisher") or "").strip()
    return p

def load_list_lower(path: Optional[str]) -> List[str]:
    vals: List[str] = []
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                t = line.strip()
                if t:
                    vals.append(t.lower())
    return vals

def ensure_parent_dir_for(path: str) -> None:
    """Create parent directory of a file path if it doesn't exist."""
    # v2.0 BUG 4 FIX: bare except pass silently swallowed all errors
    # (Reviewer 1 — system suppresses export errors)
    # Now logs a warning so users know when directory creation fails
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
    except OSError as e:
        print(f"⚠️  Warning: could not create directory for {path}: {e}")
    except Exception as e:
        print(f"⚠️  Unexpected error creating directory for {path}: {e}")

# -------------------- HGNC registry --------------------
class GeneRegistry:
    # v2.0 FIX: 2-letter symbols that are NOT real oncogenes
    # These appear in HGNC but are common abbreviations that get
    # false-matched in biomedical text (PC, HR, SI, F3, etc.)
    _FALSE_POSITIVES = {
        "SI","HR","PC","AR","AN","AS","AT","CA","CD","CS",
        "CT","CL","CM","CR","DC","DR","DM","DS","EP","ER",
        "ET","EX","FA","FB","FC","FN","FP","GC","GD","GH",
        "GL","GM","GN","GP","GR","GS","HB","HC","HD","HF",
        "IC","ID","IF","LA","LB","LC","MA","MB","MC","MD",
        "MS","MT","MU","MY","PA","PB","PR","PT","PV","SA",
        "SC","SD","SN","SP","ST","SV","TA","TB","TC","TD",
        "TE","TF","TG","TH","TI","TK","TL","TM","TN","TO",
        "TQ","TR","TS","TT","TU","TV","TW","TX","TY","F3",
        "DCR","CON","AKA","DCA",
        # Common words that match HGNC symbols but are not oncogenes
        "CCK","EGF","CBL","PPL","H19","USP","EML","SCRN",
        "LDOC","PTK","ROR","AP1","EML4",
    }
    # Valid short oncogene symbols — always keep these
    _VALID_SHORT = {
        "MET","ALK","RET","RAS","RAF","AKT","MYC","BCL","ABL",
        "JAK","KIT","FLT","CDK","MDM","RB1","NF1","NF2","VHL",
        "APC","ATM","ATR","WNT","SRC","YES","FOS","JUN","MAP",
    }

    def __init__(self, symbols: Iterable[str]):
        self.symbols = set(s.upper() for s in symbols if self._ok(s))

    @staticmethod
    def _ok(s: str) -> bool:
        u = re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()
        if not (bool(u) and not u.isdigit() and
                re.match(r"^[A-Z][A-Z0-9]{1,}$", u)):
            return False
        # Reject known false positives
        if u in GeneRegistry._FALSE_POSITIVES:
            return False
        # Reject 2-letter symbols not in known oncogene list
        if len(u) == 2 and u not in GeneRegistry._VALID_SHORT:
            return False
        return True
    @classmethod
    def from_hgnc_api(cls, cache_path: Optional[str] = None, refresh=False, timeout=60):
        docs = None
        if cache_path and os.path.exists(cache_path) and not refresh:
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    js = json.load(fh); docs = js.get("response", {}).get("docs", [])
            except Exception: pass
        if docs is None:
            r = http_get(HGNC_APPROVED, headers={"Accept":"application/json"}, timeout=timeout); r.raise_for_status()
            js = r.json(); docs = js.get("response", {}).get("docs", [])
            if cache_path:
                try:
                    with open(cache_path, "w", encoding="utf-8") as fh: json.dump(js, fh)
                except Exception: pass
        syms = [d.get("symbol","") for d in docs if d.get("symbol")]
        return cls(syms)
    def is_valid(self, s: str) -> bool:
        return re.sub(r"[^A-Za-z0-9]", "", s or "").upper() in self.symbols
    def normalize(self, s: str) -> str:
        u = re.sub(r"[^A-Za-z0-9]", "", s or "").upper()
        return u if u in self.symbols else ""

# -------------------- Drugs --------------------
DEFAULT_ONCO_DRUGS = {
    "gefitinib","erlotinib","afatinib","dacomitinib","osimertinib","lazertinib",
    "crizotinib","alectinib","ceritinib","brigatinib","lorlatinib",
    "entrectinib","selpercatinib","pralsetinib","capmatinib","tepotinib","larotrectinib",
    "sotorasib","adagrasib","imatinib","dasatinib","nilotinib","bosutinib","ponatinib",
    "dabrafenib","trametinib","vemurafenib","encorafenib","binimetinib",
    "tucatinib","neratinib","lapatinib","trastuzumab","pertuzumab","trastuzumab deruxtecan",
    "pembrolizumab","nivolumab","atezolizumab","durvalumab","avelumab","cemiplimab","ipilimumab",
    "docetaxel","paclitaxel","carboplatin","cisplatin","pemetrexed","gemcitabine","oxaliplatin","irinotecan",
    "lenvatinib","sorafenib","regorafenib","pazopanib","sunitinib","everolimus","temsirolimus",
    "ibrutinib","acalabrutinib","zanubrutinib",
    "olaparib","niraparib","rucaparib","talazoparib",
}
def load_drug_whitelist(path: Optional[str]) -> Dict[str,str]:
    """
    Load drug whitelist. Priority:
    1. Explicit path if provided
    2. Auto-detect drug_whitelist_v2.json next to extractor.py
    3. Fall back to DEFAULT_ONCO_DRUGS (64 drugs)
    """
    import json as _json

    out = {}

    # Try explicit path
    if path and os.path.exists(path):
        if path.lower().endswith(".csv"):
            import csv as _csv
            with open(path, newline="", encoding="utf-8") as fh:
                for row in _csv.DictReader(fh):
                    name = (row.get("name") or row.get("Name") or "").strip()
                    if name: out[name.lower()] = name
        elif path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
                drugs = data.get("drugs", data if isinstance(data, list) else [])
                for n in drugs:
                    if n: out[str(n).lower()] = str(n)
                # Add synonyms
                for brand, generic in data.get("synonyms", {}).items():
                    if brand: out[brand.lower()] = generic
        else:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    name = line.strip()
                    if name: out[name.lower()] = name

    # Auto-detect v2 whitelist alongside this file
    if not out:
        v2_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "data", "drug_whitelist_v2.json"
        )
        v2_path = os.path.normpath(v2_path)
        if os.path.exists(v2_path):
            try:
                with open(v2_path, "r", encoding="utf-8") as fh:
                    data = _json.load(fh)
                    for n in data.get("drugs", []):
                        if n: out[str(n).lower()] = str(n)
                    for brand, generic in data.get("synonyms", {}).items():
                        if brand: out[brand.lower()] = generic
                print(f"   📦 Auto-loaded drug_whitelist_v2.json ({len(out)} drugs)")
            except Exception as e:
                print(f"   ⚠️  Could not load v2 whitelist: {e}")

    # Final fallback
    if not out:
        for n in DEFAULT_ONCO_DRUGS:
            out[n.lower()] = n
        print(f"   ⚠️  Using DEFAULT_ONCO_DRUGS fallback ({len(out)} drugs)")

    return out

def filter_drugs(drugs: Iterable[str], whitelist: Dict[str,str]) -> List[str]:
    seen = set(); out = []
    for d in drugs:
        if not d: continue
        dl = d.lower()
        if dl in DRUG_STOPWORDS: continue
        if dl in whitelist and dl not in seen:
            seen.add(dl); out.append(whitelist[dl])
    return out

# -------------------- PubMed/EPMC/PubTator --------------------
def _query_has_date_filter(q: str) -> bool:
    return bool(re.search(r"\[(?:dp|pdat)\]", q or "", flags=re.I))

def esearch_year_with_pagination(q: str, year: int, email: Optional[str], api_key: Optional[str]) -> List[str]:
    def dbg(msg: str):
        if os.environ.get("MM_DEBUG") == "1":
            try:
                print(f"[MMDEBUG] {msg}")
            except Exception:
                pass

    def polite_delay():
        time.sleep(0.12 if api_key else 0.34)

    def fetch_page(base_params: dict, start: int, use_hist: bool, webenv: str = "", qk: str = "") -> List[str]:
        prm = dict(base_params)
        prm.update({"retstart": start, "retmax": 10000})
        if use_hist:
            prm.update({"WebEnv": webenv, "query_key": qk})
        polite_delay()
        r = http_get(PUBMED_ESEARCH, prm)
        if r.status_code != 200:
            dbg(f"page status={r.status_code} start={start}")
            return []
        try:
            return (r.json().get("esearchresult", {}) or {}).get("idlist", []) or []
        except Exception:
            dbg("page json parse error")
            return []

    def page_until_empty(base_params: dict, start0: int, use_hist: bool, webenv: str = "", qk: str = "") -> List[str]:
        out: List[str] = []
        start = start0
        while True:
            ids = fetch_page(base_params, start, use_hist, webenv, qk)
            if not ids:
                break
            out.extend(ids)
            start += 10000
        return out

    term = q if _query_has_date_filter(q) else f"({q}) AND ({year}/01/01:{year}/12/31[dp])"
    base = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "tool": "mega-mine-free",
        "email": email or "noreply@example.com",
        "sort": "pub+date",
        "usehistory": "y",
    }
    if api_key:
        base["api_key"] = api_key

    prm0 = dict(base); prm0["retmax"] = 0
    r0 = http_get(PUBMED_ESEARCH, prm0)
    js0 = {}
    if r0.status_code == 200:
        try:
            js0 = r0.json().get("esearchresult", {}) or {}
        except Exception:
            dbg("primary probe json parse error")

    try:
        count = int(js0.get("count", "0"))
    except Exception:
        count = 0
    webenv = js0.get("webenv") or ""
    qk = js0.get("querykey") or ""
    have_hist = bool(webenv and qk)

    def clipped(n: int) -> bool:
        return n >= 9999 and (n % 10000 == 0 or n == 9999)

    if count > 0:
        if have_hist:
            if clipped(count):
                return page_until_empty(base, 0, True, webenv, qk)
            else:
                out: List[str] = []
                for start in range(0, count, 10000):
                    out.extend(fetch_page(base, start, True, webenv, qk))
                return out
        else:
            dbg("primary probe missing WebEnv/QueryKey → manual paging")
            base_manual = dict(base); base_manual.pop("usehistory", None)
            if clipped(count):
                return page_until_empty(base_manual, 0, False)
            else:
                out: List[str] = []
                for start in range(0, count, 10000):
                    out.extend(fetch_page(base_manual, start, False))
                return out

    # Retry with param date filter
    base2 = {
        "db": "pubmed",
        "term": q,
        "retmode": "json",
        "tool": "mega-mine-free",
        "email": email or "noreply@example.com",
        "sort": "pub+date",
        "usehistory": "y",
        "datetype": "pdat",
        "mindate": f"{year}/01/01",
        "maxdate": f"{year}/12/31",
    }
    if api_key:
        base2["api_key"] = api_key

    prm0b = dict(base2); prm0b["retmax"] = 0
    r0b = http_get(PUBMED_ESEARCH, prm0b)
    js0b = {}
    if r0b.status_code == 200:
        try:
            js0b = r0b.json().get("esearchresult", {}) or {}
        except Exception:
            dbg("retry probe json parse error")

    try:
        count2 = int(js0b.get("count", "0"))
    except Exception:
        count2 = 0
    webenv2 = js0b.get("webenv") or ""
    qk2 = js0b.get("querykey") or ""
    have_hist2 = bool(webenv2 and qk2)

    if count2 <= 0:
        dbg("retry probe count==0 → giving up for this year")
        return []

    if have_hist2:
        if clipped(count2):
            return page_until_empty(base2, 0, True, webenv2, qk2)
        else:
            out: List[str] = []
            for start in range(0, count2, 10000):
                out.extend(fetch_page(base2, start, True, webenv2, qk2))
            return out

    dbg("retry probe missing WebEnv/QueryKey → manual paging")
    base2_manual = dict(base2); base2_manual.pop("usehistory", None)
    if clipped(count2):
        return page_until_empty(base2_manual, 0, False)
    else:
        out: List[str] = []
        for start in range(0, count2, 10000):
            out.extend(fetch_page(base2_manual, start, False))
        return out

def pubmed_esearch(q: str, years: Tuple[int,int], max_records: int, email: Optional[str], api_key: Optional[str]) -> List[str]:
    params = {"db":"pubmed","term":f"({q}) AND ({years[0]}/01/01:{years[1]}/12/31[dp])","retmode":"json",
              "retmax":min(max_records,100000),"tool":"mega-mine-free","email":email or "noreply@example.com","sort":"pub+date"}
    if api_key: params["api_key"] = api_key
    r = http_get(PUBMED_ESEARCH, params); r.raise_for_status()
    return (r.json().get("esearchresult", {}).get("idlist", []) or [])[:max_records]

def pubmed_esearch_year_binned(q: str, years: Tuple[int,int], per_year_max: int, email: Optional[str], api_key: Optional[str]) -> List[str]:
    ids, seen = [], set()

    def fetch_year(y: int) -> List[str]:
        # v2.0 BUG 2 FIX: per_year_max was never enforced (Reviewer 1)
        # Broad queries fetched ALL records per year — skewed toward
        # earlier years. Now capped here, not just at the end.
        try:
            results = esearch_year_with_pagination(q, y, email, api_key)
            return results[:per_year_max]
        except Exception:
            return []

    with cf.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 6)) as ex:
        for res in tqdm(ex.map(fetch_year, range(years[0], years[1]+1)),
                        total=years[1]-years[0]+1,
                        desc="ESearch (year bins)",
                        unit="yr"):
            for i in res:
                if i not in seen:
                    seen.add(i)
                    ids.append(i)
    return ids

def pubmed_esummary(pmids: List[str], api_key: Optional[str]) -> Dict[str, dict]:
    out = {}
    def do_chunk(chunk):
        prm = {"db":"pubmed","retmode":"json","id":",".join(chunk)}
        if api_key: prm["api_key"] = api_key
        try:
            r = http_get(PUBMED_ESUMMARY, prm)
            if r.status_code != 200: return {}
            dat = r.json().get("result", {})
            return {k:v for k,v in dat.items() if k != "uids"}
        except Exception:
            return {}
    chunks = list(batched(pmids, 500))
    with cf.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 4)) as ex:
        for res in tqdm(ex.map(do_chunk, chunks), total=len(chunks), desc="ESummary", unit="chunk"):
            out.update(res)
    return out

def _extract_abstract_sections(xml: str) -> List[Tuple[str,str]]:
    secs = []
    for m in re.finditer(r'<AbstractText([^>]*)>(.*?)</AbstractText>', xml, flags=re.S|re.I):
        attr, txt = m.group(1) or "", m.group(2) or ""
        lbl = ""
        labm = re.search(r'Label="([^"]+)"', attr) or re.search(r'NlmCategory="([^"]+)"', attr)
        if labm: lbl = labm.group(1).strip()
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt: secs.append((lbl or "ABSTRACT", txt))
    return secs

def _extract_mesh_terms(xml_article: str) -> List[str]:
    terms = []
    mh = re.search(r"<MeshHeadingList>(.*?)</MeshHeadingList>", xml_article, flags=re.S|re.I)
    if not mh: return terms
    for dm in re.finditer(r"<DescriptorName[^>]*>(.*?)</DescriptorName>", mh.group(1), flags=re.S|re.I):
        t = re.sub(r"\s+", " ", dm.group(1)).strip()
        if t: terms.append(t)
    return terms

def pubmed_efetch_abstracts(pmids: List[str], api_key: Optional[str]) -> Dict[str, dict]:
    out = {}
    def do_chunk(chunk):
        prm = {"db":"pubmed","retmode":"xml","id":",".join(chunk)}
        if api_key: prm["api_key"] = api_key
        try:
            r = http_get(PUBMED_EFETCH, prm)
            if r.status_code != 200: return {}
            txt = r.text
            articles = re.split(r"</PubmedArticle>\s*", txt)
            local = {}
            for art in articles:
                mpmid = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
                if not mpmid: continue
                pmid = mpmid.group(1)
                title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, flags=re.S)
                title = re.sub(r"\s+", " ", (title_m.group(1) if title_m else "")).strip()
                secs = _extract_abstract_sections(art)
                abst = " ".join(s for _, s in secs) if secs else ""
                y = ""
                mdate = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>.*?</PubDate>", art, flags=re.S)
                if mdate: y = mdate.group(1)
                doi = ""; pmcid = ""; publisher = ""; journal = ""
                for id_m in re.finditer(r'<ArticleId IdType="(doi|pmcid|pii)">(.*?)</ArticleId>', art):
                    idt = id_m.group(1).lower(); val = id_m.group(2).strip()
                    if idt == "doi": doi = val
                    elif idt == "pmcid": pmcid = val
                pm = re.search(r"<PublisherName>(.*?)</PublisherName>", art, flags=re.S|re.I)
                if pm: publisher = re.sub(r"\s+", " ", pm.group(1)).strip()
                jm = re.search(r"<Title>(.*?)</Title>", art, flags=re.S|re.I)
                if jm: journal = re.sub(r"\s+", " ", jm.group(1)).strip()
                mesh = _extract_mesh_terms(art)
                local[pmid] = {
                    "title": title, "abstract": abst, "sections": secs, "year": int(y or 0),
                    "doi": doi, "pmcid": pmcid, "mesh": mesh,
                    "publisher": publisher, "journal": journal
                }
            return local
        except Exception:
            return {}
    chunks = list(batched(pmids, 150))
    with cf.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 4)) as ex:
        for local in tqdm(ex.map(do_chunk, chunks), total=len(chunks), desc="EFetch", unit="chunk"):
            out.update(local)
    return out

def pubtator_for_pmids(pmids: List[str]) -> Dict[str, dict]:
    out = {}
    def do_chunk(chunk):
        try:
            r = http_get(PTC_EXPORT, {"pmids": ",".join(chunk)})
            if r.status_code != 200 or not r.text.strip(): return {}
            docs = json.loads(r.text)
            local = {}
            for d in docs:
                pmid = str(d.get("sourceid") or d.get("passages",[{"infons":{}}])[0].get("infons",{}).get("article-id") or "")
                if pmid: local[pmid] = d
            return local
        except Exception:
            return {}
    chunks = list(batched(pmids, 150))
    with cf.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 4)) as ex:
        for local in tqdm(ex.map(do_chunk, chunks), total=len(chunks), desc="PubTator", unit="chunk"):
            out.update(local)
    return out

def epmc_annotations_for_pmids(pmids: List[str]) -> Dict[str, List[dict]]:
    def one(pmid: str):
        prm = {"idType":"pmid","id": pmid,"source":"MED","format":"JSON","entity":"Gene_Protein,Disease,Chemical,Mutation"}
        try:
            r = http_get(EPMC_ANN, prm)
            if r.status_code != 200 or not r.text.strip(): return pmid, []
            return pmid, (r.json().get("annotations", []) or [])
        except Exception:
            return pmid, []
    out = {}
    with cf.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 8)) as ex:
        for pmid, anns in tqdm(ex.map(one, pmids), total=len(pmids), desc="EPMC ann", unit="pmid"):
            out[pmid] = anns
    return out

def epmc_fetch_fulltexts(pmcids: List[str]) -> Dict[str, List[Tuple[str,str]]]:
    def one(pmcid: str):
        try:
            url = EPMC_FTEXT.format(pmcid=pmcid)
            r = http_get(url)
            if r.status_code != 200 or not r.text.strip(): return pmcid, []
            xml = r.text
            secs: List[Tuple[str,str]] = []
            for m in re.finditer(r"<sec\b[^>]*>(.*?)</sec>", xml, flags=re.S|re.I):
                block = m.group(1) or ""
                t_m = re.search(r"<title[^>]*>(.*?)</title>", block, flags=re.S|re.I)
                title = re.sub(r"<[^>]+>", " ", t_m.group(1)).strip() if t_m else ""
                text = re.sub(r"<[^>]+>", " ", block).strip()
                if text: secs.append(((title or "SEC").upper(), re.sub(r"\s+", " ", text)))
            for fm in re.finditer(r"<fig\b[^>]*>(.*?)</fig>", xml, flags=re.S|re.I):
                cap_m = re.search(r"<caption\b[^>]*>(.*?)</caption>", fm.group(1), flags=re.S|re.I)
                cap = re.sub(r"<[^>]+>", " ", cap_m.group(1)).strip() if cap_m else ""
                if cap: secs.append(("FIGURE", re.sub(r"\s+", " ", cap)))
            for tm in re.finditer(r"<table-wrap\b[^>]*>(.*?)</table-wrap>", xml, flags=re.S|re.I):
                cap_m = re.search(r"<caption\b[^>]*>(.*?)</caption>", tm.group(1), flags=re.S|re.I)
                cap = re.sub(r"<[^>]+>", " ", cap_m.group(1)).strip() if cap_m else ""
                if cap: secs.append(("TABLE", re.sub(r"\s+", " ", cap)))
            return pmcid, secs
        except Exception:
            return pmcid, []
    out = {}
    with cf.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 6)) as ex:
        for pmcid, secs in tqdm(ex.map(one, pmcids), total=len(pmcids), desc="PMC fulltext", unit="pmcid"):
            out[pmcid] = secs
    return out

# -------------------- Extraction helpers --------------------
def extract_drugs(text: str) -> List[str]:
    raw = set(d.group(1) for d in DRUG_WORD.finditer(text or ""))
    raw |= set(d.group(1) for d in DRUG_INHIB.finditer(text or ""))
    return sorted(set(x.strip("-") for x in raw if x))

def extract_cancers_from_all_sources(title: str,
                                     sections: List[Tuple[str, str]],
                                     ptc_doc: Optional[dict],
                                     epmc_ann: Optional[List[dict]]) -> List[str]:
    def tidy_long(s: str) -> str:
        s = re.sub(r"\s+", " ", s.strip().lower()).replace("tumour", "tumor")
        s = SYNONYMS.get(s, s)
        return s
    out: List[str] = []
    seen: set = set()
    # v2.0 BUG 1 FIX: paper titles appeared as cancer names (Reviewer 1)
    # Three filters now validate every candidate cancer string
    TITLE_KEYWORDS = {
        "study", "trial", "assess", "efficacy", "safety",
        "patients", "treatment", "therapy", "analysis",
        "randomized", "double-blind", "placebo", "versus",
        "combination", "phase", "clinical", "novel", "approach",
        "method", "results", "outcome", "response", "testing",
        "reading", "scoring", "predict", "established", "means",
        "selecting", "optimal", "wild-type"
    }
    def add_long_or_acronym(s: str):
        if not s: return
        # Filter 1: max 6 words — real cancer names are short
        if len(s.strip().split()) > 6: return
        # Filter 2: no study/trial keywords — those are paper titles
        if any(kw in s.lower() for kw in TITLE_KEYWORDS): return
        # Filter 3: max 2 digits — cancer names rarely contain numbers
        if sum(c.isdigit() for c in s) > 2: return
        s_l = tidy_long(s)
        if s_l in SYNONYMS: s_l = tidy_long(SYNONYMS[s_l])
        short = ACRONYM_MAP.get(s_l)
        long_title = s_l.title()
        val = f"{long_title}; {short}" if short else long_title
        if val not in seen:
            seen.add(val); out.append(val)
    if ptc_doc:
        for p in ptc_doc.get("passages", []) or []:
            for a in p.get("annotations", []) or []:
                tpe = (a.get("infons", {}).get("type") or a.get("type") or "").lower()
                if tpe == "disease": add_long_or_acronym(a.get("text",""))
    if epmc_ann:
        for a in epmc_ann:
            if (a.get("type") or "").lower() == "disease":
                add_long_or_acronym(a.get("exact") or a.get("text") or "")
    full = " ".join([title or ""] + [t for _, t in (sections or [])])
    for m in CANCER_SUFFIX.finditer(full): add_long_or_acronym(m.group(1))
    for m in ACRONYM_SHORT.finditer(full): add_long_or_acronym(m.group(0))
    return out

def extract_mutation_signals(text: str) -> Tuple[set, set, List[Tuple[int,str]]]:
    snv = set(AA_SNV.findall(text or ""))
    for m in AA_SNV_3L.finditer(text or ""):
        a1 = aa3_to1(m.group(1)); pos = m.group(2); a2 = aa3_to1(m.group(3))
        if a1 and a2: snv.add(f"{a1}{pos}{a2}")

    nucs = set(NUC_CHANGE.findall(text or ""))
    for m in NUC_CDNA_SIMPLE.finditer(text or ""): nucs.add(m.group(0))
    for m in SPLICE_IVS.finditer(text or ""):      nucs.add(m.group(0))
    for m in NUC_DELINS.finditer(text or ""):      nucs.add(m.group(0))
    for m in NUC_DUP.finditer(text or ""):         nucs.add(m.group(0))
    for m in NUC_DEL.finditer(text or ""):         nucs.add(m.group(0))

    exons = []
    for m in EXON_EVENT.finditer(text or ""):
        exons.append((int(m.group(1)), "deletion" if m.group(2).lower().startswith("del") else "insertion"))
    for m in E_DEL.finditer(text or ""):
        exons.append((int(m.group(1)), "deletion"))
    return snv, nucs, exons

def _strip_gene_suffix(g: str) -> str:
    return re.sub(r"\s+(alpha|beta)$", "", g, flags=re.I)

def canonical_fusions_hgnc(text: str, registry: GeneRegistry) -> List[Tuple[str, List[str]]]:
    out = []
    seen_pairs = set()
    for m in FUSION_PAIR.finditer(text or ""):
        a_raw, delim, b_raw = m.group(1), m.group(3), m.group(4)
        a = _strip_gene_suffix(a_raw.upper())
        b = _strip_gene_suffix(b_raw.upper())
        a_ok, b_ok = registry.is_valid(a), registry.is_valid(b)
        if a_ok and b_ok:
            x, y = sorted([a,b])
            key = (x,y)
            if key not in seen_pairs:
                seen_pairs.add(key)
                out.append((f"{x}{EN_DASH}{y}", [x,y]))
        elif a_ok ^ b_ok:
            valid = a if a_ok else b
            out.append(("", [valid]))
    for g in list(registry.symbols):
        pat = rf"\b{re.escape(g)}(?:\s+(alpha|beta))?\b[^\.\n]{{0,60}}\bfusion\b"
        if re.search(pat, text or "", re.I):
            out.append(("", [g]))
    return out

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text: return []
    return re.split(SENT_SPLIT, text)

def classify_sentence_context(sent: str, is_review_like: bool) -> str:
    if SAFETY_TERMS.search(sent):
        return "toxicity" if re.search(r"toxicit", sent, re.I) else "safety"
    if EFFICACY_VERBS.search(sent):
        return "efficacy"
    if is_review_like and REVIEW_TERMS.search(sent):
        return "review"
    return "background"

def evidence_priority(et: str) -> int:
    return {"efficacy": 5, "safety": 3, "toxicity": 3, "review": 2, "background": 1}.get(et or "", 0)

# --------------- Patient parsing ---------------
AGE_PAT1  = re.compile(r"\bage\s*(\d{1,3})\s*y", re.I)
AGE_PAT2  = re.compile(r"(\d{1,3})\s*[-]?\s*year[- ]old", re.I)
SEX_M_PAT = re.compile(r"\b(male|man)\b", re.I)
SEX_F_PAT = re.compile(r"\b(female|woman)\b", re.I)

def parse_patient_age_sex(text: str) -> Tuple[str, str]:
    age = ""
    m = AGE_PAT1.search(text or "") or AGE_PAT2.search(text or "")
    if m:
        try:
            val = int(m.group(1))
            if 0 < val < 120:
                age = str(val)
        except Exception:
            pass
    sex = ""
    if SEX_M_PAT.search(text or ""): sex = "male"
    if SEX_F_PAT.search(text or ""): sex = "female" if not sex else sex
    return age, sex

# -------------------- Reference genome / TMB / MSI helpers --------------------
def extract_reference_genome(text: str) -> str:
    if not text:
        return ""
    hits = [m.group(0) for m in REF_GENOME_PAT.finditer(text)]
    if not hits:
        return ""
    normed = []
    for s in hits:
        s = s.strip()
        m_build = re.search(r"\b(?:build|b)\s*?(\d+)\b", s, re.I)
        if m_build:
            num = m_build.group(1)
            if num == "38": normed.append("GRCh38"); continue
            if num == "37": normed.append("GRCh37"); continue
        if re.search(r"\bhg[\s-]?38\b", s, re.I): normed.append("hg38"); continue
        if re.search(r"\bhg[\s-]?19\b", s, re.I): normed.append("hg19"); continue
        m_grch = re.search(r"\bgrch[\s-]?(37|38)(?:\s*\(?p\.?\s*(\d+)\)?|p(\d+))?\b", s, re.I)
        if m_grch:
            ver = m_grch.group(1)
            p1 = m_grch.group(2); p2 = m_grch.group(3)
            if ver == "38":
                if p1 or p2:
                    patch = p1 or p2
                    normed.append(f"GRCh38.p{patch}")
                else:
                    normed.append("GRCh38")
            else:
                normed.append("GRCh37")
    if not normed:
        return ""
    def score(x: str) -> int:
        if x.startswith("GRCh38.p"): return 5
        return {"GRCh38": 4, "hg38": 3, "GRCh37": 2, "hg19": 1}.get(x, 0)
    normed.sort(key=score, reverse=True)
    return normed[0]

def extract_tmb(text: str) -> Tuple[str, str, str]:
    if not text: return "", "", ""
    value = ""; unit = ""; state = ""
    m_state = TMB_STATE_PAT.search(text or "")
    if m_state:
        s1, s2 = m_state.group(1), m_state.group(2)
        st = (s1 or s2 or "").lower()
        if st in {"high","low"}:
            state = st
    for m in TMB_PAT.finditer(text or ""):
        num = m.group(1) or m.group(3)
        uni = m.group(2) or m.group(4)
        if num and uni:
            value = num
            u = uni.lower().replace("mutations", "mut").replace(" ", "")
            unit = "per Mb" if "permb" in u else "mut/Mb"
            break
    return value, unit, state

def extract_msi_state(text: str) -> str:
    if not text: return ""
    states = []
    for m in MSI_PAT.finditer(text or ""):
        if re.search(r"\bMSS\b", m.group(0)): states.append("MSS"); continue
        g = (m.group(1) or "").lower()
        if g in {"high","low","stable"}: states.append(g)
    if not states: return ""
    if "MSS" in states: return "MSS"
    if "high" in states: return "high"
    if "low" in states: return "low"
    return "stable"

# -------------------- Evidence extraction helpers --------------------
def section_sentence_hits(title: str, sections: List[Tuple[str,str]], genes: Iterable[str], drugs: Iterable[str], is_review_like: bool):
    blocks: List[Tuple[str, str]] = []
    if title:
        blocks.append(("Abstract", title))
    for lbl, txt in sections or []:
        lab_norm = map_to_primary_section(lbl)
        blocks.append((lab_norm, txt))

    for sec, txt in blocks:
        candidates: List[Tuple[int, str, str, str, str]] = []
        for sent in split_sentences(txt):
            for g in genes:
                if re.search(rf"\b{re.escape(g)}\b", sent):
                    for d in drugs:
                        if re.search(rf"\b{re.escape(d)}\b", sent, re.I):
                            et = classify_sentence_context(sent, is_review_like)
                            candidates.append((evidence_priority(et), sent.strip(), g, d, et))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            prio, sent, g, d, et = candidates[0]
            yield sec, sent, g, d, et

def short_cancer_name(cancer_type: str) -> str:
    if not cancer_type: return ""
    return (cancer_type.split(";")[0] or "").strip()

def craft_conclusion(drug: str, gene: str, alteration: str, cancer_type: str, evidence_type: str, sent: str) -> str:
    ct = short_cancer_name(cancer_type)
    scope = f"in {gene} {alteration} {ct}".strip()
    scope = re.sub(r"\s+", " ", scope).strip()
    if evidence_type == "efficacy":
        if re.search(r"\bOS\b|\boverall survival\b", sent, re.I): verb = "improved OS"
        elif re.search(r"\bPFS\b|\bprogression[- ]free survival\b", sent, re.I): verb = "improved PFS"
        elif re.search(r"\bORR\b|\bresponse\b|\bCR\b|\bPR\b|\bDCR\b", sent, re.I): verb = "showed responses"
        else: verb = "showed activity"
        return f"{drug} {verb} {scope} patients".strip()
    if evidence_type in {"safety","toxicity"}:
        return f"{drug} showed {evidence_type} signals {scope}".strip()
    if evidence_type == "review":
        return f"Review mention of {drug} {scope}; no direct efficacy evidence".strip()
    return f"{drug} mentioned {scope} without direct evidence".strip()

def normalize_alteration_type(alt_type: str, mut_type: str) -> str:
    alt_type = (alt_type or "").lower()
    mut_type = (mut_type or "").lower()
    if alt_type in {"fusion","amplification","deletion","insertion","duplication"}: return alt_type
    if alt_type == "point_mutation":
        if mut_type in {"missense","nonsense"}: return mut_type
        return "missense"
    if "exon" in alt_type:
        return "deletion" if "del" in alt_type else ("insertion" if "ins" in alt_type else "")
    return alt_type or ""

def therapy_type_from(drug_primary: str, is_combo: bool, therapy_by_drug: Dict[str,str]) -> str:
    if is_combo: return "combination"
    base = therapy_by_drug.get(drug_primary, "") or therapy_class(drug_primary)
    return base if base in {"targeted","immunotherapy","chemotherapy"} else "other"

# -------------------- Build rows --------------------
def build_rows_for_pmid(pmid: str, meta: Dict, fb: Dict,
                        ptc_doc: Optional[dict], epmc_ann: Optional[List[dict]],
                        registry: GeneRegistry, drug_dict: Dict[str,str],
                        efficacy_only: bool=False) -> List[dict]:
    title = (fb.get("title") or meta.get("title") or "").strip()
    sections_raw = fb.get("sections") or []
    sections = [(map_to_primary_section(lbl), txt) for (lbl, txt) in sections_raw]
    abst = fb.get("abstract") or ""
    combined = " ".join(t for _, t in sections) if sections else ""
    full = " ".join([title, abst, combined])

    # NEW: paper-level efficacy-like signal (for fallback rows)
    has_efficacy_signal = bool(
        EFFICACY_VERBS.search(full or "")
        or RESPONSE_PAT.search(full or "")
        or RESIST_PAT.search(full or "")
    )

    year  = int(fb.get("year") or 0) or first_year_in(meta.get("epubdate"), meta.get("pubdate"))
    journal = resolve_journal(meta, fb)
    doi = fb.get("doi") or ""
    pmcid = fb.get("pmcid") or (meta.get("pmcid") or "")
    if not doi:
        el = meta.get("elocationid") or ""
        mdoi = re.search(r"(10\.\S+)", el)
        if mdoi: doi = mdoi.group(1)
    for aid in meta.get("articleids", []) or []:
        typ = (aid.get("idtype") or "").lower(); val = (aid.get("value") or "").strip()
        if typ == "doi" and not doi: doi = val
        if typ == "pmcid" and not pmcid: pmcid = val

    pubtypes = [t.lower() for t in (meta.get("pubtype") or [])]
    is_review_like = any(t for t in pubtypes if "review" in t or "meta-analysis" in t or "systematic" in t)

    cancers = extract_cancers_from_all_sources(title, sections, ptc_doc, epmc_ann)
    cancer_type = "; ".join(cancers)

    # Per-paper meta (genome/TMB/MSI and authority mentions)
    reference_genome = extract_reference_genome(full)
    tmb_value, tmb_unit, tmb_state = extract_tmb(full)
    msi_state = extract_msi_state(full)
    drug_accessibility = extract_regulatory_authorities(full)

    # Genes (HGNC gated)
    gene_candidates = set()
    for m in GENE_TOKEN.finditer(full):
        tok = m.group(0).upper()
        if registry.is_valid(tok): gene_candidates.add(registry.normalize(tok))
    if ptc_doc:
        for p in ptc_doc.get("passages", []) or []:
            for a in p.get("annotations", []) or []:
                tpe = (a.get("infons", {}).get("type") or a.get("type") or "").lower()
                if tpe in ("gene","gene/protein","protein","gene_protein"):
                    up = registry.normalize(a.get("text") or "")
                    if up: gene_candidates.add(up)
    if epmc_ann:
        for a in epmc_ann:
            if (a.get("type") or "").lower() == "gene_protein":
                up = registry.normalize((a.get("exact") or a.get("text") or ""))
                if up: gene_candidates.add(up)
    genes = set(gene_candidates)
    if not genes: return []

    # Drugs
    drugs_raw = extract_drugs(full)
    drugs_all = filter_drugs(drugs_raw, drug_dict)
    if not drugs_all:
        return []
    therapy_by_drug = {d: therapy_class(d) for d in drugs_all}

    # Mutations / fusions
    snv_set, nucs, exons = extract_mutation_signals(full)
    nuc_str = "; ".join(sorted(nucs)) if nucs else ""
    fusions = canonical_fusions_hgnc(full, registry)

    # Clinical context (paper-level)
    study = "RCT" if RCT_PAT.search(full) else ("trial" if TRIAL_PAT.search(full) else ("observational" if OBS_PAT.search(full) else ("preclinical" if PRECLIN.search(full) else ("in_vitro" if IN_VITRO.search(full) else ("in_silico" if IN_SILICO.search(full) else ("case_report" if CASE_REPORT_PAT.search(full) else ""))))))
    m_ph = PHASE_PAT.search(full); phase = (m_ph.group(1) or m_ph.group(2) or "").upper() if m_ph else ""
    stage_text = (STAGE_TXT.search(full).group(0) if STAGE_TXT.search(full) else "")
    tnm_tuple = None
    m_tnm = TNM_PAT.search(full)
    if m_tnm:
        try: tnm_tuple = (int(m_tnm.group(1)), int(m_tnm.group(2)), int(m_tnm.group(3)))
        except Exception: pass

    def map_tnm_to_stage(tnm_tuple: Optional[Tuple[int,int,int]], stage_text: str) -> str:
        if tnm_tuple:
            T,N,M = tnm_tuple
            if M == 1: return "Stage IV"
            if T >= 4 or N >= 3: return "Stage III"
            if T >= 2 or N >= 1: return "Stage II"
            return "Stage I"
        if stage_text:
            s = stage_text.upper()
            m = re.search(r"(I{1,4}|[1-4])", s)
            if m:
                roman = {"1":"I","2":"II","3":"III","4":"IV"}.get(m.group(1), m.group(1))
                if roman in {"I","II","III","IV"}: return f"Stage {roman}"
        return ""

    stage_norm = map_tnm_to_stage(tnm_tuple, stage_text) or ""
    resect = (RESECT_PAT.search(full).group(1) if RESECT_PAT.search(full) else "")
    metastatic = "yes" if (META_PAT.search(full) or "stage iv" in stage_text.lower()) else ""
    m_line = LINE_PAT.search(full)
    if m_line:
        raw = m_line.group(0).lower()
        if raw.startswith("1") or "first" in raw: line_num, line_lab = "1","1L"
        elif raw.startswith("2") or "second" in raw: line_num, line_lab = "2","2L"
        elif raw.startswith("3") or "third" in raw: line_num, line_lab = "3","3L"
        elif raw.startswith("4") or "fourth" in raw: line_num, line_lab = "4","4L"
        else: line_num, line_lab = "",""
    else:
        line_num, line_lab = "",""
    setting = "; ".join(sorted(set(m.group(1).lower() for m in SETTING_PAT.finditer(full or ""))))

    hist = ""
    for w in ["adenocarcinoma","squamous cell carcinoma","glioblastoma","melanoma","lymphoma","leukemia","myeloma","glioma","urothelial","cholangiocarcinoma","hepatocellular carcinoma","sarcoma"]:
        if re.search(rf"\b{re.escape(w)}\b", full, re.I):
            hist = f"{hist}; {w}" if hist else w

    omics = "; ".join(sorted(set(m.group(0) for m in OMICS_PAT.finditer(full)))) if OMICS_PAT.search(full) else ""

    if IMMUNE_PAT.search(full):
        raw_tokens = set(m.group(0) for m in IMMUNE_PAT.finditer(full))
        norm_tokens = set()
        for t in raw_tokens:
            s = t.strip().lower()
            if re.fullmatch(r"pd[- ]?l1", s):      norm_tokens.add("PD-L1")
            elif s == "cd274":                     norm_tokens.add("CD274")
            elif s == "tmb" or "tumor mutation burden" in s or "tumour mutation burden" in s:
                                                  norm_tokens.add("TMB")
            elif s == "msi" or "microsatellite instability" in s:
                                                  norm_tokens.add("MSI")
            elif s == "mss":                       norm_tokens.add("MSS")
            elif s == "dmmr" or "mismatch repair" in s:
                                                  norm_tokens.add("dMMR")
            elif s == "ctla-4":                    norm_tokens.add("CTLA-4")
            elif "cd8+" in s:                      norm_tokens.add("CD8+ T")
            else:                                  norm_tokens.add(t)
        immune = "; ".join(sorted(norm_tokens))
    else:
        immune = ""

    resp_flag = bool(RESPONSE_PAT.search(full or ""))
    resist_flag = bool(RESIST_PAT.search(full or ""))

    species_val = "Human" if re.search(r"\b(human|patient)s?\b", full, re.I) else ("Mouse" if re.search(r"\b(mouse|mice|murine)\b", full, re.I) else "")
    model_val = "PDX" if re.search(r"\bPDX\b|patient[- ]derived xenograft", full, re.I) else ("Cell line" if re.search(r"\bcell[- ]line(s)?\b", full, re.I) else ("Organoid" if re.search(r"\borganoid(s)?\b", full, re.I) else ("xenograft" if re.search(r"\bxenograft(s)?\b", full, re.I) else "")))

    per_section_hits = list(section_sentence_hits(title or abst, sections, genes, drugs_all, is_review_like))

    proto_rows: List[dict] = []

    def add_row(g: str, d: str, alteration: str, alt_type: str, mut_type: str, mut_id: str, fusion_genes: str,
                gene_type: str, section: str, evidence_type: str, sent: str):
        alt_type_norm = normalize_alteration_type(alt_type, mut_type)
        therapeutic_active = "yes" if evidence_type == "efficacy" else "no"
        p_age, p_sex = parse_patient_age_sex(full)
        drug_all_list = list(OrderedDict.fromkeys(drugs_all))
        is_combo = "yes" if len(drug_all_list) > 1 else "no"
        combo_partners = [x for x in drug_all_list if x.lower() != d.lower()]
        therapy_t = therapy_type_from(d, is_combo == "yes", therapy_by_drug)
        conclusion = craft_conclusion(d, g, alteration, cancer_type, evidence_type, sent or "")

        cna_normalized = ""
        fusion_normalized = ""
        alt_low = (alteration or "").lower()

        if "homozygous deletion" in alt_low:
            cna_normalized = normalize_cna_term("homozygous deletion")
        elif re.search(r"\bloss\b", alt_low):
            cna_normalized = normalize_cna_term("loss")
        elif "deletion" in alt_low:
            cna_normalized = normalize_cna_term("deletion")
        elif "amplification" in alt_low:
            cna_normalized = normalize_cna_term("amplification")
        elif re.search(r"\bgain\b", alt_low):
            cna_normalized = normalize_cna_term("gain")
        elif "duplication" in alt_low:
            cna_normalized = normalize_cna_term("duplication")

        if "fusion" in alt_low or (fusion_genes or "").strip():
            fusion_normalized = normalize_cna_term("fusion")

        proto_rows.append({
            "biomarker": g,
            "biomarker_type": "gene",
            "gene_type": gene_type,
            "alteration": alteration,
            "alteration_type": alt_type_norm,
            "nucleotide_change": nuc_str,
            "fusion_genes": fusion_genes or "",
            "cna_normalized": cna_normalized or "",
            "fusion_normalized": fusion_normalized or "",
            "cancer_type": cancer_type,
            "histology": hist,

            "drug_primary": d,
            "drug_all": "; ".join(drug_all_list),
            "is_combination": is_combo,
            "combination_drugs": "; ".join(combo_partners),

            "therapy_type": therapy_t,

            # Regulatory authorities (FDA/PMDA/MFDS)
            "drug_accessibility": drug_accessibility or "",

            "line_of_therapy_num": line_num,
            "line_of_therapy": line_lab,
            "treatment_setting": setting,

            "study_design": study,
            "trial_phase": phase,
            "stage": stage_norm,
            "resectability": resect,
            "metastatic": metastatic,

            "patient_age": p_age,
            "patient_sex": p_sex,
            "species": species_val,
            "model_system": model_val,

            "omics_used": omics,
            "immune_features": immune,

            "reference_genome": reference_genome or "",
            "TMB_value": tmb_value or "",
            "TMB_unit": tmb_unit or "",
            "TMB_state": tmb_state or "",
            "MSI_state": msi_state or "",

            "resistance_observed": "yes" if resist_flag else ("no" if resp_flag else ""),

            "evidence_section": section,
            "evidence_type": evidence_type,
            "therapeutic_active": therapeutic_active,

            "summary_sentence": sent or "",
            "conclusion": conclusion,

            "pmid": str(pmid),
            "pmcid": pmcid,
            "doi": doi,
            "year": year,
            "journal": journal,
        })

    if not per_section_hits:
        sent_fallback = ""
        # UPDATED: if the paper clearly discusses response/resistance, treat as efficacy in fallback
        if has_efficacy_signal:
            et_fallback = "efficacy"
        elif is_review_like:
            et_fallback = "review"
        else:
            et_fallback = "background"

        for g in genes:
            for d in drugs_all:
                for s in sorted(snv_set):
                    add_row(g, d, s, "point_mutation", "missense", s, "", "DNA", "Abstract", et_fallback, sent_fallback)
                for ex, kind in exons:
                    add_row(g, d, f"exon {ex} {kind}", kind, "", "", "", "DNA", "Abstract", et_fallback, sent_fallback)
                for fusion_genes, partners in fusions:
                    if g in partners or (not fusion_genes and g in partners):
                        add_row(
                            g, d,
                            (f"{fusion_genes} fusion" if fusion_genes else f"{g} fusion"),
                            "fusion", "", "", fusion_genes,
                            "RNA", "Abstract", et_fallback, sent_fallback
                        )
    else:
        for section, sent, g_hit, d_hit, et in per_section_hits:
            if efficacy_only and et != "efficacy":
                continue
            hit_any = False
            for s in sorted(snv_set):
                if re.search(rf"\b{re.escape(s)}\b", sent):
                    add_row(g_hit, d_hit, s, "point_mutation", "missense", s, "", "DNA", section, et, sent)
                    hit_any = True
            for ex, kind in exons:
                token = f"exon {ex}"
                if re.search(rf"\b{re.escape(token)}\b", sent, re.I):
                    add_row(g_hit, d_hit, f"exon {ex} {kind}", kind, "", "", "", "DNA", section, et, sent)
                    hit_any = True
            for fusion_genes, partners in fusions:
                if (fusion_genes and re.search(r"\bfusion\b", sent, re.I) and (g_hit in partners)) or (not fusion_genes and g_hit in partners):
                    add_row(g_hit, d_hit, (f"{fusion_genes} fusion" if fusion_genes else f"{g_hit} fusion"),
                            "fusion", "", "", fusion_genes, "RNA", section, et, sent)
                    hit_any = True
            if not hit_any:
                add_row(g_hit, d_hit, "", "", "", "", "", "DNA", section, et, sent)

    if not proto_rows:
        return []

    merged: Dict[Tuple, dict] = {}
    for r in proto_rows:
        key = (
            r["pmid"], r["biomarker"], r["alteration"], r["cancer_type"],
            r["evidence_section"], r["evidence_type"], r["therapeutic_active"], r["summary_sentence"]
        )
        if key not in merged:
            merged[key] = dict(r)
            merged[key]["_drug_counts"] = Counter([r["drug_primary"]])
            merged[key]["drug_all"] = r["drug_all"]
            merged[key]["combination_drugs"] = r["combination_drugs"]
        else:
            m = merged[key]
            m["_drug_counts"].update([r["drug_primary"]])
            da = set([x.strip() for x in (m["drug_all"] or "").split(";") if x.strip()])
            da |= set([x.strip() for x in (r["drug_all"] or "").split(";") if x.strip()])
            m["drug_all"] = "; ".join(sorted(da))
            cb = set([x.strip() for x in (m["combination_drugs"] or "").split(";") if x.strip()])
            cb |= set([x.strip() for x in (r["combination_drugs"] or "").split(";") if x.strip()])
            m["combination_drugs"] = "; ".join(sorted(cb))
            for fld in ["alteration_type","fusion_genes","gene_type","cna_normalized","fusion_normalized","drug_accessibility"]:
                if not m.get(fld) and r.get(fld): m[fld] = r.get(fld)

    final_rows: List[dict] = []
    for key, m in merged.items():
        most = m.pop("_drug_counts")
        if most:
            top_freq = most.most_common()
            maxn = top_freq[0][1]
            cands = sorted([d for d,c in top_freq if c == maxn], key=lambda x: x.lower())
            primary = cands[0]
        else:
            primary = (m.get("drug_all","").split(";")[0] or "").strip()
        m["drug_primary"] = primary
        da_list = [x.strip() for x in (m.get("drug_all","") or "").split(";") if x.strip()]
        m["is_combination"] = "yes" if len(da_list) > 1 else "no"
        partners = [x for x in da_list if x.lower() != primary.lower()]
        m["combination_drugs"] = "; ".join(partners)
        therapy_by_drug_local = {d: therapy_class(d) for d in da_list}
        m["therapy_type"] = therapy_type_from(primary, m["is_combination"] == "yes", therapy_by_drug_local)
        m["biomarker_type"] = "gene" if m.get("biomarker_type","").lower() not in {"protein","pathway"} else m["biomarker_type"].lower()
        gt = (m.get("gene_type") or "").upper()
        m["gene_type"] = gt if gt in {"DNA","RNA","PROTEIN"} else ("DNA" if gt == "" else gt)
        final_rows.append(m)

    return final_rows

# -------------------- Backfill: journal names for 'Unknown Journal' --------------------
def backfill_unknown_journals(pmids: List[str], esum: Dict[str, dict], ef: Dict[str, dict], api_key: Optional[str]) -> None:
    need = []
    for p in pmids:
        j = resolve_journal(esum.get(p, {}) or {}, ef.get(p, {}) or {})
        if not j or j == "Unknown Journal":
            need.append(p)
    if not need:
        return
    extra = pubmed_esummary(need, api_key)
    for p in need:
        meta = extra.get(p, {})
        if meta:
            esum[p] = meta
            j = (meta.get("fulljournalname") or meta.get("source") or "").strip()
            if j:
                if p not in ef: ef[p] = {}
                ef[p]["journal"] = j

# -------------------- Driver --------------------
def parse_years(s: str) -> Tuple[int,int]:
    s = s.strip()
    if "-" in s:
        a,b = s.split("-",1); return (int(a), int(b))
    y = int(s); return (y,y)

def looks_like_known_drug(name: str, whitelist: Dict[str,str]) -> bool:
    return (name or "").lower() in whitelist

def main():
    ap = argparse.ArgumentParser(description="HGNC-validated, context-aware gene–drug–cancer miner (v7.3.11; robust I/O + year-binned ESearch pagination).")
    ap.add_argument("--q", required=True, help="PubMed query string")
    ap.add_argument("--all-cancers", dest="all_cancers", action="store_true")
    ap.add_argument("--years", required=True, help="YYYY or YYYY-YYYY")
    ap.add_argument("--max-records", type=int, default=300)
    ap.add_argument("--year-binned", action="store_true")
    ap.add_argument("--use-pmc-fulltext", action="store_true")
    ap.add_argument("--email", default=None)
    ap.add_argument("--ncbi-api-key", default=None)
    ap.add_argument("--out", required=True, help="Output prefix (without extension)")
    ap.add_argument("--strict-year", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-gene-and-drug", action="store_true")
    ap.add_argument("--require-known-drug", action="store_true")
    ap.add_argument("--efficacy-only", action="store_true", help="Keep only rows with sentence-level efficacy context")
    ap.add_argument("--topn-export", type=int, default=0, help="Keep top N rows per (biomarker, cancer_type, evidence_section)")
    ap.add_argument("--hgnc-cache", default=None)
    ap.add_argument("--hgnc-refresh", action="store_true")
    ap.add_argument("--drug-whitelist", default=None, help="Optional drug list file (.txt lines or .csv name column)")
    ap.add_argument("--exclude-predatory-publishers", default=None,
                    help="Optional file with publisher names to exclude (one per line, case-insensitive, partial match)")
    args = ap.parse_args()

    print("🔁 Fetching HGNC Approved genes…")
    registry = GeneRegistry.from_hgnc_api(cache_path=args.hgnc_cache, refresh=args.hgnc_refresh)
    print(f"✅ HGNC symbols: {len(registry.symbols):,}")

    drug_dict = load_drug_whitelist(args.drug_whitelist)
    print(f"✅ Drug whitelist loaded: {len(drug_dict):,} names")

    q = args.q
    years = parse_years(args.years)
    if args.all_cancers:
        q = f"({q}) AND ({CANCER_CLAUSE})"
    print(f"🔎 Query: {q}")
    print(f"📅 Years: {years[0]}–{years[1]}")
    print(f"🔢 Max records: {args.max_records}")

    if args.year_binned:
        per_year = max(1, ceil(args.max_records / max(1, (years[1]-years[0]+1))))
        pmids = pubmed_esearch_year_binned(q, years, per_year, args.email, args.ncbi_api_key)
        pmids = pmids[:args.max_records]
    else:
        pmids = pubmed_esearch(q, years, args.max_records, args.email, args.ncbi_api_key)
    if not pmids:
        print("No PMIDs found.")

    else:
        print(f"🔗 PMIDs: {len(pmids)}")

    esum: Dict[str, dict] = {}
    ef: Dict[str, dict] = {}
    ptc: Dict[str, dict] = {}
    epmc: Dict[str, List[dict]] = {}

    if pmids:
        esum = pubmed_esummary(pmids, args.ncbi_api_key)
        ef   = pubmed_efetch_abstracts(pmids, args.ncbi_api_key)
        backfill_unknown_journals(pmids, esum, ef, args.ncbi_api_key)
        ptc  = pubtator_for_pmids(pmids)
        epmc = epmc_annotations_for_pmids(pmids)

    if args.use_pmc_fulltext and pmids:
        pmcids = []
        for p in pmids:
            pmcid = (ef.get(p, {}) or {}).get("pmcid") or (esum.get(p, {}) or {}).get("pmcid") or ""
            if pmcid and isinstance(pmcid, str): pmcids.append(pmcid)
        if pmcids:
            pmc_secs = epmc_fetch_fulltexts(pmcids)
            for p in pmids:
                fb = ef.get(p, {})
                pmcid = fb.get("pmcid") or (esum.get(p, {}) or {}).get("pmcid") or ""
                if pmcid and pmc_secs.get(pmcid):
                    fb2 = dict(fb); fb2["sections"] = (fb.get("sections") or []) + pmc_secs[pmcid]; ef[p] = fb2

    print("🧩 Assembling rows…")
    all_rows: List[dict] = []
    if pmids:
        with cf.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 8)) as ex:
            futures = []
            for p in pmids:
                meta = esum.get(p, {})
                fb   = ef.get(p, {"title": meta.get("title",""), "abstract": "", "sections": [], "year": 0, "doi":"", "pmcid":"", "mesh": [], "publisher":"", "journal":""})
                futures.append(ex.submit(build_rows_for_pmid, p, meta, fb, ptc.get(p), epmc.get(p), registry, drug_dict, args.efficacy_only))
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Rows", unit="pmid"):
                all_rows.extend(fut.result() or [])

    # Strict year filter
    if all_rows and args.strict_year:
        all_rows = [r for r in all_rows if isinstance(r.get("year"), int) and years[0] <= r["year"] <= years[1]]

    # Filters
    if all_rows and args.require_gene_and_drug:
        keep_pmids = {r["pmid"] for r in all_rows if r.get("biomarker") and r.get("drug_primary")}
        all_rows = [r for r in all_rows if r["pmid"] in keep_pmids]

    if all_rows and args.require_known_drug:
        all_rows = [r for r in all_rows if looks_like_known_drug(r.get("drug_primary","").lower(), drug_dict)]

    # Predatory publisher filter
    predatory_publishers = load_list_lower(args.exclude_predatory_publishers)
    if all_rows and predatory_publishers:
        pmid_to_publisher = {}
        for r in all_rows:
            p = r.get("pmid")
            if p not in pmid_to_publisher:
                pmid_to_publisher[p] = resolve_publisher(esum.get(p, {}) or {}, ef.get(p, {}) or {})
        def is_pred_publisher(pub: str) -> bool:
            pl = (pub or "").lower()
            return any(x in pl for x in predatory_publishers) if pl else False
        before = len(all_rows)
        all_rows = [r for r in all_rows if not is_pred_publisher(pmid_to_publisher.get(r.get("pmid"), ""))]
        removed = before - len(all_rows)
        print(f"🧹 Publisher filter: removed {removed} row(s) out of {before} using {len(predatory_publishers)} publisher name(s).")

    # Top-N per group
    if all_rows and args.topn_export and args.topn_export > 0:
        df_tmp = pd.DataFrame(all_rows)
        df_tmp["year_sort"] = pd.to_numeric(df_tmp["year"], errors="coerce").fillna(0).astype(int)
        df_tmp = (df_tmp.sort_values(["year_sort","pmid"], ascending=[False, True])
                        .groupby(["biomarker","cancer_type","evidence_section"], as_index=False, sort=False)
                        .head(args.topn_export)
                        .reset_index(drop=True)
                 )
        all_rows = df_tmp.drop(columns=["year_sort"]).to_dict(orient="records")

    # Final column order (unchanged schema from v7.3.10)
    cols = [
        "biomarker", "biomarker_type", "gene_type", "alteration", "alteration_type",
        "nucleotide_change", "fusion_genes",
        "cna_normalized", "fusion_normalized",
        "cancer_type", "histology",
        "drug_primary", "drug_all", "is_combination", "combination_drugs",
        "therapy_type", "drug_accessibility",
        "line_of_therapy_num", "line_of_therapy", "treatment_setting",
        "study_design", "trial_phase", "stage", "resectability", "metastatic",
        "patient_age", "patient_sex", "species", "model_system",
        "omics_used", "immune_features",
        "reference_genome", "TMB_value", "TMB_unit", "TMB_state", "MSI_state",
        "resistance_observed",
        "evidence_section", "evidence_type", "therapeutic_active",
        "summary_sentence", "conclusion",
        "pmid", "pmcid", "doi", "year", "journal",
    ]

    out_prefix = args.out
    rows_path_xlsx = f"{out_prefix}.xlsx"
    rows_path_csvgz = f"{out_prefix}.rows.csv.gz"

    # Always attempt to write an output file (even if zero rows)
    if all_rows is None:
        all_rows = []

    print(f"🧮 Final rows: {len(all_rows):,}")

    try:
        if len(all_rows) > 1_000_000:
            ensure_parent_dir_for(rows_path_csvgz)
            with gzip.open(rows_path_csvgz, "wt", newline="", encoding="utf-8") as gz:
                w = csv.writer(gz); w.writerow(cols)
                for r in tqdm(all_rows, desc="Write CSV.GZ", unit="row"):
                    w.writerow([r.get(k, "") for k in cols])
            print(f"⚠️ Large output written: {rows_path_csvgz}")
        else:
            ensure_parent_dir_for(rows_path_xlsx)
            df = pd.DataFrame(all_rows, columns=cols)
            with pd.ExcelWriter(rows_path_xlsx, engine="openpyxl") as xl:
                # Main sheet (Rows) — will be empty if no rows
                df.to_excel(xl, sheet_name="Rows", index=False)

                # PublicationInfo — based on pmids we touched (if any)
                kept_pmids = list(OrderedDict.fromkeys(df["pmid"].tolist())) if not df.empty else (pmids or [])
                pub_records = []
                for p in kept_pmids:
                    pub_records.append({
                        "pmid": p,
                        "title": ( ( (esum.get(p, {}) or {}).get("title") ) or ( (ef.get(p, {}) or {}).get("title") ) or "" ),
                        "year": (ef.get(p, {}) or {}).get("year") or first_year_in((esum.get(p, {}) or {}).get("epubdate"),
                                                                                        (esum.get(p, {}) or {}).get("pubdate")),
                        "journal": resolve_journal(esum.get(p, {}) or {}, ef.get(p, {}) or {}),
                        "doi": (ef.get(p, {}) or {}).get("doi") or "",
                        "pmcid": (ef.get(p, {}) or {}).get("pmcid") or (esum.get(p, {}) or {}).get("pmcid") or "",
                    })
                pd.DataFrame(pub_records).to_excel(xl, sheet_name="PublicationInfo", index=False)

                # Run metadata — always present
                runinfo = pd.DataFrame([{
                    "query": q, "years": f"{years[0]}-{years[1]}",
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "n_pmids": len(pmids or []), "n_rows": len(df),
                    "year_binned": args.year_binned, "use_pmc_fulltext": args.use_pmc_fulltext,
                    "strict_year": args.strict_year, "require_gene_and_drug": args.require_gene_and_drug,
                    "require_known_drug": args.require_known_drug, "efficacy_only": args.efficacy_only,
                    "hgnc_cache": args.hgnc_cache or "", "drug_whitelist": args.drug_whitelist or "",
                    "exclude_predatory_publishers": args.exclude_predatory_publishers or "",
                    "schema": "normalized_v1",
                    "script": "mega_mine_free_plus_v7_3_11.py"
                }])
                runinfo.to_excel(xl, sheet_name="RunInfo", index=False)
            print(f"✅ Wrote {rows_path_xlsx}  (rows: {len(df):,}, PMIDs: {len(kept_pmids):,})")
    except Exception as e:
        # Final guard: never crash on save; at least tell the user what happened.
        print(f"❌ Failed to write output file(s): {e}")

if __name__ == "__main__":
    main()

