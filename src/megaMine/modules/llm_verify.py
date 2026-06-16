"""
llm_verify.py — megaMine v2.0
LLM Verification Layer Module

PURPOSE:
    Adds a semantic verification step on top of rule-based
    extraction. Rule-based extraction finds candidate
    gene-drug-cancer relationships. This module passes
    each candidate to an LLM for confirmation.

    This creates a hybrid architecture:
      Step 1: Rules  → fast, interpretable, traceable candidates
      Step 2: LLM    → semantic confirmation + confidence score

    The hybrid approach combines the traceability of
    rule-based extraction with the semantic understanding
    of modern language models. This directly addresses
    the concern that pure rule-based extraction may miss
    implicit or contextually complex relationships.

DESIGN PRINCIPLE:
    LLM is used as a VERIFIER, not a primary extractor.
    Rules find candidates. LLM confirms or rejects them.
    This keeps outputs traceable — every relationship
    can be traced to a specific sentence and rule,
    plus an LLM confidence score.

SUPPORTED MODELS:
    Option 1: Claude API (claude-haiku — fast, low cost)
    Option 2: GPT-4o-mini (OpenAI API)
    Option 3: Offline mode (no API — returns neutral score)

FLAG: --llm-verify (off by default, opt-in)
FLAG: --llm-model {claude, gpt4mini, offline}
FLAG: --llm-threshold 0.70 (filter below this confidence)

OUTPUT FIELDS ADDED:
    llm_verified    : yes / no / skipped
    llm_confidence  : 0.0 to 1.0
    llm_reason      : one sentence explanation
    llm_model       : which model was used

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import time
import json
import pandas as pd
import requests
from typing import Optional, Tuple

# ─── API settings ─────────────────────────────────────────────
CLAUDE_API_URL   = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL     = "claude-haiku-4-5-20251001"  # full stable API ID
OPENAI_API_URL   = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL     = "gpt-4o-mini"

# Rate limiting
SLEEP_BETWEEN    = 0.5   # seconds between API calls
API_TIMEOUT      = 30    # seconds per request
MAX_RETRIES      = 2     # retry failed calls once

# Confidence threshold
DEFAULT_THRESHOLD = 0.70  # below this = not verified

# ─── Prompt template ──────────────────────────────────────────
VERIFY_PROMPT = """You are a biomedical expert reviewing literature evidence for precision oncology.

Read this sentence carefully:
"{sentence}"

Task: Determine if this sentence provides direct evidence that:
  Gene/Biomarker: {gene}
  Drug: {drug}
  Cancer type: {cancer}
  Claimed relationship: {evidence_type}

Answer in this exact JSON format with no other text:
{{
  "verified": "yes" or "no",
  "confidence": a number between 0.0 and 1.0,
  "direct_relation": "yes" or "no",
  "negated": "yes" or "no",
  "speculative": "yes" or "no",
  "evidence_type": "efficacy" or "resistance" or "toxicity" or "background",
  "reason": "one sentence explanation of your decision"
}}

Guidelines:
- verified=yes means the sentence clearly supports the claimed relationship
- verified=no means it does not support it, is ambiguous, negated, or speculative
- direct_relation=yes means gene AND drug AND cancer all appear in direct relation
- negated=yes means the sentence uses negation (did not respond, failed to, no response)
- speculative=yes means the sentence uses speculation (may, might, could, possibly)
- evidence_type is your independent assessment of the relationship direction
- confidence=1.0 means completely certain
- confidence=0.5 means ambiguous
- confidence=0.0 means clearly contradicts the claim
- If negated=yes or speculative=yes then verified must be no
- Be strict — only verify if the evidence is explicit, direct, and non-speculative"""


def _build_prompt(
    sentence: str,
    gene: str,
    drug: str,
    cancer: str,
    evidence_type: str,
) -> str:
    """Build the verification prompt for a single candidate."""
    return VERIFY_PROMPT.format(
        sentence     = sentence[:500],  # cap length
        gene         = gene,
        drug         = drug,
        cancer       = cancer.split(";")[0].strip(),
        evidence_type= evidence_type,
    )


def _parse_llm_response(response_text: str) -> Tuple[str, float, str, dict]:
    """
    Parse LLM JSON response into (verified, confidence, reason, extras).
    Extracts expanded fields: negated, speculative, direct_relation,
    evidence_type — addressing Reviewer 2 concern about negation
    and speculation handling.
    Returns safe defaults if parsing fails.
    """
    try:
        # Strip any markdown code blocks
        text = response_text.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        data = json.loads(text)
        verified   = str(data.get("verified", "no")).lower()
        confidence = float(data.get("confidence", 0.5))
        reason     = str(data.get("reason", ""))[:200]

        # NEW: expanded fields for negation/speculation detection
        extras = {
            "llm_direct_relation": str(data.get("direct_relation", "")).lower(),
            "llm_negated":         str(data.get("negated", "")).lower(),
            "llm_speculative":     str(data.get("speculative", "")).lower(),
            "llm_evidence_type":   str(data.get("evidence_type", "")).lower(),
        }

        # Validate
        if verified not in ("yes", "no"):
            verified = "no"

        # If negated or speculative — force verified=no
        if extras["llm_negated"] == "yes":
            verified   = "no"
            confidence = max(0.0, confidence - 0.3)
            reason     = f"[NEGATED] {reason}"
        if extras["llm_speculative"] == "yes":
            verified   = "no"
            confidence = max(0.0, confidence - 0.2)
            reason     = f"[SPECULATIVE] {reason}"

        confidence = max(0.0, min(1.0, confidence))
        return verified, confidence, reason, extras

    except Exception:
        return "no", 0.5, "Could not parse LLM response", {
            "llm_direct_relation": "",
            "llm_negated":         "",
            "llm_speculative":     "",
            "llm_evidence_type":   "",
        }


def verify_with_claude(
    sentence: str,
    gene: str,
    drug: str,
    cancer: str,
    evidence_type: str,
    api_key: str,
) -> Tuple[str, float, str, str]:
    """
    Verify a candidate relationship using Claude API.

    Returns
    -------
    tuple: (verified, confidence, reason, model_used)
    """
    prompt = _build_prompt(sentence, gene, drug, cancer, evidence_type)

    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model":      CLAUDE_MODEL,
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(SLEEP_BETWEEN)
            resp = requests.post(
                CLAUDE_API_URL,
                headers = headers,
                json    = payload,
                timeout = API_TIMEOUT,
            )

            if resp.status_code == 200:
                data = resp.json()
                text = data["content"][0]["text"]
                verified, confidence, reason, extras = _parse_llm_response(text)
                return verified, confidence, reason, CLAUDE_MODEL, extras

            elif resp.status_code == 429:
                # Rate limited — wait longer
                time.sleep(5 * (attempt + 1))
                continue

            else:
                break

        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            continue

    return "skipped", 0.5, "API call failed", CLAUDE_MODEL


def verify_with_gpt4mini(
    sentence: str,
    gene: str,
    drug: str,
    cancer: str,
    evidence_type: str,
    api_key: str,
) -> Tuple[str, float, str, str]:
    """
    Verify a candidate relationship using GPT-4o-mini.

    Returns
    -------
    tuple: (verified, confidence, reason, model_used)
    """
    prompt = _build_prompt(sentence, gene, drug, cancer, evidence_type)

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens":  200,
        "temperature": 0.1,  # low temp for consistent verification
    }

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(SLEEP_BETWEEN)
            resp = requests.post(
                OPENAI_API_URL,
                headers = headers,
                json    = payload,
                timeout = API_TIMEOUT,
            )

            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                verified, confidence, reason, extras = _parse_llm_response(text)
                return verified, confidence, reason, OPENAI_MODEL, extras

            elif resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            else:
                break

        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            continue

    return "skipped", 0.5, "API call failed", OPENAI_MODEL


def verify_offline(
    sentence: str,
    gene: str,
    drug: str,
    cancer: str,
    evidence_type: str,
) -> Tuple[str, float, str, str]:
    """
    Offline verification — no API call.
    Uses heuristic rules as a fallback.

    IMPORTANT: Checks negation and speculation BEFORE
    checking for evidence keywords. This prevents the
    classic NLP mistake of detecting "respond" without
    checking "did not respond".

    This directly addresses Reviewer 2 concern about
    negation and speculation handling.

    This is NOT a replacement for real LLM verification.
    Use only for testing or when API is unavailable.
    """
    sentence_lower = sentence.lower()
    gene_lower     = gene.lower()
    drug_lower     = drug.lower()

    # ── Step 1: Check gene and drug presence ──────────────────
    gene_present = gene_lower in sentence_lower
    drug_present = drug_lower in sentence_lower

    if not gene_present or not drug_present:
        return (
            "no", 0.20,
            "Gene or drug not found in sentence",
            "offline_heuristic",
            {"llm_direct_relation":"no",
             "llm_negated":"",
             "llm_speculative":"",
             "llm_evidence_type":""}
        )

    # ── Step 2: Negation detection — BEFORE keyword detection ─
    # This fixes the classic mistake: seeing "respond" but missing
    # "did not respond". Check full phrases, not single words.
    NEGATION_PATTERNS = [
        "did not respond",
        "did not improve",
        "did not show",
        "no response",
        "no clinical benefit",
        "no significant",
        "not associated with response",
        "not associated with benefit",
        "failed to respond",
        "failure to respond",
        "lack of response",
        "lack of efficacy",
        "non-responder",
        "nonresponder",
        "non-responders",
        "nonresponders",
        "was not effective",
        "were not effective",
        "is not effective",
        "are not effective",
        "showed no",
        "show no",
        "demonstrated no",
        "no benefit",
        "no improvement",
        "no activity",
        "without benefit",
        "without response",
        "unable to respond",
    ]

    is_negated = any(pat in sentence_lower for pat in NEGATION_PATTERNS)

    # ── Step 3: Speculative language detection ─────────────────
    SPECULATIVE_PATTERNS = [
        "may ",
        "might ",
        "could ",
        "potentially ",
        "potential benefit",
        "suggests that",
        "suggesting that",
        "suggesting a",
        "possible",
        "possibly",
        "warrants further",
        "warrants investigation",
        "needs further",
        "requires further",
        "remains to be",
        "yet to be",
        "hypothesize",
        "speculate",
        "unclear whether",
        "unknown whether",
    ]

    is_speculative = any(pat in sentence_lower for pat in SPECULATIVE_PATTERNS)

    # ── Step 4: If negated — return NO with high confidence ───
    # A negated efficacy claim IS a resistance/lack-of-response signal
    if is_negated:
        # Recast evidence type if negated efficacy
        recast_type = "resistance" if evidence_type == "efficacy" else evidence_type
        return (
            "no", 0.85,
            "Negated efficacy / lack-of-response statement detected",
            "offline_heuristic",
            {"llm_direct_relation": "yes",
             "llm_negated":         "yes",
             "llm_speculative":     "no",
             "llm_evidence_type":   recast_type}
        )

    # ── Step 5: If speculative — reduce confidence, may still verify
    # Speculative language = uncertain, not zero evidence
    speculative_penalty = 0.20 if is_speculative else 0.0

    # ── Step 6: Evidence keyword matching ─────────────────────
    # Now safe to check keywords — negation already handled above
    EFFICACY_KEYWORDS = [
        "responded", "responding", "response",
        "efficacy", "efficacious",
        "benefit", "benefited", "benefiting",
        "improved", "improvement",
        "partial response", "complete response",
        "objective response",
        "sensitive", "sensitivity",
        "tumor shrinkage", "tumor reduction",
        "progression-free", "overall survival benefit",
        "clinical activity", "antitumor activity",
    ]

    RESISTANCE_KEYWORDS = [
        "resist", "resistance", "resistant",
        "refractory",
        "progressed", "progression on",
        "escape", "escaped",
        "bypass", "bypassed",
        "non-responder", "nonresponder",
        "acquired resistance",
        "primary resistance",
        "treatment failure",
        "disease progression",
    ]

    if evidence_type == "efficacy":
        if any(w in sentence_lower for w in EFFICACY_KEYWORDS):
            base_confidence = 0.75 - speculative_penalty
            return (
                "no" if base_confidence < DEFAULT_THRESHOLD else "yes",
                base_confidence,
                ("Speculative efficacy language detected"
                 if is_speculative else
                 "Direct efficacy keywords found with gene and drug"),
                "offline_heuristic",
                {"llm_direct_relation": "yes",
                 "llm_negated":         "no",
                 "llm_speculative":     "yes" if is_speculative else "no",
                 "llm_evidence_type":   "efficacy"}
            )
        else:
            return (
                "no", 0.40,
                "No direct efficacy keywords found",
                "offline_heuristic",
                {"llm_direct_relation": "no",
                 "llm_negated":         "no",
                 "llm_speculative":     "yes" if is_speculative else "no",
                 "llm_evidence_type":   ""}
            )

    elif evidence_type == "resistance":
        if any(w in sentence_lower for w in RESISTANCE_KEYWORDS):
            base_confidence = 0.75 - speculative_penalty
            return (
                "no" if base_confidence < DEFAULT_THRESHOLD else "yes",
                base_confidence,
                ("Speculative resistance language detected"
                 if is_speculative else
                 "Direct resistance keywords found with gene and drug"),
                "offline_heuristic",
                {"llm_direct_relation": "yes",
                 "llm_negated":         "no",
                 "llm_speculative":     "yes" if is_speculative else "no",
                 "llm_evidence_type":   "resistance"}
            )
        else:
            return (
                "no", 0.40,
                "No direct resistance keywords found",
                "offline_heuristic",
                {"llm_direct_relation": "no",
                 "llm_negated":         "no",
                 "llm_speculative":     "yes" if is_speculative else "no",
                 "llm_evidence_type":   ""}
            )

    # ── Step 7: Fallback — gene and drug present but unclear ───
    base_confidence = 0.55 - speculative_penalty
    return (
        "no" if base_confidence < DEFAULT_THRESHOLD else "yes",
        base_confidence,
        "Gene and drug present but context is unclear",
        "offline_heuristic",
        {"llm_direct_relation": "yes",
         "llm_negated":         "no",
         "llm_speculative":     "yes" if is_speculative else "no",
         "llm_evidence_type":   evidence_type}
    )


def run_llm_verification(
    df: pd.DataFrame,
    model: str = "offline",
    api_key: Optional[str] = None,
    confidence_threshold: float = DEFAULT_THRESHOLD,
    max_rows: Optional[int] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main entry point for LLM verification.

    Runs verification on every row in the megaMine output
    and adds llm_verified, llm_confidence, llm_reason,
    llm_model columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw megaMine output
    model : str
        "claude", "gpt4mini", or "offline"
    api_key : str, optional
        API key for claude or gpt4mini
    confidence_threshold : float
        Rows below this confidence get llm_verified=no
    max_rows : int, optional
        Limit rows to verify — useful for testing
    output_path : str, optional
        Save results to Excel if provided

    Returns
    -------
    pd.DataFrame
        Input df with 4 new LLM columns added
    """
    print(f"🤖 Running LLM verification (model: {model})...")

    if model in ("claude", "gpt4mini") and not api_key:
        print(f"   ⚠️  No API key provided — switching to offline mode")
        model = "offline"

    df = df.copy()
    results = []

    # Work on subset if max_rows set
    work_df = df.head(max_rows) if max_rows else df
    total   = len(work_df)

    verified_count = 0
    skipped_count  = 0

    for idx, (_, row) in enumerate(work_df.iterrows()):
        # Progress indicator every 10 rows
        if idx % 10 == 0:
            print(f"   Processing {idx+1}/{total}...", end="\r")

        sentence     = str(row.get("summary_sentence", "") or "")
        gene         = str(row.get("biomarker", "") or "")
        drug         = str(row.get("drug_primary", "") or "")
        cancer       = str(row.get("cancer_type", "") or "")
        evidence_type= str(row.get("evidence_type", "") or "")

        # Skip rows without a sentence
        if not sentence or not gene or not drug:
            results.append({
                "llm_verified":        "skipped",
                "llm_confidence":      0.0,
                "llm_reason":          "Missing sentence, gene, or drug",
                "llm_model":           model,
                "llm_direct_relation": "",
                "llm_negated":         "",
                "llm_speculative":     "",
                "llm_evidence_type":   "",
            })
            skipped_count += 1
            continue

        # Run verification
        if model == "claude":
            verified, confidence, reason, model_used, extras = verify_with_claude(
                sentence, gene, drug, cancer, evidence_type, api_key
            )
        elif model == "gpt4mini":
            verified, confidence, reason, model_used, extras = verify_with_gpt4mini(
                sentence, gene, drug, cancer, evidence_type, api_key
            )
        else:
            verified, confidence, reason, model_used, extras = verify_offline(
                sentence, gene, drug, cancer, evidence_type
            )

        # Apply confidence threshold
        if verified == "yes" and confidence < confidence_threshold:
            verified = "no"
            reason   = f"Below confidence threshold ({confidence:.2f} < {confidence_threshold})"

        if verified == "yes":
            verified_count += 1

        results.append({
            "llm_verified":        verified,
            "llm_confidence":      round(confidence, 3),
            "llm_reason":          reason,
            "llm_model":           model_used,
            "llm_direct_relation": extras.get("llm_direct_relation", ""),
            "llm_negated":         extras.get("llm_negated", ""),
            "llm_speculative":     extras.get("llm_speculative", ""),
            "llm_evidence_type":   extras.get("llm_evidence_type", ""),
        })

    print(f"   ✅ Verification complete ({total} rows)")

    # Add results to dataframe
    result_df = pd.DataFrame(results)

    # Handle case where max_rows < len(df)
    if max_rows and max_rows < len(df):
        # Fill remaining rows as skipped
        remaining = len(df) - max_rows
        skip_rows = pd.DataFrame([{
            "llm_verified":        "skipped",
            "llm_confidence":      0.0,
            "llm_reason":          "Beyond max_rows limit",
            "llm_model":           model,
            "llm_direct_relation": "",
            "llm_negated":         "",
            "llm_speculative":     "",
            "llm_evidence_type":   "",
        }] * remaining)
        result_df = pd.concat(
            [result_df, skip_rows], ignore_index=True
        )

    # Attach to original dataframe — includes new negation/speculation fields
    for col in ["llm_verified","llm_confidence","llm_reason","llm_model",
                "llm_direct_relation","llm_negated","llm_speculative","llm_evidence_type"]:
        df[col] = result_df[col].values

    # Print summary
    total_verified = (df["llm_verified"] == "yes").sum()
    total_rejected = (df["llm_verified"] == "no").sum()
    total_skipped  = (df["llm_verified"] == "skipped").sum()
    avg_confidence = df[df["llm_verified"]!="skipped"]["llm_confidence"].mean()

    print(f"\n   📊 Verification summary:")
    print(f"      ✅ Verified:  {total_verified}")
    print(f"      ❌ Rejected:  {total_rejected}")
    print(f"      ⏭️  Skipped:   {total_skipped}")
    print(f"      📈 Avg confidence: {avg_confidence:.3f}")

    if output_path:
        df.to_excel(output_path, index=False)
        print(f"   💾 Saved to {output_path}")

    return df
