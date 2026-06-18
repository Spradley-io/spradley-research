"""
evals_core.py -- Spradley Eval Pipeline: judge prompts and eval functions.

Imported by Pipeline_Evals.ipynb.
Organised by eval ID (E-numbers match the notebook cells).
To modify a judge prompt or eval logic, Ctrl+F its E-number here.
"""

import pipeline_core

JUDGE_MODEL = "claude-sonnet-4-6"


# ── E2b: L2 semantic merge quality ────────────────────────────────────────────

MERGE_JUDGE = (
    "You are a qualitative research auditor.\n\n"
    "A researcher merged the following L1 codes into a single L2 label.\n"
    "L1 codes (source): {sources}\n"
    "L2 label (result): {label}\n\n"
    "Does the L2 label faithfully capture the meaning of the L1 codes it absorbed?\n"
    "Respond only with valid JSON:\n"
    '{{"verdict": "faithful" or "unfaithful", '
    '"certainty": 0.0 to 1.0, '
    '"reason": "one concise sentence"}}'
)

def run_merge_quality_check(interview_store: dict) -> dict:
    """Judge every L2 merge for semantic faithfulness. Returns verdicts dict."""
    verdicts = []
    for interview_id, store in interview_store.items():
        for l2 in store["l2_codes"]:
            sources_str = ", ".join(l2["merged_from_l1"]) if l2["merged_from_l1"] else "(none)"
            prompt = MERGE_JUDGE.format(sources=sources_str, label=l2["code"])
            raw = pipeline_core.call_llm(prompt, model=JUDGE_MODEL)
            try:
                result = pipeline_core.parse_json_safe(raw)
            except Exception:
                result = {"verdict": "error", "certainty": 0.0, "reason": raw[:120]}
            verdicts.append({
                "interview":  interview_id[:8],
                "l2_code":    l2["code"],
                "sources":    l2["merged_from_l1"],
                **result,
            })
    unfaithful    = [v for v in verdicts if v.get("verdict") == "unfaithful"]
    low_certainty = [v for v in verdicts if v.get("verdict") == "faithful" and v.get("certainty", 1.0) < 0.8]
    return {"all": verdicts, "unfaithful": unfaithful, "low_certainty": low_certainty}


# ── E5: Narrative faithfulness ────────────────────────────────────────────────

FAITHFULNESS_JUDGE = (
    "You are a qualitative research auditor.\n\n"
    "EVIDENCE (anonymised employee answers):\n{context}\n\n"
    "NARRATIVE:\n{narrative}\n\n"
    "Does the narrative assert any sentiment, claim, or fact not clearly supported "
    "by the evidence above?\n"
    "Respond only with valid JSON:\n"
    '{{"verdict": "faithful" or "unfaithful", '
    '"certainty": 0.0 to 1.0 (how confident you are), '
    '"reason": "one concise sentence"}}'
)

def run_faithfulness_check(clusters: dict, lineage: dict, db: dict) -> dict:
    """Judge each cluster narrative for faithfulness to source Q&A. Returns verdicts dict."""
    verdicts = []
    for name, data in clusters.items():
        narrative = data.get("story") or data.get("summary", "")
        if not narrative:
            continue
        context = qa_context_for_cluster(name, lineage, db)
        prompt  = FAITHFULNESS_JUDGE.format(context=context, narrative=narrative)
        raw     = pipeline_core.call_llm(prompt, model=JUDGE_MODEL)
        try:
            result = pipeline_core.parse_json_safe(raw)
        except Exception:
            result = {"verdict": "error", "certainty": 0.0, "reason": raw[:120]}
        verdicts.append({"cluster": name, **result})
    unfaithful    = [v for v in verdicts if v.get("verdict") == "unfaithful"]
    low_certainty = [v for v in verdicts if v.get("verdict") == "faithful" and v.get("certainty", 1.0) < 0.8]
    return {"all": verdicts, "unfaithful": unfaithful, "low_certainty": low_certainty}


# ── E5b: Sentiment overclaim ──────────────────────────────────────────────────

OVERCLAIM_JUDGE = (
    "You are a qualitative research auditor.\n\n"
    "EVIDENCE (anonymised employee answers):\n{context}\n\n"
    "NARRATIVE:\n{narrative}\n\n"
    "Does this narrative assert positive sentiment (e.g. 'employees feel supported', "
    "'high morale') that the evidence does not clearly support? "
    "Include cases where the evidence is thin, neutral, or deflecting.\n"
    "Respond only with valid JSON:\n"
    '{{"verdict": "supported" or "unsupported", '
    '"certainty": 0.0 to 1.0, '
    '"reason": "one concise sentence"}}'
)

def run_overclaim_check(clusters: dict, lineage: dict, db: dict) -> dict:
    """Judge each narrative for unsupported positive sentiment. Returns verdicts dict."""
    verdicts = []
    for name, data in clusters.items():
        narrative = data.get("story") or data.get("summary", "")
        if not narrative:
            continue
        context = qa_context_for_cluster(name, lineage, db)
        prompt  = OVERCLAIM_JUDGE.format(context=context, narrative=narrative)
        raw     = pipeline_core.call_llm(prompt, model=JUDGE_MODEL)
        try:
            result = pipeline_core.parse_json_safe(raw)
        except Exception:
            result = {"verdict": "error", "certainty": 0.0, "reason": raw[:120]}
        verdicts.append({"cluster": name, **result})
    unsupported   = [v for v in verdicts if v.get("verdict") == "unsupported"]
    low_certainty = [v for v in verdicts if v.get("verdict") == "supported" and v.get("certainty", 1.0) < 0.8]
    return {"all": verdicts, "unsupported": unsupported, "low_certainty": low_certainty}


# ── E6: Metamorphic invariance ────────────────────────────────────────────────

PARAPHRASE_PROMPT = (
    "Paraphrase the following text. Preserve the exact meaning but change all "
    "wording substantially so no phrase matches the original.\n\n"
    "Original:\n{text}\n\nParaphrase:"
)

def paraphrase(text: str) -> str:
    """Return a paraphrase of text via JUDGE_MODEL."""
    return pipeline_core.call_llm(PARAPHRASE_PROMPT.format(text=text), model=JUDGE_MODEL)


# ── E7: Re-identification probe ───────────────────────────────────────────────

REID_PROMPT = (
    "You are a privacy auditor testing re-identification risk.\n\n"
    "ANONYMISED REPORT EXCERPT:\n{report_text}\n\n"
    "ROSTER ({n} people who were interviewed):\n{roster}\n\n"
    "For each paraphrased quote in the report, guess which person most likely said it.\n"
    "Return a JSON array:\n"
    '[{{"quote_preview": "first 8 words of quote", "guess": "Name", "confidence": 0.0-1.0}}, ...]'
)

def run_reidentification_probe(report_text: str, roster: list) -> dict:
    """Run the leakage probe. roster = [{'name': ..., 'role': ...}, ...]."""
    roster_str = "\n".join(f"- {p['name']}: {p['role']}" for p in roster)
    prompt     = REID_PROMPT.format(n=len(roster), report_text=report_text[:4000], roster=roster_str)
    raw        = pipeline_core.call_llm(prompt, model=JUDGE_MODEL)
    try:
        guesses = pipeline_core.parse_json_safe(raw)
        if not isinstance(guesses, list):
            guesses = guesses.get("guesses", [])
    except Exception:
        guesses = []
    high_conf = [g for g in guesses if g.get("confidence", 0) >= 0.7]
    return {"guesses": guesses, "high_conf": high_conf}


# ── Shared helper ─────────────────────────────────────────────────────────────

def qa_context_for_cluster(name: str, lineage: dict, db: dict) -> str:
    """Return concatenated Q&A text for all source entries of a cluster."""
    lin    = lineage.get(name, {})
    iq_ids = list({e["interview_question_id"] for e in lin.get("l1_codes", [])})
    lines  = []
    for iq_id in iq_ids:
        if iq_id in db:
            e = db[iq_id]
            lines.append(f"Q: {e['question']}\nA: {e['anonymised_answer']}")
    return "\n\n".join(lines) or "(no source Q&A found)"
