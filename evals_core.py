"""
Shared evaluation helpers for Pipeline_Evals.ipynb.
Import everything from here; keep the notebook cells thin.
"""
import os, json, io, base64, datetime, html as _html
import pipeline_core


# ── HTML escape ────────────────────────────────────────────────────────────────

def H(s: object) -> str:
    return _html.escape(str(s))


# ── File / cache helpers ───────────────────────────────────────────────────────

def load_json(output_dir: str, name: str):
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_eval_cache(cache_path: str) -> dict:
    try:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_eval_cache(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def save_eval_history(eval_results: dict, eval_html: str, history_dir: str) -> str:
    """Write a full eval snapshot (WITH figures + rendered report HTML) to history_dir.

    Each file is keyed by ISO datetime so runs can be compared or restored.
    """
    os.makedirs(history_dir, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path     = os.path.join(history_dir, f"{ts}.json")
    snapshot = dict(eval_results)          # shallow copy -- figures included
    snapshot["_report_html"] = eval_html
    snapshot["_run_ts"]      = ts
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    return path

def save_eval_results(eval_results: dict, output_dir: str) -> str:
    """Write a stable eval_results.json alongside the pipeline data files.

    Always overwrites the previous run. Figures are included so the checkpoint
    can fully regenerate the HTML report. Use eval_history/ for per-run snapshots.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    return path


# ── Chart / embedding helpers ──────────────────────────────────────────────────

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def partition_labels(cluster_map: dict, code_list: list) -> list:
    code_to_label = {}
    for idx, (name, codes) in enumerate(cluster_map.items()):
        for c in codes:
            code_to_label[c] = idx
    return [code_to_label.get(c, -1) for c in code_list]

def fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def qa_context_for_cluster(name: str, lineage: dict, db: dict) -> str:
    lin    = lineage.get(name, {})
    iq_ids = lin.get("l1_qa_ids", [])
    lines  = []
    for iq_id in iq_ids:
        if iq_id in db:
            e = db[iq_id]
            lines.append(f"Q: {e['question']}\nA: {e['anonymised_answer']}")
    return "\n\n".join(lines) or "(no source Q&A found)"


# ── Embedding model (sentence-transformers) ────────────────────────────────────

_ST_MODEL = None

def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _ST_MODEL

def soft_jaccard(a: list, b: list, tau: float = 0.7) -> float:
    """Cosine soft-Jaccard: fraction of codes with a semantic match above tau.

    Binary after threshold: each code in A is matched (or not) to any code in B
    by cosine >= tau. Same 0-1 range as exact Jaccard.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    import numpy as np
    model  = _get_st_model()
    emb_a  = model.encode(a, normalize_embeddings=True)
    emb_b  = model.encode(b, normalize_embeddings=True)
    sims   = emb_a @ emb_b.T          # shape (|a|, |b|)
    m_a    = int((sims.max(axis=1) >= tau).sum())
    m_b    = int((sims.max(axis=0) >= tau).sum())
    s_int  = (m_a + m_b) / 2
    s_uni  = len(a) + len(b) - s_int
    return float(s_int / s_uni) if s_uni > 0 else 1.0


# ── Judge prompts ──────────────────────────────────────────────────────────────

MERGE_JUDGE = (
    "You are auditing a qualitative coding step.\n\n"
    "An analyst read the following Q&A turns from one employee interview "
    "and assigned the code label shown below.\n\n"
    "Source Q&A turns:\n{sources}\n\n"
    "Assigned code label: \"{label}\"\n\n"
    "Is the label a faithful summary of the themes in those turns?\n"
    "Reply with JSON only:\n"
    '{{\"verdict\": \"faithful\"|\"unfaithful\", '
    '"certainty": 0.0-1.0, "reason": "one sentence"}}'
)

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

PARAPHRASE_PROMPT = (
    "Paraphrase the following text. Preserve the exact meaning but change all "
    "wording substantially so no phrase matches the original.\n\n"
    "Original:\n{text}\n\nParaphrase:"
)

NEGATE_PROMPT = (
    "Rewrite the following employee survey answer so that its sentiment or meaning is reversed.\n"
    "Keep the same length and style. Preserve any placeholder like [NAME] exactly as-is.\n\n"
    "Original: {answer}\n\n"
    "Reversed version (only the text, no explanation):"
)

PROFILE_GEN_PROMPT = (
    "You are a privacy analyst.\n\n"
    "Read this employee's complete interview answers and write a brief factual profile "
    "capturing their most distinctive characteristics: role signals, unique situations "
    "they described, specific experiences, and any tenure or team hints. "
    "Do not speculate beyond what is stated.\n\n"
    "Interview answers:\n{qa_text}\n\n"
    "Write a 3-5 sentence profile. Profile:"
)

REID_PROMPT = (
    "You are a team colleague who has read the profiles of everyone who was interviewed.\n"
    "Your task: for each paraphrased quote from the team survey report, "
    "decide which person most likely said it.\n\n"
    "A quote is linkable if its detail, phrasing, or situation narrows it to one person. "
    "Assign a confidence score (0.0-1.0): 0.7+ = quite sure, 0.4-0.69 = reasonable suspicion, "
    "below 0.4 = guessing.\n\n"
    "Team profiles:\n{profiles_text}\n\n"
    "Quotes from the report to match:\n{quotes_text}\n\n"
    "Return only valid JSON:\n"
    '{{\"matches\": [{{'
    '"quote": "first 12 words of quote...", '
    '"person_id": "interview_prefix or unknown", '
    '"confidence": 0.0, '
    '"reasoning": "one sentence"'
    '}}]}}'
)

POLARITY_SCREEN_PROMPT = (
    "You are reviewing a list of employee survey answers.\n"
    "Identify the IDs of answers that contain CLEAR positive or negative sentiment "
    "(happiness, frustration, pride, disappointment, enthusiasm, stress, etc.).\n"
    "Skip neutral or factual answers where no sentiment is expressed.\n\n"
    "Answers to review:\n{answers_text}\n\n"
    "Return only valid JSON:\n"
    '{{\"polarity_ids\": [\"id1\", \"id2\", ...]}}'
)


# ── Eval functions ─────────────────────────────────────────────────────────────

def paraphrase(text: str, model: str) -> str:
    return pipeline_core.call_llm(PARAPHRASE_PROMPT.format(text=text), model=model)


def run_merge_quality_check(interview_store: dict, db: dict, model: str) -> dict:
    """Judge whether each L2 label faithfully represents its cited Q&A turns."""
    verdicts = []
    for interview_id, store in interview_store.items():
        for l2 in store.get("l2_codes", []):
            qa_ids = l2.get("source_qa_ids", [])
            qa_lines = []
            for iq_id in qa_ids:
                if iq_id in db:
                    e = db[iq_id]
                    qa_lines.append(f"Q: {e['question']}\nA: {e['anonymised_answer']}")
            if not qa_lines:
                continue
            sources_str = "\n\n".join(qa_lines)
            prompt = MERGE_JUDGE.format(sources=sources_str, label=l2["code"])
            raw = pipeline_core.call_llm(prompt, model=model)
            try:
                result = pipeline_core.parse_json_safe(raw)
            except Exception:
                result = {"verdict": "error", "certainty": 0.0, "reason": raw[:120]}
            verdicts.append({
                "interview": interview_id[:8],
                "l2_code":  l2["code"],
                "sources":  qa_ids,
                "qa_pairs": [
                    {"q": db[iq]["question"], "a": db[iq]["anonymised_answer"]}
                    for iq in qa_ids if iq in db
                ],
                **result,
            })
    unfaithful    = [v for v in verdicts if v.get("verdict") == "unfaithful"]
    low_certainty = [v for v in verdicts if v.get("verdict") == "faithful"
                     and v.get("certainty", 1.0) < 0.8]
    return {"all": verdicts, "unfaithful": unfaithful, "low_certainty": low_certainty}


def run_faithfulness_check(clusters: dict, lineage: dict, db: dict, model: str) -> dict:
    verdicts = []
    for name, data in clusters.items():
        narrative = data.get("story") or data.get("summary", "")
        if not narrative:
            continue
        context  = qa_context_for_cluster(name, lineage, db)
        iq_ids   = lineage.get(name, {}).get("l1_qa_ids", [])
        qa_pairs = [
            {"q": db[iq]["question"], "a": db[iq]["anonymised_answer"]}
            for iq in iq_ids if iq in db
        ][:8]
        prompt  = FAITHFULNESS_JUDGE.format(context=context, narrative=narrative)
        raw     = pipeline_core.call_llm(prompt, model=model)
        try:
            result = pipeline_core.parse_json_safe(raw)
        except Exception:
            result = {"verdict": "error", "certainty": 0.0, "reason": raw[:120]}
        verdicts.append({"cluster": name, "qa_pairs": qa_pairs, **result})
    unfaithful    = [v for v in verdicts if v.get("verdict") == "unfaithful"]
    low_certainty = [v for v in verdicts if v.get("verdict") == "faithful"
                     and v.get("certainty", 1.0) < 0.8]
    return {"all": verdicts, "unfaithful": unfaithful, "low_certainty": low_certainty}


def run_overclaim_check(clusters: dict, lineage: dict, db: dict, model: str) -> dict:
    verdicts = []
    for name, data in clusters.items():
        narrative = data.get("story") or data.get("summary", "")
        if not narrative:
            continue
        context  = qa_context_for_cluster(name, lineage, db)
        iq_ids   = lineage.get(name, {}).get("l1_qa_ids", [])
        qa_pairs = [
            {"q": db[iq]["question"], "a": db[iq]["anonymised_answer"]}
            for iq in iq_ids if iq in db
        ][:8]
        prompt  = OVERCLAIM_JUDGE.format(context=context, narrative=narrative)
        raw     = pipeline_core.call_llm(prompt, model=model)
        try:
            result = pipeline_core.parse_json_safe(raw)
        except Exception:
            result = {"verdict": "error", "certainty": 0.0, "reason": raw[:120]}
        verdicts.append({"cluster": name, "qa_pairs": qa_pairs, **result})
    unsupported   = [v for v in verdicts if v.get("verdict") == "unsupported"]
    low_certainty = [v for v in verdicts if v.get("verdict") == "supported"
                     and v.get("certainty", 1.0) < 0.8]
    return {"all": verdicts, "unsupported": unsupported, "low_certainty": low_certainty}


def run_reidentification_probe(
    clusters: dict, interview_store: dict, db: dict, model: str
) -> dict:
    """Generate realistic interview profiles then run adversarial quote matching.

    Step 1: One LLM call per interview to build a factual profile from their Q&A.
    Step 2: One adversary call to match report quotes against all profiles.
    Returns {matches, high_conf, at_risk, profiles}.
    """
    from collections import defaultdict

    # Build by_interview map
    by_interview: dict = defaultdict(list)
    for iq_id, entry in db.items():
        by_interview[entry["interview_id"]].append((iq_id, entry))

    # Step 1: generate one profile per interview
    profiles = {}
    for iid, entries in by_interview.items():
        sorted_entries = sorted(entries, key=lambda x: int(x[1]["turn_number"]))
        qa_text = "\n\n".join(
            f"Q: {e['question']}\nA: {e['anonymised_answer']}"
            for _, e in sorted_entries
        )
        raw = pipeline_core.call_llm(PROFILE_GEN_PROMPT.format(qa_text=qa_text), model=model)
        profiles[iid[:8]] = raw.strip()

    # Step 2: extract quotes from clusters
    quotes = []
    for data in clusters.values():
        quotes.extend(data.get("quotes", []))
    if not quotes:
        return {"matches": [], "high_conf": [], "at_risk": [], "profiles": profiles}

    profiles_text = "\n\n".join(f"[{pid}]\n{p}" for pid, p in profiles.items())
    quotes_text   = "\n".join(f"- {q}" for q in quotes)
    raw = pipeline_core.call_llm(
        REID_PROMPT.format(profiles_text=profiles_text, quotes_text=quotes_text),
        model=model,
    )
    try:
        parsed  = pipeline_core.parse_json_safe(raw)
        matches = parsed if isinstance(parsed, list) else parsed.get("matches", [])
    except Exception:
        matches = []

    high_conf = [m for m in matches if m.get("confidence", 0) >= 0.7]
    at_risk   = [m for m in matches if m.get("confidence", 0) >= 0.4]
    return {"matches": matches, "high_conf": high_conf, "at_risk": at_risk, "profiles": profiles}


# ── HTML report ────────────────────────────────────────────────────────────────

EVAL_DESCRIPTIONS = {
    "E1": (
        "Re-runs the L2 coder 3 times per interview and compares the resulting code sets. "
        "Cosine soft-Jaccard is the headline: it counts meaning-preserving relabels as matches, "
        "separating genuine drift from cosmetic variation. Exact Jaccard is shown alongside "
        "as a secondary reference. A large gap between the two means apparent instability "
        "was mostly cosmetic relabeling, not genuine drift. Measures consistency, not correctness."
    ),
    "E2b": (
        "A Claude judge reads each L2 code label alongside the actual Q&A turns it was derived "
        "from, and decides whether the label faithfully summarises the content. Failures indicate "
        "a code label that misrepresents or over-generalises its source material."
    ),
    "EL": (
        "Audits structural integrity across all three coding layers. "
        "Checks that every Q&A source ID cited by an L2 code exists in the database, "
        "tracks how many codes survive each funnel step (Q&A to L2 to L3 to cluster), "
        "and flags duplicate assignments or broken pointers. "
        "Dropped codes are listed layer-by-layer in the detail section."
    ),
    "E4": (
        "Re-groups the same L3 codes 3 times and measures whether the resulting cluster "
        "assignments agree. ARI (Adjusted Rand Index) subtracts the agreement expected by "
        "chance, so 0 = random grouping and 1 = identical. Low ARI means reported themes "
        "are partly an artefact of random LLM sampling."
    ),
    "E5": (
        "A Claude judge reads each cluster narrative alongside the anonymised employee "
        "responses that support it, and flags claims not clearly grounded in the evidence."
    ),
    "E5b": (
        "Same as E5, focused specifically on positive sentiment assertions. "
        "Positive framing unsupported by evidence is the most likely overclaim direction "
        "in a management-facing report."
    ),
    "E6": (
        "Two robustness tests. E6a: rewrites sample answers in different words per interview "
        "and checks whether the same themes still emerge (cosine soft-Jaccard, gate 0.5). "
        "E6b: shuffles the L3 code list and re-clusters to check that groupings are "
        "order-independent (ARI, gate 0.7)."
    ),
    "E7": (
        "Generates a realistic profile for each interviewee from their own answers, then "
        "an adversarial Claude tries to match paraphrased quotes in the report to those profiles. "
        "A match with confidence >= 0.7 is a privacy flag. Matches >= 0.4 are flagged for review. "
        "This tests whether the report retains enough distinctive detail for a knowledgeable "
        "reader to re-identify individuals."
    ),
    "E8": (
        "Negates clearly positive or negative statements and re-codes the interview with the "
        "negated turn substituted in. A correct coder should respond: low code-set overlap "
        "with the original is a pass. High overlap means the coder missed the negation and "
        "would report the opposite of what was actually said."
    ),
}

EVAL_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,sans-serif;
     background:#fff;color:#111;font-size:15px;line-height:1.6}
.hdr{display:flex;align-items:center;gap:14px;padding:14px 40px;
     border-bottom:1px solid #e5e5e5;background:#fff}
.logo{width:36px;height:36px;background:#000;border-radius:8px;color:#fff;
      font-weight:800;font-size:13px;display:flex;align-items:center;justify-content:center}
.hdr-t{font-size:16px;font-weight:700}
.hdr-m{font-size:12px;color:#999}
.wrap{max-width:860px;margin:0 auto;padding:36px 24px 60px}
.ev-links{font-size:12px;color:#888;margin-bottom:18px}
.ev-links a{color:#0369a1;text-decoration:none}
.ev-links a:hover{text-decoration:underline}
.ev-caveat{font-size:12px;color:#b45309;background:#fef3c7;border-radius:6px;
           padding:9px 13px;margin-bottom:22px;border-left:3px solid #f59e0b}
.sec{margin-bottom:28px;border:1px solid #e8e8e8;border-radius:10px;overflow:hidden}
.sec-hdr{display:flex;align-items:center;gap:10px;padding:12px 18px;
         border-bottom:1px solid #e8e8e8;background:#fafafa}
.pass{background:#dcfce7;color:#15803d;border-radius:20px;padding:2px 10px;
      font-size:12px;font-weight:700}
.fail{background:#fee2e2;color:#dc2626;border-radius:20px;padding:2px 10px;
      font-size:12px;font-weight:700}
.warn{background:#fef3c7;color:#b45309;border-radius:20px;padding:2px 10px;
      font-size:12px;font-weight:700}
.sec-title{font-size:14px;font-weight:700}
.sec-sub{font-size:11px;color:#aaa;margin-left:auto}
.sec-body{padding:14px 18px}
.eval-desc{font-size:12px;color:#666;padding:9px 12px;background:#f9f9f9;
           border-radius:6px;border-left:3px solid #d4d4d4;margin-bottom:10px}
.sec-summary{font-size:13px;color:#555;margin-bottom:8px}
.fig img{max-width:100%;border-radius:6px;margin-top:8px}
.footer{text-align:center;font-size:12px;color:#ccc;padding:28px 0 8px}
.e1-subhdr{font-size:13px;font-weight:600;color:#333;margin:16px 0 8px}
.e1-scroll{overflow-x:auto}
.e1-tbl{width:100%;border-collapse:collapse;font-size:12px}
.e1-tbl th{background:#f4f4f4;padding:6px 10px;text-align:left;
           border-bottom:2px solid #e0e0e0;white-space:nowrap;font-weight:600}
.e1-tbl td{padding:8px 10px;border-bottom:1px solid #efefef;vertical-align:top}
.e1-score{font-weight:700;font-size:13px;color:#dc2626}
.e1-score2{font-weight:600;font-size:11px;color:#888}
.e1-id{font-size:10px;color:#aaa;font-family:monospace}
.e1-codes{font-size:11px;color:#444;line-height:1.8}
.tag{display:inline-block;background:#e0f2fe;color:#0369a1;border-radius:4px;
     padding:1px 7px;margin:2px 2px 0 0;font-size:11px;white-space:nowrap}
.tag-warn{background:#fef3c7;color:#b45309}
.tag-fail{background:#fee2e2;color:#dc2626}
.vt-scroll{overflow-x:auto;margin-top:10px}
.vt-tbl{width:100%;border-collapse:collapse;font-size:12px}
.vt-tbl th{background:#f4f4f4;padding:6px 10px;text-align:left;
           border-bottom:2px solid #e0e0e0;font-weight:600;white-space:nowrap}
.vt-tbl td{padding:8px 10px;border-bottom:1px solid #efefef;vertical-align:top}
.vt-badge{display:inline-block;border-radius:4px;padding:1px 8px;
          font-size:11px;font-weight:700}
.vt-bad{background:#fee2e2;color:#dc2626}
.vt-warn{background:#fef3c7;color:#b45309}
.vt-cluster{font-weight:600;font-size:12px;min-width:140px}
.vt-reason{font-size:12px;color:#444}
.vt-cert{font-size:12px;color:#888;white-space:nowrap;text-align:right}
.lin-layer{font-size:12px;font-weight:700;margin:14px 0 6px;padding:4px 8px;
           border-radius:4px;display:inline-block}
.lin-l1{background:#fef3c7;color:#b45309}
.lin-l2{background:#fef3c7;color:#b45309}
.lin-l3{background:#fee2e2;color:#dc2626}
.lin-struct{background:#fee2e2;color:#dc2626}
.lin-qa-item{font-size:11px;padding:5px 0;border-bottom:1px solid #f5f5f5;color:#555}
.lin-qa-id{font-family:monospace;font-size:10px;color:#aaa;margin-right:6px}
.lin-qa-q{color:#777;font-size:11px}
details.det{margin-top:10px;border-top:1px solid #efefef;padding-top:8px}
details.det summary.det-sum{font-size:12px;font-weight:600;color:#0369a1;
  cursor:pointer;user-select:none;list-style:none;display:inline-flex;
  align-items:center;gap:6px;padding:2px 0}
details.det summary.det-sum::-webkit-details-marker{display:none}
details.det summary.det-sum::before{content:"\\25B6";font-size:9px;
  display:inline-block;transition:transform 0.15s}
details.det[open] summary.det-sum::before{transform:rotate(90deg)}
details.det .det-body{padding-top:10px}
details.qa-collapse{margin:5px 0 4px 0}
details.qa-collapse summary{cursor:pointer;font-size:11px;color:#0369a1;font-weight:600;
  user-select:none;list-style:none;display:inline-flex;align-items:center;gap:5px;padding:1px 0}
details.qa-collapse summary::-webkit-details-marker{display:none}
details.qa-collapse summary::before{content:"\\25B6";font-size:8px;display:inline-block;
  transition:transform 0.15s}
details.qa-collapse[open] summary::before{transform:rotate(90deg)}
.qa-block{padding:8px 12px;background:#f8f8f8;border-left:3px solid #d4d4d4;
  margin-top:5px;font-size:11px;line-height:1.65;border-radius:0 4px 4px 0}
.qa-block p{margin:2px 0}
.qa-neg{color:#b00020}
"""


def _e1_worst_html(details: list) -> str:
    """Render 'worst N interviews' table for E1. Each entry: {iid, cosine_j, exact_j, runs}."""
    if not details:
        return ""
    n_runs = len(details[0].get("runs", []))
    run_headers = "".join(f"<th>Run {i+1} codes</th>" for i in range(n_runs))
    rows = ""
    for d in details:
        run_cols = "".join(
            f'<td class="e1-codes">{"<br>".join(H(c) for c in codes)}</td>'
            for codes in d.get("runs", [])
        )
        rows += (
            f"<tr>"
            f'<td><span class="e1-score">soft-J={d.get("cosine_j", 0):.2f}</span><br>'
            f'<span class="e1-score2">exact-J={d.get("exact_j", 0):.2f}</span><br>'
            f'<span class="e1-id">{H(d.get("iid", ""))}</span></td>'
            f"{run_cols}"
            f"</tr>"
        )
    return (
        f'<h4 class="e1-subhdr">Least stable interviews (by cosine soft-Jaccard)</h4>'
        f'<div class="e1-scroll"><table class="e1-tbl">'
        f"<thead><tr><th>Scores</th>{run_headers}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>"
    )


def _qa_collapse_html(qa_pairs: list) -> str:
    """Collapsible source Q&A block for embedding inside a table cell."""
    if not qa_pairs:
        return ""
    pairs_html = "".join(
        f'<div class="qa-block">'
        f'<p><strong>Q:</strong> {H(p.get("q", ""))}</p>'
        f'<p><strong>A:</strong> {H(p.get("a", ""))}</p>'
        f'</div>'
        for p in qa_pairs
    )
    return (
        f'<details class="qa-collapse">'
        f'<summary>Source Q&amp;A ({len(qa_pairs)} pairs)</summary>'
        f'{pairs_html}'
        f'</details>'
    )


def _verdict_table_html(items_fail: list, items_review: list, fail_label: str = "FAIL") -> str:
    if not items_fail and not items_review:
        return ""
    rows = ""
    for v in items_fail:
        cert     = v.get("certainty")
        cert_str = f"{cert:.0%}" if isinstance(cert, float) else str(cert)
        qa_html  = _qa_collapse_html(v.get("qa_pairs", []))
        rows += (
            f"<tr>"
            f'<td><span class="vt-badge vt-bad">{H(fail_label)}</span></td>'
            f'<td class="vt-cluster">{H(v.get("cluster", ""))}{qa_html}</td>'
            f'<td class="vt-reason">{H(v.get("reason", ""))}</td>'
            f'<td class="vt-cert">{cert_str}</td>'
            f"</tr>"
        )
    for v in items_review:
        cert     = v.get("certainty")
        cert_str = f"{cert:.0%}" if isinstance(cert, float) else str(cert)
        qa_html  = _qa_collapse_html(v.get("qa_pairs", []))
        rows += (
            f"<tr>"
            f'<td><span class="vt-badge vt-warn">REVIEW</span></td>'
            f'<td class="vt-cluster">{H(v.get("cluster", ""))}{qa_html}</td>'
            f'<td class="vt-reason">{H(v.get("reason", ""))}</td>'
            f'<td class="vt-cert">{cert_str}</td>'
            f"</tr>"
        )
    return (
        f'<div class="vt-scroll"><table class="vt-tbl">'
        f"<thead><tr><th>Status</th><th>Theme</th><th>Reason</th><th>Certainty</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>"
    )


def _e2b_table_html(unfaithful: list, low_certainty: list) -> str:
    if not unfaithful and not low_certainty:
        return ""
    rows = ""
    for v in unfaithful + low_certainty:
        is_fail  = v in unfaithful
        badge    = ('<span class="vt-badge vt-bad">UNFAITHFUL</span>' if is_fail
                    else '<span class="vt-badge vt-warn">REVIEW</span>')
        cert     = v.get("certainty")
        cert_str = f"{cert:.0%}" if isinstance(cert, float) else str(cert)
        # show full Q&A pairs when available (new format), else fall back to IDs
        qa_pairs = v.get("qa_pairs", [])
        if qa_pairs:
            src_html = "".join(
                f'<div class="lin-qa-item" style="margin-bottom:8px;padding-bottom:6px;'
                f'border-bottom:1px solid #f0f0f0">'
                f'<div class="lin-qa-q" style="margin-bottom:2px">Q: {H(pair["q"][:120])}</div>'
                f'<div style="font-size:11px;color:#444">A: {H(pair["a"][:160])}</div>'
                f'</div>'
                for pair in qa_pairs
            )
        else:
            src_html = "".join(
                f'<div class="lin-qa-item"><span class="lin-qa-id">{H(src)}</span></div>'
                for src in v.get("sources", [])
            )
        rows += (
            f"<tr>"
            f"<td>{badge}</td>"
            f'<td class="vt-cluster">{H(v.get("l2_code", ""))}'
            f'<div style="font-size:10px;color:#aaa;font-weight:400;margin-top:2px">'
            f'interview {H(v.get("interview", ""))}</div></td>'
            f'<td style="font-size:11px;line-height:1.6;max-width:320px">{src_html or "<em>none</em>"}</td>'
            f'<td class="vt-reason">{H(v.get("reason", ""))}</td>'
            f'<td class="vt-cert">{cert_str}</td>'
            f"</tr>"
        )
    return (
        f'<div class="vt-scroll"><table class="vt-tbl">'
        f"<thead><tr><th>Status</th><th>L2 Label</th><th>Source Q&amp;A</th>"
        f"<th>Reason</th><th>Certainty</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>"
    )


def _lineage_integrity_html(
    uncited_qa: list,   # [{iq_id, question}]
    dropped_l2: list,   # [str]
    dropped_l3: list,   # [str]
    structural: list,   # [str] - duplicate/bad pointer messages
) -> str:
    if not any([uncited_qa, dropped_l2, dropped_l3, structural]):
        return ""
    parts = []

    if uncited_qa:
        items_html = "".join(
            f'<div class="lin-qa-item">'
            f'<span class="lin-qa-id">{H(item["iq_id"])}</span>'
            f'<span class="lin-qa-q">{H(item["question"][:80])}</span>'
            f'</div>'
            for item in uncited_qa
        )
        parts.append(
            f'<div><span class="lin-layer lin-l1">L1: Uncited Q&amp;A turns ({len(uncited_qa)})</span>'
            f'<div>{items_html}</div></div>'
        )

    if dropped_l2:
        tags = "".join(
            f'<span class="tag tag-warn">{H(c)}</span>' for c in dropped_l2
        )
        parts.append(
            f'<div style="margin-top:12px">'
            f'<span class="lin-layer lin-l2">L2: Codes not absorbed into L3 ({len(dropped_l2)})</span>'
            f'<div style="margin-top:5px">{tags}</div></div>'
        )

    if dropped_l3:
        tags = "".join(
            f'<span class="tag tag-fail">{H(c)}</span>' for c in dropped_l3
        )
        parts.append(
            f'<div style="margin-top:12px">'
            f'<span class="lin-layer lin-l3">L3: Codes not assigned to a cluster ({len(dropped_l3)})</span>'
            f'<div style="margin-top:5px">{tags}</div></div>'
        )

    if structural:
        items_html = "".join(
            f'<div class="lin-qa-item">{H(msg)}</div>' for msg in structural
        )
        parts.append(
            f'<div style="margin-top:12px">'
            f'<span class="lin-layer lin-struct">Structural issues ({len(structural)})</span>'
            f'<div>{items_html}</div></div>'
        )

    return "".join(parts)


def _e7_html(at_risk: list, profiles: dict) -> str:
    """Render E7 results: matches with confidence >= 0.4 + collapsible profiles."""
    if not at_risk:
        html = (
            '<p style="font-size:13px;color:#15803d;font-weight:600">'
            'No quotes matched with meaningful confidence (>= 0.4) -- privacy check passed.</p>'
        )
    else:
        rows = ""
        for m in at_risk:
            conf      = m.get("confidence", 0)
            conf_str  = f"{conf:.0%}"
            row_style = 'style="background:#fff5f5"' if conf >= 0.7 else 'style="background:#fffbeb"'
            badge     = ('<span class="vt-badge vt-bad">HIGH RISK</span>' if conf >= 0.7
                         else '<span class="vt-badge vt-warn">REVIEW</span>')
            rows += (
                f"<tr {row_style}>"
                f"<td>{badge}</td>"
                f'<td style="font-size:12px;max-width:280px">{H(m.get("quote", ""))}</td>'
                f'<td style="font-size:12px;font-family:monospace">{H(m.get("person_id", ""))}</td>'
                f'<td style="font-size:12px;font-weight:700;text-align:right">{conf_str}</td>'
                f'<td class="vt-reason">{H(m.get("reasoning", ""))}</td>'
                f"</tr>"
            )
        html = (
            f'<div class="vt-scroll"><table class="vt-tbl">'
            f"<thead><tr><th>Risk</th><th>Quote</th><th>Matched to</th>"
            f"<th>Confidence</th><th>Reasoning</th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            f"</table></div>"
        )

    # Collapsible profiles section
    profiles_html = "".join(
        f'<div style="margin-bottom:10px">'
        f'<div style="font-size:11px;font-family:monospace;color:#888;margin-bottom:3px">[{H(pid)}]</div>'
        f'<div style="font-size:12px;color:#444">{H(p)}</div></div>'
        for pid, p in profiles.items()
    )
    profiles_block = (
        f'<details class="det" style="margin-top:14px">'
        f'<summary class="det-sum">Generated interview profiles</summary>'
        f'<div class="det-body">{profiles_html}</div></details>'
    ) if profiles_html else ""

    return html + profiles_block


def _e8_html(items: list) -> str:
    """Render E8 negation results: worst 5 MISSED first, then RESPONDED. Collapsible Q&A per row."""
    if not items:
        return ""
    missed    = sorted([x for x in items if not x.get("responded", True)],
                       key=lambda x: -x.get("score", 0))[:5]
    responded = [x for x in items if x.get("responded", False)]
    display   = missed + responded
    rows = ""
    for i, item in enumerate(display, 1):
        score     = item.get("score", 0)
        responded_ = item.get("responded", False)
        badge     = ('<span class="vt-badge" style="background:#dcfce7;color:#15803d">RESPONDED</span>'
                     if responded_
                     else '<span class="vt-badge vt-bad">MISSED</span>')
        # Collapsible Q&A block (graceful fallback for old cache entries)
        q_text    = item.get("q_text", "")
        orig_ans  = item.get("orig_ans", "")
        neg_ans   = item.get("neg_ans", "")
        orig_codes = ", ".join(H(c) for c in item.get("orig_codes", []))
        neg_codes  = ", ".join(H(c) for c in item.get("neg_codes", []))
        qa_inner = ""
        if q_text:
            qa_inner += f'<p><strong>Q:</strong> {H(q_text)}</p>'
        if orig_ans:
            qa_inner += f'<p><strong>Original A:</strong> {H(orig_ans)}</p>'
        if neg_ans:
            qa_inner += f'<p class="qa-neg"><strong>Negated A:</strong> {H(neg_ans)}</p>'
        if orig_codes:
            qa_inner += f'<p><strong>Original codes:</strong> {orig_codes}</p>'
        if neg_codes:
            qa_inner += f'<p><strong>Negated codes:</strong> {neg_codes}</p>'
        qa_block = (
            f'<details class="qa-collapse">'
            f'<summary>Source Q&amp;A</summary>'
            f'<div class="qa-block">{qa_inner}</div>'
            f'</details>'
        ) if qa_inner else ""
        rows += (
            f"<tr>"
            f"<td style='font-size:12px;color:#888'>{i}</td>"
            f"<td>{badge}</td>"
            f'<td style="font-size:11px;color:#888;font-family:monospace">'
            f'{H(item.get("iq_id",""))}{qa_block}</td>'
            f'<td style="font-size:12px;text-align:right">{score:.2f}</td>'
            f"</tr>"
        )
    return (
        f'<div class="vt-scroll"><table class="vt-tbl">'
        f"<thead><tr><th>#</th><th>Result</th><th>Q&amp;A ID</th>"
        f"<th>Soft-Jaccard (lower = better)</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>"
    )


def _e4_detail_html(worst: list) -> str:
    """Table of the 5 least-stable L3 codes across clustering runs."""
    if not worst:
        return ""
    top5    = worst[:5]
    n_runs  = len(top5[0].get("assignments", [])) if top5 else 3
    run_hdr = "".join(f"<th>Run {i+1}</th>" for i in range(n_runs))
    rows    = ""
    for item in top5:
        assignments = item.get("assignments", [])
        all_same    = len(set(assignments)) == 1
        row_style   = 'style="background:#fffbeb"' if not all_same else ""
        run_cols    = "".join(
            f'<td style="font-size:11px;color:#444">{H(a)}</td>'
            for a in assignments
        )
        stab     = item.get("stability", 0)
        stab_pct = f"{stab:.0%}"
        stab_col = "#15803d" if stab >= 1.0 else "#dc2626"
        rows += (
            f"<tr {row_style}>"
            f'<td style="font-size:11px;font-weight:600;max-width:200px">{H(item.get("code",""))}</td>'
            f"{run_cols}"
            f'<td style="font-weight:700;color:{stab_col};text-align:right">{stab_pct}</td>'
            f"</tr>"
        )
    return (
        f'<h4 style="font-size:12px;font-weight:700;margin:14px 0 8px;color:#333">'
        f'5 least-stable codes (yellow = placed in different clusters across runs)</h4>'
        f'<div class="vt-scroll"><table class="vt-tbl">'
        f"<thead><tr><th>L3 Code</th>{run_hdr}<th>Stability</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>"
    )


def _e6_detail_html(worst_para: list, worst_order: dict) -> str:
    """Detail blocks for E6a (worst paraphrase cases) and E6b (codes that shifted on reorder)."""
    parts = []

    if worst_para:
        rows = ""
        for item in worst_para[:5]:
            score      = item.get("score", 0)
            orig_list  = item.get("orig_codes", [])
            para_list  = item.get("para_codes", [])
            orig_str   = H(", ".join(orig_list[:5]) + ("..." if len(orig_list) > 5 else ""))
            para_str   = H(", ".join(para_list[:5]) + ("..." if len(para_list) > 5 else ""))
            score_col  = "#dc2626" if score < 0.5 else "#444"
            rows += (
                f"<tr>"
                f'<td style="font-size:11px;font-family:monospace;color:#888">'
                f'{H(str(item.get("iid",""))[:8])}</td>'
                f'<td style="font-weight:700;color:{score_col};text-align:right">{score:.2f}</td>'
                f'<td style="font-size:11px;color:#444">{orig_str}</td>'
                f'<td style="font-size:11px;color:#444">{para_str}</td>'
                f"</tr>"
            )
        parts.append(
            f'<h4 style="font-size:12px;font-weight:700;margin:14px 0 8px;color:#333">'
            f'E6a: Paraphrase robustness -- worst cases</h4>'
            f'<div class="vt-scroll"><table class="vt-tbl">'
            f"<thead><tr><th>Interview</th><th>Score</th>"
            f"<th>Original codes (first 5)</th><th>Paraphrased codes (first 5)</th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            f"</table></div>"
        )

    if worst_order:
        shifted   = worst_order.get("codes_shifted", [])
        n_shifted = worst_order.get("n_shifted", len(shifted))
        ari       = worst_order.get("ari", 0.0)
        if shifted:
            tags = "".join(f'<span class="tag tag-warn">{H(c)}</span>' for c in shifted[:20])
            parts.append(
                f'<h4 style="font-size:12px;font-weight:700;margin:18px 0 8px;color:#333">'
                f'E6b: Order invariance -- {n_shifted} code(s) placed differently after shuffle '
                f'(ARI={ari:.3f})</h4>'
                f'<div style="margin-top:5px">{tags}</div>'
            )
        else:
            parts.append(
                f'<p style="font-size:12px;color:#15803d;margin-top:10px;font-weight:600">'
                f'E6b: No codes shifted -- perfect order invariance (ARI={ari:.3f}).</p>'
            )

    return "".join(parts)


def _sec(eval_results: dict, eid: str, title: str, dyn: str = "", extra_html: str = "") -> str:
    r     = eval_results.get(eid, {})
    ok    = r.get("passed")
    badge = ('<span class="pass">PASS</span>' if ok is True
             else '<span class="fail">FAIL</span>' if ok is False
             else '<span class="warn">NOT RUN</span>')
    sub       = f'<span class="sec-sub">{H(dyn)}</span>' if dyn else ""
    desc      = EVAL_DESCRIPTIONS.get(eid, "")
    desc_html = f'<p class="eval-desc">{H(desc)}</p>' if desc else ""
    summ      = H(r.get("summary", "Not run."))
    figs      = "".join(
        f'<div class="fig"><img src="data:image/png;base64,{fig}" alt="{H(eid)}"></div>'
        for fig in r.get("figures", [])
    )
    extra_block = (
        f'<details class="det"><summary class="det-sum">Show details</summary>'
        f'<div class="det-body">{extra_html}</div></details>'
    ) if extra_html else ""
    return (
        f'<div class="sec"><div class="sec-hdr">{badge}'
        f'<span class="sec-title">{H(eid)}: {H(title)}</span>{sub}</div>'
        f'<div class="sec-body">{desc_html}'
        f'<p class="sec-summary">{summ}</p>{figs}{extra_block}</div></div>'
    )


def build_eval_html(eval_results: dict, run_meta: dict,
                    N: int = 0, SMALL_N: int = 30) -> str:
    meta_str   = " | ".join(f"{k}: {v}" for k, v in run_meta.items()) if run_meta else "no run_meta"
    all_passed = all(r.get("passed") is True for r in eval_results.values())
    overall    = ('<span class="pass">ALL PASS</span>' if all_passed
                  else '<span class="fail">ISSUES FOUND</span>')

    # Sample-size caveat
    if N and N < SMALL_N:
        caveat = (
            f"This eval ran on {N} interviews. At this size every score is a rough indicator, "
            f"not a stable estimate; one unusual interview can swing a metric. "
            f"Treat gate pass/fail as directional and re-run on a larger set."
        )
    elif N:
        caveat = (
            f"This eval ran on {N} interviews. "
            f"Scores are reasonably stable at this size; "
            f"per-interviewee charts still show single observations."
        )
    else:
        caveat = ""
    caveat_html = f'<div class="ev-caveat">{H(caveat)}</div>' if caveat else ""

    # Section extras
    e1_extra = _e1_worst_html(eval_results.get("E1", {}).get("worst", []))

    e2b_data  = eval_results.get("E2b", {}).get("data", {})
    e2b_extra = _e2b_table_html(
        e2b_data.get("unfaithful", []), e2b_data.get("low_certainty", [])
    )

    el_data     = eval_results.get("EL", {})
    el_extra    = _lineage_integrity_html(
        el_data.get("uncited_qa", []),
        el_data.get("dropped_l2", []),
        el_data.get("dropped_l3", []),
        el_data.get("structural", []),
    )

    e4_extra = _e4_detail_html(eval_results.get("E4", {}).get("worst", []))

    e5_data  = eval_results.get("E5", {}).get("data", {})
    e5_extra = _verdict_table_html(
        e5_data.get("unfaithful", []), e5_data.get("low_certainty", []), "UNFAITHFUL"
    )
    e5b_data  = eval_results.get("E5b", {}).get("data", {})
    e5b_extra = _verdict_table_html(
        e5b_data.get("unsupported", []), e5b_data.get("low_certainty", []), "OVERCLAIM"
    )

    e6_data  = eval_results.get("E6", {})
    e6_extra = _e6_detail_html(
        e6_data.get("worst_para", []),
        e6_data.get("worst_order", {}),
    )

    e7_data   = eval_results.get("E7", {})
    e7_extra  = _e7_html(e7_data.get("at_risk", []), e7_data.get("profiles", {}))

    e8_data   = eval_results.get("E8", {})
    e8_extra  = _e8_html(e8_data.get("items", []))

    body = (
        _sec(eval_results, "E1",  "L2 Code Set Stability",        "dynamic",           e1_extra)
        + _sec(eval_results, "E2b", "L2 Semantic Label Quality",  "static, judge",     e2b_extra)
        + _sec(eval_results, "EL",  "Lineage Integrity",          "static",            el_extra)
        + _sec(eval_results, "E4",  "Clustering Stability",       "dynamic",           e4_extra)
        + _sec(eval_results, "E5",  "Narrative Faithfulness",     "static, judge",     e5_extra)
        + _sec(eval_results, "E5b", "Sentiment Overclaim",        "static, judge",     e5b_extra)
        + _sec(eval_results, "E6",  "Metamorphic Invariance",     "dynamic",           e6_extra)
        + _sec(eval_results, "E7",  "Re-identification Probe",    "static, adversary", e7_extra)
        + _sec(eval_results, "E8",  "Negation Sensitivity",       "dynamic",           e8_extra)
    )

    links_html = (
        '<p class="ev-links">Further reading: '
        '<a href="https://doi.org/10.1111/j.1469-8137.1912.tb05611.x">Jaccard index</a>'
        ' &middot; '
        '<a href="https://doi.org/10.1007/BF01908075">Adjusted Rand Index</a>'
        '</p>'
    )

    return (
        '<!DOCTYPE html>\n<html lang="en"><head>\n'
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>Spradley: Eval Report</title>\n'
        f'<style>{EVAL_CSS}</style></head><body>\n'
        f'<header class="hdr"><div class="logo">SP</div><div>\n'
        f'<div class="hdr-t">Pipeline Eval Report</div>\n'
        f'<div class="hdr-m">{H(meta_str)}</div></div>\n'
        f'<div style="margin-left:auto">{overall}</div></header>\n'
        f'<div class="wrap">{links_html}{caveat_html}{body}\n'
        '<div class="footer">Spradley &middot; app.spradley.io</div></div>\n'
        '</body></html>'
    )
