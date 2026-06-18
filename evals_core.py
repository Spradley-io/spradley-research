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

def save_eval_history(eval_results: dict, history_dir: str) -> str:
    os.makedirs(history_dir, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
    path = os.path.join(history_dir, f"{ts}.json")
    # strip base64 figure blobs -- everything else is kept
    snapshot = {
        k: {kk: vv for kk, vv in v.items() if kk != "figures"}
        for k, v in eval_results.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    return path

def save_eval_results(eval_results: dict, output_dir: str) -> str:
    """Write a stable eval_results.json to output_dir alongside the pipeline data files.

    Always overwrites the previous run. Figures are included so the checkpoint
    can fully regenerate the HTML report. Use eval_history/ for compact snapshots.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    return path


# ── Chart helpers ──────────────────────────────────────────────────────────────

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
    iq_ids = list({e["interview_question_id"] for e in lin.get("l1_codes", [])})
    lines  = []
    for iq_id in iq_ids:
        if iq_id in db:
            e = db[iq_id]
            lines.append(f"Q: {e['question']}\nA: {e['anonymised_answer']}")
    return "\n\n".join(lines) or "(no source Q&A found)"


# ── Judge prompts ──────────────────────────────────────────────────────────────

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

REID_PROMPT = (
    "You are a privacy auditor testing re-identification risk.\n\n"
    "ANONYMISED REPORT EXCERPT:\n{report_text}\n\n"
    "ROSTER ({n} people who were interviewed):\n{roster}\n\n"
    "For each paraphrased quote in the report, guess which person most likely said it.\n"
    "Return a JSON array:\n"
    '[{{"quote_preview": "first 8 words of quote", "guess": "Name", "confidence": 0.0-1.0}}, ...]'
)


# ── Eval functions ─────────────────────────────────────────────────────────────

def paraphrase(text: str, model: str) -> str:
    return pipeline_core.call_llm(PARAPHRASE_PROMPT.format(text=text), model=model)


def run_merge_quality_check(interview_store: dict, model: str) -> dict:
    verdicts = []
    for interview_id, store in interview_store.items():
        for l2 in store["l2_codes"]:
            sources_str = ", ".join(l2["merged_from_l1"]) if l2["merged_from_l1"] else "(none)"
            prompt = MERGE_JUDGE.format(sources=sources_str, label=l2["code"])
            raw = pipeline_core.call_llm(prompt, model=model)
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


def run_faithfulness_check(clusters: dict, lineage: dict, db: dict, model: str) -> dict:
    verdicts = []
    for name, data in clusters.items():
        narrative = data.get("story") or data.get("summary", "")
        if not narrative:
            continue
        context = qa_context_for_cluster(name, lineage, db)
        prompt  = FAITHFULNESS_JUDGE.format(context=context, narrative=narrative)
        raw     = pipeline_core.call_llm(prompt, model=model)
        try:
            result = pipeline_core.parse_json_safe(raw)
        except Exception:
            result = {"verdict": "error", "certainty": 0.0, "reason": raw[:120]}
        verdicts.append({"cluster": name, **result})
    unfaithful    = [v for v in verdicts if v.get("verdict") == "unfaithful"]
    low_certainty = [v for v in verdicts if v.get("verdict") == "faithful" and v.get("certainty", 1.0) < 0.8]
    return {"all": verdicts, "unfaithful": unfaithful, "low_certainty": low_certainty}


def run_overclaim_check(clusters: dict, lineage: dict, db: dict, model: str) -> dict:
    verdicts = []
    for name, data in clusters.items():
        narrative = data.get("story") or data.get("summary", "")
        if not narrative:
            continue
        context = qa_context_for_cluster(name, lineage, db)
        prompt  = OVERCLAIM_JUDGE.format(context=context, narrative=narrative)
        raw     = pipeline_core.call_llm(prompt, model=model)
        try:
            result = pipeline_core.parse_json_safe(raw)
        except Exception:
            result = {"verdict": "error", "certainty": 0.0, "reason": raw[:120]}
        verdicts.append({"cluster": name, **result})
    unsupported   = [v for v in verdicts if v.get("verdict") == "unsupported"]
    low_certainty = [v for v in verdicts if v.get("verdict") == "supported" and v.get("certainty", 1.0) < 0.8]
    return {"all": verdicts, "unsupported": unsupported, "low_certainty": low_certainty}


def run_reidentification_probe(report_text: str, roster: list, model: str) -> dict:
    roster_str = "\n".join(f"- {p['name']}: {p['role']}" for p in roster)
    prompt     = REID_PROMPT.format(n=len(roster), report_text=report_text[:4000], roster=roster_str)
    raw        = pipeline_core.call_llm(prompt, model=model)
    try:
        guesses = pipeline_core.parse_json_safe(raw)
        if not isinstance(guesses, list):
            guesses = guesses.get("guesses", [])
    except Exception:
        guesses = []
    high_conf = [g for g in guesses if g.get("confidence", 0) >= 0.7]
    return {"guesses": guesses, "high_conf": high_conf}


# ── HTML report ────────────────────────────────────────────────────────────────

EVAL_DESCRIPTIONS = {
    "E1": (
        "Runs the qualitative coder 3 times on every interview exchange and measures how consistently "
        "it assigns the same codes. Low consistency means the extracted codes are sensitive to random "
        "variation rather than grounded in what was actually said."
    ),
    "E2": (
        "Checks that every merged theme has a valid trail back to its source L1 codes, and that no "
        "L1 code was silently dropped. Failures mean the audit trail between raw answers and merged "
        "themes is broken."
    ),
    "E2b": (
        "An AI auditor checks whether each merged theme label faithfully represents the L1 source "
        "codes it absorbed. Failures indicate that two unrelated codes were incorrectly grouped under "
        "a single label."
    ),
    "E3": (
        "Tracks how many distinct codes survive the full funnel (L1 to L2 to L3 to cluster). "
        "The chart shows the count at each stage. High suppression means minority viewpoints from "
        "interviews were silently dropped before reaching the report."
    ),
    "E4": (
        "Re-groups the same L3 codes 3 times and measures whether the resulting themes are consistent "
        "across runs. Low consistency means the reported themes are partly an artifact of random LLM "
        "sampling, not the data itself."
    ),
    "E4b": (
        "Structural audit: checks for codes that appear in multiple clusters simultaneously, codes "
        "never assigned to any cluster, and lineage pointers referencing non-existent entries. "
        "These are data integrity errors, not interpretation errors."
    ),
    "E5": (
        "An AI auditor reads each cluster's narrative summary and compares it against the actual "
        "anonymised interview excerpts. Failures mean the summary makes claims that the underlying "
        "interview data does not clearly support."
    ),
    "E5b": (
        "Specifically checks whether cluster narratives assert positive sentiment ('employees feel "
        "supported', 'high morale') that the evidence does not clearly back. Particularly important "
        "for management-facing reports where overclaiming can erode trust."
    ),
    "E6": (
        "Two robustness tests: (a) rewrites sample answers in different words and checks whether "
        "the same codes still emerge; (b) shuffles the code list and checks whether the same themes "
        "cluster. Both test that the pipeline responds to meaning, not surface phrasing."
    ),
    "E7": (
        "An adversarial AI tries to match anonymised report quotes to fictional employee personas. "
        "High-confidence matches suggest the report retains enough distinctive detail that a real "
        "reader with a real roster could potentially identify who said what."
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
.e1-id{font-size:10px;color:#aaa;font-family:monospace}
.e1-q{font-size:12px;color:#555;margin-bottom:3px}
.e1-a{font-size:12px;color:#111;font-style:italic}
.e1-codes{font-size:11px;color:#444;line-height:1.8}
.e1-shared{font-size:11px}
.tag{display:inline-block;background:#e0f2fe;color:#0369a1;border-radius:4px;
     padding:1px 7px;margin:2px 2px 0 0;font-size:11px;white-space:nowrap}
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
details.det{margin-top:10px;border-top:1px solid #efefef;padding-top:8px}
details.det summary.det-sum{font-size:12px;font-weight:600;color:#0369a1;
  cursor:pointer;user-select:none;list-style:none;display:inline-flex;
  align-items:center;gap:6px;padding:2px 0}
details.det summary.det-sum::-webkit-details-marker{display:none}
details.det summary.det-sum::before{content:"▶";font-size:9px;
  display:inline-block;transition:transform 0.15s}
details.det[open] summary.det-sum::before{transform:rotate(90deg)}
details.det .det-body{padding-top:10px}
"""


def _e1_worst_html(details: list) -> str:
    if not details:
        return ""
    n_runs = len(details[0]["run_codes"]) if details else 0
    run_headers = "".join(f"<th>Run {i+1}</th>" for i in range(n_runs))
    rows = ""
    for d in details:
        run_cols = "".join(
            f'<td class="e1-codes">{"<br>".join(H(c) for c in codes)}</td>'
            for codes in d["run_codes"]
        )
        shared_tags = "".join(f'<span class="tag">{H(c)}</span>' for c in d["shared"])
        rows += (
            f"<tr>"
            f'<td><span class="e1-score">J={d["score"]:.2f}</span><br>'
            f'<span class="e1-id">{H(d["iq_id"])}</span></td>'
            f'<td><div class="e1-q">Q: {H(d["question"])}</div>'
            f'<div class="e1-a">A: {H(d["answer"])}</div></td>'
            f"{run_cols}"
            f'<td class="e1-shared">{shared_tags or "<em>none</em>"}</td>'
            f"</tr>"
        )
    return (
        f'<h4 class="e1-subhdr">5 least stable Q&amp;A entries</h4>'
        f'<div class="e1-scroll"><table class="e1-tbl">'
        f"<thead><tr><th>Score</th><th>Q / A</th>{run_headers}<th>Shared codes</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>"
    )


def _verdict_table_html(items_fail: list, items_review: list, fail_label: str = "FAIL") -> str:
    if not items_fail and not items_review:
        return ""
    rows = ""
    for v in items_fail:
        cert = v.get("certainty")
        cert_str = f"{cert:.0%}" if isinstance(cert, float) else str(cert)
        rows += (
            f"<tr>"
            f'<td><span class="vt-badge vt-bad">{H(fail_label)}</span></td>'
            f'<td class="vt-cluster">{H(v.get("cluster", ""))}</td>'
            f'<td class="vt-reason">{H(v.get("reason", ""))}</td>'
            f'<td class="vt-cert">{cert_str}</td>'
            f"</tr>"
        )
    for v in items_review:
        cert = v.get("certainty")
        cert_str = f"{cert:.0%}" if isinstance(cert, float) else str(cert)
        rows += (
            f"<tr>"
            f'<td><span class="vt-badge vt-warn">REVIEW</span></td>'
            f'<td class="vt-cluster">{H(v.get("cluster", ""))}</td>'
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
        badge    = '<span class="vt-badge vt-bad">UNFAITHFUL</span>' if is_fail else '<span class="vt-badge vt-warn">REVIEW</span>'
        cert     = v.get("certainty")
        cert_str = f"{cert:.0%}" if isinstance(cert, float) else str(cert)
        src_tags = "".join(f'<span class="tag">{H(s)}</span>' for s in v.get("sources", []))
        rows += (
            f"<tr>"
            f"<td>{badge}</td>"
            f'<td class="vt-cluster">{H(v.get("l2_code", ""))}'
            f'<div style="font-size:10px;color:#aaa;font-weight:400;margin-top:2px">'
            f'interview {H(v.get("interview", ""))}</div></td>'
            f'<td style="font-size:11px;line-height:1.9">{src_tags}</td>'
            f'<td class="vt-reason">{H(v.get("reason", ""))}</td>'
            f'<td class="vt-cert">{cert_str}</td>'
            f"</tr>"
        )
    return (
        f'<div class="vt-scroll"><table class="vt-tbl">'
        f"<thead><tr><th>Status</th><th>L2 Label</th><th>L1 Sources</th>"
        f"<th>Reason</th><th>Certainty</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>"
    )


def _e3_dropped_html(dropped_l3: list, dropped_l2: list) -> str:
    if not dropped_l3 and not dropped_l2:
        return ""
    parts = []
    if dropped_l3:
        tags = "".join(
            f'<span class="tag" style="background:#fee2e2;color:#dc2626">{H(c)}</span>'
            for c in dropped_l3
        )
        parts.append(
            f'<div style="margin-top:10px">'
            f'<div style="font-size:12px;font-weight:600;color:#dc2626;margin-bottom:5px">'
            f'L3 codes not assigned to any cluster ({len(dropped_l3)})</div>'
            f'<div>{tags}</div></div>'
        )
    if dropped_l2:
        tags = "".join(
            f'<span class="tag" style="background:#fef3c7;color:#b45309">{H(c)}</span>'
            for c in dropped_l2
        )
        parts.append(
            f'<div style="margin-top:10px">'
            f'<div style="font-size:12px;font-weight:600;color:#b45309;margin-bottom:5px">'
            f'L2 codes not absorbed into any L3 ({len(dropped_l2)})</div>'
            f'<div>{tags}</div></div>'
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


def build_eval_html(eval_results: dict, run_meta: dict) -> str:
    meta_str   = " | ".join(f"{k}: {v}" for k, v in run_meta.items()) if run_meta else "no run_meta"
    all_passed = all(r.get("passed") is True for r in eval_results.values())
    overall    = ('<span class="pass">ALL PASS</span>' if all_passed
                  else '<span class="fail">ISSUES FOUND</span>')

    e1_extra = _e1_worst_html(eval_results.get("E1", {}).get("worst", []))

    e2b_data  = eval_results.get("E2b", {}).get("data", {})
    e2b_extra = _e2b_table_html(
        e2b_data.get("unfaithful", []), e2b_data.get("low_certainty", [])
    )

    e3_data   = eval_results.get("E3", {})
    e3_extra  = _e3_dropped_html(
        e3_data.get("dropped_l3_codes", []), e3_data.get("dropped_l2_codes", [])
    )

    e5_data  = eval_results.get("E5", {}).get("data", {})
    e5_extra = _verdict_table_html(
        e5_data.get("unfaithful", []), e5_data.get("low_certainty", []), "UNFAITHFUL"
    )
    e5b_data  = eval_results.get("E5b", {}).get("data", {})
    e5b_extra = _verdict_table_html(
        e5b_data.get("unsupported", []), e5b_data.get("low_certainty", []), "OVERCLAIM"
    )

    body = (
        _sec(eval_results, "E1",  "L1 Coding Stability",          "dynamic",         e1_extra)
        + _sec(eval_results, "E2",  "L2 Lineage Integrity",        "static")
        + _sec(eval_results, "E2b", "L2 Semantic Merge Quality",   "static, judge",  e2b_extra)
        + _sec(eval_results, "E3",  "L3 Coverage and Suppression", "static",         e3_extra)
        + _sec(eval_results, "E4",  "C9 Clustering Stability",     "dynamic")
        + _sec(eval_results, "E4b", "Lineage Integrity",           "static")
        + _sec(eval_results, "E5",  "Narrative Faithfulness",      "static, judge",   e5_extra)
        + _sec(eval_results, "E5b", "Sentiment Overclaim",         "static, judge",   e5b_extra)
        + _sec(eval_results, "E6",  "Metamorphic Invariance",      "dynamic")
        + _sec(eval_results, "E7",  "Leakage / Re-identification", "static, adversary")
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
        f'<div class="wrap">{body}\n'
        '<div class="footer">Spradley &middot; app.spradley.io</div></div>\n'
        '</body></html>'
    )
