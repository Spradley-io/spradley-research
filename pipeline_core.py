"""
pipeline_core.py — Spradley Coding Pipeline: shared config, prompts, and functions.

Imported by both Pipeline_Execution.ipynb and Pipeline_Evals.ipynb.
Organised by pipeline stage (C-numbers match the notebook cells).
To modify a stage's prompt or logic, Ctrl+F its C-number here.
"""

import re
import json
import os
import html as _html_mod

# ── Fixed paths ───────────────────────────────────────────────────────────────
CSV_PATH   = "pilot_transcripts.csv"
KEYS_ENV   = "keys.env"
OUTPUT_DIR = "pipeline_output"

# ── Runtime config (C0 in the notebook overrides these per run) ───────────────
CONFIG = {
    "LLM_PROVIDER":    "anthropic",
    "LLM_MODEL":       "claude-haiku-4-5-20251001",
    "LLM_TEMPERATURE": 0.2,
    "L1_CODES_RANGE":  (1, 10),
    "L2_CODES_RANGE":  (20, 30),
    "L3_CODES_RANGE":  (40, 80),
    "CLUSTERS_RANGE":  (7, 12),
}


# ── C2: LLM client ────────────────────────────────────────────────────────────

def call_llm(prompt: str, system: str = "You are a qualitative research assistant.",
             model: str | None = None, temperature: float | None = None) -> str:
    """Single LLM call. Uses CONFIG by default; override model/temperature for eval judges."""
    _model       = model       or CONFIG["LLM_MODEL"]
    _temperature = temperature if temperature is not None else CONFIG["LLM_TEMPERATURE"]
    provider     = CONFIG["LLM_PROVIDER"]

    if provider == "anthropic":
        import anthropic, httpx
        client = anthropic.Anthropic(
            http_client=httpx.Client(verify=False, trust_env=False)
        )
        resp = client.messages.create(
            model=_model, max_tokens=2048,
            temperature=_temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text
    # elif provider == "openai":
    #     import openai, httpx
    #     client = openai.OpenAI(http_client=httpx.Client(verify=False, trust_env=False))
    #     resp = client.chat.completions.create(
    #         model=_model, max_tokens=2048, temperature=_temperature,
    #         messages=[{"role": "system", "content": system},
    #                   {"role": "user",   "content": prompt}]
    #     )
    #     return resp.choices[0].message.content
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


# ── C3: Data ingestion ────────────────────────────────────────────────────────

def load_interviews(csv_path: str = CSV_PATH) -> list:
    """Load and validate CSV → standardised interviews list."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    required = {"session_id", "turn_number", "speaker", "message"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    if df["session_id"].isnull().any():
        raise ValueError("CSV contains null session_id values")
    if not (df["turn_number"] > 0).all():
        raise ValueError("turn_number must be positive integers")

    interviews = []
    for interview_id, group in df.groupby("session_id"):
        group = group.sort_values("turn_number")
        bot_msgs = {
            int(row["turn_number"]): row["message"]
            for _, row in group.iterrows()
            if row["speaker"] == "Bot"
        }
        qa_pairs = []
        for _, row in group.iterrows():
            if row["speaker"] != "User":
                continue
            turn     = int(row["turn_number"])
            question = bot_msgs.get(turn - 1, "[initial mood opener]")
            qa_pairs.append({"turn_number": turn, "question": question,
                              "answer": str(row["message"])})
        if not qa_pairs:
            raise ValueError(f"Interview {interview_id} has no User turns")
        interviews.append({"interview_id": interview_id, "qa_pairs": qa_pairs})

    if not interviews:
        raise ValueError("No interviews loaded — check csv_path")
    return interviews


# ── C4: Anonymizer ────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',                         "[NAME]",  0),
    (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z]{2,}\b',  "[EMAIL]", re.IGNORECASE),
    (r'\b(?:\+?\d[\d\s\-().]{7,}\d)\b',                      "[PHONE]", 0),
]

def anonymize(text: str) -> str:
    for pattern, replacement, flags in _PII_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=flags)
    return text


# ── C5/C6: DB init + L1 Coder ─────────────────────────────────────────────────

def parse_json_safe(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)

PROMPT_CODER = (
    "You are a qualitative researcher performing open coding of employee interview responses.\n"
    "Generate between {l1_min} and {l1_max} short open codes (2-5 word noun phrases) that\n"
    "capture the key conceptual ideas in the answer. Be specific and grounded in the text.\n\n"
    "--- EXAMPLE ---\n"
    "Question: How would you describe your relationship with your direct manager?\n"
    "Answer: She's always approachable and gives honest feedback. I feel trusted to make decisions.\n"
    "Output:\n"
    '{{"codes": ["managerial approachability", "honest feedback culture", "autonomy and trust"]}}\n'
    "--- END EXAMPLE ---\n\n"
    "Now code the following:\n"
    "Question: {question}\n"
    "Answer: {anonymised_answer}\n\n"
    "Return only valid JSON — no other text:\n"
    '{{"codes": ["code 1", "code 2"]}}'
)

def code_one_answer(question: str, answer: str) -> list:
    """Run L1 open coding on a single Q&A pair. Returns list of code strings."""
    l1_min, l1_max = CONFIG["L1_CODES_RANGE"]
    prompt = PROMPT_CODER.format(
        l1_min=l1_min, l1_max=l1_max,
        question=question,
        anonymised_answer=answer
    )
    raw    = call_llm(prompt)
    result = parse_json_safe(raw)
    return result["codes"]


# ── C7: User Consolidator (L2) ───────────────────────────────────────────────

PROMPT_L2 = (
    "You are a qualitative researcher consolidating open codes from one employee interview into\n"
    "a unified set of between {l2_min} and {l2_max} codes. Merge overlapping or synonymous codes;\n"
    "preserve meaningfully distinct concepts. Use 2-5 word noun-phrase labels.\n"
    "You MUST list which source L1 codes each new code absorbs.\n\n"
    "--- EXAMPLE ---\n"
    'Input codes: ["managerial approachability", "manager accessibility", "open door policy", "task clarity", "clear role expectations"]\n'
    "Output:\n"
    '{{"consolidated_codes": [\n'
    '  {{"code": "accessible and open management", "merged_from_l1": ["managerial approachability", "manager accessibility", "open door policy"]}},\n'
    '  {{"code": "role and task clarity", "merged_from_l1": ["task clarity", "clear role expectations"]}}\n'
    "]}}\n"
    "--- END EXAMPLE ---\n\n"
    "All L1 codes from this interview:\n"
    "{l1_codes_list}\n\n"
    "Return only valid JSON — no other text:\n"
    '{{"consolidated_codes": [\n'
    '  {{"code": "new label", "merged_from_l1": ["source l1 code"]}},\n'
    "  ...\n"
    "]}}"
)

def consolidate_l2(all_l1_codes: list) -> list:
    """Consolidate a flat list of L1 codes into L2 codes with merge lineage."""
    l2_min, l2_max = CONFIG["L2_CODES_RANGE"]
    l1_list_str    = "\n".join(f"- {c}" for c in all_l1_codes)
    prompt         = PROMPT_L2.format(l2_min=l2_min, l2_max=l2_max, l1_codes_list=l1_list_str)
    raw            = call_llm(prompt)
    return parse_json_safe(raw)["consolidated_codes"]


# ── C8: Global Consolidator (L3) ─────────────────────────────────────────────

PROMPT_L3 = (
    "You are a qualitative researcher consolidating codes from {n_interviews} employee interviews\n"
    "into a final set of between {l3_min} and {l3_max} codes. Merge highly similar codes across\n"
    "interviews; keep meaningfully distinct concepts separate. Use 2-5 word noun-phrase labels.\n"
    "You MUST list which source L2 codes each new code absorbs.\n\n"
    "--- EXAMPLE ---\n"
    "Input L2 codes:\n"
    "- team and workplace culture\n"
    "- accessible and open management\n"
    "- collegial support\n"
    "- role and task clarity\n\n"
    "Output:\n"
    '{{"consolidated_codes": [\n'
    '  {{"code": "supportive management and team culture", "merged_from_l2": ["team and workplace culture", "accessible and open management", "collegial support"]}},\n'
    '  {{"code": "role clarity and expectations", "merged_from_l2": ["role and task clarity"]}}\n'
    "]}}\n"
    "--- END EXAMPLE ---\n\n"
    "All L2 codes (one per line):\n"
    "{l2_codes_list}\n\n"
    "Return only valid JSON — no other text:\n"
    '{{"consolidated_codes": [\n'
    '  {{"code": "new label", "merged_from_l2": ["source l2 code"]}},\n'
    "  ...\n"
    "]}}"
)

def consolidate_l3(all_l2_codes: list, n_interviews: int) -> list:
    """Consolidate all L2 code strings into global L3 codes with merge lineage."""
    l3_min, l3_max = CONFIG["L3_CODES_RANGE"]
    l2_list_str    = "\n".join(f"- {c}" for c in all_l2_codes)
    prompt         = PROMPT_L3.format(
        n_interviews=n_interviews, l3_min=l3_min, l3_max=l3_max,
        l2_codes_list=l2_list_str
    )
    raw    = call_llm(prompt)
    return parse_json_safe(raw)["consolidated_codes"]


# ── C9: Theme Clustering ──────────────────────────────────────────────────────

PROMPT_CLUSTER = (
    "You are a qualitative researcher grouping final codes into thematic clusters.\n"
    "Group into between {clusters_min} and {clusters_max} clusters.\n"
    "Each cluster must have a 3-6 word name that reads as a natural section header: "
    "specific, concrete, and slightly engaging rather than an academic label. "
    "Title-case each word. It must contain at least 2 codes.\n\n"
    "--- EXAMPLE ---\n"
    'Input L3 codes: ["supportive management culture", "open leadership style", "role clarity and expectations",\n'
    '                 "workload balance", "growth opportunities", "career development support"]\n'
    "Output:\n"
    '{{"clusters": [\n'
    '  {{"name": "Accessible and Open Leadership", "codes": ["supportive management culture", "open leadership style"]}},\n'
    '  {{"name": "Clear Roles and Workload Balance", "codes": ["role clarity and expectations", "workload balance"]}},\n'
    '  {{"name": "Career Growth and Support",       "codes": ["growth opportunities", "career development support"]}}\n'
    "]}}\n"
    "--- END EXAMPLE ---\n\n"
    "Final codes (L3):\n"
    "{l3_codes_list}\n\n"
    "Return only valid JSON — no other text:\n"
    '{{"clusters": [\n'
    '  {{"name": "Cluster Name", "codes": ["l3 code 1", "l3 code 2"]}},\n'
    "  ...\n"
    "]}}"
)

def cluster_l3_codes(l3_list: list) -> dict:
    """Group L3 code strings into named clusters. Returns {cluster_name: [l3_codes]}."""
    clusters_min, clusters_max = CONFIG["CLUSTERS_RANGE"]
    l3_list_str = "\n".join(f"- {c}" for c in l3_list)
    prompt      = PROMPT_CLUSTER.format(
        clusters_min=clusters_min, clusters_max=clusters_max,
        l3_codes_list=l3_list_str
    )
    raw    = call_llm(prompt)
    result = parse_json_safe(raw)
    return {cl["name"]: cl["codes"] for cl in result["clusters"]}


# ── C10: LLM Explainer ────────────────────────────────────────────────────────

PROMPT_FINDING = (
    "You are a qualitative researcher writing structured findings for an HR report.\n\n"
    "Cluster: {cluster_name}\n"
    "Codes in this cluster: {codes_list}\n\n"
    "Supporting employee responses:\n"
    "{qa_pairs_text}\n\n"
    "Return valid JSON with exactly this structure:\n"
    '{{\n'
    '  "category": "working_well",\n'
    '  "summary": "3-5 sentence analytical narrative for an HR audience. Explain the pattern, why it matters, and any nuance.",\n'
    '  "quotes": ["paraphrased quote 1", "paraphrased quote 2"],\n'
    '  "tag": "1-2 word theme label e.g. Culture, Development, Wellbeing"\n'
    '}}\n\n'
    'Use exactly one of these values for category: "working_well", "needs_work", or "mixed".\n'
    "For quotes: paraphrase 2-4 representative responses. Preserve meaning, remove identifying details.\n"
    "Never use em dashes in any text.\n"
    "Return only valid JSON. No other text."
)

PROMPT_EXPERIMENTS = (
    "You are an HR consultant reviewing employee interview findings and proposing actionable experiments.\n\n"
    "Findings that need attention:\n"
    "{findings_text}\n\n"
    "Propose 2-4 concrete, low-cost experiments the team could run to address these findings.\n"
    "Each experiment should be actionable within 1-2 weeks with a clear success signal.\n\n"
    "Return valid JSON:\n"
    '{{"experiments": [\n'
    '  {{"title": "Short experiment name",\n'
    '   "summary": "One-sentence description",\n'
    '   "rationale": "2-3 sentences: connection to findings and expected outcome",\n'
    '   "tag": "theme label e.g. Culture, Development"}},\n'
    "  ...\n"
    "]}}\n\n"
    "Never use em dashes in any text.\n"
    "Return only valid JSON. No other text."
)

def explain_cluster(name: str, l3_codes: list, qa_pairs_text: str) -> dict:
    """Generate structured finding for one cluster. Returns raw LLM result dict."""
    codes_list = ", ".join(l3_codes)
    prompt     = PROMPT_FINDING.format(
        cluster_name=name, codes_list=codes_list, qa_pairs_text=qa_pairs_text
    )
    raw = call_llm(prompt)
    return parse_json_safe(raw)

def propose_experiments(needs_attention: list) -> list:
    """Generate experiment proposals for needs_work and mixed clusters.
    needs_attention: [(name, data_dict), ...]
    """
    if not needs_attention:
        return []
    parts = []
    for name, data in needs_attention:
        parts.append(
            f"Finding: {name}\n"
            f"Category: {data['category']}\n"
            f"Summary: {data['summary']}"
        )
    prompt = PROMPT_EXPERIMENTS.format(findings_text="\n\n---\n\n".join(parts))
    raw    = call_llm(prompt)
    return parse_json_safe(raw).get("experiments", [])


# ── C12 / C13: HTML report generation ────────────────────────────────────────

def H(s: object) -> str:
    return _html_mod.escape(str(s))

# Inline (notebook) report CSS — includes <style> tags for direct HTML concatenation.
REPORT_CSS = """<style>
.sp{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,sans-serif;max-width:740px;margin:0 auto;color:#111}
.sp-hdr{display:flex;align-items:center;gap:14px;padding:22px 0 18px;border-bottom:2px solid #000;margin-bottom:28px}
.sp-logo{width:40px;height:40px;background:#000;border-radius:9px;color:#fff;font-weight:800;font-size:14px;letter-spacing:-1px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.sp h1{font-size:20px;font-weight:700;margin:0}
.sp-meta{font-size:12px;color:#888;margin:2px 0 0}
.sp h2{font-size:16px;font-weight:700;margin:32px 0 16px}
.sp h2:first-of-type{margin-top:0}
hr.sp-hr{border:none;border-top:1px solid #ebebeb;margin:18px 0}
.sp-card{margin-bottom:18px}
.sp-hl{font-size:16px;font-weight:600;margin:0 0 3px}
.sp-sub{font-size:12px;color:#777;margin:0 0 8px}
.sp-badge{background:#f2f2f2;border-radius:20px;padding:2px 9px;font-size:11px;font-weight:500}
details.sp-d{border:1px solid #e8e8e8;border-radius:7px}
details.sp-d>summary{padding:8px 13px;cursor:pointer;font-size:12px;font-weight:500;color:#666;list-style:none;user-select:none}
details.sp-d>summary::-webkit-details-marker{display:none}
details.sp-d>summary::before{content:'&#9654; ';font-size:9px}
details[open].sp-d>summary::before{content:'&#9660; '}
.sp-db{padding:2px 14px 14px}
.sp-db h4{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#bbb;margin:12px 0 5px}
.sp-db p{font-size:13px;color:#333;line-height:1.7}
.sp-ql{list-style:none;padding:0;margin:0}
.sp-ql li{font-size:13px;color:#444;padding:6px 0 6px 11px;border-left:3px solid #e0e0e0;margin-bottom:6px}
.sp-exp-hl{font-size:15px;font-weight:600;margin:0 0 3px}
.sp-about{background:#f8f8f8;border-radius:9px;padding:18px 22px;margin-top:32px}
.sp-about h3{font-size:14px;font-weight:700;margin:0 0 8px}
.sp-about p,.sp-about ol{font-size:12px;color:#666;line-height:1.7}
.sp-about ol{padding-left:16px;margin-top:6px}
.sp-foot{text-align:center;font-size:11px;color:#ccc;padding:28px 0 8px}
</style>"""

# Standalone web app CSS — no <style> tags; embedded in the HTML template.
APP_CSS = (
    "*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}\n"
    "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;"
    "background:#fff;color:#111;font-size:15px;line-height:1.6}\n"
    ".app-header{display:flex;align-items:center;gap:14px;padding:14px 40px;"
    "border-bottom:1px solid #e5e5e5;position:sticky;top:0;background:#fff;z-index:100}\n"
    ".logo{width:36px;height:36px;background:#000;border-radius:8px;flex-shrink:0;"
    "color:#fff;font-weight:800;font-size:13px;letter-spacing:-1px;"
    "display:flex;align-items:center;justify-content:center}\n"
    ".logo-img{width:36px;height:36px;border-radius:8px;object-fit:contain;flex-shrink:0}\n"
    ".app-title{font-size:16px;font-weight:700}\n"
    ".app-meta{font-size:12px;color:#999;margin-top:1px}\n"
    ".tabs{margin-left:auto;display:flex;gap:4px}\n"
    ".tab-btn{padding:6px 15px;border:1px solid #ddd;background:#fff;border-radius:6px;"
    "font-size:13px;font-weight:500;cursor:pointer;color:#555;transition:background .1s}\n"
    ".tab-btn:hover{background:#f5f5f5}\n"
    ".tab-btn.active{background:#000;color:#fff;border-color:#000}\n"
    ".tab-pane{display:none;max-width:760px;margin:0 auto;padding:36px 24px 60px}\n"
    ".tab-pane.active{display:block}\n"
    "h2{font-size:17px;font-weight:700;margin:36px 0 16px}\n"
    "h2:first-of-type{margin-top:0}\n"
    ".card{margin-bottom:18px}\n"
    ".hl{font-size:16px;font-weight:600;margin-bottom:3px}\n"
    ".meta{font-size:12px;color:#777;margin-bottom:8px}\n"
    ".badge{background:#f2f2f2;border-radius:20px;padding:2px 9px;font-size:11px;font-weight:500}\n"
    "hr.div{border:none;border-top:1px solid #ebebeb;margin:16px 0}\n"
    "details.det>summary{padding:9px 13px;cursor:pointer;font-size:13px;font-weight:500;"
    "color:#666;list-style:none;border:1px solid #e8e8e8;"
    "border-radius:7px;user-select:none;display:block}\n"
    "details[open].det>summary{border-bottom-left-radius:0;border-bottom-right-radius:0}\n"
    "details.det>summary::-webkit-details-marker{display:none}\n"
    "details.det>summary::before{content:'&#9654;  ';font-size:9px}\n"
    "details[open].det>summary::before{content:'&#9660;  '}\n"
    ".det-body{padding:4px 14px 14px;border:1px solid #e8e8e8;border-top:none;"
    "border-radius:0 0 7px 7px}\n"
    ".det-body h4{font-size:10px;font-weight:700;text-transform:uppercase;"
    "letter-spacing:.5px;color:#bbb;margin:12px 0 5px}\n"
    ".det-body p{font-size:14px;color:#333;line-height:1.7}\n"
    ".quotes{list-style:none;padding:0;margin:0}\n"
    ".quotes li{font-size:14px;color:#444;padding:6px 0 6px 12px;"
    "border-left:3px solid #e0e0e0;margin-bottom:7px}\n"
    ".exp-hl{font-size:15px;font-weight:600;margin-bottom:3px}\n"
    ".about{background:#f8f8f8;border-radius:10px;padding:20px 22px;margin-top:36px}\n"
    ".about h3{font-size:14px;font-weight:700;margin-bottom:10px}\n"
    ".about p,.about ol{font-size:13px;color:#666;line-height:1.65}\n"
    ".about ol{padding-left:18px;margin-top:8px}\n"
    ".footer{text-align:center;font-size:12px;color:#ccc;padding:32px 0 8px}\n"
    ".lineage-intro{font-size:13px;color:#666;margin-bottom:24px;line-height:1.7;max-width:600px}\n"
    "details.tr{margin-bottom:4px}\n"
    "details.tr-cluster{border-left:3px solid #000;padding-left:16px;margin-bottom:14px}\n"
    "details.tr-l3{border-left:2px solid #c8d8ff;padding-left:14px;margin:4px 0}\n"
    "details.tr-l2{border-left:2px solid #c8f0d8;padding-left:14px;margin:4px 0}\n"
    "details.tr-l1{border-left:2px solid #ffe8c8;padding-left:14px;margin:4px 0}\n"
    "details.tr>summary{list-style:none;cursor:pointer;padding:5px 4px;"
    "display:flex;align-items:center;gap:7px;flex-wrap:wrap;user-select:none}\n"
    "details.tr>summary::-webkit-details-marker{display:none}\n"
    "details[open].tr>summary{color:#000}\n"
    "details.tr-cluster>summary{font-size:15px;font-weight:600;padding:7px 4px}\n"
    ".lv{border-radius:4px;font-size:10px;font-weight:700;padding:1px 6px;"
    "flex-shrink:0;letter-spacing:.3px}\n"
    ".lv-3{background:#e8f0fe;color:#1a56db}\n"
    ".lv-2{background:#dcfce7;color:#15803d}\n"
    ".lv-1{background:#fef3c7;color:#b45309}\n"
    ".lv-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}\n"
    ".lv-dot-g{background:#16a34a}\n"
    ".lv-dot-r{background:#dc2626}\n"
    ".lv-dot-a{background:#d97706}\n"
    ".int-tag{font-size:11px;color:#aaa;font-family:monospace}\n"
    ".src-n{margin-left:auto;font-size:11px;color:#bbb;white-space:nowrap}\n"
    ".tr-body{padding:6px 0 2px 8px}\n"
    ".qa-blk{background:#fafafa;border-radius:6px;padding:10px 12px;margin-bottom:8px}\n"
    ".iq-tag{font-size:10px;font-family:monospace;background:#f0f0f0;border-radius:4px;"
    "padding:1px 5px;display:inline-block;margin-bottom:5px;color:#999}\n"
    ".qa-ln{font-size:13px;color:#444;margin-bottom:3px}\n"
    ".empty{font-size:12px;color:#ccc;font-style:italic;padding:4px 0}\n"
)


def _inline_card(name: str, data: dict) -> str:
    hl = H(name)
    tg = H(data.get("tag", ""))
    v  = data.get("voice_count", 0)
    sm = H(data.get("summary", ""))
    ql = "".join(f'<li>{H(q)}</li>' for q in data.get("quotes", []))
    return (
        f'<div class="sp-card"><p class="sp-hl">{hl}</p>'
        f'<p class="sp-sub"><span class="sp-badge">{tg}</span>'
        f' &middot; {v} voice{"s" if v != 1 else ""}</p>'
        f'<details class="sp-d"><summary>Read summary and quotes</summary>'
        f'<div class="sp-db"><h4>Summary</h4><p>{sm}</p>'
        f'<h4>Paraphrased quotes</h4>'
        f'<ol class="sp-ql">{ql}</ol></div></details></div>'
        f'<hr class="sp-hr">'
    )


def _inline_exp_card(exp: dict) -> str:
    t = H(exp.get("title", ""))
    s = H(exp.get("summary", ""))
    r = H(exp.get("rationale", ""))
    g = H(exp.get("tag", ""))
    return (
        f'<div class="sp-card"><p class="sp-exp-hl">{t}</p>'
        f'<p class="sp-sub"><span class="sp-badge">{g}</span> {s}</p>'
        f'<details class="sp-d"><summary>Read rationale</summary>'
        f'<div class="sp-db"><p>{r}</p></div></details></div>'
    )


def build_inline_report_html(clusters: dict, interviews: list, experiments: list) -> str:
    """Build styled HTML for inline display inside the notebook (C12)."""
    by_cat: dict = {"working_well": [], "needs_work": [], "mixed": []}
    for n, d in clusters.items():
        by_cat.get(d.get("category", "mixed"), by_cat["mixed"]).append((n, d))

    body = ""
    if by_cat["working_well"]:
        body += "<h2>What&#x2019;s working well</h2>"
        for n, d in by_cat["working_well"]:
            body += _inline_card(n, d)
    if by_cat["needs_work"]:
        body += "<h2>What needs work</h2>"
        for n, d in by_cat["needs_work"]:
            body += _inline_card(n, d)
    if by_cat["mixed"]:
        body += "<h2>Mixed signals &amp; tensions</h2>"
        for n, d in by_cat["mixed"]:
            body += _inline_card(n, d)
    if experiments:
        body += "<h2>Experiments</h2>"
        for exp in experiments:
            body += _inline_exp_card(exp)

    n_iv = len(interviews)
    body += (
        '<div class="sp-about"><h3>Spradley guide: How the analysis works</h3>'
        f'<p>Insights are based on qualitative analysis of {n_iv} employee AI interview '
        'transcripts. Rather than survey scores, we surface <strong>patterns</strong> '
        'from what people say.</p>'
        '<ol>'
        '<li>Transcripts are clustered to identify recurring patterns across the dataset.</li>'
        '<li>Patterns are distilled into clear findings.</li>'
        '<li>Findings are labelled based on matching workplace themes.</li>'
        '<li>Paraphrased quotes give context while preserving anonymity.</li>'
        '</ol></div>'
    )

    return (
        REPORT_CSS
        + '<div class="sp">'
        + '<div class="sp-hdr"><div class="sp-logo">SP</div><div>'
        + '<h1>Employee Insights Report</h1>'
        + f'<p class="sp-meta">Spradley &middot; {len(clusters)} themes &middot; {n_iv} interviews</p>'
        + '</div></div>'
        + body
        + '<div class="sp-foot">Spradley &middot; app.spradley.io</div>'
        + '</div>'
    )


def _rcard(name: str, data: dict) -> str:
    hl = H(name)
    tg = H(data.get("tag", ""))
    v  = data.get("voice_count", 0)
    sm = H(data.get("summary", ""))
    ql = "".join(f"<li>{H(q)}</li>" for q in data.get("quotes", []))
    return (
        f'<div class="card"><p class="hl">{hl}</p>'
        f'<p class="meta"><span class="badge">{tg}</span>'
        f' &middot; {v} voice{"s" if v != 1 else ""}</p>'
        f'<details class="det"><summary>Read summary and quotes</summary>'
        f'<div class="det-body"><h4>Summary</h4><p>{sm}</p>'
        f'<h4>Paraphrased quotes</h4>'
        f'<ol class="quotes">{ql}</ol></div></details></div>'
        f'<hr class="div">'
    )


def _ecard(exp: dict) -> str:
    t = H(exp.get("title", ""))
    s = H(exp.get("summary", ""))
    r = H(exp.get("rationale", ""))
    g = H(exp.get("tag", ""))
    return (
        f'<div class="card"><p class="exp-hl">{t}</p>'
        f'<p class="meta"><span class="badge">{g}</span> {s}</p>'
        f'<details class="det"><summary>Read rationale</summary>'
        f'<div class="det-body"><p>{r}</p></div></details></div>'
    )


def _ltree(
    lineage: dict, clusters: dict,
    global_store: dict, interview_store: dict, db: dict
) -> str:
    l3_map = {item["code"]: item for item in global_store["l3_codes"]}
    l2_map: dict = {}
    for iv_id, store in interview_store.items():
        for item in store["l2_codes"]:
            l2_map[item["code"]] = {"interview_id": iv_id, **item}

    dot_cls = {"working_well": "lv-dot-g", "needs_work": "lv-dot-r", "mixed": "lv-dot-a"}
    out = ""

    for cname, lin in lineage.items():
        n_src   = len({e["interview_question_id"] for e in lin["l1_codes"]})
        cl_data = clusters.get(cname, {})
        cl_dot  = dot_cls.get(cl_data.get("category", "mixed"), "lv-dot-a")
        cl_tag  = H(cl_data.get("tag", ""))

        l3h = ""
        for l3c in lin["l3_codes"]:
            l3i = l3_map.get(l3c, {})
            ml2 = l3i.get("merged_from_l2", [])
            l2h = ""
            for l2c in ml2:
                l2i  = l2_map.get(l2c, {})
                ivid = l2i.get("interview_id", "unknown")
                ml1  = l2i.get("merged_from_l1", [])
                l1h  = ""
                for l1c in ml1:
                    iqs = [iq for iq, e in db.items() if l1c in e.get("l1_codes", [])]
                    qah = ""
                    for iq in iqs:
                        e = db[iq]
                        qah += (
                            f'<div class="qa-blk">'
                            f'<span class="iq-tag">{H(iq)}</span>'
                            f'<p class="qa-ln"><strong>Q:</strong> {H(e["question"])}</p>'
                            f'<p class="qa-ln"><strong>A:</strong> {H(e["anonymised_answer"])}</p>'
                            f'</div>'
                        )
                    l1h += (
                        f'<details class="tr tr-l1">'
                        f'<summary><span class="lv lv-1">L1</span> {H(l1c)}</summary>'
                        f'<div class="tr-body">'
                        f'{qah or "<p class=empty>No source Q&amp;A found.</p>"}'
                        f'</div></details>'
                    )
                l2h += (
                    f'<details class="tr tr-l2">'
                    f'<summary><span class="lv lv-2">L2</span> {H(l2c)}'
                    f' <span class="int-tag">{H(ivid[:8])}</span></summary>'
                    f'<div class="tr-body">'
                    f'{l1h or "<p class=empty>No L1 codes.</p>"}'
                    f'</div></details>'
                )
            l3h += (
                f'<details class="tr tr-l3">'
                f'<summary><span class="lv lv-3">L3</span> {H(l3c)}</summary>'
                f'<div class="tr-body">'
                f'{l2h or "<p class=empty>No L2 codes.</p>"}'
                f'</div></details>'
            )
        out += (
            f'<details class="tr tr-cluster">'
            f'<summary>{H(cname)}'
            f' <span class="lv-dot {cl_dot}"></span>'
            f' <span class="badge">{cl_tag}</span>'
            f' <span class="src-n">{n_src} sources</span></summary>'
            f'<div class="tr-body">'
            f'{l3h or "<p class=empty>No L3 codes.</p>"}'
            f'</div></details>'
        )
    return out


def build_report_html(
    clusters: dict, interviews: list, experiments: list,
    global_store: dict, interview_store: dict, lineage: dict, db: dict
) -> str:
    """Build the full standalone HTML web app (C13). Returns complete HTML string."""
    logo_candidates = ["spradley_logo.png", "spradley_logo.svg", "logo.png", "logo.svg"]
    logo_file = next(
        (f for f in logo_candidates if os.path.exists(os.path.join("assets", f))),
        None
    )
    logo_html = (
        f'<img src="../assets/{H(logo_file)}" class="logo-img" alt="Spradley">'
        if logo_file else '<div class="logo">SP</div>'
    )

    by_cat: dict = {"working_well": [], "needs_work": [], "mixed": []}
    for n, d in clusters.items():
        by_cat.get(d.get("category", "mixed"), by_cat["mixed"]).append((n, d))

    report = ""
    if by_cat["working_well"]:
        report += "<h2>What&#x2019;s working well</h2>"
        for n, d in by_cat["working_well"]:
            report += _rcard(n, d)
    if by_cat["needs_work"]:
        report += "<h2>What needs work</h2>"
        for n, d in by_cat["needs_work"]:
            report += _rcard(n, d)
    if by_cat["mixed"]:
        report += "<h2>Mixed signals &#x26; tensions</h2>"
        for n, d in by_cat["mixed"]:
            report += _rcard(n, d)
    if experiments:
        report += "<h2>Experiments</h2>"
        for exp in experiments:
            report += _ecard(exp)

    n_iv = len(interviews)
    report += (
        '<div class="about"><h3>Spradley guide: How the analysis works</h3>'
        f'<p>Insights are based on qualitative analysis of {n_iv} employee AI interview '
        'transcripts. Rather than survey scores, we surface <strong>patterns</strong> '
        'from what people say.</p>'
        '<ol>'
        '<li>Transcripts are clustered to identify recurring patterns.</li>'
        '<li>Patterns are distilled into clear findings.</li>'
        '<li>Findings are labelled based on matching workplace themes.</li>'
        '<li>Paraphrased quotes give context while preserving anonymity.</li>'
        '</ol></div>'
        '<div class="footer">Spradley &middot; app.spradley.io</div>'
    )

    lineage_tab = (
        '<p class="lineage-intro">Expand any cluster to trace a finding back to its source '
        'interview answers. Each level is independently collapsible.</p>'
        + _ltree(lineage, clusters, global_store, interview_store, db)
    )

    meta = H(f"Spradley · {len(clusters)} themes · {n_iv} interviews")
    js = (
        "function showTab(name,btn){"
        "document.querySelectorAll('.tab-pane').forEach(function(el){el.classList.remove('active');});"
        "document.querySelectorAll('.tab-btn').forEach(function(el){el.classList.remove('active');});"
        "document.getElementById('pane-'+name).classList.add('active');"
        "btn.classList.add('active');}"
    )

    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Spradley: Employee Insights</title>\n"
        "<style>" + APP_CSS + "</style>\n"
        "</head>\n<body>\n\n"
        "<header class=\"app-header\">\n"
        "  " + logo_html + "\n"
        "  <div>\n"
        "    <div class=\"app-title\">Employee Insights Report</div>\n"
        "    <div class=\"app-meta\">" + meta + "</div>\n"
        "  </div>\n"
        "  <nav class=\"tabs\">\n"
        "    <button class=\"tab-btn active\" onclick=\"showTab('report',this)\">Report</button>\n"
        "    <button class=\"tab-btn\" onclick=\"showTab('lineage',this)\">Data Lineage</button>\n"
        "  </nav>\n"
        "</header>\n\n"
        "<div id=\"pane-report\" class=\"tab-pane active\">\n" + report + "\n</div>\n\n"
        "<div id=\"pane-lineage\" class=\"tab-pane\">\n" + lineage_tab + "\n</div>\n\n"
        "<script>\n" + js + "\n</script>\n\n"
        "</body>\n</html>"
    )
