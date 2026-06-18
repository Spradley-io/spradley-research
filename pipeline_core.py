"""
pipeline_core.py — Spradley Coding Pipeline: shared config, prompts, and functions.

Imported by both Pipeline_Execution.ipynb and Pipeline_Evals.ipynb.
Organised by pipeline stage (C-numbers match the notebook cells).
To modify a stage's prompt or logic, Ctrl+F its C-number here.
"""

import re
import json
import os

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
