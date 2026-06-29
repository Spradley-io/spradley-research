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
import datetime

# ── Fixed paths ───────────────────────────────────────────────────────────────
INPUT_DIR  = "interview_input"
KEYS_ENV   = "keys.env"
OUTPUT_DIR = "pipeline_output"

# ── Runtime config (C0 in the notebook overrides these per run) ───────────────
CONFIG = {
    "LLM_PROVIDER":    "anthropic",
    "LLM_MODEL":       "claude-haiku-4-5-20251001",
    "LLM_TEMPERATURE": 0.2,
    "L2_CODES_RANGE":  (20, 30),
    "L3_CODES_RANGE":  (40, 80),
    "CLUSTERS_RANGE":  (7, 12),
    # ── Input ─────────────────────────────────────────────────────────────────
    # Set INPUT_FILE to the filename inside interview_input/.
    # Set INPUT_FORMAT to the matching parser key (see _PARSERS below).
    "INPUT_FILE":        "the-office-2.csv",
    "INPUT_FORMAT":      "spradley_v2",
    # ── Experiments ───────────────────────────────────────────────────────────
    # Total experiment proposals across all needs_work / mixed clusters.
    "MAX_EXPERIMENTS":   3,
}


# ── C2: LLM client ────────────────────────────────────────────────────────────

def call_llm(prompt: str, system: str = "You are a qualitative research assistant.",
             model: str | None = None, temperature: float | None = None) -> str:
    """Single LLM call. Uses CONFIG by default; override model/temperature for eval judges."""
    import time
    _model       = model       or CONFIG["LLM_MODEL"]
    _temperature = temperature if temperature is not None else CONFIG["LLM_TEMPERATURE"]
    provider     = CONFIG["LLM_PROVIDER"]

    if provider == "anthropic":
        import anthropic, httpx
        client = anthropic.Anthropic(
            http_client=httpx.Client(verify=False, trust_env=False)
        )
        for attempt in range(5):
            try:
                resp = client.messages.create(
                    model=_model, max_tokens=8192,
                    temperature=_temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.content[0].text
            except anthropic.RateLimitError:
                if attempt == 4:
                    raise
                wait = 2 ** attempt * 5
                print(f"  [rate limit] retrying in {wait}s...")
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code == 529 and attempt < 4:
                    wait = 2 ** attempt * 5
                    print(f"  [overloaded] retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
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

def _parse_legacy(df: "pd.DataFrame") -> list:
    """Parser for the original pilot_transcripts.csv format.
    Columns: session_id, turn_number, speaker (User/Bot), message.
    Bot message at turn N-1 becomes the question for User turn N.
    """
    required = {"session_id", "turn_number", "speaker", "message"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns for 'legacy' format: {missing}")

    interviews = []
    for interview_id, group in df.groupby("session_id"):
        group    = group.sort_values("turn_number")
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
        if qa_pairs:
            interviews.append({"interview_id": interview_id, "qa_pairs": qa_pairs})
    return interviews


def _parse_spradley_v2(df: "pd.DataFrame") -> list:
    """Parser for the Spradley platform export format (e.g. the-office-2.csv).
    Columns include: employee_id, thread, turn_index, question_asked,
    answer_text, is_skipped, participant_status, is_test.
    Each non-skipped row with a non-empty answer becomes one Q&A pair.
    """
    required = {"employee_id", "thread", "turn_index", "question_asked",
                "answer_text", "is_skipped", "participant_status", "is_test"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns for 'spradley_v2' format: {missing}")

    df = df[
        (df["participant_status"] == "completed") &
        (df["is_test"].astype(str).str.lower() != "true") &
        (df["is_skipped"].astype(str).str.lower() != "true") &
        (df["answer_text"].notna()) &
        (df["answer_text"].astype(str).str.strip() != "")
    ].copy()

    interviews = []
    for interview_id, group in df.groupby("employee_id"):
        group    = group.sort_values(["thread", "turn_index"])
        qa_pairs = []
        for turn_number, (_, row) in enumerate(group.iterrows(), start=1):
            qa_pairs.append({
                "turn_number": turn_number,
                "question":    str(row["question_asked"]),
                "answer":      str(row["answer_text"]),
            })
        if qa_pairs:
            interviews.append({"interview_id": str(interview_id), "qa_pairs": qa_pairs})
    return interviews


_PARSERS = {
    "legacy":       _parse_legacy,
    "spradley_v2":  _parse_spradley_v2,
}


def load_interviews() -> list:
    """Load interviews from interview_input/ using CONFIG["INPUT_FILE"] and CONFIG["INPUT_FORMAT"].
    Returns the standardised interviews list consumed by all downstream cells.
    To add a new format: write a _parse_<name>(df) function and register it in _PARSERS.
    """
    import pandas as pd

    fmt      = CONFIG.get("INPUT_FORMAT", "legacy")
    filename = CONFIG.get("INPUT_FILE", "")
    path     = os.path.join(INPUT_DIR, filename)

    if fmt not in _PARSERS:
        raise ValueError(f"Unknown INPUT_FORMAT {fmt!r}. Available: {list(_PARSERS)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    interviews = _PARSERS[fmt](df)

    if not interviews:
        raise ValueError(f"No interviews loaded from {path!r} with format {fmt!r}")
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


# ── C5: DB init ──────────────────────────────────────────────────────────────

def parse_json_safe(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:200] + ("..." if len(text) > 200 else "")
        raise json.JSONDecodeError(
            f"{e.msg} — LLM output truncated or malformed "
            f"(got {len(text)} chars, likely hit max_tokens). "
            f"Preview: {preview!r}",
            e.doc, e.pos
        ) from None


# ── C6: L2 Per-interview Coder ───────────────────────────────────────────────

PROMPT_L2_DIRECT = (
    "You are a qualitative researcher performing thematic coding of a complete employee interview.\n\n"
    "Your task: read all Q&A turns below and generate between {l2_min} and {l2_max} open codes\n"
    "(2-5 word noun phrases) that together cover all meaningful topics raised in the interview.\n"
    "Be exhaustive -- do not drop a theme just because it appears in only one or two turns.\n\n"
    "For each code, list the IDs of the turns that support it (\"source_qa_ids\").\n"
    "A turn may be left uncited if it contains nothing codeable\n"
    "(e.g. a purely procedural exchange or a turn where the employee has nothing to add).\n\n"
    "Polarity rule: your code label must reflect the direction of the employee's actual experience.\n"
    "A positive statement (\"my manager supports my growth\") and a negated one\n"
    "(\"my manager does not support my growth\") must map to codes with opposite polarity --\n"
    "e.g. \"supportive management\" vs \"lack of management support\".\n"
    "Never assign the same code to statements that contradict each other.\n\n"
    "--- EXAMPLE ---\n"
    "Interview turns:\n"
    "[abc1_t1] Q: How would you describe your relationship with your manager?\n"
    "          A: She is always approachable and gives honest, specific feedback.\n"
    "[abc1_t2] Q: What does a productive day look like for you?\n"
    "          A: One where I finish a feature end-to-end without interruptions.\n"
    "[abc1_t3] Q: Anything else you would like to share?\n"
    "          A: Not really, I think that covers it.\n\n"
    "Output:\n"
    '{{"codes": [\n'
    '  {{"code": "accessible and honest management", "source_qa_ids": ["abc1_t1"]}},\n'
    '  {{"code": "uninterrupted deep work", "source_qa_ids": ["abc1_t2"]}}\n'
    "]}}\n"
    "(abc1_t3 is uncited -- the employee provided no codeable content)\n"
    "--- END EXAMPLE ---\n\n"
    "Complete interview ({n_turns} turns):\n"
    "{interview_turns}\n\n"
    "Return only valid JSON -- no other text:\n"
    '{{"codes": [{{"code": "label", "source_qa_ids": ["turn_id"]}}, ...]}}'
)

def code_one_interview(interview_id_prefix: str, qa_entries: list) -> list:
    """Code a complete interview in one pass. Returns L2 codes with Q&A source lineage.

    qa_entries: list of (iq_id, question, anonymised_answer) tuples in turn order.
    Returns:    list of {"code": str, "source_qa_ids": [str]} dicts.
    """
    l2_min, l2_max = CONFIG["L2_CODES_RANGE"]
    turns = "\n".join(
        f"[{iq_id}] Q: {question}\n          A: {answer}"
        for iq_id, question, answer in qa_entries
    )
    prompt = PROMPT_L2_DIRECT.format(
        l2_min=l2_min, l2_max=l2_max,
        n_turns=len(qa_entries),
        interview_turns=turns,
    )
    raw = call_llm(prompt)
    return parse_json_safe(raw)["codes"]


# ── C7: Global Consolidator (L3) ─────────────────────────────────────────────

PROMPT_L3 = (
    "You are a qualitative researcher consolidating codes from {n_interviews} employee interviews\n"
    "into a final set of between {l3_min} and {l3_max} codes. Merge highly similar codes across\n"
    "interviews; keep meaningfully distinct concepts separate. Use 2-5 word noun-phrase labels.\n"
    "You MUST list which source L2 codes each new code absorbs.\n\n"
    "Three rules -- apply before merging:\n"
    "1. Treat the list order as arbitrary. Do not give codes that appear earlier any priority.\n"
    "2. Never merge codes with opposite polarity. \"Access to training\" and \"lack of training\"\n"
    "   must remain separate codes even though they share a topic.\n"
    "3. Normalise wording before deciding: \"growth opportunities\", \"career development support\",\n"
    "   and \"professional growth\" are likely the same concept -- merge them unless context clearly\n"
    "   distinguishes them.\n\n"
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
    "Return only valid JSON -- no other text:\n"
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


# ── C8: Theme Clustering ──────────────────────────────────────────────────────

PROMPT_CLUSTER = (
    "You are a qualitative researcher grouping final codes into thematic clusters.\n"
    "Group into between {clusters_min} and {clusters_max} clusters.\n"
    "Each cluster must have a 3-6 word name that reads as a natural section header: "
    "specific, concrete, and slightly engaging rather than an academic label. "
    "Title-case each word. It must contain at least 2 codes.\n\n"
    "Rules: (1) Treat the input order as arbitrary.\n"
    "(2) Never place codes that reflect opposite experiences in the same cluster --\n"
    "positive and negative framings of the same topic belong in separate clusters.\n\n"
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
    "Return only valid JSON -- no other text:\n"
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


# ── C9: LLM Explainer ────────────────────────────────────────────────────────

PROMPT_FINDING = (
    "You are a senior HR consultant presenting findings to a business leader. "
    "Write in the style of a top-tier strategy consulting firm: conclusion first, specific evidence, zero filler.\n"
    "Domain: employee satisfaction survey (anonymised AI interview transcripts).\n\n"
    "Cluster: {cluster_name}\n"
    "Codes in this cluster: {codes_list}\n\n"
    "Supporting employee responses:\n"
    "{qa_pairs_text}\n\n"
    "Return valid JSON with exactly this structure:\n"
    '{{\n'
    '  "category": "working_well",\n'
    '  "tagline": "...",\n'
    '  "summary": "...",\n'
    '  "tension": null,\n'
    '  "quotes": ["...", "..."],\n'
    '  "tag": "..."\n'
    '}}\n\n'
    'Use exactly one of these values for category: "working_well", "needs_work", or "mixed".\n\n'
    "--- WRITING RULES ---\n"
    "tagline (slide action title):\n"
    "  - One complete sentence stating the key takeaway -- what management should know or do.\n"
    "  - Lead with the conclusion, not the topic label.\n"
    "    Bad: 'Management and feedback.'\n"
    "    Good: 'Unclear feedback cycles are quietly stalling team development.'\n"
    "  - Max 15 words. No hedging adverbs (somewhat, quite, fairly, seems).\n"
    "summary (Pyramid Principle -- 3 sentences max):\n"
    "  - Sentence 1: restate the answer directly (the conclusion in one line).\n"
    "  - Sentence 2: cite 1-2 specific evidence patterns from the responses above.\n"
    "  - Sentence 3: state the business implication or the action priority.\n"
    "  - No filler openers ('The data shows...', 'It appears that...'). State claims directly.\n"
    "  - If evidence is mixed, sentence 2 must name both directions explicitly.\n"
    "  - Ground every claim in the cited responses. If only some employees mentioned something,\n"
    "    say 'some' or 'a few', not 'employees' or 'the team'.\n"
    "tension (optional -- set null in most cases):\n"
    "  - Include ONLY when the responses above reveal a genuine co-existing tension: the same\n"
    "    people simultaneously want or experience two contradictory things.\n"
    "  - This is NOT variation ('some feel X, others feel Y') and NOT a mixed picture.\n"
    "  - It IS a tension when people express both sides themselves -- e.g. wanting more\n"
    "    involvement in decisions while also feeling overwhelmed by too many meetings.\n"
    "  - Both sides of the contradiction must be clearly present in the cited responses.\n"
    "  - If genuine: one sentence only. State both sides plainly. No dramatic openers.\n"
    "  - When in doubt, set null. Fewer is better -- only flag real tensions.\n"
    "quotes:\n"
    "  - 2-3 paraphrases, as specific and concrete as possible. Prefer concrete over abstract.\n"
    "  - Strip filler words ('you know', 'like', 'I mean'). Preserve meaning and anonymity.\n"
    "tag: 1-2 word theme label e.g. Culture, Development, Wellbeing.\n"
    "Never use em dashes, dramatic openers ('The tension is acute:', 'This reveals a divide:',\n"
    "'The stakes are high:'), or AI-sounding framing. Write plainly.\n"
    "Return only valid JSON. No other text."
)

PROMPT_EXPERIMENTS = (
    "You are a senior HR consultant reviewing employee interview findings and proposing actionable experiments.\n\n"
    "Ground each experiment in the specific finding it addresses. Do not overstate the problem "
    "or promise more than a small experiment can deliver. Be specific and concrete -- vague suggestions are not useful.\n\n"
    "Findings that need attention:\n"
    "{findings_text}\n\n"
    "Choose the {max_n} most impactful experiments across all findings above. "
    "Prioritise by likely impact and feasibility within 1-2 weeks.\n\n"
    "Experiment title rule: imperative verb, plain everyday language, specific, max 10 words. "
    "No jargon. Example: 'Share the promotion criteria in writing with the team.'\n\n"
    "Return valid JSON:\n"
    '{{"experiments": [\n'
    '  {{"title": "Imperative-verb title, max 10 words",\n'
    '   "source_cluster": "exact finding name from the list above",\n'
    '   "insight": "1-2 sentences: which finding this addresses and why it matters for that team.",\n'
    '   "try_this": "The concrete action: what to do, how often, who does it. One paragraph.",\n'
    '   "working_when": "One sentence: the observable change that signals this is working.",\n'
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

def propose_experiments(needs_attention: list, max_n: int | None = None) -> list:
    """Generate experiment proposals for needs_work and mixed clusters.
    needs_attention: [(name, data_dict), ...]
    max_n: total experiments to return; defaults to CONFIG["MAX_EXPERIMENTS"].
    """
    if not needs_attention:
        return []
    if max_n is None:
        max_n = CONFIG.get("MAX_EXPERIMENTS", 3)
    parts = []
    for name, data in needs_attention:
        parts.append(
            f"Finding: {name}\n"
            f"Category: {data['category']}\n"
            f"Summary: {data['summary']}"
        )
    prompt = PROMPT_EXPERIMENTS.format(
        findings_text="\n\n---\n\n".join(parts),
        max_n=max_n,
    )
    raw = call_llm(prompt)
    return parse_json_safe(raw).get("experiments", [])


EMOTIONAL_DIMENSIONS = ["satisfaction", "stress", "worry", "trust", "motivation", "fairness"]

PROMPT_DIMENSIONS = (
    "You are analysing one anonymised employee interview.\n\n"
    "For each of the 6 emotional dimensions below, decide whether this specific employee "
    "personally expressed that emotion or concern in their answers.\n\n"
    "Rules:\n"
    "  - Score 1 only if the employee directly expresses the emotion in first person.\n"
    "  - Score 0 if they only report it about others or the organisation in general.\n"
    "  - Score 0 if the topic is mentioned neutrally with no emotional signal.\n\n"
    "Dimensions:\n"
    "  satisfaction: personal fulfilment, enjoyment, or positive engagement with work or team\n"
    "  stress:       personal pressure, overload, anxiety, or overwhelm\n"
    "  worry:        personal uncertainty about the future, job security, or direction\n"
    "  trust:        direct personal expression of trust or distrust in management / colleagues / processes\n"
    "  motivation:   personal sense of purpose, drive, desire for recognition or growth\n"
    "  fairness:     personal perception of equal treatment, transparent criteria, or consistent standards\n\n"
    "Interview Q&A:\n"
    "{interview_turns}\n\n"
    'Return only valid JSON: {{"satisfaction":0,"stress":0,"worry":0,"trust":0,"motivation":0,"fairness":0}}\n'
    "Use 1 for expressed, 0 for not expressed. No other text."
)


def score_dimensions_one_interview(interview_id_prefix: str, qa_entries: list) -> dict:
    """Score 6 emotional dimensions for one interview. Returns {dim: 0|1}.

    qa_entries: list of (iq_id, question, answer) tuples.
    """
    turns = "\n".join(
        f"[{iq_id}] Q: {question}\n          A: {answer}"
        for iq_id, question, answer in qa_entries
    )
    prompt = PROMPT_DIMENSIONS.format(interview_turns=turns)
    raw    = call_llm(prompt)
    result = parse_json_safe(raw)
    return {d: int(bool(result.get(d, 0))) for d in EMOTIONAL_DIMENSIONS}


PROMPT_HEADLINE = (
    "You are a senior HR consultant writing the opening and closing narrative for a management briefing "
    "on employee satisfaction. Write in the style of a top-tier strategy consulting firm.\n\n"
    "Use the SCQA structure across the opening (headline field):\n"
    "  - Situation (paragraph 1): one grounding sentence setting the context "
    "(e.g. 'This team of N employees was interviewed about...'). Then 1-2 sentences on the broad picture.\n"
    "  - Complication (paragraph 2): the key tension or surprise the findings reveal. "
    "What is most concerning or unexpected?\n"
    "  - Question + Answer (paragraph 3): state the implicit management question, then give your synthesis "
    "and main recommendation. Put the answer first -- do not build to it.\n"
    "Keep paragraphs tight (2-3 sentences each). No superlatives. No filler. "
    "No dramatic openers or AI-sounding phrasing ('The tension is clear:', 'This signals a crisis:'). "
    "Do not list findings -- tell a story. Ground every claim in the evidence below.\n\n"
    "Number of employees interviewed: {n_interviews}\n\n"
    "Findings:\n"
    "{findings_text}\n\n"
    "Return valid JSON with exactly two string fields:\n"
    '  "headline": 3 paragraphs following the SCQA structure above. '
    "Separate paragraphs with a newline character.\n"
    '  "note_to_protect": 1-2 paragraphs identifying what is genuinely positive and specific '
    "in the data and why it matters. Grounded, not cheerleading.\n\n"
    'Return format: {{"headline": "...", "note_to_protect": "..."}}\n\n'
    "Never use em dashes in any text.\n"
    "Return only valid JSON. No other text."
)


def generate_headline(clusters: dict, n_interviews: int, model: str) -> dict:
    """Generate the report headline and closing note from all cluster findings.

    Returns {"headline": str, "note_to_protect": str}.
    Store results in global_store["headline"] and global_store["note_to_protect"].
    """
    parts = []
    for name, data in clusters.items():
        cat     = data.get("category", "mixed")
        tagline = data.get("tagline", "")
        summary = data.get("summary", "")
        parts.append(
            f"Finding: {name}\n"
            f"Category: {cat}\n"
            f"Tagline: {tagline}\n"
            f"Summary: {summary}"
        )
    findings_text = "\n\n---\n\n".join(parts)
    prompt = PROMPT_HEADLINE.format(n_interviews=n_interviews, findings_text=findings_text)
    raw    = call_llm(prompt, model=model)
    return parse_json_safe(raw)


# ── C11 / C12: HTML report generation ────────────────────────────────────────

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
details.sp-d>summary::before{content:'\\25B6  ';font-size:9px}
details[open].sp-d>summary::before{content:'\\25BC  '}
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
    "details.det>summary::before{content:'\\25B6  ';font-size:9px}\n"
    "details[open].det>summary::before{content:'\\25BC  '}\n"
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
    # ── Report tab document-style classes (rpt- prefix) ───────────────────────
    ".rpt-title{font-size:26px;font-weight:800;margin-bottom:6px}\n"
    ".rpt-sub{font-size:13px;color:#888;margin-bottom:28px}\n"
    ".rpt-sec{font-size:20px;font-weight:700;margin:32px 0 14px}\n"
    ".rpt-hr{border:none;border-top:1px solid #e8e8e8;margin:24px 0}\n"
    ".rpt-num{font-size:16px;font-weight:700;margin:0 0 4px;color:#111}\n"
    ".rpt-tagline{font-size:15px;font-weight:700;color:#111;margin-bottom:10px}\n"
    ".rpt-body{font-size:14px;line-height:1.75;color:#444;margin-bottom:8px}\n"
    ".rpt-qlabel{font-size:10px;font-weight:700;text-transform:uppercase;"
    "letter-spacing:.6px;color:#bbb;margin:14px 0 6px}\n"
    ".rpt-quote{font-size:13px;color:#666;font-style:italic;margin-bottom:6px}\n"
    ".rpt-exp-title{font-size:15px;font-weight:700;margin:20px 0 8px}\n"
    ".rpt-intro{font-size:14px;color:#555;margin-bottom:16px}\n"
    ".rpt-accent-ww{border-left:3px solid #16a34a;padding-left:14px;margin-bottom:18px}\n"
    ".rpt-accent-nw{border-left:3px solid #dc2626;padding-left:14px;margin-bottom:18px}\n"
    ".rpt-accent-mx{border-left:3px solid #d97706;padding-left:14px;margin-bottom:18px}\n"
    ".rpt-tension-label{font-size:10px;font-weight:700;text-transform:uppercase;"
    "letter-spacing:.6px;color:#aaa;margin:10px 0 3px}\n"
    ".rpt-tension{font-size:13px;color:#666;font-style:italic;line-height:1.6;margin-bottom:6px}\n"
    ".rpt-pct{display:inline-block;font-size:11px;font-weight:600;color:#fff;"
    "border-radius:12px;padding:1px 8px;margin-left:8px;vertical-align:middle;opacity:0.85}\n"
    ".rpt-pct-ww{background:#16a34a}.rpt-pct-nw{background:#dc2626}.rpt-pct-mx{background:#d97706}\n"
    ".topics-section{margin-bottom:48px}\n"
    ".topics-section h3{font-size:15px;font-weight:700;margin-bottom:8px;color:#111}\n"
    ".topics-intro{font-size:13px;color:#666;margin-bottom:16px;line-height:1.6;max-width:640px}\n"
    ".topics-svg{width:100%;height:auto;display:block}\n"
    ".radar-wrap{max-width:560px;margin:0 auto;position:relative}\n"
    "#radar-tooltip{position:absolute;background:#fff;border:1px solid #e5e7eb;border-radius:6px;"
    "padding:10px 14px;font-size:12px;line-height:1.6;color:#333;"
    "box-shadow:0 2px 8px rgba(0,0,0,0.10);pointer-events:none;display:none;"
    "max-width:220px;z-index:10}\n"
    "#radar-tooltip strong{display:block;font-size:13px;margin-bottom:4px;color:#111}\n"
    "#radar-tooltip .rt-pct{font-size:15px;font-weight:700;color:#4f8ef7;margin-bottom:6px;display:block}\n"
    "#radar-tooltip .rt-def{color:#666;font-size:11px;line-height:1.5}\n"
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
        n_src   = len(lin.get("l1_qa_ids", []))
        cl_data = clusters.get(cname, {})
        cl_dot  = dot_cls.get(cl_data.get("category", "mixed"), "lv-dot-a")
        cl_tag  = H(cl_data.get("tag", ""))

        l3h = ""
        for l3c in lin["l3_codes"]:
            l3i = l3_map.get(l3c, {})
            ml2 = l3i.get("merged_from_l2", [])
            l2h = ""
            for l2c in ml2:
                l2i    = l2_map.get(l2c, {})
                ivid   = l2i.get("interview_id", "unknown")
                src_qa = l2i.get("source_qa_ids", [])
                qah = ""
                for iq_id in src_qa:
                    e = db.get(iq_id)
                    if e:
                        qah += (
                            f'<div class="qa-blk">'
                            f'<span class="iq-tag">{H(iq_id)}</span>'
                            f'<p class="qa-ln"><strong>Q:</strong> {H(e["question"])}</p>'
                            f'<p class="qa-ln"><strong>A:</strong> {H(e["anonymised_answer"])}</p>'
                            f'</div>'
                        )
                l2h += (
                    f'<details class="tr tr-l2">'
                    f'<summary><span class="lv lv-2">L2</span> {H(l2c)}'
                    f' <span class="int-tag">{H(ivid[:8])}</span></summary>'
                    f'<div class="tr-body">'
                    f'{qah or "<p class=empty>No source Q&amp;A found.</p>"}'
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


def _rpt_insight(n: int, name: str, data: dict, n_iv: int = 0) -> str:
    """Render one numbered insight block in the document-style report tab."""
    tagline  = H(data.get("tagline", ""))
    summary  = H(data.get("summary", ""))
    tension  = data.get("tension") or ""
    quotes   = data.get("quotes", [])
    category = data.get("category", "mixed")
    voice    = data.get("voice_count", 0)

    accent_cls = {"working_well": "rpt-accent-ww", "needs_work": "rpt-accent-nw"}.get(
        category, "rpt-accent-mx"
    )
    pct_cls = {"working_well": "rpt-pct-ww", "needs_work": "rpt-pct-nw"}.get(
        category, "rpt-pct-mx"
    )
    pct_val   = round(voice / n_iv * 10) * 10 if n_iv else 0
    pct_badge = (
        f'<span class="rpt-pct {pct_cls}">~{pct_val}% of people</span>'
        if n_iv and voice else ""
    )

    tension_block = (
        f'<p class="rpt-tension-label">Tension</p>'
        f'<p class="rpt-tension">{H(tension)}</p>'
    ) if tension else ""

    ql = "".join(f'<p class="rpt-quote">&ldquo;{H(q)}&rdquo;</p>' for q in quotes)
    ql_block = f'<p class="rpt-qlabel">What people said:</p>{ql}' if quotes else ""
    return (
        f'<div class="{accent_cls}">'
        f'<h3 class="rpt-num">{n}. {H(name)}{pct_badge}</h3>'
        f'<p class="rpt-tagline">{tagline}</p>'
        f'{tension_block}'
        f'<p class="rpt-body">{summary}</p>'
        f'{ql_block}'
        f'</div>'
        f'<hr class="rpt-hr">'
    )


def _rpt_exp(n: int, exp: dict) -> str:
    """Render one experiment block in the document-style report tab."""
    title        = H(exp.get("title", ""))
    insight      = H(exp.get("insight") or exp.get("summary", ""))
    try_this     = H(exp.get("try_this") or exp.get("rationale", ""))
    working_when = H(exp.get("working_when", ""))
    ww_block = (
        f'<p class="rpt-body"><strong>You\'ll know it\'s working when</strong> {working_when}</p>'
    ) if working_when else ""
    return (
        f'<h3 class="rpt-exp-title">Experiment {n}: {title}</h3>'
        f'<p class="rpt-body"><strong>The insight:</strong> {insight}</p>'
        f'<p class="rpt-body"><strong>Try this:</strong> {try_this}</p>'
        f'{ww_block}'
        f'<hr class="rpt-hr">'
    )


def _bubble_pane(clusters: dict, lineage: dict, global_store: dict,
                  interview_store: dict, n_iv: int) -> str:
    """SVG bubble chart: free-scatter with force-directed collision avoidance."""
    import math, random

    l3_to_cluster: dict = {}
    for cname, cdata in clusters.items():
        for l3c in cdata.get("l3_codes", []):
            l3_to_cluster[l3c] = cname

    l2_to_ivs: dict = {}
    for iid, store in interview_store.items():
        for item in store.get("l2_codes", []):
            l2_to_ivs.setdefault(item["code"], set()).add(iid)

    l3_to_count: dict = {}
    for l3item in global_store.get("l3_codes", []):
        l3c = l3item["code"]
        ivs: set = set()
        for l2c in l3item.get("merged_from_l2", []):
            ivs |= l2_to_ivs.get(l2c, set())
        l3_to_count[l3c] = len(ivs)

    filtered = [(l3c, cnt) for l3c, cnt in l3_to_count.items()
                if n_iv and cnt / n_iv > 0.10]
    if not filtered:
        return '<p class="topics-intro">No topics met the 10% coverage threshold.</p>'

    counts  = [cnt for _, cnt in filtered]
    min_cnt = min(counts)
    max_cnt = max(counts)
    R_MIN, R_MAX = 20, 68

    def _r(cnt: int) -> float:
        if max_cnt == min_cnt:
            return (R_MIN + R_MAX) / 2
        t = math.sqrt((cnt - min_cnt) / (max_cnt - min_cnt))
        return R_MIN + t * (R_MAX - R_MIN)

    SVG_W, SVG_H = 900, 560
    AXIS_Y = SVG_H - 38
    x_bands_px = {
        "needs_work":   (int(SVG_W * 0.09), int(SVG_W * 0.32)),
        "mixed":        (int(SVG_W * 0.36), int(SVG_W * 0.64)),
        "working_well": (int(SVG_W * 0.67), int(SVG_W * 0.91)),
    }
    colors = {
        "working_well": ("#16a34a", "rgba(22,163,74,0.13)"),
        "needs_work":   ("#dc2626", "rgba(220,38,38,0.13)"),
        "mixed":        ("#d97706", "rgba(217,119,6,0.13)"),
    }

    DROP_R = 24   # bubbles below this radius have no useful label -- drop them
    rng = random.Random(42)
    bubbles = []
    for l3c, cnt in sorted(filtered, key=lambda x: -x[1]):
        cname    = l3_to_cluster.get(l3c, "")
        cat      = clusters.get(cname, {}).get("category", "mixed")
        xlo, xhi = x_bands_px.get(cat, (int(SVG_W * 0.36), int(SVG_W * 0.64)))
        r        = _r(cnt)
        if r < DROP_R:
            continue
        cx_init  = rng.uniform(xlo + r, xhi - r)
        cy_frac  = 1 - (cnt - min_cnt) / max((max_cnt - min_cnt), 1)
        cy_init  = R_MAX + 20 + cy_frac * (AXIS_Y - R_MAX - 60 - r)
        pct      = round(cnt / n_iv * 10) * 10
        stroke, fill = colors.get(cat, ("#888", "rgba(136,136,136,0.13)"))
        bubbles.append({
            "code": l3c, "r": r, "pct": pct,
            "cx": cx_init, "cy": cy_init,
            "xlo": xlo, "xhi": xhi,
            "stroke": stroke, "fill": fill,
        })

    def _wrap(label: str) -> list:
        """Split into at most 2 balanced lines. Never truncates."""
        words = label.split()
        if len(words) <= 1:
            return [label]
        best, best_diff = 0, 9999
        for k in range(1, len(words)):
            l1 = " ".join(words[:k])
            l2 = " ".join(words[k:])
            diff = abs(len(l1) - len(l2))
            if diff < best_diff:
                best, best_diff = k, diff
        return [" ".join(words[:best]), " ".join(words[best:])]

    circle_els = ""
    for b in bubbles:
        r      = b["r"]
        # Font size proportional to radius; larger bubbles get bigger text.
        # Text is always rendered inside the bubble and may overflow slightly for long labels.
        fs     = max(8, min(14, int(r * 0.28)))
        fs_pct = max(7, fs - 2)
        lines  = _wrap(b["code"])

        circle_el = (
            f'<circle class="bbl-c" cx="{b["cx"]:.1f}" cy="{b["cy"]:.1f}" r="{r:.1f}" '
            f'fill="{b["fill"]}" stroke="{b["stroke"]}" stroke-width="1.5"/>'
        )

        n_lines = len(lines)
        line_h  = fs * 1.3
        block_h = n_lines * line_h + fs_pct * 1.5
        y0      = b["cy"] - block_h / 2 + fs * 0.85
        text_el = ""
        for j, ln in enumerate(lines):
            text_el += (
                f'<text x="{b["cx"]:.1f}" y="{y0 + j * line_h:.1f}" '
                f'text-anchor="middle" font-size="{fs}" font-family="Arial,sans-serif" '
                f'fill="{b["stroke"]}" font-weight="600" '
                f'stroke="white" stroke-width="3" paint-order="stroke fill">{H(ln)}</text>'
            )
        pct_y = y0 + n_lines * line_h + 2
        text_el += (
            f'<text x="{b["cx"]:.1f}" y="{pct_y:.1f}" '
            f'text-anchor="middle" font-size="{fs_pct}" font-family="Arial,sans-serif" '
            f'fill="{b["stroke"]}" opacity="0.8" '
            f'stroke="white" stroke-width="2" paint-order="stroke fill">~{b["pct"]}%</text>'
        )

        circle_els += (
            f'<g class="bbl" '
            f'data-cx="{b["cx"]:.1f}" data-cy="{b["cy"]:.1f}" data-r="{r:.1f}" '
            f'data-xmin="{b["xlo"]}" data-xmax="{b["xhi"]}">'
            f'{circle_el}{text_el}</g>'
        )

    axis = (
        f'<defs>'
        f'<marker id="arrowR" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">'
        f'<polygon points="0 0, 8 3, 0 6" fill="#bbb"/></marker>'
        f'<marker id="arrowL" markerWidth="8" markerHeight="6" refX="0" refY="3" orient="auto">'
        f'<polygon points="8 0, 0 3, 8 6" fill="#bbb"/></marker>'
        f'</defs>'
        f'<line x1="24" y1="{AXIS_Y}" x2="{SVG_W - 24}" y2="{AXIS_Y}" '
        f'stroke="#bbb" stroke-width="1" marker-end="url(#arrowR)" marker-start="url(#arrowL)"/>'
        f'<text x="28" y="{AXIS_Y + 14}" font-size="10" font-family="Arial,sans-serif" '
        f'fill="#dc2626" font-weight="500">Needs Work</text>'
        f'<text x="{SVG_W // 2}" y="{AXIS_Y + 14}" font-size="10" font-family="Arial,sans-serif" '
        f'fill="#d97706" text-anchor="middle">Mixed Signals</text>'
        f'<text x="{SVG_W - 28}" y="{AXIS_Y + 14}" font-size="10" font-family="Arial,sans-serif" '
        f'fill="#16a34a" text-anchor="end" font-weight="500">Working Well</text>'
        f'<text x="{SVG_W // 2}" y="{SVG_H - 4}" font-size="9" font-family="Arial,sans-serif" '
        f'fill="#ccc" text-anchor="middle">'
        f'Bubble size and percentage reflect share of participants who raised this topic.</text>'
    )

    js_data = "[" + ",".join(
        f'[{b["cx"]:.1f},{b["cy"]:.1f},{b["r"]:.1f},{b["xlo"]},{b["xhi"]}]'
        for b in bubbles
    ) + "]"

    js = (
        "(function(){"
        + f"var AXIS_Y={AXIS_Y};"
        + f"var D={js_data};"
        + "var gs=document.querySelectorAll('.bbl');"
        "for(var iter=0;iter<80;iter++){"
        "for(var i=0;i<D.length;i++){"
        "for(var j=i+1;j<D.length;j++){"
        "var dx=D[j][0]-D[i][0],dy=D[j][1]-D[i][1];"
        "var dist=Math.sqrt(dx*dx+dy*dy)||0.01;"
        "var mn=D[i][2]+D[j][2]+18;"
        "if(dist<mn){"
        "var push=(mn-dist)/2,nx=dx/dist,ny=dy/dist;"
        "D[i][0]-=nx*push;D[i][1]-=ny*push;"
        "D[j][0]+=nx*push;D[j][1]+=ny*push;}"
        "}"
        "D[i][0]=Math.max(D[i][3]+D[i][2],Math.min(D[i][4]-D[i][2],D[i][0]));"
        "D[i][1]=Math.max(D[i][2]+8,Math.min(AXIS_Y-D[i][2]-14,D[i][1]));"
        "}"
        "}"
        "gs.forEach(function(g,i){"
        "var cx=D[i][0].toFixed(1),cy=D[i][1].toFixed(1),r=D[i][2];"
        "var circ=g.querySelector('.bbl-c');"
        "if(circ){circ.setAttribute('cx',cx);circ.setAttribute('cy',cy);}"
        "var origCy=parseFloat(g.dataset.cy);"
        "var shift=parseFloat(cy)-origCy;"
        "g.querySelectorAll('text').forEach(function(t){"
        "t.setAttribute('x',cx);"
        "t.setAttribute('y',(parseFloat(t.getAttribute('y'))+shift).toFixed(1));"
        "});"
        "});"
        "})();"
    )

    return (
        f'<svg viewBox="0 0 {SVG_W} {SVG_H}" class="topics-svg">'
        f'{circle_els}{axis}</svg>'
        f'<script>{js}</script>'
    )


def _radar_pane(dimension_store: dict, n_iv: int) -> str:
    """SVG radar chart for 6 emotional dimensions with hover tooltips."""
    import math, json as _json

    total = dimension_store.get("_total", {})
    if not total or not n_iv:
        return (
            '<p class="topics-intro" style="color:#bbb;font-style:italic;">'
            'Emotional dimension data not yet available. Run C9c in Pipeline_Execution.ipynb first.</p>'
        )

    CX, CY, R = 300, 255, 175
    LABEL_R    = R + 30
    RINGS      = [20, 40, 60, 80, 100]
    DIM_LABELS = ["Satisfaction", "Stress", "Worry", "Trust", "Motivation", "Fairness"]
    DIM_DEFS   = {
        "satisfaction": "Personal fulfilment, enjoyment, or positive engagement with work or team.",
        "stress":       "Personal pressure, overload, anxiety, or overwhelm.",
        "worry":        "Uncertainty about the future, job security, or direction.",
        "trust":        "Direct expression of trust or distrust in management, colleagues, or processes.",
        "motivation":   "Sense of purpose, drive, desire for recognition or growth.",
        "fairness":     "Perceived equal treatment, transparent criteria, or consistent standards.",
    }
    anchors  = ["middle", "start", "start", "middle", "end", "end"]
    dy_extra = [-8, 4, 4, 18, 4, 4]

    scores = [total.get(d, 0) / n_iv for d in EMOTIONAL_DIMENSIONS]

    out = '<svg viewBox="0 0 600 510" class="topics-svg" id="radar-svg">\n'

    # Concentric rings
    label_angle = math.radians(-90 + 50)   # fixed position between top and right axes
    for pct in RINGS:
        ring_r = pct / 100 * R
        lx = CX + ring_r * math.cos(label_angle) + 3
        ly = CY + ring_r * math.sin(label_angle) + 3
        out += (
            f'<circle cx="{CX}" cy="{CY}" r="{ring_r:.1f}" '
            f'fill="none" stroke="#e8e8e8" stroke-width="1"/>\n'
            f'<rect x="{lx - 3:.1f}" y="{ly - 9:.1f}" width="26" height="12" '
            f'fill="white" rx="2"/>\n'
            f'<text x="{lx:.1f}" y="{ly:.1f}" '
            f'font-size="9" fill="#aaa" font-family="Arial,sans-serif">{pct}%</text>\n'
        )

    # Axis lines + labels (visible only -- hit elements rendered after polygon)
    for i, dim in enumerate(EMOTIONAL_DIMENSIONS):
        angle = math.radians(-90 + i * 60)
        ax = CX + R * math.cos(angle)
        ay = CY + R * math.sin(angle)
        out += (
            f'<line x1="{CX}" y1="{CY}" x2="{ax:.1f}" y2="{ay:.1f}" '
            f'stroke="#d8d8d8" stroke-width="1"/>\n'
        )
        lx = CX + LABEL_R * math.cos(angle)
        ly = CY + LABEL_R * math.sin(angle) + dy_extra[i]
        out += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchors[i]}" '
            f'font-size="11" font-family="Arial,sans-serif" fill="#333" font-weight="500">'
            f'{DIM_LABELS[i]}</text>\n'
        )

    # Filled polygon
    poly_pts = []
    for i in range(len(EMOTIONAL_DIMENSIONS)):
        angle = math.radians(-90 + i * 60)
        dist  = scores[i] * R
        poly_pts.append((CX + dist * math.cos(angle), CY + dist * math.sin(angle)))

    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in poly_pts)
    out += (
        f'<polygon points="{pts_str}" fill="rgba(79,142,247,0.15)" '
        f'stroke="#4f8ef7" stroke-width="2"/>\n'
    )

    # Vertex dots + percentage annotations
    for i, (px, py) in enumerate(poly_pts):
        out += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#4f8ef7"/>\n'
        angle   = math.radians(-90 + i * 60)
        ann_off = 14
        ann_x   = CX + (scores[i] * R + ann_off) * math.cos(angle)
        ann_y   = CY + (scores[i] * R + ann_off) * math.sin(angle)
        pct_val = round(scores[i] * 10) * 10
        out += (
            f'<text x="{ann_x:.1f}" y="{ann_y + 4:.1f}" text-anchor="middle" '
            f'font-size="10" font-family="Arial,sans-serif" fill="#4f8ef7" font-weight="600">'
            f'~{pct_val}%</text>\n'
        )

    # Invisible hit elements -- rendered last so they sit on top of the polygon fill
    # Users naturally hover near the polygon edge/vertex, not just the background axis lines
    for i, (px, py) in enumerate(poly_pts):
        angle = math.radians(-90 + i * 60)
        ax    = CX + R * math.cos(angle)
        ay    = CY + R * math.sin(angle)
        dim   = EMOTIONAL_DIMENSIONS[i]
        out += (
            f'<line x1="{CX}" y1="{CY}" x2="{ax:.1f}" y2="{ay:.1f}" '
            f'stroke="transparent" stroke-width="20" class="radar-hit" '
            f'data-dim="{dim}" data-idx="{i}"/>\n'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="22" '
            f'fill="transparent" stroke="none" class="radar-hit" '
            f'data-dim="{dim}" data-idx="{i}"/>\n'
        )

    # SVG caption
    out += (
        f'<text x="{CX}" y="498" text-anchor="middle" font-size="9" '
        f'font-family="Arial,sans-serif" fill="#ccc">'
        f'Blue area shows share of participants who directly expressed each dimension.</text>\n'
    )

    out += '</svg>\n'

    # Build dimension data for JS tooltip
    dim_data = {}
    for i, dim in enumerate(EMOTIONAL_DIMENSIONS):
        pct_val = round(scores[i] * 10) * 10
        dim_data[dim] = {
            "label": DIM_LABELS[i],
            "pct":   pct_val,
            "def":   DIM_DEFS[dim],
        }

    js = (
        "(function(){"
        "var wrap=document.querySelector('.radar-wrap');"
        "if(!wrap)return;"
        "var tip=document.createElement('div');"
        "tip.id='radar-tooltip';wrap.appendChild(tip);"
        f"var DIM={_json.dumps(dim_data)};"
        "document.querySelectorAll('.radar-hit').forEach(function(el){"
        "el.addEventListener('mouseenter',function(e){"
        "var d=DIM[el.dataset.dim];if(!d)return;"
        "tip.innerHTML='<strong>'+d.label+'</strong>"
        "<span class=\"rt-pct\">~'+d.pct+'% of participants</span>"
        "<span class=\"rt-def\">'+d.def+'</span>';"
        "tip.style.display='block';"
        "var rect=wrap.getBoundingClientRect();"
        "tip.style.left=(e.clientX-rect.left+12)+'px';"
        "tip.style.top=(e.clientY-rect.top-10)+'px';"
        "});"
        "el.addEventListener('mousemove',function(e){"
        "var rect=wrap.getBoundingClientRect();"
        "tip.style.left=(e.clientX-rect.left+12)+'px';"
        "tip.style.top=(e.clientY-rect.top-10)+'px';"
        "});"
        "el.addEventListener('mouseleave',function(){"
        "tip.style.display='none';});"
        "});"
        "})();"
    )

    return out + f'<script>{js}</script>\n'


def build_report_html(
    clusters: dict, n_iv: int, experiments: list,
    global_store: dict, interview_store: dict, lineage: dict, db: dict,
    dimension_store: dict | None = None,
) -> str:
    """Build the full standalone HTML web app. Returns complete HTML string."""
    logo_candidates = ["spradley_logo.png", "spradley_logo.svg", "logo.png", "logo.svg"]
    logo_file = next(
        (f for f in logo_candidates if os.path.exists(os.path.join("assets", f))),
        None
    )
    logo_html = (
        f'<img src="../assets/{H(logo_file)}" class="logo-img" alt="Spradley">'
        if logo_file else '<div class="logo">SP</div>'
    )

    n_themes = len(clusters)
    date_str = datetime.datetime.now().strftime("%B %Y")

    # Title block
    report = (
        f'<h1 class="rpt-title">Team Insights Report</h1>'
        f'<p class="rpt-sub">Based on: {n_iv} confidential employee conversations'
        f' &middot; Themes identified: {n_themes}'
        f' &middot; Date: {date_str}</p>'
        f'<hr class="rpt-hr">'
    )

    # The headline section
    headline_text = global_store.get("headline", "")
    if headline_text:
        paras = [p.strip() for p in headline_text.split("\n") if p.strip()]
        report += f'<h2 class="rpt-sec">The headline</h2>'
        report += "".join(f'<p class="rpt-body">{H(p)}</p>' for p in paras)
        report += '<hr class="rpt-hr">'

    # Insights -- working_well first, then mixed, then needs_work
    ordered = (
        [(name, d) for name, d in clusters.items() if d.get("category") == "working_well"]
        + [(name, d) for name, d in clusters.items() if d.get("category") == "mixed"]
        + [(name, d) for name, d in clusters.items() if d.get("category") == "needs_work"]
    )
    if ordered:
        report += f'<h2 class="rpt-sec">Insights</h2><hr class="rpt-hr">'
        for idx, (name, d) in enumerate(ordered, 1):
            report += _rpt_insight(idx, name, d, n_iv)

    # Experiments
    if experiments:
        n_exp     = len(experiments)
        exp_label = "experiment" if n_exp == 1 else "experiments"
        report += f'<h2 class="rpt-sec">{n_exp} {exp_label} to consider</h2>'
        report += (
            '<p class="rpt-intro">These are lightweight, low-risk actions you can try '
            'in the next 2 to 4 weeks. Each one addresses a pattern from the insights above.</p>'
        )
        for idx, exp in enumerate(experiments, 1):
            report += _rpt_exp(idx, exp)

    # A note on what to protect
    note_text = global_store.get("note_to_protect", "")
    if note_text:
        note_paras = [p.strip() for p in note_text.split("\n") if p.strip()]
        report += f'<h2 class="rpt-sec">A note on what to protect</h2>'
        report += "".join(f'<p class="rpt-body">{H(p)}</p>' for p in note_paras)
        report += '<hr class="rpt-hr">'

    # Methodology block
    report += (
        '<div class="about"><h3>How this analysis works</h3>'
        f'<p>Insights are based on qualitative analysis of {n_iv} confidential employee '
        'conversations. Rather than survey scores, Spradley surfaces <strong>patterns</strong> '
        'from what people actually said.</p>'
        '<ol>'
        '<li>Conversations are coded to surface recurring topics from each interview.</li>'
        '<li>Codes are consolidated and grouped into thematic clusters across all interviews.</li>'
        '<li>Each cluster is analysed to produce a finding grounded in the source Q&amp;A.</li>'
        '<li>Paraphrased quotes provide context while protecting anonymity.</li>'
        '</ol></div>'
        '<div class="footer">Spradley &middot; app.spradley.io</div>'
    )

    lineage_tab = (
        '<p class="lineage-intro">Expand any cluster to trace a finding back to its source '
        'interview answers. Each level is independently collapsible.</p>'
        + _ltree(lineage, clusters, global_store, interview_store, db)
    )

    topics_tab = (
        '<div class="topics-section">'
        '<h3>Sentiment Dimensions</h3>'
        '<p class="topics-intro">Share of participants who directly expressed each dimension '
        'in their own words. Based on individual interview scoring -- only first-person '
        'expression counts.</p>'
        '<div class="radar-wrap">' + _radar_pane(dimension_store or {}, n_iv) + '</div>'
        '</div>'
        '<div class="topics-section">'
        '<h3>Topic Map</h3>'
        '<p class="topics-intro">Each bubble is a recurring sub-theme mentioned by more than '
        '10% of participants. Size reflects mention frequency. Position shows sentiment: '
        'left needs attention, right is working well.</p>'
        + _bubble_pane(clusters, lineage, global_store, interview_store, n_iv) +
        '</div>'
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
        "    <button class=\"tab-btn\" onclick=\"showTab('topics',this)\">Topics</button>\n"
        "    <button class=\"tab-btn\" onclick=\"showTab('lineage',this)\">Data Lineage</button>\n"
        "  </nav>\n"
        "</header>\n\n"
        "<div id=\"pane-report\" class=\"tab-pane active\">\n" + report + "\n</div>\n\n"
        "<div id=\"pane-topics\" class=\"tab-pane\">\n" + topics_tab + "\n</div>\n\n"
        "<div id=\"pane-lineage\" class=\"tab-pane\">\n" + lineage_tab + "\n</div>\n\n"
        "<script>\n" + js + "\n</script>\n\n"
        "</body>\n</html>"
    )
