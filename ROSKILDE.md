# Roskilde Festival 2026 -- Branch Context

## What this branch is

A one-time adaptation of the Spradley employee insights pipeline to generate a
qualitative research report on festival waste behavior at Roskilde Festival 2026.
This branch (`roskilde`) forks from `main` and will not be merged back.
When done, strip C0c and all Roskilde-specific changes to return `main` to
employee-only use.

## Research context

**Project:** Festival waste Roskilde 2026
**Operator:** Spradley (dev environment, spradley-dev Supabase project)
**Field period:** June 2026, answers received through 2026-06-30
**Respondents:** 40 completed sessions out of 156 total (short intercept interviews
at the festival, ~60-90 seconds on a phone)

**The problem being studied:**
Camping-area waste is about 75% of Roskilde's ~2,200 tonnes of annual festival
trash. The primary drivers are belief ("it'll be donated / it's recyclable") and
convenience (throwaway buying, hassle of packing wet gear) rather than indifference.
The research focuses on single-use tent, mat, sleeping bag, and chair abandonment.

**Interview structure (3 AI-adaptive turns):**
1. Think back to packing for Roskilde -- what did you bring, and where did most
   of it come from (new, borrowed, or stuff you already had)?
2. When the festival wraps up, what are you planning to take home, and what will
   you leave here? Be honest.
3. Do you think waste is a personal or festival responsibility?

## Data

**Local CSV:** `interview_input/roskilde-festival-2026.csv`
Format: `spradley_v2` columns -- `employee_id` (= anonymous session UUID),
`thread`, `turn_index`, `question_asked`, `answer_text`, `is_skipped`,
`participant_status`, `is_test`.

**Source:** Supabase dev project `jbifjmcsausdhyfydrnu`, tables
`research_sessions` / `research_answers` / `research_questions`,
project UUID `c6b21bb1-01b4-4878-8b0d-0e2cdddccda0`.

**Pull cell:** C0c in `Pipeline_Execution.ipynb` (Roskilde-specific, remove on main).

## Notebook run sequence (Roskilde)

```
C0 -> C0c -> C1 -> C2 -> C5 -> C6 -> C8 -> C8b -> C9 -> C9b -> C11 -> R0 -> R1
```
Skip C3 (load_interviews) -- C0c sets `interviews` directly from Supabase.
Skip C4 (anonymizer) -- respondents are already anonymous (session UUIDs, no PII).

## Planned pipeline adaptations (to be planned separately)

The following pipeline_core.py and report elements need reworking for a festival
audience. A full plan will be created in a future session.

| Area | Employee version | Festival version |
|---|---|---|
| Report framing | Management briefing, employee satisfaction | Research debrief, waste behavior drivers |
| Headline prompt | What employees experience at work | What visitors do and believe about waste |
| Finding categories | needs_work / mixed / working_well | concern / ambiguous / positive_signal |
| Emotional dimensions | satisfaction, stress, worry, trust, motivation, fairness | guilt, convenience, deflection, agency, social_norm, intent |
| Bubble chart axis | sentiment axis (left=negative, right=positive) | behavior axis (left=leave behind, right=take home) |
| Audience | HR / management | Festival sustainability team |
| Anonymity rule | ~X0% rounding, no exact counts | Same rule applies |

## What stays the same

- L2/L3/cluster coding pipeline (qualitative grounded theory approach is format-agnostic)
- HTML report structure and visual layout
- BCG-style observational tone: findings only, no blame, no prescriptive headlines
- Anonymity rounding (~X0%)
- Bubble chart + radar chart visuals (content changes, mechanics stay)
