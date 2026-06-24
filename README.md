# Spradley — Qualitative Interview Coding Pipeline

Automated open coding of employee interviews using an LLM, following grounded theory principles. Raw transcripts go in; thematic clusters with full source lineage come out.

![Pipeline overview](assets/Coding_Pipeline_Architecture_simple.svg)

---

## How it works

The pipeline processes interview transcripts in four stages:

**1. Ingest and anonymise:** Transcripts are loaded from a CSV, paired into question-answer turns, and scrubbed of PII (names, emails, phone numbers) before anything is stored.

**2. Open coding (L1 to L2 to L3):** The LLM reads each Q&A pair and generates 1-10 inductive codes grounded in the employee's exact words (L1). These are consolidated per interview into 20-30 broader codes (L2), then merged across all interviews into 40-80 global codes (L3). Every merge step records which source codes it absorbed, keeping the lineage intact.

**3. Theme clustering:** L3 codes are grouped into 7-12 named thematic clusters, each accompanied by a narrative written by the LLM and grounded in the original Q&A pairs.

**4. Persist:** All data structures are written to `pipeline_output/` as JSON, including the full lineage chain from cluster name down to individual interview question IDs.

| Coding level | Scope | Range |
|---|---|---|
| L1 | Open codes per Q&A pair | 1-10 |
| L2 | Consolidated codes per interview | 20-30 |
| L3 | Global codes across all interviews | 40-80 |
| Clusters | Thematic clusters | 7-12 |

---

## Architecture

![Technical architecture](assets/Coding_Pipeline_Architecture.svg)

Each box corresponds to a single notebook cell (C0-C13), keeping stages independently re-runnable. State is held in Python dicts and persisted to JSON after each run.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m ipykernel install --user --name=spradley-venv --display-name "Python (Spradley)"
```

Create `keys.env` in the project root (never committed):

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running

Open `Pipeline_Execution.ipynb`, select the **Python (Spradley)** kernel, and run cells **C0 → C13** in order. Each cell is self-contained and can be re-run independently after adjusting config in C0.

### Output files

| File | Contents | Committed |
|------|----------|-----------|
| `pipeline_output/db.json` | Per Q&A store: anonymised answer + L1 codes | No (gitignored) |
| `pipeline_output/interview_store.json` | L2 codes per interview (with merge lineage) | No (gitignored) |
| `pipeline_output/global_store.json` | L3 codes across all interviews (with merge lineage) | No (gitignored) |
| `pipeline_output/lineage.json` | Full chain: cluster to L3 to L2 to L1 to Q&A ID | No (gitignored) |
| `pipeline_output/clusters.json` | Final clusters with headline, summary, quotes, category | No (gitignored) |
| `pipeline_output/experiments.json` | Proposed experiments for needs\_work and mixed clusters | No (gitignored) |
| `pipeline_output/report.html` | Standalone report, served via GitHub Pages | Yes |
| `pipeline_output/eval_report.html` | Eval results report with per-eval detail and Q&A traceability | Yes |
| `pipeline_output/eval_results.json` | Latest eval run results (auto-restored on next E0 run) | No (gitignored) |
| `pipeline_output/eval_cache.json` | Cached LLM calls for E6a and E8 (paraphrase + negation) | No (gitignored) |
| `pipeline_output/eval_history/` | Full snapshots per eval run (JSON + rendered HTML) | No (gitignored) |

---

## Swapping the LLM

Change `LLM_PROVIDER` and `LLM_MODEL` in cell **C0**. The client implementation lives in `pipeline_core.py` (search `# C2`); uncomment the alternative provider branch there. No notebook cells need to change.

---

## Evaluating the pipeline

Open `Pipeline_Evals.ipynb` and run cells **E0 to Final** in order. It reads from `pipeline_output/` (no re-runs of the main pipeline needed) and writes `pipeline_output/eval_report.html`.

### Eval cells

| Cell | Name | What it tests |
|------|------|---------------|
| E0 | Setup | Loads pipeline outputs; auto-restores `eval_results.json` from the last run |
| E1 | L2 Code Stability | Reruns L2 coding N times per interview; measures cosine soft-Jaccard (gate >= 0.6) |
| E2b | L2 Label Quality | Claude judge checks each L2 label against its source Q&A turns |
| EL | Lineage Integrity | Checks that every cluster code traces back to at least one Q&A pair |
| E4 | Clustering Stability | Reruns L3 clustering N times; measures pairwise ARI (gate >= 0.7) |
| E5 | Faithfulness | Claude judge checks cluster narratives are grounded in source Q&A |
| E5b | Sentiment Overclaim | Claude judge checks narratives do not overstate positive or negative sentiment |
| E6 | Metamorphic Invariance | Paraphrase robustness (gate >= 0.5) and order invariance (gate >= 0.7) |
| E7 | Re-identification Probe | Adversarial Claude tries to re-identify employees from the report |
| E8 | Negation Sensitivity | Negates polarity-bearing answers; checks codes change accordingly (gate >= 60%) |
| Final | Report | Generates `eval_report.html` and saves a full history snapshot |

### Restoring a previous run

E0 auto-restores from `eval_results.json` on kernel start. To load a specific historical run, set `HISTORY_RUN` at the top of E0:

```python
HISTORY_RUN = "2026-06-23T14-30-00"  # paste a timestamp from the history-list cell
```

Run the history-list cell (just below E0) to see all available timestamps.
