# Claude Code instructions for this repo

## Git commits
- Never add `Co-Authored-By` attribution to commit messages.

## Style
- Never use em dashes (—) in any generated text, prompts, or HTML output.

## Security
- `pilot_transcripts.csv` must never be committed. It contains raw interview data.
- `keys.env` must never be committed. It contains `ANTHROPIC_API_KEY`.

## Project
- Company name is **Spradley** (not Spreadley).
- Output directory is `pipeline_output/`. JSON data files are gitignored; `{dataset-id}_report.html` and `{dataset-id}_eval_report.html` are committed (e.g. `the-office-2_report.html`).
- Logo file lives at `assets/spradley_logo.png`.
- See `memory/project_spradley_pipeline.md` for cell map and data structure reference.
