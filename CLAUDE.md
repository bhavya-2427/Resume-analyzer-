# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All backend commands run from `backend/`:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # NOT in requirements.txt — separate step, required
python app.py                             # dev server on http://127.0.0.1:5000
python evaluate.py                        # the only verification harness (see below)
gunicorn app:app                          # production (Render)
```

Frontend is a single static file — open `frontend/index.html` directly in a browser. No build step, no npm, no bundler.

There is no test suite, linter, or CI. `evaluate.py` is the closest thing to a test: it imports the real scoring functions from `app.py` and scores 12 hand-labeled resume/JD pairs, printing MAE and correlation against the manual labels. Run it after touching anything in the scoring path — a jump in MAE means the change regressed accuracy.

## Architecture

Three files carry the whole project: `backend/app.py` (all backend logic), `backend/evaluate.py` (scoring validation), `frontend/index.html` (UI + Chart.js, vanilla JS inline).

### The scoring pipeline

`analyze_resume(file, job_desc)` in `app.py` is the single entry point for both endpoints (`/upload` for one resume, `/upload-batch` for up to 20 ranked by score). Any change to matching behavior belongs there, not in the route handlers. The pipeline:

1. Text extraction — pdfplumber for `.pdf`, python-docx for `.docx`. Raises `ValueError` with a user-facing message on unsupported type or empty extraction; routes turn `ValueError` into a 400 and anything else into a generic 500.
2. `normalize_text` collapses `-`, `_`, `/` to spaces so "problem-solving" matches "problem solving".
3. `split_sections` — regex header detection into education/experience/skills/projects/other.
4. Skill extraction, then two independent scores combined as **30% keyword + 70% semantic**.

**Keyword score** = fraction of JD-required skills present in the resume, where "required" means the skill string appears in the JD.

**Semantic score** = cosine similarity between the resume's *Skills + Projects sections only* (falling back to the full text if those sections came back empty) and the JD, then passed through `calibrate_score`.

### Two things that are easy to get wrong

**Both sides of the match go through `PhraseMatcher`; keep it that way.** Resume text and job description are both extracted with `extract_skills`, which is token-boundary aware — so "java" doesn't match inside "javascript". `build_matcher` registers one matcher label per skill so matches map back to the **canonical** skill string from the skill lists, not the raw matched text. `category_stats` and missing-skill detection both rely on that (they test membership against `TECHNICAL_SKILLS` / `SOFT_SKILLS` / `TOOLS`), so don't change `extract_skills` to return span text. Related: the route handlers lowercase `job_desc` before passing it in; `analyze_resume` assumes that has already happened.

**`normalize_text` must be applied to both the pattern and the text.** It rewrites `-`, `_`, `/` to spaces *and* collapses all whitespace runs to a single space, so `build_matcher` runs skill patterns through it too. Both halves are load-bearing: without pattern normalization, `"ci/cd"` and `"scikit-learn"` can never match anything (the text side has already had those characters rewritten); without whitespace collapsing, multi-word skills silently fail whenever the source text has a double space, tab, or line wrap between the words, because spaCy attaches only a lone space to the preceding token and makes any other whitespace run its own token. Flattening newlines is safe *only* because this output feeds skill matching alone — `split_sections` and the semantic embedding read the raw text. A useful invariant to re-check after touching any of this: every skill in `ALL_SKILLS` must match its own name when passed through `normalize_text` + `extract_skills`.

**`calibrate_score(raw, low=15, high=75)` is empirically tuned, not arbitrary.** Raw MiniLM cosine similarity clusters in a narrow band, so this linearly stretches the observed 15–75 range to 0–100 and clamps. The 15/75 bounds came out of `evaluate.py` runs and are what dropped MAE from 24.74 to 18.85 (README documents this). Changing them, the 30/70 weights, or the embedding model invalidates those published numbers — re-run `evaluate.py` and update the README table.

### Embedding model choice

Uses `fastembed` (`TextEmbedding` with quantized ONNX MiniLM), deliberately **not** `sentence-transformers`, to stay inside a 512MB free-tier RAM limit — torch + sentence-transformers needs 500MB–1GB, fastembed ~100MB. Don't swap it back for convenience. Note the README's tech-stack table still says "Sentence-Transformers"; that's stale prose, `app.py` is authoritative.

### Skill lists

`TECHNICAL_SKILLS`, `SOFT_SKILLS`, `TOOLS` at the top of `app.py` are the source of truth for matchers, `category_breakdown` (which drives the frontend coverage chart), and missing-skill detection. Adding a skill means appending to exactly one of the three lists — nothing else needs updating. `SKILL_RESOURCES` optionally maps a skill to a learning suggestion, with a generic fallback for anything unlisted.

## Deployment

Frontend deploys to GitHub Pages; the repo-root `index.html` exists only as a meta-refresh redirect into `frontend/index.html`. Backend runs on Render.

**The backend URL is hardcoded in two places** in `frontend/index.html` (the `/upload` and `/upload-batch` fetch calls, currently pointing at the Render deployment). For local development both must be pointed at `http://127.0.0.1:5000` — and reverted before committing. There is no environment-based switch.

No `Procfile` or `render.yaml` in the repo, so Render's build and start commands are configured in its dashboard. The build command must include the `spacy download en_core_web_sm` step, since `requirements.txt` doesn't cover it.
