# 📄 Smart Resume Analyzer

An AI-powered resume-to-job matching tool that goes beyond simple keyword search. It uses **NLP (spaCy)** for accurate skill extraction and **Sentence-BERT embeddings** for true semantic matching — understanding that "ML" and "Machine Learning" mean the same thing, not just comparing exact text.

**[Live Demo](#)** ← *(add your deployed link here after Day 7 deployment)*

---

## Why this project

Most "resume matcher" projects are just `if skill in text` — that's substring search, not AI. This project was built to actually understand meaning:

- ❌ v1 (keyword-only): "problem-solving" ≠ "problem solving" — missed valid matches
- ✅ v2 (this version): Sentence-BERT embeddings understand semantic similarity, not just exact text

## Features

- **Resume parsing** — supports both PDF and DOCX
- **Accurate skill extraction** — spaCy `PhraseMatcher` (word-boundary aware, so "java" never wrongly matches inside "javascript")
- **Dual scoring system**:
  - *Keyword Score* — direct skill overlap between resume and job description
  - *Semantic Score* — Sentence-BERT (`all-MiniLM-L6-v2`) cosine similarity between resume and JD meaning, **calibrated** to correct for SBERT's natural score compression
  - Final score = 30% keyword + 70% semantic
- **Explainability** — every score comes with a plain-language reason, not just a number
- **Multi-resume ranking** — upload several resumes for one job, get them ranked best-to-worst (recruiter-style shortlisting)
- **Learning recommendations** — for every missing skill, get a suggested resource on how to learn it
- **Visual dashboard** — score charts and skills-coverage-by-category chart (Chart.js)

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| NLP / Skill Extraction | spaCy (`PhraseMatcher`, `en_core_web_sm`) |
| Semantic Matching | fastembed — quantized ONNX `all-MiniLM-L6-v2` |
| File Parsing | pdfplumber, python-docx |
| Frontend | HTML, CSS, JavaScript, Chart.js |

Embeddings run through **fastembed** rather than the `sentence-transformers` library: it serves the same `all-MiniLM-L6-v2` model as quantized ONNX with no PyTorch dependency, using roughly 100MB instead of 500MB–1GB. That's what makes the backend fit in a 512MB free-tier instance.

## How the Matching Works

1. Resume text is extracted (PDF/DOCX) and split into rough sections (Skills, Projects, Experience, Education) using header detection.
2. Skills are extracted using spaCy's `PhraseMatcher` against a 100+ skill list (technical, soft skills, tools) — this is boundary-aware, so it doesn't have the classic "java matches inside javascript" bug.
3. The Skills + Projects sections are embedded using Sentence-BERT and compared to the job description via cosine similarity — this is the **semantic score**.
4. Because raw SBERT cosine similarity naturally clusters in a narrow band (~15–75) rather than spanning 0–100, a **calibration step** rescales it to use the full range — this was found and fixed during model evaluation (see below).
5. Final score = weighted combination of keyword overlap (30%) and semantic similarity (70%).

## Model Evaluation

The semantic matching was validated against 12 manually-labeled resume-JD pairs (`evaluate.py`):

| Metric | Before Calibration | After Calibration |
|---|---|---|
| Mean Absolute Error (MAE) | 24.74 | 18.95 |
| Correlation with manual judgment | 0.80 | 0.80 |

This showed the model ranks candidates in roughly the correct order (0.80 correlation), and that a simple calibration step meaningfully improved score accuracy.

## Project Structure

```
resume-analyzer/
├── backend/
│   ├── app.py              # Flask API — parsing, matching, scoring
│   ├── evaluate.py         # Model evaluation script
│   └── requirements.txt
├── frontend/
│   └── index.html          # UI (vanilla JS + Chart.js)
└── README.md
```

## Running Locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
```
Server runs at `http://127.0.0.1:5000`

**Frontend:**
Just open `frontend/index.html` in a browser.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Analyze a single resume against a job description |
| `/upload-batch` | POST | Analyze and rank multiple resumes against one job description |

## Possible Future Improvements

- Fine-tune the embedding model on a larger, domain-specific labeled dataset
- Extract candidate name/contact info automatically using NER
- Support for LinkedIn profile URLs as input

---

Built by Bhavya  — B.Tech CSE (AI), GNIOT
