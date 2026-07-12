from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import docx  # for reading .docx (Word) files
import spacy
from spacy.matcher import PhraseMatcher
import re
from fastembed import TextEmbedding
import numpy as np

# ---------------- SKILL LISTS ----------------
# (Expanded from ~30 to 100+ skills so more resumes get recognized properly)

TECHNICAL_SKILLS = [
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "html", "css", "sql", "mysql", "postgresql", "mongodb", "sqlite",
    "flask", "django", "fastapi", "react", "angular", "vue", "node",
    "node.js", "express", "spring", "spring boot", "next.js",
    "machine learning", "deep learning", "data science", "data analysis",
    "data structures", "algorithms", "dsa", "oop", "operating systems",
    "computer networks", "dbms", "nlp", "natural language processing",
    "computer vision", "pandas", "numpy", "scikit-learn", "sklearn",
    "tensorflow", "pytorch", "keras", "matplotlib", "seaborn",
    "opencv", "rest api", "api", "microservices", "graphql",
    "linux", "bash", "shell scripting", "php", "r", "swift", "kotlin",
    "golang", "go", "rust", "scala", "hadoop", "spark", "kafka",
    "power bi", "tableau", "big data"
]

SOFT_SKILLS = [
    "communication", "teamwork", "leadership",
    "problem solving", "time management",
    "adaptability", "critical thinking", "collaboration",
    "creativity", "decision making", "conflict resolution",
    "presentation skills", "public speaking", "negotiation",
    "attention to detail", "multitasking", "flexibility",
    "work ethic", "mentoring", "project management"
]

TOOLS = [
    "git", "github", "gitlab", "docker", "kubernetes", "aws",
    "azure", "gcp", "google cloud", "excel", "powerpoint", "figma",
    "postman", "jira", "confluence", "vs code", "vscode",
    "jenkins", "ci/cd", "terraform", "ansible", "slack",
    "notion", "trello", "firebase", "heroku", "vercel", "netlify",
    "colab", "jupyter", "anaconda"
]

# ---------------- SPACY SETUP ----------------
# Why spaCy PhraseMatcher instead of "if skill in text": plain substring
# matching has a bug -- "java" would wrongly match inside "javascript".
# PhraseMatcher matches on actual word/token boundaries, so it's accurate.

nlp = spacy.load("en_core_web_sm")

def build_matcher(skill_list):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_list]
    matcher.add("SKILLS", patterns)
    return matcher

tech_matcher = build_matcher(TECHNICAL_SKILLS)
soft_matcher = build_matcher(SOFT_SKILLS)
tools_matcher = build_matcher(TOOLS)

# ---------------- SEMANTIC EMBEDDING SETUP (fastembed) ----------------
# Using fastembed instead of sentence-transformers here: fastembed runs
# quantized ONNX models with no PyTorch dependency, so it uses far less
# memory (~100MB vs 500MB-1GB+ for the full torch + sentence-transformers
# stack). This matters for deploying on free-tier hosting (512MB RAM limit).
# Same underlying model family (MiniLM), so scoring behavior stays consistent.
embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def semantic_similarity(text1, text2):
    """Returns similarity score (0-100) between two texts based on meaning, not exact words."""
    embeddings = list(embed_model.embed([text1, text2]))
    a, b = embeddings[0], embeddings[1]
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    score = max(0, float(cos_sim)) * 100
    return round(score, 2)

def calibrate_score(raw_score, low=15, high=75):
    """
    Rescales SBERT's raw similarity score to use the full 0-100 range.

    WHY THIS IS NEEDED: Sentence-BERT cosine similarity scores naturally
    cluster in a narrow middle band (observed ~15-75 in our evaluation),
    even for very good or very bad matches. This stretches that observed
    range to a full 0-100 scale, making scores more interpretable.
    Values are clamped to stay within 0-100.
    """
    stretched = (raw_score - low) / (high - low) * 100
    return round(max(0, min(100, stretched)), 2)

def extract_skills(text, matcher, skill_list):
    """Run a PhraseMatcher over text and return matched skills (deduplicated)."""
    doc = nlp(text)
    matches = matcher(doc)
    found = set()
    for match_id, start, end in matches:
        span = doc[start:end].text.lower()
        found.add(span)
    return list(found)

# ---------------- SECTION DETECTION ----------------
# Simple regex-based header detection to split resume into rough sections.
# Not perfect, but good enough to tag which section a skill came from.

SECTION_HEADERS = {
    "education": r"(education|academic background|qualification)",
    "experience": r"(experience|work history|employment|internship)",
    "skills": r"(skills|technical skills|competencies)",
    "projects": r"(projects|academic projects)"
}

def split_sections(text):
    """Break resume text into sections based on common headers."""
    lines = text.split("\n")
    sections = {"education": "", "experience": "", "skills": "", "projects": "", "other": ""}
    current = "other"

    for line in lines:
        line_clean = line.strip().lower()
        matched_header = None
        for name, pattern in SECTION_HEADERS.items():
            if re.fullmatch(pattern, line_clean) or (len(line_clean) < 30 and re.search(pattern, line_clean)):
                matched_header = name
                break
        if matched_header:
            current = matched_header
            continue
        sections[current] += line + "\n"

    return sections

def normalize_text(text):
    """Fixes small mismatches like 'problem-solving' vs 'problem solving'
    by converting hyphens/underscores to spaces before matching."""
    return re.sub(r"[-_/]", " ", text)

# ---------------- LEARNING RESOURCES ----------------
# Maps common skills to a short, practical suggestion on how to learn them.
# Not exhaustive -- falls back to a generic suggestion for anything not listed.

SKILL_RESOURCES = {
    "python": "Apna College or CodeWithHarry's Python playlist (YouTube) is a solid beginner-friendly start.",
    "java": "Apna College's Java DSA playlist covers both language basics and interview-relevant DSA.",
    "flask": "Official Flask docs + build one small project (like a to-do API) to learn it hands-on.",
    "django": "Django official tutorial (polls app) is the standard starting point.",
    "sql": "SQLBolt or Mode Analytics SQL tutorial for interactive practice.",
    "mysql": "freeCodeCamp's MySQL course covers joins, normalization, and queries well.",
    "machine learning": "Andrew Ng's ML course (Coursera) or the IBM ML Professional Certificate for structured learning.",
    "data science": "Kaggle's free micro-courses are great for hands-on data science basics.",
    "git": "'Git and GitHub for Beginners' (freeCodeCamp) covers everything needed for resumes/projects.",
    "github": "Practice by pushing your own projects -- that's the best way to learn Git/GitHub workflows.",
    "docker": "Docker's official 'Get Started' guide + containerize one of your own projects.",
    "aws": "AWS Cloud Practitioner Essentials (free on AWS Skill Builder) is a good starting point.",
    "rest api": "Build a small Flask/Express REST API yourself -- theory alone won't stick as well.",
    "dbms": "Your coursework + a normalization/SQL joins revision should cover most interview questions.",
    "computer networks": "Gate Smashers (YouTube) has a well-structured Computer Networks playlist.",
    "data structures": "Apna College or Striver's DSA sheet for structured, interview-focused practice.",
    "algorithms": "Striver's A2Z DSA course covers this alongside data structures.",
    "teamwork": "Look for group project or hackathon opportunities to build and demonstrate this.",
    "problem solving": "Regular DSA practice (LeetCode/GFG) builds this naturally over time.",
}

def suggest_learning_plan(missing_skills):
    """For each missing skill, return a short suggestion on what/how to learn it."""
    plan = []
    for skill in missing_skills:
        suggestion = SKILL_RESOURCES.get(
            skill,
            f"Search for a beginner-friendly tutorial or course on '{skill}' (YouTube/Coursera are good starting points)."
        )
        plan.append({"skill": skill, "suggestion": suggestion})
    return plan

# ---------------- APP SETUP ----------------

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload size

ALLOWED_EXTENSIONS = (".pdf", ".docx")

@app.route('/')
def home():
    return "Backend is running!"

def analyze_resume(file, job_desc):
    """
    Core analysis logic: takes one uploaded file + job description text,
    returns a dict with skills, scores, and explanation.
    Reused by both the single-resume endpoint and the batch-ranking endpoint.
    Raises ValueError with a user-friendly message on bad input.
    """
    filename = file.filename.lower()

    if not filename.endswith(ALLOWED_EXTENSIONS):
        raise ValueError(f"Unsupported file type: {file.filename}. Only .pdf and .docx are supported.")

    text = ""

    # -------- Extract text: PDF or DOCX --------
    if filename.endswith(".pdf"):
        with pdfplumber.open(file.stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif filename.endswith(".docx"):
        document = docx.Document(file.stream)
        for para in document.paragraphs:
            if para.text:
                text += para.text + "\n"

    if not text.strip():
        raise ValueError(f"Could not extract any text from {file.filename}. The file may be empty, scanned, or corrupted.")

    text_normalized = normalize_text(text)

    # -------- Section Detection --------
    sections = split_sections(text)

    # -------- Skill Extraction --------
    tech_found = extract_skills(text_normalized, tech_matcher, TECHNICAL_SKILLS)
    soft_found = extract_skills(text_normalized, soft_matcher, SOFT_SKILLS)
    tools_found = extract_skills(text_normalized, tools_matcher, TOOLS)

    # -------- MATCHING LOGIC --------
    job_desc_clean = job_desc.replace(",", " ")
    job_desc_normalized = normalize_text(job_desc_clean)

    job_skills = []
    for skill in TECHNICAL_SKILLS + SOFT_SKILLS + TOOLS:
        if skill in job_desc_normalized:
            job_skills.append(skill)
    job_skills = list(set(job_skills))

    all_resume_skills = list(set(tech_found + soft_found + tools_found))

    matched = sum(1 for skill in job_skills if skill in all_resume_skills)
    keyword_score = (matched / len(job_skills)) * 100 if job_skills else 0

    relevant_text = (sections.get("skills", "") + " " + sections.get("projects", "")).strip()
    if not relevant_text:
        relevant_text = text

    semantic_score = semantic_similarity(relevant_text, job_desc_clean) if job_desc_clean.strip() else 0
    semantic_score = calibrate_score(semantic_score)

    match_score = round((0.3 * keyword_score) + (0.7 * semantic_score), 2)

    missing_skills = [s for s in job_skills if s not in all_resume_skills]
    matched_skills = [s for s in job_skills if s in all_resume_skills]

    # -------- Category-wise breakdown (for the coverage chart) --------
    def category_stats(skill_list):
        required = [s for s in job_skills if s in skill_list]
        matched_in_cat = [s for s in required if s in all_resume_skills]
        return {"required": len(required), "matched": len(matched_in_cat)}

    category_breakdown = {
        "technical": category_stats(TECHNICAL_SKILLS),
        "soft": category_stats(SOFT_SKILLS),
        "tools": category_stats(TOOLS)
    }

    if match_score >= 70:
        summary = "Strong match! Most required skills are present, and overall meaning aligns well with the job description."
    elif match_score >= 40:
        summary = "Moderate match. Some required skills are present, but a few important ones are missing."
    else:
        summary = "Low match. Many required skills are missing, and overall profile alignment with the job is weak."

    explanation = {
        "summary": summary,
        "keyword_score": round(keyword_score, 2),
        "semantic_score": semantic_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "how_calculated": "Final score = 30% keyword overlap + 70% semantic (meaning-based) similarity between your Skills/Projects sections and the job description."
    }

    return {
        "filename": file.filename,
        "content": text,
        "technical_skills": tech_found,
        "soft_skills": soft_found,
        "tools": tools_found,
        "match_score": match_score,
        "keyword_score": round(keyword_score, 2),
        "semantic_score": semantic_score,
        "missing_skills": missing_skills,
        "sections": sections,
        "explanation": explanation,
        "learning_plan": suggest_learning_plan(missing_skills),
        "category_breakdown": category_breakdown
    }

# ---------------- MAIN API (single resume) ----------------

@app.route('/upload', methods=['POST'])
def upload_resume():
    file = request.files.get('file')
    job_desc = request.form.get("jobDesc", "").lower()

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        result = analyze_resume(file, job_desc)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": "Something went wrong while analyzing the resume. Please try again."}), 500

# ---------------- BATCH API (multiple resumes, ranked) ----------------

@app.route('/upload-batch', methods=['POST'])
def upload_batch():
    """
    Accepts multiple resume files under the 'files' field, plus a jobDesc field.
    Returns all results sorted by match_score (best match first) -- useful for
    a recruiter-style "rank these candidates" use case.
    """
    files = request.files.getlist('files')
    job_desc = request.form.get("jobDesc", "").lower()

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    if len(files) > 20:
        return jsonify({"error": "Please upload 20 files or fewer at a time."}), 400

    results = []
    errors = []

    for file in files:
        try:
            result = analyze_resume(file, job_desc)
            results.append(result)
        except ValueError as ve:
            errors.append({"filename": file.filename, "error": str(ve)})
        except Exception as e:
            errors.append({"filename": file.filename, "error": "Failed to process this file."})

    # Rank best match first
    results.sort(key=lambda r: r["match_score"], reverse=True)

    # Add rank position
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    return jsonify({
        "ranked_results": results,
        "errors": errors,
        "total_processed": len(results)
    })

# ---------------- RUN SERVER ----------------

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)