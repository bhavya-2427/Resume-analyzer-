from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber

# ---------------- SKILL LISTS ----------------

TECHNICAL_SKILLS = [
    "python", "java", "c", "c++", "javascript",
    "html", "css", "sql", "mysql", "mongodb",
    "flask", "django", "react", "node",
    "machine learning", "data science"
]

SOFT_SKILLS = [
    "communication", "teamwork", "leadership",
    "problem solving", "time management",
    "adaptability", "critical thinking"
]

TOOLS = [
    "git", "github", "docker", "aws",
    "excel", "powerpoint", "figma"
]

# ---------------- APP SETUP ----------------

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Backend is running!"

# ---------------- MAIN API ----------------

@app.route('/upload', methods=['POST'])
def upload_resume():
    file = request.files.get('file')
    job_desc = request.form.get("jobDesc", "").lower()

    print("JOB DESCRIPTION:", job_desc)

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    text = ""

    try:
        # -------- Extract text from PDF --------
        with pdfplumber.open(file.stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        text_lower = text.lower()

        # -------- Skill Extraction --------
        tech_found = []
        soft_found = []
        tools_found = []

        # Technical skills
        for skill in TECHNICAL_SKILLS:
            if skill in text_lower:
                tech_found.append(skill)

        # Soft skills
        for skill in SOFT_SKILLS:
            if skill in text_lower:
                soft_found.append(skill)

        # Tools
        for tool in TOOLS:
            if tool in text_lower:
                tools_found.append(tool)

        # Remove duplicates
        tech_found = list(set(tech_found))
        soft_found = list(set(soft_found))
        tools_found = list(set(tools_found))

        print("RESUME SKILLS:", tech_found + soft_found + tools_found)

        # -------- MATCHING LOGIC --------

        # Clean job description (remove commas)
        job_desc = job_desc.replace(",", " ")

        # Extract job skills
        job_skills = []
        for skill in TECHNICAL_SKILLS + SOFT_SKILLS + TOOLS:
            if skill in job_desc:
                job_skills.append(skill)

        # Remove duplicates
        job_skills = list(set(job_skills))

        print("JOB SKILLS:", job_skills)

        # Combine resume skills
        all_resume_skills = list(set(tech_found + soft_found + tools_found))

        # Calculate match score
        matched = 0
        for skill in job_skills:
            if skill in all_resume_skills:
                matched += 1

        if len(job_skills) > 0:
            match_score = (matched / len(job_skills)) * 100
        else:
            match_score = 0

        print("MATCH SCORE:", match_score)

        # Find missing skills
        missing_skills = []
        for skill in job_skills:
            if skill not in all_resume_skills:
                missing_skills.append(skill)

        # -------- RETURN RESPONSE --------

        return jsonify({
            "content": text,
            "technical_skills": tech_found,
            "soft_skills": soft_found,
            "tools": tools_found,
            "match_score": match_score,
            "missing_skills": missing_skills
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)})

# ---------------- RUN SERVER ----------------

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)