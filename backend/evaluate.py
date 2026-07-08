"""
EVALUATION SCRIPT for Resume Analyzer
--------------------------------------
Why this file exists:
Any ML project needs proof that it actually works, not just a demo.
This script runs your model on labeled examples (resume snippet + job
description + a manual score YOU gave it) and checks how close the
model's predicted score is to your manual judgment.

This reuses the same functions from app.py, so it tests the REAL logic,
not a separate copy of it.

Run this with: python evaluate.py
"""

from app import (
    semantic_similarity,
    calibrate_score,
    normalize_text,
    tech_matcher, soft_matcher, tools_matcher,
    extract_skills,
    TECHNICAL_SKILLS, SOFT_SKILLS, TOOLS
)

DATASET = [
    (
        "Skilled in Python, Flask, SQL, and Machine Learning. Built a fraud detection project using Pandas and Scikit-learn.",
        "Looking for a Python developer with Flask and Machine Learning experience.",
        90
    ),
    (
        "Proficient in Java, Data Structures and Algorithms, and OOP concepts. Solved 200+ LeetCode problems.",
        "Hiring a backend engineer skilled in Java and strong DSA fundamentals.",
        85
    ),
    (
        "Experience with HTML, CSS, and basic JavaScript for building static websites.",
        "Looking for a Python developer with Flask and Machine Learning experience.",
        15
    ),
    (
        "Built REST APIs using Django and PostgreSQL. Familiar with Docker and AWS deployment.",
        "Seeking backend developer with Django, REST API, and cloud deployment experience.",
        88
    ),
    (
        "Strong in NLP, spaCy, NLTK, and Sentence-BERT for text classification projects.",
        "Looking for an NLP engineer familiar with transformer-based embedding models.",
        75
    ),
    (
        "Worked with React, Node.js, and MongoDB to build a full-stack e-commerce site.",
        "Hiring a backend engineer skilled in Java and strong DSA fundamentals.",
        10
    ),
    (
        "Good communication, teamwork, and leadership skills. Organized college hackathons.",
        "Looking for a candidate with strong communication and teamwork abilities.",
        70
    ),
    (
        "Knowledge of Computer Networks, DBMS, and Operating Systems from coursework.",
        "Looking for a candidate with strong CS fundamentals: DBMS, OS, and Networks.",
        80
    ),
    (
        "Experience with Tableau and Power BI for data visualization and business reporting.",
        "Seeking backend developer with Django, REST API, and cloud deployment experience.",
        8
    ),
    (
        "Built a resume analyzer using Flask, spaCy, and Sentence-BERT with semantic matching.",
        "Looking for an NLP engineer familiar with transformer-based embedding models.",
        82
    ),
    (
        "Basic Excel and PowerPoint skills, some exposure to data entry tasks.",
        "Hiring a backend engineer skilled in Java and strong DSA fundamentals.",
        5
    ),
    (
        "Solid Python and Machine Learning background, plus Git and GitHub for version control.",
        "Looking for a Python developer with Flask and Machine Learning experience.",
        65
    ),
]


def evaluate():
    print(f"Evaluating on {len(DATASET)} labeled examples...\n")
    print(f"{'#':<3} {'Manual':<8} {'Model':<8} {'Diff':<8}")
    print("-" * 35)

    errors = []
    manual_scores = []
    model_scores = []

    for i, (resume_text, job_desc, manual_score) in enumerate(DATASET, start=1):
        model_score = semantic_similarity(resume_text, job_desc)
        model_score = calibrate_score(model_score)

        diff = abs(manual_score - model_score)
        errors.append(diff)
        manual_scores.append(manual_score)
        model_scores.append(model_score)

        print(f"{i:<3} {manual_score:<8} {model_score:<8.2f} {diff:<8.2f}")

    mae = sum(errors) / len(errors)

    n = len(manual_scores)
    mean_manual = sum(manual_scores) / n
    mean_model = sum(model_scores) / n

    numerator = sum((manual_scores[i] - mean_manual) * (model_scores[i] - mean_model) for i in range(n))
    denom_manual = sum((m - mean_manual) ** 2 for m in manual_scores) ** 0.5
    denom_model = sum((m - mean_model) ** 2 for m in model_scores) ** 0.5
    correlation = numerator / (denom_manual * denom_model) if denom_manual and denom_model else 0

    print("-" * 35)
    print(f"\nMean Absolute Error (MAE): {mae:.2f}")
    print(f"Correlation with manual scores: {correlation:.2f}")
    print("\nWhat this means:")
    print("- MAE tells you, on average, how far off the model's score is from your judgment (lower = better).")
    print("- Correlation (closer to 1.0) tells you whether the model ranks good/bad matches in the right order.")


if __name__ == "__main__":
    evaluate()