"""
ingest_knowledge.py
Chunks persona_knowledge.json and ingests into ChromaDB.

Chunking strategy:
- Each top-level section → independent retrieval units
- hard_problem_solved → own chunk (high-value interview content)
- eval_metrics → own chunk (queried independently)
- FAQ → one chunk per Q&A pair
- Skills → one chunk per sub-section

Run:
    python ingestion/scripts/ingest_knowledge.py
"""

import json
import sys
import os
import logging

# Allow imports from backend/
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from backend.rag.embeddings import embed_texts
from backend.rag.vector_store import add_chunks, reset_collection, get_collection_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "../data/persona_knowledge.json")


# ─────────────────────────────────────────
# CHUNKING HELPERS
# ─────────────────────────────────────────

def make_chunk(chunk_id: str, text: str, section: str, sub_section: str = "") -> dict:
    """Helper to create a standardised chunk dict."""
    return {
        "id": chunk_id,
        "text": text.strip(),
        "metadata": {
            "section": section,
            "sub_section": sub_section,
            "source": "persona_knowledge.json"
        }
    }


def chunk_personal(data: dict) -> list:
    p = data["personal"]
    text = f"""
Name: {p['name']}
Email: {p['email']}
Phone: {p['phone']}
Location: {p['location']}
LinkedIn: {p['linkedin']}
GitHub: {p['github']}
Calendar Booking: {p['calendar_booking_link']}
Availability: {p['availability_note']}
""".strip()
    return [make_chunk("personal_001", text, "personal")]


def chunk_education(data: dict) -> list:
    e = data["education"]
    coursework = ", ".join(e["relevant_coursework"])
    text = f"""
Degree: {e['degree']}
Institution: {e['institution']}
Duration: {e['duration']}
GPA: {e['gpa']}
Graduation Year: {e['graduation_year']}
Relevant Coursework: {coursework}
""".strip()
    return [make_chunk("education_001", text, "education")]


def chunk_why_scaler(data: dict) -> list:
    w = data["why_scaler"]
    text = f"""
Why Scaler:
{w['summary']}

Deeper Context:
{w['deeper_context']}
""".strip()
    return [make_chunk("why_scaler_001", text, "why_scaler")]


def chunk_experience(data: dict) -> list:
    chunks = []

    for exp_idx, exp in enumerate(data["experience"]):
        company_slug = exp["company"].replace(" ", "_").lower()[:20]

        # Company overview chunk
        overview_text = f"""
Role: {exp['role']}
Company: {exp['company']}
Duration: {exp['duration']}
Location: {exp['location']}
Type: {exp['type']}
""".strip()
        chunks.append(make_chunk(
            f"experience_{exp_idx}_overview",
            overview_text,
            "experience",
            f"{exp['company']} — overview"
        ))

        # Each project gets its own chunk
        for proj_idx, proj in enumerate(exp["projects"]):
            proj_text = f"""
Company: {exp['company']}
Project: {proj['name']}
Description: {proj['description']}
Impact: {proj['impact']}
Tech Stack: {', '.join(proj['tech_stack']) if isinstance(proj['tech_stack'], list) else ', '.join(proj['tech_stack'])}
Key Contributions:
""".strip()
            proj_text += "\n" + "\n".join(f"- {c}" for c in proj["key_contributions"])

            chunks.append(make_chunk(
                f"experience_{exp_idx}_project_{proj_idx}",
                proj_text,
                "experience",
                f"{exp['company']} — {proj['name']}"
            ))

            # hard_problem_solved gets its own dedicated chunk
            if "hard_problem_solved" in proj:
                hps = proj["hard_problem_solved"]
                hps_text = f"""
Hard Problem Solved at {exp['company']} — {proj['name']}:

Problem: {hps['problem']}

Root Cause: {hps['root_cause']}

Solution: {hps['solution']}

Outcome: {hps['outcome']}

Key Insight: {hps['insight']}
""".strip()
                chunks.append(make_chunk(
                    f"experience_{exp_idx}_project_{proj_idx}_hard_problem",
                    hps_text,
                    "experience",
                    f"{exp['company']} — hard problem solved"
                ))

    return chunks


def chunk_projects(data: dict) -> list:
    chunks = []

    for proj_idx, proj in enumerate(data["projects"]):
        proj_slug = proj["name"][:30].replace(" ", "_").lower()

        # Project overview chunk
        tech = proj["tech_stack"]
        if isinstance(tech, dict):
            tech_str = "; ".join([f"{k}: {', '.join(v)}" for k, v in tech.items()])
        else:
            tech_str = ", ".join(tech)

        overview_text = f"""
Project: {proj['name']}
Status: {proj['status']}
GitHub: {proj.get('github_url', 'N/A')}
Description: {proj['description']}
Tech Stack: {tech_str}
Key Features:
""".strip()

        if "key_features" in proj:
            overview_text += "\n" + "\n".join(f"- {f}" for f in proj["key_features"])
        elif "key_contributions" in proj:
            overview_text += "\n" + "\n".join(f"- {c}" for c in proj["key_contributions"])

        if "architecture" in proj:
            overview_text += f"\nArchitecture: {proj['architecture']}"

        if "design_tradeoffs" in proj:
            overview_text += "\nDesign Tradeoffs:\n" + "\n".join(f"- {t}" for t in proj["design_tradeoffs"])

        if "what_id_do_differently" in proj:
            overview_text += "\nWhat I'd Do Differently:\n" + "\n".join(f"- {t}" for t in proj["what_id_do_differently"])

        chunks.append(make_chunk(
            f"project_{proj_idx}_overview",
            overview_text,
            "projects",
            proj["name"]
        ))

        # eval_metrics gets own chunk — queried independently
        if "eval_metrics" in proj:
            metrics = proj["eval_metrics"]
            metrics_text = f"""
Eval Metrics — {proj['name']}:
""".strip()
            metrics_text += "\n" + "\n".join(f"{k}: {v}" for k, v in metrics.items())

            chunks.append(make_chunk(
                f"project_{proj_idx}_eval_metrics",
                metrics_text,
                "projects",
                f"{proj['name']} — eval metrics"
            ))

        # results chunk if present (non-tutoring projects)
        if "results" in proj:
            results = proj["results"]
            results_text = f"""
Results — {proj['name']}:
""".strip()
            results_text += "\n" + "\n".join(f"{k}: {v}" for k, v in results.items())

            chunks.append(make_chunk(
                f"project_{proj_idx}_results",
                results_text,
                "projects",
                f"{proj['name']} — results"
            ))

    return chunks


def chunk_skills(data: dict) -> list:
    chunks = []
    skills = data["skills"]

    for section_key, section_val in skills.items():
        text = f"Skills — {section_key}:\n"
        for k, v in section_val.items():
            if isinstance(v, list):
                text += f"{k}: {', '.join(v)}\n"
            else:
                text += f"{k}: {v}\n"

        chunks.append(make_chunk(
            f"skills_{section_key}",
            text.strip(),
            "skills",
            section_key
        ))

    return chunks


def chunk_persona_guidelines(data: dict) -> list:
    pg = data["persona_guidelines"]
    strengths = "\n".join(f"- {s}" for s in pg["strengths_to_highlight"])
    honesty = "\n".join(f"- {r}" for r in pg["honesty_rules"])

    text = f"""
Persona Guidelines:
Tone: {pg['tone']}

Strengths to Highlight:
{strengths}

Booking Instruction: {pg['booking_instruction']}

Graduation Status: {pg['graduation_status']}

Honesty Rules:
{honesty}
""".strip()

    return [make_chunk("persona_guidelines_001", text, "persona_guidelines")]


def chunk_faq(data: dict) -> list:
    chunks = []
    for idx, (question, answer) in enumerate(data["faq"].items()):
        text = f"Q: {question}\nA: {answer}"
        chunks.append(make_chunk(
            f"faq_{idx:03d}",
            text,
            "faq",
            question[:50]
        ))
    return chunks


# ─────────────────────────────────────────
# MAIN INGESTION
# ─────────────────────────────────────────

def build_all_chunks(data: dict) -> list:
    """Run all chunkers and return combined list."""
    all_chunks = []
    all_chunks.extend(chunk_personal(data))
    all_chunks.extend(chunk_education(data))
    all_chunks.extend(chunk_why_scaler(data))
    all_chunks.extend(chunk_experience(data))
    all_chunks.extend(chunk_projects(data))
    all_chunks.extend(chunk_skills(data))
    all_chunks.extend(chunk_persona_guidelines(data))
    all_chunks.extend(chunk_faq(data))
    return all_chunks


def run_ingestion(reset: bool = False):
    logger.info("Starting knowledge base ingestion...")

    # Load JSON
    with open(KNOWLEDGE_BASE_PATH, "r") as f:
        data = json.load(f)
    logger.info("Loaded persona_knowledge.json")

    # Optionally reset collection
    if reset:
        logger.info("Resetting ChromaDB collection...")
        reset_collection()

    # Build chunks
    chunks = build_all_chunks(data)
    logger.info(f"Built {len(chunks)} chunks total.")

    # Log chunk breakdown
    from collections import Counter
    section_counts = Counter(c["metadata"]["section"] for c in chunks)
    for section, count in section_counts.items():
        logger.info(f"  {section}: {count} chunks")

    # Embed
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    logger.info(f"Generated {len(embeddings)} embeddings.")

    # Store
    add_chunks(chunks, embeddings)

    # Stats
    stats = get_collection_stats()
    logger.info(f"Ingestion complete. Collection stats: {stats}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset collection before ingesting")
    args = parser.parse_args()
    run_ingestion(reset=args.reset)