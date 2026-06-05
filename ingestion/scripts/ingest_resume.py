"""
ingest_resume.py
Extracts text from ML_Resume.pdf and ingests into ChromaDB.

The resume is already captured in persona_knowledge.json, but ingesting
the raw PDF text as well improves retrieval for verbatim questions
(e.g. exact dates, company names, GPA phrasing).

Run:
    python ingestion/scripts/ingest_resume.py --pdf ingestion/data/ML_Resume.pdf
"""

import sys
import os
import logging
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from backend.rag.embeddings import embed_texts
from backend.rag.vector_store import add_chunks, get_collection_stats
from backend.rag.retriever import invalidate_bm25_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PDF_PATH = os.path.join(os.path.dirname(__file__), "../data/ML_Resume.pdf")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from PDF using pdfplumber."""
    try:
        import pdfplumber
        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text.strip())
        content = "\n\n".join(full_text)
        logger.info(f"Extracted {len(content)} chars from {pdf_path}")
        return content
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        sys.exit(1)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        sys.exit(1)


def chunk_resume_text(text: str) -> list:
    """
    Chunk resume by natural sections.
    Resume sections are typically: Summary, Experience, Projects, Skills, Education.
    Split on uppercase section headers.
    """
    import re

    # Common resume section headers
    section_pattern = re.compile(
        r'^(Summary|Experience|Projects|Education|Skills|Certifications|Achievements)\s*$',
        re.MULTILINE | re.IGNORECASE
    )

    chunks = []
    parts = section_pattern.split(text)

    # parts alternates: [pre-header text, header, content, header, content, ...]
    if len(parts) <= 1:
        # No clear sections found — chunk by paragraph instead
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
        for idx, para in enumerate(paragraphs):
            chunks.append({
                "id": f"resume_para_{idx:03d}",
                "text": f"Resume Content:\n{para}",
                "metadata": {
                    "section": "resume",
                    "sub_section": f"paragraph_{idx}",
                    "source": "ML_Resume.pdf"
                }
            })
        return chunks

    # Header-based chunking
    i = 1  # Skip pre-header content if empty
    section_idx = 0
    while i < len(parts) - 1:
        header = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        if content and len(content) > 50:
            chunks.append({
                "id": f"resume_{header.lower()}_{section_idx:03d}",
                "text": f"Resume — {header}:\n{content}",
                "metadata": {
                    "section": "resume",
                    "sub_section": header,
                    "source": "ML_Resume.pdf"
                }
            })
            section_idx += 1
        i += 2

    # Also add the full resume as one chunk for holistic queries
    chunks.append({
        "id": "resume_full",
        "text": f"Full Resume of Mohd Kaif:\n{text[:3000]}",  # First 3000 chars
        "metadata": {
            "section": "resume",
            "sub_section": "full_resume",
            "source": "ML_Resume.pdf"
        }
    })

    logger.info(f"Chunked resume into {len(chunks)} sections.")
    return chunks


def run_resume_ingestion(pdf_path: str):
    logger.info(f"Starting resume ingestion from: {pdf_path}")

    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found at: {pdf_path}")
        logger.error("Place ML_Resume.pdf in ingestion/data/ and try again.")
        sys.exit(1)

    # Extract
    text = extract_text_from_pdf(pdf_path)

    # Chunk
    chunks = chunk_resume_text(text)
    logger.info(f"Generated {len(chunks)} resume chunks.")

    # Embed
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # Store (upsert)
    add_chunks(chunks, embeddings)

    # Invalidate BM25 cache
    invalidate_bm25_cache()

    stats = get_collection_stats()
    logger.info(f"Resume ingestion complete. Collection stats: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        default=DEFAULT_PDF_PATH,
        help="Path to resume PDF"
    )
    args = parser.parse_args()
    run_resume_ingestion(args.pdf)