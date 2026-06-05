"""
ingest_github.py
Fetches README content from Mohd Kaif's GitHub repos and ingests into ChromaDB.

Repos targeted:
- intelligent-tutoring-system
- bp_prediction_using_video
- Computer-vision-based-parking-parking-lot-management
- canada-immigration-lead-agent

Run:
    python ingestion/scripts/ingest_github.py
"""

import sys
import os
import logging
import requests
import base64

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from backend.rag.embeddings import embed_texts
from backend.rag.vector_store import add_chunks, get_collection_stats
from backend.rag.retriever import invalidate_bm25_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

GITHUB_USERNAME = "mohdkaif-bit"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # Optional — avoids rate limits

# Repos to ingest — add more as needed
TARGET_REPOS = [
    "intelligent-tutoring-system",
    "bp_prediction_using_video",
    "Computer-vision-based-parking-parking-lot-management",
    "canada-immigration-lead-agent",
]

# Files to fetch per repo (in priority order)
TARGET_FILES = ["README.md", "readme.md", "README.MD"]


def get_headers() -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def fetch_readme(repo_name: str) -> str | None:
    """Fetch README content for a repo via GitHub API."""
    for filename in TARGET_FILES:
        url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{filename}"
        resp = requests.get(url, headers=get_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            content_b64 = data.get("content", "")
            content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            logger.info(f"Fetched README for {repo_name} ({len(content)} chars)")
            return content
    logger.warning(f"No README found for {repo_name}")
    return None


def chunk_readme(repo_name: str, content: str) -> list:
    """
    Chunk a README into meaningful sections.
    Split on markdown headers (## or ###).
    Each section becomes one chunk.
    """
    chunks = []
    lines = content.split("\n")

    current_section_title = "overview"
    current_lines = []
    section_idx = 0

    def flush_section():
        nonlocal section_idx
        text = "\n".join(current_lines).strip()
        if len(text) < 50:  # Skip very short sections
            return
        chunk_id = f"github_{repo_name}_{section_idx:03d}"
        chunks.append({
            "id": chunk_id,
            "text": f"GitHub Repo: {repo_name}\nSection: {current_section_title}\n\n{text}",
            "metadata": {
                "section": "github",
                "sub_section": f"{repo_name} — {current_section_title}",
                "source": f"github/{repo_name}/README.md",
                "repo": repo_name
            }
        })
        section_idx += 1

    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            flush_section()
            current_section_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush_section()  # Last section

    logger.info(f"Chunked {repo_name} into {len(chunks)} sections.")
    return chunks


def run_github_ingestion():
    logger.info("Starting GitHub README ingestion...")

    all_chunks = []

    for repo in TARGET_REPOS:
        logger.info(f"Processing repo: {repo}")
        readme = fetch_readme(repo)
        if readme:
            chunks = chunk_readme(repo, readme)
            all_chunks.extend(chunks)

    if not all_chunks:
        logger.error("No chunks generated. Check repo names and GitHub access.")
        return

    logger.info(f"Total GitHub chunks: {len(all_chunks)}")

    # Embed
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # Store (upsert — safe to re-run)
    add_chunks(all_chunks, embeddings)

    # Invalidate BM25 cache so it rebuilds with new chunks
    invalidate_bm25_cache()

    stats = get_collection_stats()
    logger.info(f"GitHub ingestion complete. Collection stats: {stats}")


if __name__ == "__main__":
    run_github_ingestion()