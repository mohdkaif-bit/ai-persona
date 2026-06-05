"""
embeddings.py
Handles all embedding logic for the RAG pipeline.
Model: sentence-transformers/all-MiniLM-L6-v2
- Fast, lightweight, good semantic similarity
- 384-dim vectors — works well with ChromaDB
"""

from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

# Single model instance — loaded once, reused everywhere
_model: SentenceTransformer | None = None

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> SentenceTransformer:
    """
    Returns singleton embedding model.
    Lazy-loaded on first call.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text strings.
    Returns list of float vectors.

    Args:
        texts: List of strings to embed

    Returns:
        List of embedding vectors (List[float])
    """
    if not texts:
        return []

    model = get_embedding_model()
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string for retrieval.

    Args:
        query: User query string

    Returns:
        Single embedding vector (List[float])
    """
    model = get_embedding_model()
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.tolist()