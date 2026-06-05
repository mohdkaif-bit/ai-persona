"""
vector_store.py
ChromaDB setup, storage, and retrieval for the RAG pipeline.
- Persistent local storage
- Stores chunks with metadata
- Dense retrieval via ChromaDB's built-in similarity search
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

from pathlib import Path
import os

# backend directory
BASE_DIR = Path(__file__).resolve().parent.parent

# backend/chroma_db
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(BASE_DIR / "chroma_db")
)

COLLECTION_NAME = "kaif_knowledge"


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client.    Data survives restarts at CHROMA_PERSIST_DIR.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Gets existing collection or creates a new one.
    Uses cosine similarity — better than L2 for text embeddings.
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Collection '{COLLECTION_NAME}' ready. Count: {collection.count()}")
    return collection


def add_chunks(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> None:
    """
    Add chunks + embeddings to ChromaDB.

    Args:
        chunks: List of dicts with keys:
            - id (str): unique chunk ID
            - text (str): chunk content
            - metadata (dict): section, source, etc.
        embeddings: Corresponding embedding vectors
    """
    if not chunks:
        logger.warning("No chunks to add.")
        return

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Upsert — safe to re-run ingestion without duplicates
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    logger.info(f"Upserted {len(chunks)} chunks into '{COLLECTION_NAME}'.")


def dense_search(
    query_embedding: List[float],
    n_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Dense vector search using ChromaDB cosine similarity.

    Args:
        query_embedding: Embedded query vector
        n_results: Number of top results to return

    Returns:
        List of dicts with keys: id, text, metadata, score
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        logger.warning("Collection is empty. Run ingestion first.")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            # ChromaDB cosine distance → convert to similarity score
            "score": 1 - results["distances"][0][i]
        })

    return hits


def get_collection_stats() -> Dict[str, Any]:
    """
    Returns basic stats about the collection.
    Useful for health checks.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    return {
        "collection": COLLECTION_NAME,
        "total_chunks": collection.count(),
        "persist_dir": CHROMA_PERSIST_DIR
    }


def reset_collection() -> None:
    """
    Deletes and recreates the collection.
    Use only during re-ingestion / development.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted collection '{COLLECTION_NAME}'.")
    except Exception:
        pass
    get_or_create_collection(client)
    logger.info(f"Recreated collection '{COLLECTION_NAME}'.")