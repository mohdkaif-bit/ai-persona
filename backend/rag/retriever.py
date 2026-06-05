"""
retriever.py
Hybrid retrieval pipeline: Dense (ChromaDB) + BM25 + Cross-encoder Reranking.

Pipeline:
    query
      → dense search (ChromaDB cosine similarity)
      → BM25 search (keyword matching)
      → RRF fusion (merge both result sets)
      → cross-encoder reranking (second-stage relevance scoring)
      → top-k final results

This mirrors exactly what was built at Acro Technologies for the VMS system.
"""

import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from backend.rag.embeddings import embed_query
from backend.rag.vector_store import dense_search, get_chroma_client, get_or_create_collection

logger = logging.getLogger(__name__)

# Cross-encoder model for reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None

# In-memory BM25 index (rebuilt per query from collection)
# For this scale (~30 chunks), this is fast enough
_bm25_index: BM25Okapi | None = None
_bm25_chunks: List[Dict[str, Any]] | None = None


def get_reranker() -> CrossEncoder:
    """Lazy-load cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker loaded.")
    return _reranker


def build_bm25_index() -> tuple[BM25Okapi, List[Dict[str, Any]]]:
    """
    Build BM25 index from all chunks in ChromaDB.
    Returns (bm25_index, all_chunks).
    Cached in memory — rebuilt only on first call.
    """
    global _bm25_index, _bm25_chunks

    if _bm25_index is not None:
        return _bm25_index, _bm25_chunks

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        logger.warning("Collection empty. Run ingestion first.")
        return None, []

    # Fetch all documents from ChromaDB
    all_data = collection.get(include=["documents", "metadatas"])
    all_chunks = [
        {
            "id": all_data["ids"][i],
            "text": all_data["documents"][i],
            "metadata": all_data["metadatas"][i]
        }
        for i in range(len(all_data["ids"]))
    ]

    # Tokenise for BM25
    tokenised = [doc["text"].lower().split() for doc in all_chunks]
    _bm25_index = BM25Okapi(tokenised)
    _bm25_chunks = all_chunks

    logger.info(f"BM25 index built with {len(all_chunks)} documents.")
    return _bm25_index, _bm25_chunks


def bm25_search(query: str, n_results: int = 10) -> List[Dict[str, Any]]:
    """
    BM25 keyword search over all chunks.

    Returns:
        List of dicts with id, text, metadata, score (normalised BM25 score)
    """
    bm25, chunks = build_bm25_index()

    if bm25 is None:
        return []

    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)

    # Pair chunks with scores, sort descending
    scored = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )[:n_results]

    results = []
    for chunk, score in scored:
        if score > 0:  # Only include chunks with non-zero BM25 score
            results.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(score)
            })

    return results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Merge dense and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank)) across all result lists.
    k=60 is standard default from the original RRF paper.

    Returns:
        Merged and re-ranked list of unique chunks.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for rank, result in enumerate(dense_results):
        cid = result["id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = result

    for rank, result in enumerate(bm25_results):
        cid = result["id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = result

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused = []
    for cid in sorted_ids:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = rrf_scores[cid]
        fused.append(chunk)

    return fused


def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Cross-encoder reranking of candidate chunks.
    Re-scores each (query, document) pair together — not just vector similarity.

    This is the key fix from the Acro Technologies VMS project:
    vector similarity alone was insufficient; joint query-document scoring improved
    retrieval relevance by 20%+.

    Args:
        query: Original user query
        candidates: List of candidate chunks from RRF fusion
        top_k: Final number of results to return

    Returns:
        Top-k reranked chunks with rerank_score added
    """
    if not candidates:
        return []

    reranker = get_reranker()

    # Build (query, document) pairs for cross-encoder
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for i, chunk in enumerate(candidates):
        chunk["rerank_score"] = float(scores[i])

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def retrieve(
    query: str,
    top_k: int = 5,
    dense_k: int = 10,
    bm25_k: int = 10,
    use_reranker: bool = True
) -> List[Dict[str, Any]]:
    """
    Full hybrid retrieval pipeline.

    Steps:
        1. Dense search (ChromaDB cosine similarity)
        2. BM25 keyword search
        3. RRF fusion
        4. Cross-encoder reranking (if enabled)
        5. Return top_k results

    Args:
        query: User query string
        top_k: Final number of results
        dense_k: How many dense results to fetch
        bm25_k: How many BM25 results to fetch
        use_reranker: Whether to apply cross-encoder reranking

    Returns:
        List of top_k most relevant chunks
    """
    logger.info(f"Retrieving for query: '{query[:80]}...' " if len(query) > 80 else f"Retrieving for query: '{query}'")

    # Step 1: Dense retrieval
    query_embedding = embed_query(query)
    dense_results = dense_search(query_embedding, n_results=dense_k)
    logger.info(f"Dense search returned {len(dense_results)} results.")

    # Step 2: BM25 retrieval
    bm25_results = bm25_search(query, n_results=bm25_k)
    logger.info(f"BM25 search returned {len(bm25_results)} results.")

    # Step 3: RRF fusion
    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    logger.info(f"RRF fusion produced {len(fused)} unique candidates.")

    # Step 4: Reranking
    if use_reranker and fused:
        final = rerank(query, fused, top_k=top_k)
        logger.info(f"Reranking complete. Returning top {len(final)} results.")
    else:
        final = fused[:top_k]

    return final


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.

    Args:
        chunks: List of retrieved chunk dicts

    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant context found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk["metadata"].get("section", "unknown")
        sub = chunk["metadata"].get("sub_section", "")
        label = f"{section} — {sub}" if sub else section
        parts.append(f"[{i}] ({label})\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def invalidate_bm25_cache():
    """
    Call this after re-ingestion to force BM25 index rebuild.
    """
    global _bm25_index, _bm25_chunks
    _bm25_index = None
    _bm25_chunks = None
    logger.info("BM25 cache invalidated.")