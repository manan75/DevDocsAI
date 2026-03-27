"""
retriever.py — Configurable retrieval over the Chroma vector store.

Supports:
  - Similarity search (cosine distance ranking)
  - MMR (Maximum Marginal Relevance) for diversity-aware retrieval

Returns LangChain Documents with scores where applicable.
"""

import logging
from typing import List, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from config import DEFAULT_TOP_K, MMR_FETCH_K, MMR_LAMBDA_MULT
from ingestion.indexer import get_vectorstore

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    search_type: str = "similarity",
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[List[Document], List[float]]:
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        query: Natural language question from the user.
        search_type: "similarity" or "mmr".
        top_k: Number of chunks to return.

    Returns:
        Tuple of (documents, scores).
        Scores are cosine-similarity floats for similarity search;
        a list of zeros for MMR (Chroma does not expose MMR scores).

    Raises:
        RuntimeError: If the vector store is empty.
    """
    vectorstore: Chroma = get_vectorstore()

    if vectorstore._collection.count() == 0:
        raise RuntimeError("Vector store is empty. Please index a repository first.")

    if search_type == "mmr":
        docs = vectorstore.max_marginal_relevance_search(
            query=query,
            k=top_k,
            fetch_k=max(MMR_FETCH_K, top_k * 4),
            lambda_mult=MMR_LAMBDA_MULT,
        )
        scores = [0.0] * len(docs)
    else:
        results = vectorstore.similarity_search_with_score(query=query, k=top_k)
        docs = [d for d, _ in results]
        # Chroma returns L2 distance; convert to similarity (0–1) for clarity.
        scores = [max(0.0, 1.0 - s) for _, s in results]

    logger.info(f"[{search_type.upper()}] Retrieved {len(docs)} chunks for: '{query[:60]}'")
    return docs, scores