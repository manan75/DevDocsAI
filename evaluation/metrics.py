"""
metrics.py — Retrieval quality metrics.

Implements:
  - Recall@K  : fraction of relevant docs retrieved in top-K
  - MRR       : Mean Reciprocal Rank of the first relevant doc
  - nDCG      : Normalized Discounted Cumulative Gain

Relevance is determined by keyword matching between the query and chunk content.
This is a proxy measure used when ground-truth labels are unavailable.
"""

import math
import re
import logging
from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Pydantic output model ────────────────────────────────────────────────────

class RetrievalMetrics(BaseModel):
    """Structured container for retrieval evaluation scores."""
    recall_at_k: float = Field(..., ge=0.0, le=1.0, description="Recall@K")
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    ndcg: float = Field(..., ge=0.0, le=1.0, description="nDCG@K")
    top_k: int = Field(..., description="K used for evaluation")
    num_relevant: int = Field(..., description="Number of docs judged relevant")


# ─── Relevance oracle ─────────────────────────────────────────────────────────

def _extract_keywords(text: str) -> set:
    """Extract lowercase alphabetic tokens (length ≥ 3) from text."""
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))


def _is_relevant(query: str, doc: Document, threshold: int = 2) -> bool:
    """
    Determine if a document is relevant to the query via keyword overlap.

    Args:
        query: User question.
        doc: Retrieved document.
        threshold: Minimum number of shared keywords to count as relevant.

    Returns:
        True if overlap ≥ threshold.
    """
    q_keywords = _extract_keywords(query)
    d_keywords = _extract_keywords(doc.page_content)
    overlap = len(q_keywords & d_keywords)
    return overlap >= threshold


# ─── Metric functions ─────────────────────────────────────────────────────────

def _compute_relevance_flags(query: str, docs: List[Document]) -> List[int]:
    """Return binary relevance list (1 = relevant, 0 = not)."""
    return [1 if _is_relevant(query, doc) else 0 for doc in docs]


def recall_at_k(relevance: List[int]) -> float:
    """
    Recall@K: fraction of retrieved docs that are relevant.

    Since we have no total relevant pool, we treat the number of
    relevant items in the retrieved set as the denominator baseline.
    """
    num_relevant = sum(relevance)
    if num_relevant == 0:
        return 0.0
    return num_relevant / len(relevance)


def mean_reciprocal_rank(relevance: List[int]) -> float:
    """
    MRR: 1/rank of the first relevant document.

    Returns 0.0 if no relevant document is found.
    """
    for rank, rel in enumerate(relevance, 1):
        if rel == 1:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance: List[int]) -> float:
    """
    nDCG@K using binary relevance.

    Args:
        relevance: Binary relevance list ordered by retrieval rank.

    Returns:
        nDCG score in [0, 1].
    """
    def dcg(rels: List[int]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

    actual_dcg = dcg(relevance)
    ideal_dcg = dcg(sorted(relevance, reverse=True))

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_retrieval_metrics(query: str, docs: List[Document]) -> RetrievalMetrics:
    """
    Compute all retrieval metrics for a query–result pair.

    Args:
        query: User's natural language question.
        docs: Retrieved documents in retrieval rank order.

    Returns:
        RetrievalMetrics Pydantic model.
    """
    relevance = _compute_relevance_flags(query, docs)
    return RetrievalMetrics(
        recall_at_k=round(recall_at_k(relevance), 4),
        mrr=round(mean_reciprocal_rank(relevance), 4),
        ndcg=round(ndcg_at_k(relevance), 4),
        top_k=len(docs),
        num_relevant=sum(relevance),
    )