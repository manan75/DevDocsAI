"""
helpers.py — Shared utility functions used across the project.
"""

import logging
import sys
from pathlib import Path
from typing import List

from langchain.schema import Document


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean, consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def format_chunks_for_display(docs: List[Document], scores: List[float]) -> str:
    """
    Format retrieved chunks into a human-readable string for the Gradio UI.

    Args:
        docs: Retrieved LangChain Documents.
        scores: Corresponding similarity scores.

    Returns:
        Formatted multi-line string.
    """
    parts = []
    for i, (doc, score) in enumerate(zip(docs, scores), 1):
        meta = doc.metadata
        file_path = meta.get("file_path", "unknown")
        symbol = meta.get("symbol_name", "")
        symbol_type = meta.get("symbol_type", "chunk")
        score_str = f"{score:.3f}" if score > 0 else "N/A (MMR)"

        header = f"━━━ [{i}] {file_path}"
        if symbol:
            header += f" › {symbol_type}:{symbol}"
        header += f"  (score: {score_str}) ━━━"

        parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(parts)


def format_metrics_for_display(retrieval_metrics, answer_scores) -> str:
    """
    Format all evaluation metrics into a readable dashboard string.

    Args:
        retrieval_metrics: RetrievalMetrics Pydantic model.
        answer_scores: AnswerQualityScores Pydantic model.

    Returns:
        Formatted metrics string.
    """
    lines = [
        "╔══════════════════════════════════════╗",
        "║       EVALUATION METRICS PANEL       ║",
        "╠══════════════════════════════════════╣",
        "║  RETRIEVAL METRICS                   ║",
        f"║  Recall@{retrieval_metrics.top_k:<2}       : {retrieval_metrics.recall_at_k:.4f}          ║",
        f"║  MRR              : {retrieval_metrics.mrr:.4f}          ║",
        f"║  nDCG@{retrieval_metrics.top_k:<2}         : {retrieval_metrics.ndcg:.4f}          ║",
        f"║  Relevant chunks  : {retrieval_metrics.num_relevant}/{retrieval_metrics.top_k}              ║",
        "╠══════════════════════════════════════╣",
        "║  ANSWER QUALITY (LLM Judge)          ║",
        f"║  Accuracy         : {answer_scores.accuracy}/5              ║",
        f"║  Completeness     : {answer_scores.completeness}/5              ║",
        f"║  Relevance        : {answer_scores.relevance}/5              ║",
        f"║  Groundedness     : {answer_scores.groundedness}/5              ║",
        f"║  Overall Score    : {answer_scores.overall:.2f}/5.00         ║",
        "╠══════════════════════════════════════╣",
        f"║  Reasoning: {answer_scores.reasoning[:38]:<38}",
        "╚══════════════════════════════════════╝",
    ]
    return "\n".join(lines)


def save_temp_file(file_bytes: bytes, filename: str) -> Path:
    """
    Save raw bytes to the uploads directory.

    Args:
        file_bytes: Raw file content.
        filename: Target filename.

    Returns:
        Path to the saved file.
    """
    from config import UPLOAD_DIR
    dest = UPLOAD_DIR / filename
    dest.write_bytes(file_bytes)
    return dest