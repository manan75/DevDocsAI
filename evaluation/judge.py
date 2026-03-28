"""
judge.py — LLM-as-a-judge answer quality evaluation.

Scores the generated answer on four dimensions (each 1–5):
  - Accuracy     : Is the answer factually correct given the context?
  - Completeness : Does it fully address the question?
  - Relevance    : Is the answer focused and on-topic?
  - Groundedness : Is every claim supported by the retrieved context?

Uses a single structured LLM call returning JSON to minimise cost.
"""

import json
import logging
from typing import List

import litellm
from pydantic import BaseModel, Field

from config import JUDGE_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)


# ─── Pydantic output model ────────────────────────────────────────────────────

class AnswerQualityScores(BaseModel):
    """Structured LLM-judge evaluation scores."""
    accuracy: int = Field(..., ge=1, le=5, description="Factual accuracy (1–5)")
    completeness: int = Field(..., ge=1, le=5, description="How fully the question is answered (1–5)")
    relevance: int = Field(..., ge=1, le=5, description="Relevance to the question (1–5)")
    groundedness: int = Field(..., ge=1, le=5, description="Claims backed by retrieved context (1–5)")
    overall: float = Field(..., description="Mean of the four scores")
    reasoning: str = Field(..., description="One-sentence justification from the judge")


_JUDGE_SYSTEM = """You are a strict, impartial evaluator of AI-generated answers about codebases.

Given:
- A user question
- Retrieved code context
- A generated answer

Score the answer on FOUR criteria, each from 1 to 5:
  accuracy     : Is every claim factually correct based on the context?
  completeness : Does the answer fully address all parts of the question?
  relevance    : Is the answer focused on the question without padding?
  groundedness : Are all claims directly supported by the retrieved context?

Respond ONLY with valid JSON matching exactly this schema (no extra keys):
{
  "accuracy": <int 1-5>,
  "completeness": <int 1-5>,
  "relevance": <int 1-5>,
  "groundedness": <int 1-5>,
  "reasoning": "<one sentence justification>"
}"""


def judge_answer(
    query: str,
    context_docs: List,
    answer: str,
) -> AnswerQualityScores:
    """
    Evaluate an LLM-generated answer using an LLM judge.

    This consumes 1 LLM call. Results are returned as a Pydantic model.

    Args:
        query: The user's original question.
        context_docs: LangChain Documents used as context.
        answer: The generated answer to evaluate.

    Returns:
        AnswerQualityScores with per-dimension scores and overall mean.
    """
    if not OPENAI_API_KEY:
        # Return neutral scores when no API key is configured.
        return AnswerQualityScores(
            accuracy=0, completeness=0, relevance=0, groundedness=0,
            overall=0.0, reasoning="No API key — evaluation skipped."
        )

    context_text = "\n\n".join(
        f"[{i+1}] {d.page_content[:400]}" for i, d in enumerate(context_docs)
    )
    user_msg = (
        f"Question: {query}\n\n"
        f"Retrieved Context:\n{context_text}\n\n"
        f"Generated Answer:\n{answer}"
    )

    try:
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.0
          
        )
        raw = response.choices[0].message.content.strip()

        # Strip potential markdown fences
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)

        scores_sum = data["accuracy"] + data["completeness"] + data["relevance"] + data["groundedness"]
        return AnswerQualityScores(
            accuracy=data["accuracy"],
            completeness=data["completeness"],
            relevance=data["relevance"],
            groundedness=data["groundedness"],
            overall=round(scores_sum / 4, 2),
            reasoning=data.get("reasoning", ""),
        )

    except Exception as e:
        logger.error(f"Judge evaluation failed: {e}")
        return AnswerQualityScores(
            accuracy=0, completeness=0, relevance=0, groundedness=0,
            overall=0.0, reasoning=f"Evaluation failed: {e}"
        )