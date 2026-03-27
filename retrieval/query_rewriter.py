"""
query_rewriter.py — Lightweight query reformulation before retrieval.

Two modes:
  1. Rule-based (free): simple heuristic expansions (default, zero cost).
  2. LLM-based (optional): one cheap LLM call to reformulate the query.

The LLM path is only invoked when explicitly requested to keep costs minimal.
"""

import re
import logging

import litellm

from config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY

logger = logging.getLogger(__name__)

# Heuristic keyword expansions (extend as needed).
_EXPANSIONS = {
    r"\bauth\b": "authentication authorization",
    r"\bdb\b": "database",
    r"\bapi\b": "API endpoint route handler",
    r"\bconfig\b": "configuration settings",
    r"\berror\b": "error exception handling",
    r"\btest\b": "unit test test case",
    r"\bdeploy\b": "deployment CI CD pipeline",
}


def rule_based_rewrite(query: str) -> str:
    """
    Apply simple regex-based expansions to common abbreviations.

    Args:
        query: Original user query.

    Returns:
        Slightly expanded query string.
    """
    rewritten = query
    for pattern, expansion in _EXPANSIONS.items():
        rewritten = re.sub(pattern, expansion, rewritten, flags=re.IGNORECASE)
    if rewritten != query:
        logger.debug(f"Rule-based rewrite: '{query}' → '{rewritten}'")
    return rewritten


def llm_rewrite(query: str) -> str:
    """
    Use a cheap LLM call to reformulate the query for better retrieval.
    This is optional and costs ~1 LLM call per query.

    Args:
        query: Original user query.

    Returns:
        Reformulated query optimised for semantic code search.
    """
    if not OPENAI_API_KEY:
        logger.warning("No API key set; falling back to rule-based rewrite.")
        return rule_based_rewrite(query)

    system_prompt = (
        "You are a search query optimizer for code repositories. "
        "Rewrite the user's question into a concise, keyword-rich query "
        "that will best match relevant code chunks. "
        "Output ONLY the rewritten query — no explanation."
    )
    try:
        response = litellm.completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=80,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"LLM rewrite: '{query}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"LLM rewrite failed ({e}); falling back to rule-based.")
        return rule_based_rewrite(query)


def rewrite_query(query: str, use_llm: bool = False) -> str:
    """
    Entry point for query rewriting.

    Args:
        query: Raw user question.
        use_llm: If True, invoke LLM rewrite (costs 1 LLM call).

    Returns:
        Rewritten query string.
    """
    if use_llm:
        return llm_rewrite(query)
    return rule_based_rewrite(query)