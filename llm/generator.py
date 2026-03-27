"""
generator.py — LLM-based answer generation from retrieved context.

Uses litellm so the model can be swapped by changing config.LLM_MODEL.
The prompt is designed to:
  - Ground the answer strictly in retrieved context
  - Reference source files by name
  - Decline gracefully when context is insufficient
"""

import logging
from typing import List, Tuple

from langchain.schema import Document
import litellm

from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, OPENAI_API_KEY

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are DevDocs AI, an expert assistant that answers questions about codebases.

Rules:
1. Answer ONLY using the provided code context. Do NOT hallucinate.
2. If the context is insufficient, say so clearly.
3. Always cite the source file(s) at the end of your answer under a "Sources:" heading.
4. Be concise and precise. Use code snippets when helpful.
5. Format code blocks with triple backticks and the appropriate language tag.
"""


def _build_context_block(docs: List[Document]) -> str:
    """
    Format retrieved documents into a structured context string for the prompt.

    Args:
        docs: Retrieved LangChain Documents.

    Returns:
        Formatted context string.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        file_path = meta.get("file_path", "unknown")
        symbol = meta.get("symbol_name", "")
        symbol_type = meta.get("symbol_type", "chunk")

        header = f"[{i}] File: {file_path}"
        if symbol:
            header += f" | {symbol_type}: {symbol}"

        parts.append(f"{header}\n```\n{doc.page_content.strip()}\n```")

    return "\n\n".join(parts)


def generate_answer(
    query: str,
    docs: List[Document],
) -> Tuple[str, List[str]]:
    """
    Generate a grounded answer from retrieved documents.

    Args:
        query: The user's natural language question.
        docs: Retrieved Document chunks (context).

    Returns:
        Tuple of (answer_text, source_file_list).

    Raises:
        RuntimeError: If the LLM call fails.
    """
    if not OPENAI_API_KEY:
        return (
            "⚠️ No OpenAI API key configured. Set the OPENAI_API_KEY environment variable.",
            [],
        )

    context_block = _build_context_block(docs)
    source_files = list({doc.metadata.get("file_path", "") for doc in docs})

    user_message = (
        f"Question: {query}\n\n"
        f"Context (retrieved code):\n{context_block}"
    )

    try:
        response = litellm.completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated answer ({len(answer)} chars) for: '{query[:60]}'")
        return answer, source_files

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise RuntimeError(f"LLM generation failed: {e}") from e