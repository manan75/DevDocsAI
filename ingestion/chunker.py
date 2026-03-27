"""
chunker.py — Code-aware document chunking.

Strategy:
  1. For Python files: split by top-level functions/classes using AST.
  2. For all other files: fall back to character-level sliding window chunks.

Each chunk is a LangChain Document with rich metadata.
"""

import ast
import logging
from typing import List, Dict

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS

logger = logging.getLogger(__name__)


def _chunk_python_by_ast(content: str, file_path: str) -> List[Document]:
    """
    Parse Python source and extract top-level functions and classes as chunks.
    Falls back to generic chunking if AST parsing fails.

    Args:
        content: Raw Python source code.
        file_path: Source file path for metadata.

    Returns:
        List of Documents, one per function/class (or fallback chunks).
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.warning(f"AST parse failed for {file_path}, using fallback chunker.")
        return _chunk_generic(content, file_path)

    lines = content.splitlines(keepends=True)
    documents: List[Document] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not isinstance(node, ast.stmt):
            continue  # skip nested; only top-level

        start = node.lineno - 1
        end = node.end_lineno
        chunk_text = "".join(lines[start:end])

        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        documents.append(Document(
            page_content=chunk_text,
            metadata={
                "file_path": file_path,
                "symbol_name": node.name,
                "symbol_type": kind,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
            }
        ))

    if not documents:
        # File has no top-level definitions (e.g. script) — use fallback
        return _chunk_generic(content, file_path)

    return documents


def _chunk_generic(content: str, file_path: str, extension: str = "") -> List[Document]:
    """
    Generic recursive character splitter for non-Python or unparseable files.

    Args:
        content: Raw file content.
        file_path: Source file path for metadata.
        extension: File extension hint (unused currently, reserved).

    Returns:
        List of overlapping text chunk Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", " ", ""],
    )
    texts = splitter.split_text(content)
    return [
        Document(
            page_content=text,
            metadata={
                "file_path": file_path,
                "symbol_name": "",
                "symbol_type": "chunk",
                "chunk_index": i,
            }
        )
        for i, text in enumerate(texts)
    ]


def chunk_documents(raw_docs: List[Dict]) -> List[Document]:
    """
    Dispatch each loaded file to the appropriate chunker.

    Args:
        raw_docs: List of dicts from loader.load_files().

    Returns:
        Flat list of LangChain Document objects ready for embedding.
    """
    all_chunks: List[Document] = []

    for doc in raw_docs:
        content = doc["content"]
        file_path = doc["file_path"]
        ext = doc.get("extension", "")

        if not content.strip():
            continue

        if ext == ".py":
            chunks = _chunk_python_by_ast(content, file_path)
        else:
            chunks = _chunk_generic(content, file_path, ext)

        all_chunks.extend(chunks)

    logger.info(f"Produced {len(all_chunks)} chunks from {len(raw_docs)} files.")
    return all_chunks