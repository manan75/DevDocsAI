"""
indexer.py — Embeds chunks and persists them in ChromaDB.

Uses HuggingFace all-MiniLM-L6-v2 (free, 384-dim).
ChromaDB is stored locally so embeddings are never recomputed
unless the collection is explicitly cleared.
"""

import logging
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import VECTOR_DB_DIR, EMBEDDING_MODEL, EMBEDDING_DEVICE, CHROMA_COLLECTION_NAME

logger = logging.getLogger(__name__)

# Module-level singleton so the embedding model is loaded only once per process.
_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return (or lazily create) the shared HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings instance for all-MiniLM-L6-v2.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def get_vectorstore() -> Chroma:
    """
    Open (or create) the persistent Chroma vector store.

    Returns:
        Chroma instance backed by the local vector_db directory.
    """
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(VECTOR_DB_DIR),
    )


def index_documents(chunks: List[Document]) -> Chroma:
    """
    Embed and insert document chunks into ChromaDB.

    Existing documents in the collection are cleared before re-indexing
    so that re-uploading a ZIP starts fresh.

    Args:
        chunks: LangChain Documents produced by the chunker.

    Returns:
        The populated Chroma vector store.
    """
    if not chunks:
        raise ValueError("No chunks to index.")

    embeddings = get_embedding_model()

    # Clear previous collection to avoid stale data on re-index.
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )
    vectorstore.delete_collection()

    # Recreate and populate.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR),
    )

    logger.info(f"Indexed {len(chunks)} chunks into Chroma collection '{CHROMA_COLLECTION_NAME}'.")
    return vectorstore


def is_index_populated() -> bool:
    """
    Check whether the Chroma collection contains any documents.

    Returns:
        True if at least one document is stored, False otherwise.
    """
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        return count > 0
    except Exception:
        return False