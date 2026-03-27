"""
loader.py — Handles ZIP extraction and file loading.

Responsibilities:
  - Extract uploaded ZIP archives
  - Filter files by allowed extensions
  - Read file contents safely
  - Return a list of raw document dicts
"""

import zipfile
import os
import logging
from pathlib import Path
from typing import List, Dict

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, UPLOAD_DIR

logger = logging.getLogger(__name__)


def extract_zip(zip_path: str) -> Path:
    """
    Extract a ZIP archive to a unique subdirectory under UPLOAD_DIR.

    Args:
        zip_path: Path to the uploaded .zip file.

    Returns:
        Path to the extraction directory.
    """
    zip_path = Path(zip_path)
    extract_dir = UPLOAD_DIR / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    logger.info(f"Extracted ZIP to: {extract_dir}")
    return extract_dir


def load_files(extract_dir: Path) -> List[Dict]:
    """
    Walk the extraction directory and load allowed source files.

    Each returned dict contains:
        - content (str): raw file text
        - file_path (str): relative path within the archive
        - extension (str): file extension

    Args:
        extract_dir: Directory containing extracted files.

    Returns:
        List of raw document dicts.
    """
    documents: List[Dict] = []
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    for root, _dirs, files in os.walk(extract_dir):
        for filename in files:
            full_path = Path(root) / filename
            ext = full_path.suffix.lower()

            if ext not in ALLOWED_EXTENSIONS:
                continue

            if full_path.stat().st_size > max_bytes:
                logger.warning(f"Skipping large file: {full_path}")
                continue

            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                relative_path = str(full_path.relative_to(extract_dir))
                documents.append({
                    "content": content,
                    "file_path": relative_path,
                    "extension": ext,
                })
            except Exception as e:
                logger.warning(f"Failed to read {full_path}: {e}")

    logger.info(f"Loaded {len(documents)} files from {extract_dir}")
    return documents