"""
app.py — Gradio UI for DevDocs AI: Codebase RAG Assistant.

Dashboard tabs:
  1. Index Repository — upload ZIP, trigger ingestion pipeline.
  2. Ask Questions    — query the indexed codebase with configurable retrieval.
  3. Compare Modes    — side-by-side similarity vs MMR retrieval.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Tuple

import gradio as gr

from config import UPLOAD_DIR, DEFAULT_TOP_K
from ingestion.loader import extract_zip, load_files
from ingestion.chunker import chunk_documents
from ingestion.indexer import index_documents, is_index_populated
from retrieval.retriever import retrieve
from retrieval.query_rewriter import rewrite_query
from llm.generator import generate_answer
from evaluation.metrics import compute_retrieval_metrics
from evaluation.judge import judge_answer
from utils.helpers import setup_logging, format_chunks_for_display, format_metrics_for_display

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

# ─── Pipeline functions ───────────────────────────────────────────────────────

def run_indexing(zip_file) -> str:
    """
    Gradio handler: extract ZIP → load files → chunk → embed → index.

    Args:
        zip_file: Gradio file object (has .name attribute with temp path).

    Returns:
        Status message string.
    """
    if zip_file is None:
        return "❌ Please upload a ZIP file first."

    try:
        # Copy uploaded file to our uploads dir
        src = Path(zip_file.name)
        dest = UPLOAD_DIR / src.name
        shutil.copy2(src, dest)

        gr.Info("📦 Extracting ZIP archive...")
        extract_dir = extract_zip(str(dest))

        gr.Info("📂 Loading source files...")
        raw_docs = load_files(extract_dir)
        if not raw_docs:
            return "⚠️ No supported source files found in the ZIP."

        gr.Info(f"✂️ Chunking {len(raw_docs)} files...")
        chunks = chunk_documents(raw_docs)

        gr.Info(f"🧠 Embedding and indexing {len(chunks)} chunks...")
        index_documents(chunks)

        return (
            f"✅ Indexing complete!\n"
            f"   • Files processed : {len(raw_docs)}\n"
            f"   • Chunks indexed  : {len(chunks)}\n"
            f"   • Ready to query!"
        )

    except Exception as e:
        logger.exception("Indexing failed")
        return f"❌ Indexing failed: {e}"


def run_query(
    query: str,
    use_mmr: bool,
    use_rewriting: bool,
    top_k: int,
    run_evaluation: bool,
) -> Tuple[str, str, str]:
    """
    Gradio handler: rewrite query → retrieve → generate answer → evaluate.

    Returns:
        Tuple of (answer_text, retrieved_context_text, metrics_text).
    """
    if not query.strip():
        return "❌ Please enter a question.", "", ""

    if not is_index_populated():
        return "❌ No index found. Please index a repository first.", "", ""

    try:
        # 1. Optional query rewriting
        effective_query = query
        if use_rewriting:
            gr.Info("🔄 Rewriting query...")
            effective_query = rewrite_query(query, use_llm=False)

        # 2. Retrieval
        search_type = "mmr" if use_mmr else "similarity"
        gr.Info(f"🔍 Retrieving with {search_type.upper()}...")
        docs, scores = retrieve(effective_query, search_type=search_type, top_k=int(top_k))

        # 3. Format retrieved context for display
        context_display = format_chunks_for_display(docs, scores)
        if effective_query != query:
            context_display = f"🔄 Rewritten query: \"{effective_query}\"\n\n" + context_display

        # 4. Answer generation
        gr.Info("💬 Generating answer...")
        answer, source_files = generate_answer(query, docs)

        # 5. Evaluation
        metrics_display = ""
        if run_evaluation:
            gr.Info("📊 Running evaluation...")
            retrieval_metrics = compute_retrieval_metrics(query, docs)
            answer_scores = judge_answer(query, docs, answer)
            metrics_display = format_metrics_for_display(retrieval_metrics, answer_scores)
        else:
            metrics_display = "ℹ️ Enable 'Run Evaluation' to see metrics."

        return answer, context_display, metrics_display

    except Exception as e:
        logger.exception("Query failed")
        return f"❌ Error: {e}", "", ""


def run_comparison(query: str, top_k: int) -> Tuple[str, str, str, str]:
    """
    Gradio handler: run both similarity and MMR side-by-side.

    Returns:
        Tuple of (sim_answer, sim_context, mmr_answer, mmr_context).
    """
    if not query.strip():
        return "❌ Please enter a question.", "", "", ""

    if not is_index_populated():
        msg = "❌ No index found."
        return msg, "", msg, ""

    try:
        k = int(top_k)

        sim_docs, sim_scores = retrieve(query, search_type="similarity", top_k=k)
        mmr_docs, mmr_scores = retrieve(query, search_type="mmr", top_k=k)

        sim_answer, _ = generate_answer(query, sim_docs)
        mmr_answer, _ = generate_answer(query, mmr_docs)

        sim_context = format_chunks_for_display(sim_docs, sim_scores)
        mmr_context = format_chunks_for_display(mmr_docs, mmr_scores)

        return sim_answer, sim_context, mmr_answer, mmr_context

    except Exception as e:
        logger.exception("Comparison failed")
        err = f"❌ Error: {e}"
        return err, "", err, ""


# ─── Gradio UI ────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.metric-box { font-family: monospace; white-space: pre; background: #1e1e2e; color: #cdd6f4; padding: 12px; border-radius: 8px; }
.chunk-box  { font-family: monospace; font-size: 0.82rem; background: #181825; color: #cdd6f4; padding: 10px; border-radius: 8px; }
.answer-box { background: #f8fafc; border-left: 4px solid #6366f1; padding: 12px; border-radius: 6px; }
footer { display: none !important; }
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CSS, title="DevDocs AI") as demo:

        gr.Markdown(
            """
            # 🤖 DevDocs AI — Codebase RAG Assistant
            Upload any code repository as a ZIP file, index it, then ask natural language questions.
            Powered by **HuggingFace embeddings** (free) + **GPT-4.1-nano** (minimal cost).
            """
        )

        # ── Tab 1: Index ──────────────────────────────────────────────────────
        with gr.Tab("📦 Index Repository"):
            gr.Markdown("### Step 1 — Upload your codebase ZIP and index it.")
            with gr.Row():
                with gr.Column(scale=2):
                    zip_input = gr.File(
                        label="Upload ZIP file",
                        file_types=[".zip"],
                        type="filepath",
                    )
                    index_btn = gr.Button("🚀 Index Repository", variant="primary", size="lg")
                with gr.Column(scale=3):
                    index_status = gr.Textbox(
                        label="Indexing Status",
                        lines=6,
                        interactive=False,
                        placeholder="Status will appear here after indexing...",
                    )

            index_btn.click(
                fn=run_indexing,
                inputs=[zip_input],
                outputs=[index_status],
            )

        # ── Tab 2: Query ──────────────────────────────────────────────────────
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("### Step 2 — Ask anything about your indexed codebase.")
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g. How does the authentication flow work?",
                        lines=2,
                    )
                with gr.Column(scale=1):
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=15, value=DEFAULT_TOP_K, step=1,
                        label="Top-K chunks",
                    )

            with gr.Row():
                use_mmr_toggle = gr.Checkbox(label="Use MMR retrieval", value=False)
                use_rewrite_toggle = gr.Checkbox(label="Use query rewriting", value=False)
                run_eval_toggle = gr.Checkbox(label="Run evaluation (costs 1 LLM call)", value=True)
                query_btn = gr.Button("🔍 Ask", variant="primary")

            with gr.Row():
                with gr.Column(scale=2):
                    answer_output = gr.Markdown(label="Answer", elem_classes=["answer-box"])
                with gr.Column(scale=1):
                    metrics_output = gr.Textbox(
                        label="📊 Evaluation Metrics",
                        lines=18,
                        interactive=False,
                        elem_classes=["metric-box"],
                    )

            context_output = gr.Textbox(
                label="📄 Retrieved Context Chunks",
                lines=15,
                interactive=False,
                elem_classes=["chunk-box"],
            )

            query_btn.click(
                fn=run_query,
                inputs=[query_input, use_mmr_toggle, use_rewrite_toggle, top_k_slider, run_eval_toggle],
                outputs=[answer_output, context_output, metrics_output],
            )

        # ── Tab 3: Compare ────────────────────────────────────────────────────
        with gr.Tab("⚖️ Compare: Similarity vs MMR"):
            gr.Markdown(
                "### Run both retrieval modes side-by-side for the same question.\n"
                "_Note: This uses 2 LLM calls._"
            )
            with gr.Row():
                cmp_query = gr.Textbox(
                    label="Question",
                    placeholder="e.g. Where is database initialisation handled?",
                    lines=2,
                    scale=4,
                )
                cmp_top_k = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Top-K", scale=1)
            cmp_btn = gr.Button("⚖️ Compare", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎯 Similarity Search")
                    sim_answer_out = gr.Markdown(elem_classes=["answer-box"])
                    sim_context_out = gr.Textbox(lines=10, interactive=False, label="Chunks", elem_classes=["chunk-box"])
                with gr.Column():
                    gr.Markdown("#### 🌈 MMR Search")
                    mmr_answer_out = gr.Markdown(elem_classes=["answer-box"])
                    mmr_context_out = gr.Textbox(lines=10, interactive=False, label="Chunks", elem_classes=["chunk-box"])

            cmp_btn.click(
                fn=run_comparison,
                inputs=[cmp_query, cmp_top_k],
                outputs=[sim_answer_out, sim_context_out, mmr_answer_out, mmr_context_out],
            )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            **DevDocs AI** | Embeddings: `all-MiniLM-L6-v2` (free) | LLM: `gpt-4.1-nano` | Vector DB: ChromaDB (local)
            """
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )