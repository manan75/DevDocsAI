"""app.py — Gradio UI for DevDocs AI: Codebase RAG Assistant.

A polished, product-like interface with a softer visual language,
modern typography, improved spacing, and clearer output cards.

Dashboard tabs:
  1. Index Repository — upload ZIP, trigger ingestion pipeline.
  2. Ask Questions    — query the indexed codebase with configurable retrieval.
  3. Compare Modes    — side-by-side similarity vs MMR retrieval.
"""

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


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline functions
# ──────────────────────────────────────────────────────────────────────────────

def run_indexing(zip_file) -> str:
    """Gradio handler: extract ZIP → load files → chunk → embed → index."""
    if zip_file is None:
        return "❌ Please upload a ZIP file first."

    try:
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
            f"✅ Indexing complete!\n\n"
            f"Files processed: {len(raw_docs)}\n"
            f"Chunks indexed: {len(chunks)}\n"
            f"Status: Ready to query"
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
    """Gradio handler: rewrite query → retrieve → generate answer → evaluate."""
    if not query.strip():
        return "❌ Please enter a question.", "", ""

    if not is_index_populated():
        return "❌ No index found. Please index a repository first.", "", ""

    try:
        effective_query = query
        if use_rewriting:
            gr.Info("🔄 Rewriting query...")
            effective_query = rewrite_query(query, use_llm=False)

        search_type = "mmr" if use_mmr else "similarity"
        gr.Info(f"🔍 Retrieving with {search_type.upper()}...")
        docs, scores = retrieve(effective_query, search_type=search_type, top_k=int(top_k))

        context_display = format_chunks_for_display(docs, scores)
        if effective_query != query:
            context_display = f"🔄 Rewritten query: \"{effective_query}\"\n\n" + context_display

        gr.Info("💬 Generating answer...")
        answer, _source_files = generate_answer(query, docs)

        metrics_display = ""
        if run_evaluation:
            gr.Info("📊 Running evaluation...")
            retrieval_metrics = compute_retrieval_metrics(query, docs)
            answer_scores = judge_answer(query, docs, answer)
            metrics_display = format_metrics_for_display(retrieval_metrics, answer_scores)
        else:
            metrics_display = "ℹ️ Enable 'Run evaluation' to see metrics."

        return answer, context_display, metrics_display

    except Exception as e:
        logger.exception("Query failed")
        return f"❌ Error: {e}", "", ""


def run_comparison(query: str, top_k: int) -> Tuple[str, str, str, str]:
    """Gradio handler: run both similarity and MMR side-by-side."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Theme + Styling
# ──────────────────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
:root {
  --bg-0: #0b1020;
  --bg-1: #11162a;
  --bg-2: #151b31;
  --card: rgba(17, 24, 39, 0.72);
  --card-strong: rgba(15, 23, 42, 0.92);
  --card-border: rgba(148, 163, 184, 0.14);
  --text-main: #e5e7eb;
  --text-soft: #94a3b8;
  --accent: #8b5cf6;
  --accent-2: #22c55e;
  --accent-3: #38bdf8;
  --danger: #f87171;
  --shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
}

html, body {
  background:
    radial-gradient(circle at top left, rgba(139,92,246,0.18), transparent 28%),
    radial-gradient(circle at top right, rgba(56,189,248,0.14), transparent 22%),
    linear-gradient(180deg, var(--bg-0), var(--bg-1) 45%, #0a0f1d 100%) !important;
  color: var(--text-main) !important;
}

.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
}

/* Main shell */
#app-shell {
  border: 1px solid var(--card-border);
  background: linear-gradient(180deg, rgba(17,24,39,0.84), rgba(15,23,42,0.74));
  box-shadow: var(--shadow);
  border-radius: 28px;
  padding: 22px;
  backdrop-filter: blur(18px);
}

/* Hero */
.hero-wrap {
  display: grid;
  grid-template-columns: 1.4fr 0.8fr;
  gap: 18px;
  align-items: stretch;
  margin-bottom: 18px;
}
.hero-card, .mini-card, .section-card {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 24px;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16);
  backdrop-filter: blur(14px);
}
.hero-card {
  padding: 24px 24px 22px;
}
.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(139,92,246,0.14);
  color: #d8b4fe;
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  margin-bottom: 14px;
}
.hero-title {
  margin: 0;
  font-size: clamp(2rem, 3vw, 3.1rem);
  line-height: 1.05;
  letter-spacing: -0.03em;
  color: #f8fafc;
}
.hero-subtitle {
  margin-top: 12px;
  color: var(--text-soft);
  font-size: 1rem;
  line-height: 1.65;
  max-width: 68ch;
}
.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}
.badge-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  border-radius: 999px;
  font-size: 0.86rem;
  color: #e2e8f0;
  background: rgba(15,23,42,0.55);
  border: 1px solid rgba(148,163,184,0.16);
}
.mini-card {
  padding: 18px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.mini-card h4 {
  margin: 0 0 8px;
  color: #f8fafc;
  font-size: 1rem;
}
.mini-card p {
  margin: 0;
  color: var(--text-soft);
  line-height: 1.6;
  font-size: 0.95rem;
}
.mini-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 14px;
}
.stat {
  border-radius: 18px;
  padding: 14px;
  background: rgba(15,23,42,0.72);
  border: 1px solid rgba(148,163,184,0.12);
}
.stat .label {
  color: var(--text-soft);
  font-size: 0.78rem;
  margin-bottom: 6px;
}
.stat .value {
  color: #f8fafc;
  font-size: 1rem;
  font-weight: 700;
}

/* Tabs */
.tab-nav {
  margin-top: 8px !important;
}
.gradio-tabs .tab-nav button {
  border-radius: 999px !important;
  border: 1px solid rgba(148,163,184,0.14) !important;
  background: rgba(15,23,42,0.55) !important;
  color: #cbd5e1 !important;
  padding: 10px 14px !important;
  transition: all 0.2s ease !important;
}
.gradio-tabs .tab-nav button.selected {
  background: linear-gradient(135deg, rgba(139,92,246,0.95), rgba(59,130,246,0.85)) !important;
  color: white !important;
  box-shadow: 0 12px 24px rgba(91, 33, 182, 0.25) !important;
}

/* Sections and widgets */
.section-card {
  padding: 18px;
  margin-bottom: 14px;
}
.section-title {
  margin: 0 0 6px;
  font-size: 1.05rem;
  color: #f8fafc;
  letter-spacing: -0.01em;
}
.section-desc {
  margin: 0;
  color: var(--text-soft);
  font-size: 0.95rem;
  line-height: 1.6;
}

textarea, input, .wrap, .prose, .markdown, .svelte-textbox, .svelte-slider, .svelte-checkbox {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

textarea, .gr-textbox textarea, .gr-textbox input, .gr-file, .gr-number input {
  background: rgba(15,23,42,0.72) !important;
  color: var(--text-main) !important;
  border: 1px solid rgba(148,163,184,0.14) !important;
  border-radius: 18px !important;
}

.gr-textbox label, .gr-slider label, .gr-checkbox label, .gr-file label {
  color: #e2e8f0 !important;
  font-weight: 600 !important;
}

.gr-button {
  border-radius: 16px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  padding: 12px 16px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em;
}
.gr-button.primary {
  background: linear-gradient(135deg, #8b5cf6, #3b82f6) !important;
  color: white !important;
  box-shadow: 0 16px 30px rgba(59,130,246,0.22) !important;
}
.gr-button:hover {
  transform: translateY(-1px);
}

/* Outputs */
.answer-box, .metric-box, .chunk-box, .output-card {
  border-radius: 22px !important;
  border: 1px solid rgba(148,163,184,0.14) !important;
  background: rgba(2, 6, 23, 0.48) !important;
  box-shadow: 0 12px 30px rgba(0,0,0,0.14);
}
.answer-box {
  padding: 16px !important;
  line-height: 1.75 !important;
}
.answer-box h1, .answer-box h2, .answer-box h3, .answer-box h4 {
  color: #f8fafc !important;
  letter-spacing: -0.02em;
}
.answer-box p, .answer-box li {
  color: #e2e8f0 !important;
}
.answer-box code, .chunk-box code, .metric-box code {
  background: rgba(15,23,42,0.9) !important;
  color: #e2e8f0 !important;
  border-radius: 8px !important;
  padding: 0.12rem 0.35rem !important;
}
.chunk-box, .metric-box {
  padding: 14px !important;
  white-space: pre-wrap !important;
  color: #cbd5e1 !important;
  line-height: 1.7 !important;
}

/* Make the built-in markdown areas feel cleaner */
.prose, .markdown {
  color: #e2e8f0 !important;
}
.prose h1, .prose h2, .prose h3, .markdown h1, .markdown h2, .markdown h3 {
  color: #f8fafc !important;
}

footer { display: none !important; }

/* Responsive */
@media (max-width: 1000px) {
  .hero-wrap { grid-template-columns: 1fr; }
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def hero_panel() -> str:
    return """
<div class="hero-wrap">
  <div class="hero-card">
    <div class="hero-kicker">✨ DevDocs AI · Codebase RAG Assistant</div>
    <h1 class="hero-title">A calm, premium workspace for exploring your codebase.</h1>
    <p class="hero-subtitle">
      Upload a repository ZIP, index it once, and ask natural-language questions with a cleaner
      reading experience. The interface keeps the workflow fast while feeling intentionally designed,
      not template-generated.
    </p>
    <div class="hero-badges">
      <span class="badge-pill">⚡ Fast indexing flow</span>
      <span class="badge-pill">🧠 Query rewriting</span>
      <span class="badge-pill">🔎 Similarity + MMR</span>
      <span class="badge-pill">📊 Built-in evaluation</span>
    </div>
  </div>
  <div class="mini-card">
    <div>
      <h4>What this interface emphasizes</h4>
      <p>
        Clear hierarchy, softer contrast, rounded surfaces, better spacing, and output cards that are easier to scan.
      </p>
    </div>
    <div class="mini-grid">
      <div class="stat">
        <div class="label">Primary feel</div>
        <div class="value">Modern glass UI</div>
      </div>
      <div class="stat">
        <div class="label">Typography</div>
        <div class="value">Inter</div>
      </div>
      <div class="stat">
        <div class="label">Tone</div>
        <div class="value">Soft + premium</div>
      </div>
      <div class="stat">
        <div class="label">Outputs</div>
        <div class="value">Readable cards</div>
      </div>
    </div>
  </div>
</div>
"""


def section_block(title: str, desc: str) -> str:
    return f"""
<div class="section-card">
  <div class="section-title">{title}</div>
  <p class="section-desc">{desc}</p>
</div>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Build UI
# ──────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CSS, title="DevDocs AI") as demo:
        with gr.Column(elem_id="app-shell"):
            gr.HTML(hero_panel())

            with gr.Tabs(elem_classes=["tab-nav"]):
                # ── Tab 1: Index ──────────────────────────────────────────────
                with gr.Tab("📦 Index Repository"):
                    gr.HTML(section_block(
                        "Step 1 — Add your codebase",
                        "Upload a ZIP file, extract it, chunk the files, and build the local vector index."
                    ))
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
                                lines=9,
                                interactive=False,
                                placeholder="Status will appear here after indexing...",
                            )

                    index_btn.click(
                        fn=run_indexing,
                        inputs=[zip_input],
                        outputs=[index_status],
                    )

                # ── Tab 2: Query ──────────────────────────────────────────────
                with gr.Tab("💬 Ask Questions"):
                    gr.HTML(section_block(
                        "Step 2 — Ask about the code",
                        "Use retrieval settings to control how the assistant searches the indexed repository."
                    ))
                    with gr.Row():
                        with gr.Column(scale=3):
                            query_input = gr.Textbox(
                                label="Your Question",
                                placeholder="e.g. How does the authentication flow work?",
                                lines=2,
                            )
                        with gr.Column(scale=1):
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=15,
                                value=DEFAULT_TOP_K,
                                step=1,
                                label="Top-K chunks",
                            )

                    with gr.Row():
                        use_mmr_toggle = gr.Checkbox(label="Use MMR retrieval", value=False)
                        use_rewrite_toggle = gr.Checkbox(label="Use query rewriting", value=False)
                        run_eval_toggle = gr.Checkbox(label="Run evaluation (costs 1 LLM call)", value=True)
                        query_btn = gr.Button("🔍 Ask", variant="primary")

                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.HTML('<div class="section-title">Answer</div>')
                            answer_output = gr.Markdown(elem_classes=["answer-box"])
                        with gr.Column(scale=1):
                            metrics_output = gr.Textbox(
                                label="📊 Evaluation Metrics",
                                lines=18,
                                interactive=False,
                                elem_classes=["metric-box"],
                            )

                    gr.HTML('<div class="section-title">Retrieved Context</div>')
                    context_output = gr.Textbox(
                        label="",
                        lines=15,
                        interactive=False,
                        elem_classes=["chunk-box"],
                    )

                    query_btn.click(
                        fn=run_query,
                        inputs=[query_input, use_mmr_toggle, use_rewrite_toggle, top_k_slider, run_eval_toggle],
                        outputs=[answer_output, context_output, metrics_output],
                    )

                # ── Tab 3: Compare ────────────────────────────────────────────
                with gr.Tab("⚖️ Compare: Similarity vs MMR"):
                    gr.HTML(section_block(
                        "Step 3 — Compare retrieval styles",
                        "Run similarity and MMR side-by-side to inspect how the context and answer change."
                    ))
                    with gr.Row():
                        cmp_query = gr.Textbox(
                            label="Question",
                            placeholder="e.g. Where is database initialisation handled?",
                            lines=2,
                            scale=4,
                        )
                        cmp_top_k = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Top-K",
                            scale=1,
                        )
                    cmp_btn = gr.Button("⚖️ Compare", variant="primary")

                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<div class="section-title">Similarity Search</div>')
                            sim_answer_out = gr.Markdown(elem_classes=["answer-box"])
                            sim_context_out = gr.Textbox(
                                lines=10,
                                interactive=False,
                                label="Chunks",
                                elem_classes=["chunk-box"],
                            )
                        with gr.Column():
                            gr.HTML('<div class="section-title">MMR Search</div>')
                            mmr_answer_out = gr.Markdown(elem_classes=["answer-box"])
                            mmr_context_out = gr.Textbox(
                                lines=10,
                                interactive=False,
                                label="Chunks",
                                elem_classes=["chunk-box"],
                            )

                    cmp_btn.click(
                        fn=run_comparison,
                        inputs=[cmp_query, cmp_top_k],
                        outputs=[sim_answer_out, sim_context_out, mmr_answer_out, mmr_context_out],
                    )

            gr.Markdown(
                """
                <div style="margin-top: 18px; padding: 14px 6px 0; color: #94a3b8; font-size: 0.9rem; line-height: 1.7;">
                    <strong style="color:#e2e8f0;">DevDocs AI</strong> · Embeddings: <code>all-MiniLM-L6-v2</code> ·
                    LLM: <code>gpt-4.1-nano</code> · Vector DB: <code>ChromaDB</code>
                </div>
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
