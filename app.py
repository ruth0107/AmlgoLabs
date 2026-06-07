"""
app.py — Streamlit RAG Chatbot with Streaming Responses
Amlgo Labs Junior AI Engineer Assignment
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent))
from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    DEFAULT_DOCUMENT,
    TOP_K,
    DISPLAY_TOP_K,
    TEMPERATURE,
    NUM_CTX,
)
from src.vectorstore import get_chunk_count, index_exists, build_index
from src.document_loader import load_pdf
from src.chunker import chunk_text, save_chunks, load_chunks
from src.pipeline import stream_answer

# Page Config
st.set_page_config(
    page_title="PolicyAI — RAG Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }



    /* Source cards */
    .source-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #a0aec0;
        line-height: 1.5;
    }

    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .score-badge {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .score-badge.high   { background: linear-gradient(135deg, #065f46, #10b981); }
    .score-badge.medium { background: linear-gradient(135deg, #92400e, #f59e0b); }
    .score-badge.low    { background: linear-gradient(135deg, #7f1d1d, #ef4444); }

    /* Sidebar metrics */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #63b3ed;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Status dot */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Title */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #63b3ed, #7c3aed, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 4px;
    }
    .main-subtitle {
        text-align: center;
        color: #718096;
        font-size: 0.9rem;
        margin-bottom: 24px;
    }

    /* Input area */
    .stTextInput input, .stChatInput textarea {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }

    /* Dividers */
    hr { border-color: rgba(255,255,255,0.08); }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 8px !important;
        color: #90cdf4 !important;
        font-size: 0.85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helpers

def score_class(score: float) -> str:
    if score >= 0.70:
        return "high"
    elif score >= 0.50:
        return "medium"
    return "low"


def render_source_cards(sources: list):
    """Render retrieved source chunks as styled cards."""
    if not sources:
        return
    display = sources[:DISPLAY_TOP_K]
    for i, src in enumerate(display, start=1):
        sc = score_class(src["score"])
        text_preview = src["text"][:280].replace("\n", " ").strip()
        if len(src["text"]) > 280:
            text_preview += "…"
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-header">
                    <strong style="color:#90cdf4;">📄 Source {i} &nbsp;·&nbsp; {src['chunk_id']}</strong>
                    <span class="score-badge {sc}">Relevance: {src['score']:.0%}</span>
                </div>
                <div>{text_preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Session State Init

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# Sidebar

with st.sidebar:
    st.markdown(
        "<h2 style='color:#63b3ed; font-weight:700; margin-bottom:4px;'>🔍 PolicyAI</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#718096; font-size:0.8rem;'>RAG Chatbot · Amlgo Labs</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Status indicator
    chunk_count = get_chunk_count()
    st.session_state.chunk_count = chunk_count
    status_color = "#10b981" if chunk_count > 0 else "#f59e0b"
    status_text  = "Index Ready" if chunk_count > 0 else "Building Index…"

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; margin-bottom:16px;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:{status_color};margin-right:8px;
                        box-shadow:0 0 6px {status_color};"></div>
            <span style="color:#e2e8f0; font-size:0.85rem; font-weight:500;">{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Technical Info (Collapsible)
    with st.expander("ℹ️ System Info", expanded=False):
        st.markdown("**🤖 Model**")
        st.code(LLM_MODEL, language=None)

        st.markdown("**🔢 Embeddings**")
        st.code(EMBEDDING_MODEL.split("/")[-1], language=None)

        st.markdown("**🗄️ Vector DB**")
        st.code("ChromaDB", language=None)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{chunk_count}</div>
                    <div class="metric-label">Chunks</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{TOP_K}</div>
                    <div class="metric-label">Top-K</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Document info
    doc_name = DEFAULT_DOCUMENT.name if DEFAULT_DOCUMENT.exists() else "No document"
    st.markdown(
        f"**📁 Document**  \n`{doc_name}`",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Index build button
    if not index_exists():
        if st.button("⚙️ Build Index", use_container_width=True):
            with st.spinner("Loading & chunking document…"):
                raw_text = load_pdf()
                chunks   = chunk_text(raw_text, source_name=DEFAULT_DOCUMENT.name)
                save_chunks(chunks)
            with st.spinner(f"Embedding {len(chunks)} chunks…"):
                build_index(chunks)
            st.session_state.index_ready = True
            st.session_state.chunk_count = get_chunk_count()
            st.success(f"✅ {st.session_state.chunk_count} chunks indexed!")
            st.rerun()
    else:
        st.session_state.index_ready = True

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        st.caption("Note: Restart app to apply model changes.")
        st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.05, key="temperature",
                  help="Lower = more factual, Higher = more creative")
        st.slider("Context Window", 512, 4096, NUM_CTX, 256, key="num_ctx",
                  help="Reduce to save VRAM (2048 recommended for 4GB GPU)")
        st.slider("Top-K Chunks", 1, 10, TOP_K, 1, key="top_k",
                  help="Number of document chunks to retrieve per query")


# Main Chat Area

# Title
st.markdown(
    "<div class='main-title'>🔍 PolicyAI — Document Chatbot</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='main-subtitle'>Ask questions about your policy documents · Powered by Mistral 7B + RAG</div>",
    unsafe_allow_html=True,
)

# Index not ready warning
if not st.session_state.index_ready and not index_exists():
    st.warning(
        "⚠️ **No index found.** Click **Build Index** in the sidebar to process and embed the document before chatting.",
        icon="⚙️",
    )
    st.stop()

# Auto-set index ready if it already exists from a previous run
if index_exists():
    st.session_state.index_ready = True

# Welcome message
if not st.session_state.messages:
    st.markdown(
        """
        <div style="background:rgba(30,64,175,0.15);border:1px solid rgba(59,130,246,0.3);
                    border-radius:14px;padding:20px 24px;margin:16px 0;color:#90cdf4;">
            <strong>👋 Welcome to PolicyAI!</strong><br/>
            <span style="color:#a0aec0;font-size:0.9rem;">
            I can answer questions based on the indexed policy document.
            Try asking things like:
            </span>
            <ul style="color:#90cdf4;margin-top:8px;font-size:0.88rem;">
                <li>How does Batch Arbitration work?</li>
                <li>What happens if eBay amends the arbitration agreement?</li>
                <li>Explain how arbitrators are selected.</li>
                <li>What happens if the Batch Arbitration clause is found unenforceable?</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🔍"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 Sources Used ({len(msg['sources'][:DISPLAY_TOP_K])} chunks)", expanded=False):
                    render_source_cards(msg["sources"])


# Chat Input
if prompt := st.chat_input("Ask a question about the document…", key="chat_input"):

    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant", avatar="🔍"):
        response_placeholder = st.empty()
        sources_placeholder  = st.empty()

        full_response = ""
        sources       = []

        with st.spinner("Searching documents…"):
            k_val = st.session_state.get("top_k", TOP_K)
            token_stream, sources = stream_answer(prompt, k=k_val)

        # Stream tokens
        for token in token_stream:
            full_response += token
            response_placeholder.markdown(full_response + "▌")

        # Final response without cursor
        response_placeholder.markdown(full_response)

        # Show sources
        if sources:
            with sources_placeholder.expander(
                f"📄 Sources Used ({len(sources[:DISPLAY_TOP_K])} chunks)", expanded=False
            ):
                render_source_cards(sources)

    # Save to session state
    st.session_state.messages.append(
        {
            "role":    "assistant",
            "content": full_response,
            "sources": sources,
        }
    )
