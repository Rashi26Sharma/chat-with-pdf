"""
app.py - Main Streamlit Application
====================================
This is the entry point of our "Chat with PDF" RAG application.
It handles the user interface and ties everything together.

RAG = Retrieval-Augmented Generation
- Retrieval: Find relevant chunks from your PDF
- Augmented: Add those chunks as context to your prompt
- Generation: LLM generates an answer using that context
"""

import streamlit as st
from utils import (
    load_pdf,
    split_text,
    create_vector_store,
    create_qa_chain,
    save_vector_store
)
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS for a clean look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .chat-message-user {
        background-color: #e8f4fd;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 8px 0;
    }
    .chat-message-bot {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 8px 0;
    }
    .source-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State Initialization
# Streamlit reruns the script on every interaction.
# Session state persists data across reruns.
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # Stores (question, answer, sources)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None        # The LangChain QA chain

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False  # Has the PDF been embedded?

# ─────────────────────────────────────────────
# Sidebar: PDF Upload & Settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # OpenAI API Key input (fallback if not in .env)
    api_key_input = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored. It lives only in this session."
    )

    # If user typed a key, override the env variable
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.markdown("---")

    # PDF File Uploader
    uploaded_file = st.file_uploader(
        "📂 Upload your PDF",
        type=["pdf"],
        help="Upload any PDF and start chatting with it!"
    )

    # Advanced settings (chunk controls)
    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200, max_value=1500, value=500, step=100,
            help="How many characters per text chunk. Larger = more context, slower."
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0, max_value=300, value=50, step=10,
            help="Overlap between chunks so context isn't lost at boundaries."
        )
        top_k = st.slider(
            "Top K Results",
            min_value=1, max_value=10, value=4,
            help="How many chunks to retrieve per question."
        )

    # Process PDF Button
    if uploaded_file and st.button("🚀 Process PDF", use_container_width=True):
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("❌ Please enter your OpenAI API Key first.")
        else:
            with st.spinner("📖 Reading and indexing your PDF..."):
                try:
                    # Step 1: Extract raw text from PDF
                    raw_text = load_pdf(uploaded_file)

                    # Step 2: Split text into manageable chunks
                    chunks = split_text(raw_text, chunk_size, chunk_overlap)

                    # Step 3: Create embeddings + FAISS vector store
                    vector_store = create_vector_store(chunks)

                    # Step 4: Save index to disk (for reuse)
                    save_vector_store(vector_store)

                    # Step 5: Build the LangChain QA chain
                    st.session_state.qa_chain = create_qa_chain(vector_store, top_k)
                    st.session_state.pdf_processed = True
                    st.session_state.chat_history = []  # Reset chat on new PDF

                    st.success(f"✅ PDF processed! {len(chunks)} chunks indexed.")
                    st.info(f"📊 Total characters extracted: {len(raw_text):,}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Status indicator
    if st.session_state.pdf_processed:
        st.success("✅ PDF Ready — Ask your questions!")
    else:
        st.info("📄 Upload a PDF to get started.")

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────
# Main Content Area
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">📄 Chat with PDF</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a PDF and ask anything about it — powered by RAG + OpenAI</div>',
    unsafe_allow_html=True
)

# Display existing chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for question, answer, sources in st.session_state.chat_history:
        st.markdown(f"""
        <div class="chat-message-user"><strong>🧑 You:</strong> {question}</div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="chat-message-bot"><strong>🤖 Assistant:</strong> {answer}</div>
        """, unsafe_allow_html=True)

        if sources:
            with st.expander(f"📚 View Sources ({len(sources)} chunks used)", expanded=False):
                for j, source in enumerate(sources):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Chunk {j+1}:</strong> {source.page_content[:300]}...
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("---")

# ─────────────────────────────────────────────
# Question Input Area
# ─────────────────────────────────────────────
if st.session_state.pdf_processed:
    st.markdown("### 🤔 Ask a Question")

    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_input(
            "Your question:",
            placeholder="What is this document about? Summarize the key points...",
            label_visibility="collapsed"
        )
        submit = st.form_submit_button("Send ➤")

    if submit and user_question.strip():
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                # ─── CORE RAG PIPELINE ───────────────────────────
                # 1. RETRIEVAL  → FAISS finds most relevant chunks
                # 2. AUGMENTED  → Chunks injected into the prompt
                # 3. GENERATION → GPT answers using that context
                # ─────────────────────────────────────────────────
                result = st.session_state.qa_chain.invoke({"query": user_question})

                answer = result["result"]
                source_docs = result.get("source_documents", [])

                # Save to session history
                st.session_state.chat_history.append((user_question, answer, source_docs))
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")

else:
    # Onboarding steps when no PDF is loaded
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📤 **Step 1:** Upload your PDF in the sidebar")
    with col2:
        st.info("⚡ **Step 2:** Click 'Process PDF' to index it")
    with col3:
        st.info("💬 **Step 3:** Ask any question about the content")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using LangChain · FAISS · OpenAI · Streamlit</small></center>",
    unsafe_allow_html=True
)
