"""
utils.py - Helper Functions for the RAG Pipeline
==================================================
This module contains all the "behind the scenes" logic.
Think of it as the engine — app.py is just the dashboard.

The RAG Pipeline has 5 key steps:
  1. load_pdf()         → Extract raw text from PDF
  2. split_text()       → Break text into chunks
  3. create_vector_store() → Embed chunks + store in FAISS
  4. save/load_vector_store() → Persist index to disk
  5. create_qa_chain()  → Build the LangChain QA chain
"""

import os
import pickle
from io import BytesIO

# PDF reading
import pdfplumber

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
VECTOR_STORE_PATH = "faiss_index"   # Folder where FAISS saves its index files


# ─────────────────────────────────────────────
# STEP 1: Load PDF
# ─────────────────────────────────────────────
def load_pdf(uploaded_file) -> str:
    """
    Extract all text from an uploaded PDF file.

    Args:
        uploaded_file: A Streamlit UploadedFile object (file-like object)

    Returns:
        raw_text (str): All text extracted from the PDF, page by page.

    Why pdfplumber?
        It handles complex PDFs (tables, columns) better than PyPDF2.
    """
    raw_text = ""

    # Read the uploaded file bytes into a BytesIO buffer
    pdf_bytes = BytesIO(uploaded_file.read())

    with pdfplumber.open(pdf_bytes) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:  # Some pages might be blank or image-only
                raw_text += f"\n--- Page {page_num + 1} ---\n"
                raw_text += page_text

    if not raw_text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It might be a scanned image PDF. Please use a text-based PDF."
        )

    return raw_text


# ─────────────────────────────────────────────
# STEP 2: Split Text into Chunks
# ─────────────────────────────────────────────
def split_text(raw_text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Split a large text into smaller overlapping chunks.

    Why do we split?
        LLMs have a limited context window (e.g., 4096 tokens).
        We can't send a 100-page PDF as a single prompt.
        Instead, we find only the relevant chunks and send those.

    Why overlap?
        If a sentence is cut at the boundary of chunk 1 and chunk 2,
        the overlap ensures neither chunk loses that sentence entirely.

    Args:
        raw_text (str): The full extracted text
        chunk_size (int): Max characters per chunk (default: 500)
        chunk_overlap (int): Characters shared between adjacent chunks (default: 50)

    Returns:
        List of LangChain Document objects (each has .page_content)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try splitting on paragraphs → sentences → words → characters
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.create_documents([raw_text])
    print(f"[INFO] Created {len(chunks)} chunks from {len(raw_text)} characters.")
    return chunks


# ─────────────────────────────────────────────
# STEP 3: Create Vector Store (FAISS)
# ─────────────────────────────────────────────
def create_vector_store(chunks):
    """
    Convert text chunks into vector embeddings and store them in FAISS.

    What is an embedding?
        A vector (list of ~1536 numbers) that represents the MEANING of a text.
        Similar meaning → similar vectors → close together in vector space.

    What is FAISS?
        Facebook AI Similarity Search — an ultra-fast library for finding
        the most similar vectors. It's like a search engine for meaning.

    What is OpenAIEmbeddings?
        The model that converts text → embedding vectors.
        We use "text-embedding-ada-002" by default (cheap and powerful).

    Args:
        chunks: List of LangChain Document objects

    Returns:
        FAISS vector store object
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API Key not found. Set it in .env or the sidebar.")

    # Initialize the embedding model
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-ada-002"  # ~$0.0001 per 1K tokens — very cheap
    )

    # Create FAISS index from our document chunks
    # Under the hood: each chunk gets embedded → stored in FAISS
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print(f"[INFO] FAISS index created with {len(chunks)} vectors.")
    return vector_store


# ─────────────────────────────────────────────
# STEP 4a: Save Vector Store to Disk
# ─────────────────────────────────────────────
def save_vector_store(vector_store, path: str = VECTOR_STORE_PATH):
    """
    Save the FAISS index to disk so we don't have to re-embed every time.

    This creates two files:
      - faiss_index/index.faiss  (the actual index)
      - faiss_index/index.pkl    (metadata like document text)

    Args:
        vector_store: FAISS object to save
        path (str): Directory to save the index
    """
    vector_store.save_local(path)
    print(f"[INFO] Vector store saved to '{path}/'")


# ─────────────────────────────────────────────
# STEP 4b: Load Vector Store from Disk
# ─────────────────────────────────────────────
def load_vector_store(path: str = VECTOR_STORE_PATH):
    """
    Load a previously saved FAISS index from disk.

    Useful for:
      - Resuming without re-processing the PDF
      - Caching expensive embedding calls

    Args:
        path (str): Directory where the index was saved

    Returns:
        FAISS vector store object (or None if not found)
    """
    if not os.path.exists(path):
        print(f"[WARNING] No saved index found at '{path}/'")
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-ada-002"
    )

    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  # Required flag for loading pickled FAISS
    )
    print(f"[INFO] Vector store loaded from '{path}/'")
    return vector_store


# ─────────────────────────────────────────────
# STEP 5: Create QA Chain
# ─────────────────────────────────────────────
def create_qa_chain(vector_store, top_k: int = 4):
    """
    Build the LangChain RetrievalQA chain.

    This chain does the full RAG loop in one call:
      1. Takes a user question
      2. Converts it to an embedding
      3. Searches FAISS for the top_k most similar chunks
      4. Builds a prompt: [System Instructions + Retrieved Chunks + User Question]
      5. Sends the prompt to GPT-3.5-Turbo
      6. Returns the generated answer + the source documents

    Args:
        vector_store: FAISS vector store with embedded chunks
        top_k (int): Number of chunks to retrieve per query

    Returns:
        A LangChain RetrievalQA chain object
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    # ── LLM: GPT-3.5-Turbo ─────────────────────────────────────────
    # temperature=0 → deterministic, factual answers (no creativity)
    # For creative tasks, use temperature=0.7 or higher
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0
    )

    # ── Custom Prompt Template ───────────────────────────────────────
    # This tells the LLM HOW to use the retrieved context.
    # {context} → replaced with retrieved chunks at runtime
    # {question} → replaced with user's question at runtime
    prompt_template = """You are a helpful assistant that answers questions based on the provided document.

Use ONLY the information from the context below to answer the question.
If the answer is not in the context, say "I couldn't find this information in the document."
Be concise, accurate, and helpful.

Context (extracted from the document):
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # ── Retriever ────────────────────────────────────────────────────
    # Converts FAISS into a LangChain Retriever object
    # search_type="similarity" → cosine similarity search
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    # ── RetrievalQA Chain ────────────────────────────────────────────
    # chain_type="stuff" → "stuff" all retrieved chunks into one prompt
    # Other options: "map_reduce", "refine" (for very long docs)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,  # Also return which chunks were used
        chain_type_kwargs={"prompt": PROMPT}
    )

    print("[INFO] QA chain created successfully.")
    return qa_chain
