# 📄 Chat with PDF — RAG Application

> A production-ready "Chat with your PDF" app built with **LangChain**, **FAISS**, **OpenAI**, and **Streamlit**.  
> Uses **Retrieval-Augmented Generation (RAG)** to answer questions grounded in your document.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧠 What is RAG?

**RAG (Retrieval-Augmented Generation)** solves a key limitation of LLMs:  
> *"The model doesn't know about your specific document."*

Instead of fine-tuning a model (expensive), RAG **retrieves** relevant information at query time and **injects** it into the prompt:

```
Your PDF ──► Text Chunks ──► Embeddings ──► FAISS Index
                                                  │
User Question ──────────────────────────► Similarity Search
                                                  │
                                         Top K Relevant Chunks
                                                  │
                                     [Question + Chunks] ──► GPT ──► Answer
```

---

## ✨ Features

- 📤 Upload any PDF directly from the browser
- 🔍 Semantic search using FAISS vector store
- 🤖 GPT-3.5-Turbo powered answers grounded in your document
- 📚 Source chunk viewer — see exactly which parts of the PDF were used
- ⚙️ Configurable chunk size, overlap, and top-K retrieval
- 💾 FAISS index saved to disk (no re-embedding on refresh)
- 🔐 API key input in sidebar (never stored or logged)

---

## 🗂️ Project Structure

```
chat_with_pdf/
│
├── app.py              # Streamlit UI & user interaction logic
├── utils.py            # RAG pipeline: PDF → Chunks → Embeddings → QA Chain
├── requirements.txt    # All Python dependencies
├── .env.example        # Template for your API key
├── .gitignore          # Prevents secrets/cache from being pushed
└── README.md           # You are here!
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/chat-with-pdf.git
cd chat-with-pdf
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Activate it:
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key
```bash
cp .env.example .env
```
Open `.env` and replace the placeholder:
```
OPENAI_API_KEY=sk-your-actual-key-here
```
Get your key from: https://platform.openai.com/api-keys

### 5. Run the App
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

---

## 📖 How to Use

1. **Upload** your PDF using the sidebar uploader
2. **Click "Process PDF"** — this extracts text, splits it into chunks, and creates embeddings
3. **Ask questions** in the chat box — the app finds relevant chunks and generates answers
4. **View Sources** — expand the source panel to see which chunks were retrieved

---

## 🔧 Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| Chunk Size | 500 | Characters per text chunk |
| Chunk Overlap | 50 | Characters shared between adjacent chunks |
| Top K Results | 4 | Number of chunks retrieved per query |

---

## ☁️ Deployment

### Deploy on Streamlit Cloud (Free & Recommended)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Connect your GitHub repo
4. Set `app.py` as the main file
5. Add your OpenAI key under **Settings → Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
6. Click **Deploy** — your app goes live in ~2 minutes!

### Deploy on Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** as the SDK
3. Upload all project files
4. Add `OPENAI_API_KEY` as a Space Secret

---

## 🏗️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI** | Streamlit | Web interface |
| **LLM** | OpenAI GPT-3.5-Turbo | Answer generation |
| **Embeddings** | OpenAI text-embedding-ada-002 | Text → Vectors |
| **Vector Store** | FAISS | Similarity search |
| **Orchestration** | LangChain | RAG pipeline glue |
| **PDF Parsing** | pdfplumber | Text extraction |

---

## 🔮 Future Improvements

- [ ] Support for multiple PDFs simultaneously
- [ ] Chat memory / multi-turn conversation
- [ ] Switch to local embeddings (HuggingFace) to reduce API costs
- [ ] Add a document summary feature
- [ ] Support for DOCX, TXT, and web URLs
- [ ] Use GPT-4 for more accurate answers
- [ ] Add re-ranking of retrieved chunks (Cohere Reranker)
- [ ] Streaming responses for better UX

---

## 💡 Interview Questions This Project Covers

1. What is RAG and why is it better than fine-tuning for Q&A?
2. What is a vector embedding and how does cosine similarity work?
3. Why do we split text into chunks? What's the tradeoff with chunk size?
4. What is FAISS and how does it find similar vectors efficiently?
5. What is LangChain and what problem does it solve?
6. What is the difference between `temperature=0` and `temperature=0.7`?
7. How would you handle a scanned (image-based) PDF?
8. How would you scale this to handle 1000 PDFs?

---

## 📄 License

MIT License — feel free to use, modify, and share.

---

## 🙋 Author

Built by [Your Name] · [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)
