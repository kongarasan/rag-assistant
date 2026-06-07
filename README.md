# RAG Assistant

A full-stack Retrieval-Augmented Generation (RAG) application that lets you upload a PDF and ask questions about it. Built with LangChain, Groq, FAISS, FastAPI, and React.

## What it does

- Upload any PDF document
- Ask natural language questions about it
- Get accurate, grounded answers powered by LLM + vector search

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (LLaMA 3.3 70B) |
| Orchestration | LangChain |
| Vector Store | FAISS |
| Backend | FastAPI |
| Frontend | React |

## Project Structure

```
rag-assistant/
├── test.py          # Day 1: first LLM call
├── loader.py        # Day 2: PDF loading + chunking
├── embeddings.py    # Day 3: embeddings + vector store
├── rag.py           # Day 4: full RAG chain
├── main.py          # Day 5: FastAPI backend
├── frontend/        # Day 6: React frontend
├── requirements.txt
└── .env             # API keys (not committed)
```

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+ (for frontend)
- Free [Groq API key](https://console.groq.com)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-assistant.git
cd rag-assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo GROQ_API_KEY=your-key-here > .env
```

### Test the LLM connection

```bash
python test.py
```

## Roadmap

- [x] Day 1 — LLM connection via Groq
- [x] Day 2 — PDF loading + chunking
- [x] Day 3 — Embeddings + FAISS vector store
- [ ] Day 4 — Full RAG chain
- [ ] Day 5 — FastAPI backend
- [ ] Day 6 — React frontend
- [ ] Day 7 — Deploy to Render + Vercel

## License

MIT
