# Hybrid RAG Intelligence

A PDF-based question-answering system that combines keyword search, semantic search, and reranking to retrieve relevant information and generate answers using an LLM API.

## Features

- PDF upload and saved PDF library
- PDF text extraction using PyMuPDF
- Text chunking with overlap
- Semantic search using Sentence Transformers and FAISS
- Keyword search using BM25
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Answer generation using Groq API
- Source filename and page references
- Interactive Streamlit chat interface

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- BM25
- Cross-Encoder
- Groq API

## Project Structure

```text
Hybrid rag/
├── app.py
├── src/
├── test/
├── data/
│   └── uploads/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

1. Clone or download this repository.
2. Create a virtual environment.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own API key. Never publish your API key.

5. Run the application:

```bash
streamlit run app.py
```

6. Open the local URL displayed in the terminal.
7. Upload a PDF and ask questions about its content.

## How It Works

PDF → Text Extraction → Chunking → BM25 + FAISS Search → RRF Fusion → Cross-Encoder Reranking → Groq Answer Generation

## Security

- API keys should remain in the local `.env` file.
- The `.env`, virtual environment, and private uploaded PDFs should not be committed to GitHub.

## Limitations

- Scanned PDFs may require OCR.
- Answer quality depends on PDF text quality and retrieval relevance.
- Groq API access requires a valid API key and network connectivity.
