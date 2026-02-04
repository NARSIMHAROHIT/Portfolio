# Project 4: Simple RAG Bot

**Phase 2: RAG Systems** | **Project 4/18** | **Difficulty: Intermediate**

## Overview

Complete Retrieval-Augmented Generation system that answers questions based on your documents. Supports PDF, TXT, and HTML files. Uses ChromaDB for vector storage and optionally integrates with Grok API for generation.

## What You'll Learn

- Complete RAG pipeline implementation
- Vector database integration (ChromaDB)
- Multi-format document loading
- Prompt engineering for RAG
- Context window management
- LLM API integration
- Evaluation of RAG responses

## Key Concepts

### RAG Pipeline

```
Documents (PDF/TXT/HTML)
    ↓
Load and Extract Text
    ↓
Chunk into Pieces (500 words, 50 overlap)
    ↓
Generate Embeddings (384d vectors)
    ↓
Store in ChromaDB
    ↓
[User asks question]
    ↓
Embed Question
    ↓
Search for Similar Chunks (top 3)
    ↓
Build Prompt: Context + Question
    ↓
Send to LLM (Grok or Mock)
    ↓
Return Answer to User
```

### Why RAG Works

- **Grounded Responses**: Answers based on your documents
- **Up-to-date**: Update documents, not retrain models
- **Explainable**: See which chunks were used
- **Cost-effective**: No model training needed
- **Flexible**: Works with any LLM

## Tech Stack

**Core Components:**
- sentence-transformers: Generate embeddings
- ChromaDB: Local vector database
- PyPDF2: PDF processing
- BeautifulSoup: HTML parsing

**LLM Integration:**
- Grok API: Text generation (optional)
- Mock mode: Testing without API

**Interfaces:**
- Streamlit: Web UI
- Typer: CLI

## Getting Started

### Installation

```bash
cd projects/04-simple-rag
pip install -r requirements.txt
```

### Quick Start - Web UI

```bash
streamlit run ui/streamlit_app.py
```

Steps:
1. Click "Initialize RAG System"
2. Upload documents (PDF/TXT/HTML)
3. Click "Process Documents"
4. Ask questions in chat
5. See retrieved context


```
04-simple-rag/
├── src/
│   ├── document_loader.py  # Load PDF/TXT/HTML (150 lines)
│   └── rag_engine.py       # Complete RAG system (250 lines)
│
├── ui/
│   └── streamlit_app.py    # Web interface (300 lines)
│
├── cli/
│   └── main.py             # CLI commands (150 lines)
│
├── data/
│   └── sample_doc.txt      # Sample document
│
└── chroma_db/              # ChromaDB storage (auto-created)
```

## Features

### Document Support

**Text Files (.txt)**
- Plain text documents
- UTF-8 and Latin-1 encoding

**PDF Files (.pdf)**
- Multi-page PDFs
- Text extraction
- Handles scanned PDFs (if text layer present)

**HTML Files (.html, .htm)**
- Web pages
- Extracts text, removes scripts/styles
- Cleans formatting

### RAG Features

**Chunking**
- 500 words per chunk
- 50 words overlap
- Prevents information loss at boundaries

**Embeddings**
- all-MiniLM-L6-v2 model
- 384 dimensions
- Fast inference

**Retrieval**
- Cosine similarity search
- Top-K results (default 3)
- Metadata tracking

**Generation**
- Grok API integration
- Context-aware prompting
- Mock mode for testing

## Usage Examples

### Python API

```python
from document_loader import DocumentLoader
from rag_engine import RAGEngine

# Initialize
loader = DocumentLoader()
rag = RAGEngine()

# Load document
doc = loader.load('report.pdf')

# Add to RAG
rag.add_documents([doc.to_dict()])

# Query
result = rag.query(
    "What are the main findings?",
    top_k=3,
    llm_provider='grok',
    api_key='your-key'
)

print(result['answer'])
print(f"Used {len(result['retrieved_chunks'])} chunks")
```


```
### Document Processing

1. **Load**: Extract text from file
2. **Chunk**: Split into 500-word pieces
3. **Embed**: Convert each chunk to 384d vector
4. **Store**: Save in ChromaDB with metadata

### Question Answering

1. **Embed Query**: Convert question to vector
2. **Search**: Find top-K similar chunks (cosine similarity)
3. **Build Context**: Combine retrieved chunks
4. **Prompt LLM**: 
   ```
   Context: [retrieved chunks]
   Question: [user question]
   Answer:
   ```
5. **Return**: LLM-generated answer + sources

### Prompt Template

```python
prompt = f"""Answer the question based on the context below. 
If you cannot answer based on the context, say 
"I don't have enough information to answer this question."

Context:
{retrieved_context}

Question: {user_question}

Answer:"""
```

## Evaluation

### Quality Metrics

**Retrieval Quality:**
- Are relevant chunks retrieved?
- Is important information included?

**Answer Quality:**
- Does answer use the context?
- Is it accurate?
- Does it avoid hallucination?

## Best Practices

### Document Preparation

- Clean PDFs (remove headers/footers)
- Use plain text when possible
- Structure documents with clear sections

### Chunking Strategy

- 500 words works for most documents
- Increase for technical documents
- Add overlap to preserve context

### Retrieval Tuning

- Start with top_k=3
- Increase if answers lack detail
- Decrease if too much irrelevant context

Be specific in questions
Include context in prompts
Tell LLM to say when it doesn't know

## Troubleshooting

PDF not loading
- Check if PDF has text layer
- Try OCR if scanned
- Convert to text first

**Poor retrieval**
- Check chunk size
- Verify documents loaded correctly
- Try different embedding model





**Current Version:**
- No conversation memory (stateless)
- Single vector space (no namespaces)
- Basic prompting (no advanced techniques)
- Local only (no cloud deployment)



\


This is the foundation for production RAG applications