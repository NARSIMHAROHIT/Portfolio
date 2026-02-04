# 🔍 Project 3: Similarity Search Engine

**Phase 1: Foundations** | **Project 3/18** | **Difficulty: Beginner-Intermediate**

## Overview

Build a production-ready similarity search engine that finds relevant documents using semantic understanding, keyword matching, or hybrid approaches. This is the final foundation project before building complete RAG systems.

- Vector similarity search algorithms
- Multiple search strategies (semantic, keyword, hybrid)
- Search result ranking and re-ranking
- Performance metrics (precision, recall, MRR, NDCG)
- Building search interfaces
- Preparing for RAG retrieval (Project 4)

### Similarity Search

Finding documents similar to a query based on:
- **Semantic similarity**: Understanding meaning through embeddings
- **Keyword matching**: Traditional TF-IDF based search
- **Hybrid**: Best of both worlds

### Why It Matters

Critical for:
- **RAG Systems**: Retrieval is the first step
- **Recommendation Engines**: Find similar items
- **Document Discovery**: Explore related content
- **Question Answering**: Find relevant context


### Features

1. **Three Search Methods**
   - Semantic search (embeddings)
   - Keyword search (TF-IDF)
   - Hybrid search (combined)

2. **Advanced Ranking**
   - Score-based ranking
   - Diversity promotion
   - Reciprocal Rank Fusion
   - Custom ranking functions

3. **Performance Metrics**
   - Precision@K
   - Recall@K
   - Mean Reciprocal Rank (MRR)
   - Normalized Discounted Cumulative Gain (NDCG)

4. **Interactive Interface**
   - Web UI with Streamlit
   - CLI for batch processing
   - Search history tracking

## Tech Stack

- **Embeddings**: sentence-transformers 
- **Keyword Search**: TF-IDF (scikit-learn)
- **Similarity Metrics**: cosine, euclidean, dot product
- **UI**: Streamlit
- **CLI**: Typer with Rich

### Installation

```bash
cd projects/03-similarity-search
pip install -r requirements.txt
```

### Quick Start - Web UI

```bash
streamlit run ui/streamlit_app.py
```

Then:
1. Initialize Engine
2. Load sample documents or upload your own
3. Index documents
4. Start searching!

### Quick Start - CLI

```bash
# Search documents
python cli/main.py search "machine learning" docs.txt

# Compare all methods
python cli/main.py compare "neural networks" docs.txt

# Find similar documents
python cli/main.py similar 0 docs.txt

# List methods
python cli/main.py methods
```

## 📁 Project Structure

```
03-similarity-search/
├── src/
│   ├── search_engine.py    # Core search engine (400 lines)
│   └── ranker.py           # Advanced ranking (350 lines)
│
├── ui/
│   └── streamlit_app.py    # Web interface (350 lines)
│
├── cli/
│   └── main.py             # CLI commands (150 lines)
│
├── tests/
│   └── test_search.py      # Unit tests
│
└── data/
    └── sample_docs.txt     # Sample documents
```

## 🎓 Learning Path

### Step 1: Understand Search Types (30 min)
- Semantic vs keyword search
- When to use each
- Hybrid approaches

### Step 2: Explore with UI (1 hour)
- Load documents
- Try different search methods
- Compare results
- Understand scoring

### Step 3: Advanced Features (1 hour)
- Result ranking
- Filtering
- Performance metrics
- Similar document search

### Step 4: CLI Usage (30 min)
- Batch searching
- Automation
- Integration workflows

### Step 5: Code Deep Dive (1 hour)
- Read search_engine.py
- Understand ranking algorithms
- Study metrics calculation

## 📊 Search Methods Explained

### Semantic Search

**How it works:**
```python
1. Embed query → vector
2. Embed all documents → vectors (done once)
3. Calculate cosine similarity
4. Return top-K most similar
```

**Best for:**
- Natural language queries
- Conceptual searches
- Paraphrased questions

**Example:**
```
Query: "What is AI?"
Matches: "Artificial intelligence is..." (semantic match)
```

### Keyword Search

**How it works:**
```python
1. Build TF-IDF matrix from documents
2. Transform query using same vocabulary
3. Calculate cosine similarity
4. Return top matches
```

**Best for:**
- Specific terms
- Technical jargon
- Exact phrase matching

**Example:**
```
Query: "neural network architecture"
Matches: Documents with these exact terms
```

### Hybrid Search

**How it works:**
```python
1. Run both semantic and keyword search
2. Normalize scores to 0-1
3. Combine: 0.7 * semantic + 0.3 * keyword
4. Return merged results
```

**Best for:**
- General purpose
- Production systems
- Robust performance

## 📈 Performance Metrics

### Precision@K

What percentage of top-K results are relevant?

```
Precision@5 = Relevant in top 5 / 5
```

### Recall@K

What percentage of all relevant docs are in top-K?

```
Recall@5 = Relevant in top 5 / Total relevant
```

### Mean Reciprocal Rank (MRR)

Average of reciprocal ranks of first relevant result.

```
MRR = 1/rank of first relevant result
```

### NDCG@K

Normalized discounted cumulative gain (considers ranking position).

##  Example Usage

```python
from search_engine import SearchEngine

# Initialize
engine = SearchEngine()

# Index documents
documents = [
    "Machine learning is AI subset",
    "Deep learning uses neural networks",
    "NLP processes human language"
]

engine.index_documents(documents)

# Search
results = engine.search(
    "What is ML?",
    method='semantic',
    top_k=2
)

for result in results.results:
    print(f"{result.rank}. {result.text} (score: {result.score:.3f})")
```

## Advanced Features

### Similar Document Search

```python
# Find documents similar to document 0
results = engine.find_similar_documents(doc_id=0, top_k=5)
```

### Filtered Search

```python
# Search with metadata filters
results = engine.search(
    query="machine learning",
    filters={'category': 'tutorial', 'level': 'beginner'}
)
```

### Reciprocal Rank Fusion

```python
from ranker import SearchRanker

ranker = SearchRanker()

# Combine results from multiple methods
semantic_results = engine.search(query, method='semantic')
keyword_results = engine.search(query, method='keyword')

fused = ranker.reciprocal_rank_fusion([
    semantic_results.results,
    keyword_results.results
])
```

## Success Criteria

By the end, you should:
- Understand three search methods
- Know when to use each
- Calculate performance metrics
- Build search interfaces
- Prepare for RAG systems

## Pipeline Connection

```
Project 1 (Embeddings) → Project 2 (Chunking) → Project 3 (Search) → Project 4 (RAG)
     Generate vectors       Split docs            Find similar        Complete
                                                  chunks              system
```

This project completes the foundation for RAG by teaching retrieval!

## 🚧 Troubleshooting

**Slow indexing**
- Normal on first run (embedding generation)
- Subsequent searches are fast

**Poor search results**
- Try different methods (semantic vs keyword vs hybrid)
- Adjust chunk size in Project 2
- Use more documents for better coverage

**Out of memory**
- Process documents in batches
- Use smaller embedding model



After completing:
- Working search engine with 3 methods
- Interactive UI and CLI
- Performance metrics implementation
- Understanding of search algorithms
- Ready for RAG (Project 4)


