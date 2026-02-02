# ✂️ Project 2: Smart Chunker

**Phase 1: Foundations** | **Project 2/18** | **Difficulty: Beginner-Intermediate**

## 📖 Overview

Smart Chunker is an interactive tool for comparing and evaluating different text chunking strategies. Learn how to split documents optimally for RAG systems, embeddings, and LLM applications.

## 🎯 What You'll Learn

- ✅ What chunking is and why it matters
- ✅ Four different chunking strategies
- ✅ How to evaluate chunk quality
- ✅ Trade-offs between strategies
- ✅ Optimal parameters for your use case
- ✅ Preparation for RAG systems (Project 4+)

## 🧠 Key Concepts

### What is Chunking?

Chunking is breaking large documents into smaller, meaningful pieces. Essential for:
- **Token Limits**: LLMs can't process infinite text
- **Better Embeddings**: Focused text = clearer meaning
- **Accurate Retrieval**: Find exact relevant sections
- **Cost Optimization**: Process only what's needed

### The Four Strategies

1. **Fixed-Size** - Split every N characters
   - ✅ Simple, predictable
   - ❌ Can break sentences

2. **Recursive** - Split by separators (paragraphs → sentences → words)
   - ✅ Respects structure
   - ❌ Variable sizes

3. **Semantic** - Split based on meaning using embeddings
   - ✅ Best quality
   - ❌ Slower

4. **Sliding Window** - Fixed-size with overlap
   - ✅ No information loss
   - ❌ Duplicate content

## 🛠️ What You'll Build

### Features

1. **Multiple Strategies**
   - Implement all 4 chunking methods
   - Compare side-by-side
   - Adjustable parameters

2. **Quality Evaluation**
   - Coherence scoring
   - Consistency metrics
   - Size distribution analysis

3. **Interactive Visualizations**
   - Chunk size distributions
   - Embedding scatter plots
   - Comparison dashboards

4. **Both UI and CLI**
   - Streamlit web interface
   - Command-line tools

## 📚 Tech Stack

- **Core**: Python 3.9+
- **Embeddings**: sentence-transformers
- **Evaluation**: cosine similarity, statistical analysis
- **Visualization**: Plotly, UMAP
- **UI**: Streamlit
- **CLI**: Typer with Rich

## 🚀 Getting Started

### Installation

```bash
cd projects/02-smart-chunker
pip install -r requirements.txt
```

### Quick Start - Web UI

```bash
streamlit run ui/streamlit_app.py
```

Then:
1. Click "Load Models" in sidebar
2. Use sample text or paste your own
3. Select strategies to compare
4. Click "Chunk Text"
5. Explore results and visualizations!

### Quick Start - CLI

```bash
# Chunk a file
python cli/main.py chunk my_document.txt --strategy recursive

# Compare all strategies
python cli/main.py compare my_document.txt

# Evaluate specific strategy
python cli/main.py evaluate my_document.txt --strategy semantic

# List available strategies
python cli/main.py strategies
```

## 📁 Project Structure

```
02-smart-chunker/
├── src/
│   ├── chunker.py          # Core chunking logic (400 lines)
│   ├── evaluator.py        # Quality evaluation (300 lines)
│   └── visualizer.py       # Plotting functions (250 lines)
│
├── ui/
│   └── streamlit_app.py    # Web interface (400 lines)
│
├── cli/
│   └── main.py             # CLI commands (200 lines)
│
├── tests/
│   └── test_chunker.py     # Unit tests
│
├── data/
│   └── sample.txt          # Sample text
│
└── docs/
    └── LEARNINGS.md        # Your learning journal
```



## 📊 Evaluation Metrics

### Coherence (0-1, higher is better)
How semantically related are sentences within chunks?
- Good chunk: sentences discuss same topic
- Bad chunk: multiple unrelated topics

### Consistency (0-1, higher is better)
How uniform are chunk sizes?
- Good: consistent sizes, easier to manage
- Bad: wildly varying sizes

### Overlap Ratio (0-1)
How much content is shared between chunks?
- Good for critical docs: 10-20%
- Too high: wasted storage

## 🎯 Best Practices You'll Learn

1. **Chunk Size: 300-800 tokens**
   - Small enough for focused meaning
   - Large enough for context

2. **Overlap: 10-20%**
   - Prevents information loss
   - Balances redundancy vs completeness

3. **Respect Boundaries**
   - Don't split mid-sentence
   - Preserve structure (headers, lists)

4. **Strategy Selection**
   - **General docs**: Recursive
   - **Best quality**: Semantic
   - **Speed**: Fixed-size
   - **Critical**: Sliding window

## 💡 Example Workflow

```python
# 1. Initialize chunker
from chunker import Chunker

chunker = Chunker(embedding_model='all-MiniLM-L6-v2')

# 2. Load your document
text = open('my_document.txt').read()

# 3. Try different strategies
recursive_result = chunker.chunk(text, strategy='recursive', chunk_size=400)
semantic_result = chunker.chunk(text, strategy='semantic', chunk_size=400)

# 4. Evaluate
from evaluator import ChunkingEvaluator

evaluator = ChunkingEvaluator()
metrics = evaluator.evaluate_chunking([c.text for c in recursive_result.chunks])

print(f"Coherence: {metrics['coherence']:.3f}")
```

## 🔍 Advanced Features

### Custom Strategies
Add your own chunking logic:

```python
def my_custom_chunker(text, chunk_size):
    # Your logic here
    pass
```

### Metadata Tracking
Each chunk includes:
- Start/end position
- Chunk ID
- Custom metadata

### Integration with Project 1
Use embeddings from Project 1 to evaluate chunks!

By the end, you should be able to:
- [ ] Explain 4 chunking strategies
- [ ] Choose appropriate strategy for your use case
- [ ] Evaluate chunk quality quantitatively
- [ ] Use both UI and CLI
- [ ] Understand preparation for RAG

##  Troubleshooting

**Model download is slow**
- First time only, models are cached

**High memory usage**
- Use smaller batch for semantic chunking
- Limit number of chunks for visualization



- Retrieval works on chunks, not documents



After completing this project:
- Working chunker with 4 strategies
- Evaluation metrics implementation
- Interactive visualizations
- Both UI and CLI
- Understanding of optimal chunking
- Documentation of learnings




---

Remember**: Good chunking is the foundation of good RAG systems!

