"""
Streamlit UI for Similarity Search Engine
Interactive search interface with multiple search methods
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json

sys.path.append(str(Path(__file__).parent.parent / "src"))

from search_engine import SearchEngine
from ranker import SearchRanker, ResultFilter

st.set_page_config(
    page_title="Similarity Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #4ECDC4 0%, #556270 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .result-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []


def load_sample_documents():
    return [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Reinforcement learning trains agents to make decisions through trial and error.",
        "Supervised learning uses labeled data to train models for classification and regression tasks.",
        "Unsupervised learning finds patterns in data without explicit labels or guidance.",
        "Transfer learning allows models trained on one task to be adapted for related tasks.",
        "Python is a popular programming language for data science and machine learning applications.",
        "TensorFlow and PyTorch are leading frameworks for building deep learning models.",
        "The Transformer architecture revolutionized natural language processing in 2017.",
        "GPT models use transformers to generate human-like text based on input prompts.",
        "BERT introduced bidirectional training of transformers for better language understanding.",
        "Convolutional neural networks excel at processing grid-like data such as images.",
        "Recurrent neural networks are designed to handle sequential data like time series.",
        "Attention mechanisms help models focus on relevant parts of the input data.",
        "Embeddings convert categorical data into dense vector representations.",
        "Gradient descent is an optimization algorithm used to train neural networks.",
        "Overfitting occurs when a model learns training data too well and fails on new data.",
        "Cross-validation helps assess model performance and prevent overfitting."
    ]


def main():
    init_session_state()
    
    st.markdown('<h1 class="main-header">🔍 Similarity Search Engine</h1>', 
                unsafe_allow_html=True)
    st.markdown("*Find similar content using semantic and keyword search*")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if st.button("🚀 Initialize Engine", type="primary"):
            with st.spinner("Loading model..."):
                st.session_state.engine = SearchEngine()
            st.success("Engine ready!")
        
        st.markdown("---")
        
        st.subheader("📚 Document Management")
        
        doc_source = st.radio(
            "Document Source",
            ["Sample Documents", "Upload File", "Paste Text"]
        )
        
        if doc_source == "Sample Documents":
            if st.button("Load Sample Docs"):
                st.session_state.documents = load_sample_documents()
                st.success(f"Loaded {len(st.session_state.documents)} documents")
        
        elif doc_source == "Upload File":
            uploaded = st.file_uploader("Upload text file (one doc per line)", type=['txt'])
            if uploaded:
                content = uploaded.read().decode('utf-8')
                st.session_state.documents = [
                    line.strip() for line in content.split('\n') if line.strip()
                ]
                st.success(f"Loaded {len(st.session_state.documents)} documents")
        
        else:
            docs_text = st.text_area("Paste documents (one per line)", height=150)
            if st.button("Load Documents"):
                st.session_state.documents = [
                    line.strip() for line in docs_text.split('\n') if line.strip()
                ]
                st.success(f"Loaded {len(st.session_state.documents)} documents")
        
        if st.session_state.documents and st.session_state.engine:
            if st.button("📇 Index Documents"):
                with st.spinner("Indexing..."):
                    st.session_state.engine.index_documents(st.session_state.documents)
                st.success("Indexing complete!")
        
        st.markdown("---")
        
        if st.session_state.engine:
            stats = st.session_state.engine.get_statistics()
            if stats.get('indexed'):
                st.subheader("📊 Statistics")
                st.metric("Documents", stats['num_documents'])
                st.metric("Dimensions", stats['embedding_dimensions'])
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Search",
        "📊 Compare Methods",
        "📈 Analytics",
        "📚 Learn"
    ])
    
    with tab1:
        st.header("Search Documents")
        
        if not st.session_state.engine or not st.session_state.documents:
            st.warning("Please initialize engine and load documents first!")
            return
        
        stats = st.session_state.engine.get_statistics()
        if not stats.get('indexed'):
            st.warning("Please index documents first!")
            return
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            query = st.text_input(
                "Enter your search query:",
                placeholder="What is machine learning?"
            )
        
        with col2:
            method = st.selectbox(
                "Method",
                ["semantic", "keyword", "hybrid"]
            )
        
        with col3:
            top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Searching..."):
                    results = st.session_state.engine.search(
                        query,
                        method=method,
                        top_k=top_k
                    )
                    
                    st.session_state.search_history.insert(0, {
                        'query': query,
                        'method': method,
                        'results': len(results.results)
                    })
                
                st.markdown("---")
                st.subheader(f"📋 Results ({len(results.results)})")
                
                if results.results:
                    for result in results.results:
                        with st.container():
                            col1, col2 = st.columns([0.9, 0.1])
                            
                            with col1:
                                st.markdown(
                                    f'<div class="result-card">'
                                    f'<strong>Rank {result.rank}</strong> | '
                                    f'Score: {result.score:.4f}<br>'
                                    f'{result.text}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            
                            with col2:
                                st.metric("Doc ID", result.doc_id)
                else:
                    st.info("No results found")
            else:
                st.error("Please enter a query")
        
        if st.session_state.search_history:
            st.markdown("---")
            st.subheader("🕐 Recent Searches")
            
            history_df = pd.DataFrame(st.session_state.search_history[:5])
            st.dataframe(history_df, use_container_width=True)
    
    with tab2:
        st.header("Compare Search Methods")
        
        if not st.session_state.engine or not st.session_state.documents:
            st.warning("Please initialize engine and load documents first!")
            return
        
        stats = st.session_state.engine.get_statistics()
        if not stats.get('indexed'):
            st.warning("Please index documents first!")
            return
        
        query = st.text_input(
            "Enter query to compare:",
            placeholder="artificial intelligence",
            key="compare_query"
        )
        
        if st.button("🔄 Compare All Methods"):
            if query.strip():
                methods = ['semantic', 'keyword', 'hybrid']
                
                cols = st.columns(3)
                
                for idx, method in enumerate(methods):
                    with cols[idx]:
                        st.subheader(f"{method.title()} Search")
                        
                        results = st.session_state.engine.search(
                            query,
                            method=method,
                            top_k=5
                        )
                        
                        for result in results.results:
                            st.write(f"**{result.rank}.** ({result.score:.3f})")
                            st.caption(result.text[:100] + "...")
                            st.markdown("---")
            else:
                st.error("Please enter a query")
    
    with tab3:
        st.header("Search Analytics")
        
        if not st.session_state.search_history:
            st.info("No search history yet. Perform some searches first!")
            return
        
        st.subheader("📊 Search Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Searches", len(st.session_state.search_history))
        
        with col2:
            avg_results = sum(h['results'] for h in st.session_state.search_history) / len(st.session_state.search_history)
            st.metric("Avg Results", f"{avg_results:.1f}")
        
        with col3:
            methods_used = [h['method'] for h in st.session_state.search_history]
            most_common = max(set(methods_used), key=methods_used.count)
            st.metric("Most Used Method", most_common)
        
        st.markdown("---")
        st.subheader("📈 Search History")
        
        history_df = pd.DataFrame(st.session_state.search_history)
        st.dataframe(history_df, use_container_width=True)
    
    with tab4:
        st.header("📚 Learn About Similarity Search")
        
        st.markdown("""
        ### What is Similarity Search?
        
        Similarity search finds documents that are similar to a query based on meaning or content.
        Unlike traditional keyword search, it understands semantics and context.
        
        ### Search Methods
        """)
        
        with st.expander("🧠 Semantic Search"):
            st.markdown("""
            **How it works:** Uses embeddings to find similar meaning
            
            **Example:**
            - Query: "What is ML?"
            - Matches: "Machine learning is..." (different words, same meaning)
            
            **Advantages:**
            - Understands synonyms and context
            - Works with paraphrased queries
            - Best for conceptual searches
            
            **Best for:** Natural language queries, conceptual understanding
            """)
        
        with st.expander("🔤 Keyword Search"):
            st.markdown("""
            **How it works:** Uses TF-IDF to match keywords
            
            **Example:**
            - Query: "machine learning"
            - Matches: Documents containing these exact terms
            
            **Advantages:**
            - Fast and efficient
            - Good for specific terms
            - Interpretable results
            
            **Best for:** Specific terms, technical searches
            """)
        
        with st.expander("🔄 Hybrid Search"):
            st.markdown("""
            **How it works:** Combines semantic and keyword search
            
            **Example:**
            - Query: "neural network architecture"
            - Combines: Semantic understanding + keyword matching
            
            **Advantages:**
            - Best of both worlds
            - Robust to different query types
            - Balanced results
            
            **Best for:** General purpose, production systems
            """)
        
        st.markdown("---")
        st.markdown("""
        ### Similarity Metrics
        
        **Cosine Similarity:**
        - Measures angle between vectors
        - Range: -1 to 1 (higher is more similar)
        - Best for: Text similarity
        
        **Euclidean Distance:**
        - Straight-line distance
        - Lower distance = more similar
        - Best for: When magnitude matters
        
        **Dot Product:**
        - Raw vector multiplication
        - Higher = more similar
        - Best for: Unnormalized vectors
        """)


if __name__ == "__main__":
    main()