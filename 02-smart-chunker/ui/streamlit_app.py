"""
Streamlit UI for Smart Chunker
Interactive tool for comparing text chunking strategies
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chunker import Chunker
from evaluator import ChunkingEvaluator
from visualizer import ChunkingVisualizer

# Page config
st.set_page_config(
    page_title="Smart Chunker",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chunk-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    if 'chunker' not in st.session_state:
        st.session_state.chunker = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'comparison' not in st.session_state:
        st.session_state.comparison = None


def load_sample_text():
    """Load sample text for demo"""
    return """Machine learning is a subset of artificial intelligence that focuses on developing systems that can learn from data. Unlike traditional programming where rules are explicitly coded, machine learning algorithms identify patterns and make decisions with minimal human intervention.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to learn the relationship between input and output. The algorithm is trained on examples where the correct answer is known, allowing it to learn patterns that can be applied to new, unseen data.

Unsupervised learning, on the other hand, works with unlabeled data. The algorithm tries to find hidden patterns or structures in the data without being told what to look for. Common applications include clustering similar items together and dimensionality reduction.

Reinforcement learning is inspired by behavioral psychology. An agent learns to make decisions by performing actions in an environment and receiving rewards or penalties. Over time, the agent learns which actions yield the best outcomes. This approach has been particularly successful in game playing and robotics.

Deep learning is a subset of machine learning that uses neural networks with multiple layers. These deep neural networks can automatically learn hierarchical representations of data, making them particularly effective for complex tasks like image recognition, natural language processing, and speech recognition.

The success of machine learning depends heavily on the quality and quantity of training data. More data generally leads to better performance, but the data must be representative of the real-world scenarios where the model will be applied. Feature engineering, the process of selecting and transforming raw data into features that better represent the underlying problem, is also crucial for traditional machine learning approaches.

Modern machine learning frameworks like TensorFlow, PyTorch, and scikit-learn have made it easier than ever to implement and experiment with machine learning algorithms. Cloud platforms provide the computational resources needed to train large models, democratizing access to advanced AI capabilities.

However, machine learning also raises important ethical considerations. Bias in training data can lead to discriminatory outcomes, and the black-box nature of some algorithms makes it difficult to understand how decisions are made. As machine learning systems become more prevalent in critical applications like healthcare and criminal justice, addressing these concerns becomes increasingly important."""


def main():
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">Smart Chunker</h1>', unsafe_allow_html=True)
    st.markdown("Compare and evaluate different text chunking strategies*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Initialize models
        if st.button(" Load Models", type="primary"):
            with st.spinner("Loading models..."):
                st.session_state.chunker = Chunker(embedding_model='all-MiniLM-L6-v2')
                st.session_state.evaluator = ChunkingEvaluator('all-MiniLM-L6-v2')
            st.success("Models loaded!")
        
        st.markdown("---")
        
        # Chunking parameters
        st.subheader(" Parameters")
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=1000,
            value=400,
            step=50,
            help="Target size for each chunk in characters"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=200,
            value=50,
            step=10,
            help="Overlap between consecutive chunks"
        )
        
        st.markdown("---")
        
        # Strategy selection
        st.subheader("🎯 Strategies")
        
        strategies_info = Chunker.get_available_strategies()
        selected_strategies = []
        
        for strategy, description in strategies_info.items():
            if st.checkbox(strategy.title(), value=True, help=description):
                selected_strategies.append(strategy)
        
        st.markdown("---")
        
        # Stats
        if st.session_state.results:
            st.subheader("Quick Stats")
            for strategy, result in st.session_state.results.items():
                st.metric(
                    f"{strategy.title()}",
                    f"{result.num_chunks} chunks",
                    f"{result.avg_chunk_size:.0f} chars avg"
                )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        " Chunk Text",
        " Compare Strategies",
        " Visualize",
        " Learn"
    ])
    
    # Tab 1: Chunk Text
    with tab1:
        st.header("Chunk Your Text")
        
        if st.session_state.chunker is None:
            st.warning(" Please load models from the sidebar first!")
            return
        
        # Input options
        input_method = st.radio(
            "Input Method",
            ["Paste Text", "Use Sample", "Upload File"],
            horizontal=True
        )
        
        if input_method == "Paste Text":
            text = st.text_area(
                "Enter your text:",
                height=200,
                placeholder="Paste your text here..."
            )
        elif input_method == "Use Sample":
            text = load_sample_text()
            st.info(" Using sample text about machine learning")
        else:
            uploaded_file = st.file_uploader("Upload text file", type=['txt'])
            text = uploaded_file.read().decode('utf-8') if uploaded_file else ""
        
        # Chunk button
        if st.button(" Chunk Text", type="primary", use_container_width=True):
            if not text.strip():
                st.error("Please provide some text!")
                return
            
            st.session_state.results = {}
            
            # Process each strategy
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, strategy in enumerate(selected_strategies):
                status_text.text(f"Processing {strategy}...")
                
                result = st.session_state.chunker.chunk(
                    text,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                st.session_state.results[strategy] = result
                progress_bar.progress((i + 1) / len(selected_strategies))
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"Chunked with {len(selected_strategies)} strategies!")
        
        # Display results
        if st.session_state.results:
            st.markdown("---")
            st.subheader("Chunking Results")
            
            # Summary table
            summary_data = []
            for strategy, result in st.session_state.results.items():
                summary_data.append({
                    'Strategy': strategy.title(),
                    'Chunks': result.num_chunks,
                    'Avg Size': f"{result.avg_chunk_size:.0f}",
                    'Total Chars': result.total_chars
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # Show chunks for selected strategy
            st.markdown("---")
            st.subheader("View Chunks")
            
            view_strategy = st.selectbox(
                "Select strategy to view:",
                list(st.session_state.results.keys()),
                format_func=lambda x: x.title()
            )
            
            result = st.session_state.results[view_strategy]
            
            for i, chunk in enumerate(result.chunks):
                with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
                    st.markdown(f'<div class="chunk-box">{chunk.text}</div>', 
                              unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"Start: {chunk.start_idx}")
                    with col2:
                        st.caption(f"End: {chunk.end_idx}")
                    with col3:
                        st.caption(f"Length: {len(chunk)}")
    
    # Tab 2: Compare Strategies
    with tab2:
        st.header("Compare Chunking Strategies")
        
        if not st.session_state.results:
            st.info("Chunk some text first to see comparisons!")
            return
        
        if st.session_state.evaluator is None:
            st.warning("Please load models from the sidebar first!")
            return
        
        # Evaluate button
        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Evaluating strategies..."):
                # Prepare chunks for evaluation
                chunks_dict = {
                    strategy: [c.text for c in result.chunks]
                    for strategy, result in st.session_state.results.items()
                }
                
                # Compare strategies
                st.session_state.comparison = st.session_state.evaluator.compare_strategies(
                    chunks_dict
                )
            
            st.success("Evaluation complete!")
        
        # Display comparison
        if st.session_state.comparison:
            st.markdown("---")
            
            # Metrics overview
            st.subheader("Metrics Overview")
            
            metrics_data = []
            for strategy, metrics in st.session_state.comparison.items():
                metrics_data.append({
                    'Strategy': strategy.title(),
                    'Coherence': f"{metrics['coherence']:.3f}",
                    'Consistency': f"{metrics['consistency']:.3f}",
                    'Overlap': f"{metrics['overlap_ratio']:.3f}",
                    'Num Chunks': metrics['num_chunks']
                })
            
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
            
            # Detailed metrics
            st.markdown("---")
            st.subheader("Detailed Analysis")
            
            selected = st.selectbox(
                "Select strategy for details:",
                list(st.session_state.comparison.keys()),
                format_func=lambda x: x.title()
            )
            
            metrics = st.session_state.comparison[selected]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Coherence", f"{metrics['coherence']:.3f}",
                         help="Internal semantic similarity (higher is better)")
            with col2:
                st.metric("Consistency", f"{metrics['consistency']:.3f}",
                         help="Uniformity of chunk sizes (higher is better)")
            with col3:
                st.metric("Overlap", f"{metrics['overlap_ratio']:.3f}",
                         help="Overlap between chunks")
            
            # Size statistics
            st.markdown("**Size Statistics:**")
            size_stats = metrics['size_stats']
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.write(f"Mean: {size_stats['mean']:.0f} chars")
                st.write(f"Median: {size_stats['median']:.0f} chars")
            with stats_col2:
                st.write(f"Std Dev: {size_stats['std']:.0f}")
                st.write(f"Min: {size_stats['min']:.0f}")
            with stats_col3:
                st.write(f"Max: {size_stats['max']:.0f}")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            # Find best strategy
            coherence_scores = {s: m['coherence'] for s, m in st.session_state.comparison.items()}
            best_coherence = max(coherence_scores, key=coherence_scores.get)
            
            st.success(f"**Best Coherence:** {best_coherence.title()} ({coherence_scores[best_coherence]:.3f})")
            
            consistency_scores = {s: m['consistency'] for s, m in st.session_state.comparison.items()}
            best_consistency = max(consistency_scores, key=consistency_scores.get)
            
            st.info(f"**Most Consistent:** {best_consistency.title()} ({consistency_scores[best_consistency]:.3f})")
    
    # Tab 3: Visualize
    with tab3:
        st.header("Visualize Chunking Results")
        
        if not st.session_state.results:
            st.info("Chunk some text first to see visualizations!")
            return
        
        viz = ChunkingVisualizer()
        
        # Chunk size distribution
        st.subheader("Chunk Size Distribution")
        
        chunks_dict = {
            strategy: [c.text for c in result.chunks]
            for strategy, result in st.session_state.results.items()
        }
        
        fig_sizes = viz.plot_chunk_sizes(chunks_dict)
        st.plotly_chart(fig_sizes, use_container_width=True)
        
        # Individual strategy distribution
        st.markdown("---")
        st.subheader("Individual Strategy Analysis")
        
        selected_viz = st.selectbox(
            "Select strategy:",
            list(st.session_state.results.keys()),
            format_func=lambda x: x.title(),
            key="viz_select"
        )
        
        chunks = [c.text for c in st.session_state.results[selected_viz].chunks]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = viz.plot_chunk_distribution(chunks, selected_viz.title())
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            fig_timeline = viz.plot_chunk_timeline(chunks, show_text=False)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Metrics comparison (if evaluated)
        if st.session_state.comparison:
            st.markdown("---")
            st.subheader("🎯 Metrics Comparison")
            
            fig_metrics = viz.plot_metrics_comparison(st.session_state.comparison)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Dashboard
            st.markdown("---")
            st.subheader("Comparison Dashboard")
            
            fig_dashboard = viz.create_comparison_dashboard(
                chunks_dict,
                st.session_state.comparison
            )
            st.plotly_chart(fig_dashboard, use_container_width=True)
    
    # Tab 4: Learn
    with tab4:
        st.header("Learn About Chunking")
        
        st.markdown("""
        ### What is Text Chunking?
        
        Chunking is splitting large documents into smaller, meaningful pieces. It's essential for:
        - **RAG Systems**: Retrieve relevant sections, not entire documents
        - **Embeddings**: Better quality on focused text
        - **Token Limits**: Fit text within model constraints
        - **Cost Optimization**: Process only what's needed
        
        ### Chunking Strategies
        """)
        
        with st.expander("Fixed-Size Chunking"):
            st.markdown("""
            **How it works:** Split every N characters
            
            **Pros:**
            - Simple to implement
            - Predictable sizes
            - Fast
            
            **Cons:**
            - Can break sentences
            - Loses context
            - Not semantic
            
            **Use when:** Quick prototypes, uniform data
            """)
        
        with st.expander(" Recursive Chunking"):
            st.markdown("""
            **How it works:** Split by separators (paragraphs → sentences → words)
            
            **Pros:**
            - Respects structure
            - Preserves sentences
            - Readable chunks
            
            **Cons:**
            - Variable sizes
            - Not meaning-aware
            
            **Use when:** General documents, Markdown, code
            """)
        
        with st.expander(" Semantic Chunking"):
            st.markdown("""
            **How it works:** Split based on meaning using embeddings
            
            **Pros:**
            - Coherent topics
            - Natural boundaries
            - Best retrieval quality
            
            **Cons:**
            - Slower (needs embeddings)
            - More complex
            - Unpredictable sizes
            
            **Use when:** High-quality RAG, research papers
            """)
        
        with st.expander("Sliding Window"):
            st.markdown("""
            **How it works:** Fixed-size chunks with overlap
            
            **Pros:**
            - Prevents info loss
            - Context preservation
            - No split phrases
            
            **Cons:**
            -  Duplicate content
            -  More storage
            -  Higher costs
            
            **Use when:** Critical applications, legal/medical docs
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### Best Practices
        
        1. **Chunk Size**: 300-800 tokens (200-600 words)
        2. **Overlap**: 10-20% of chunk size
        3. **Respect Boundaries**: Don't split sentences
        4. **Add Metadata**: Track source, page, section
        5. **Test & Evaluate**: Measure coherence and retrieval
        
        ### Evaluation Metrics
        
        - **Coherence**: How semantically related are sentences within chunks?
        - **Consistency**: How uniform are chunk sizes?
        - **Overlap**: How much do consecutive chunks share?
        
        ### In the RAG Pipeline
        
        ```
        Document → CHUNK → Embed → Store → [Query] → Retrieve → LLM
                   ↑
                Here!
        ```
        """)


if __name__ == "__main__":
    main()