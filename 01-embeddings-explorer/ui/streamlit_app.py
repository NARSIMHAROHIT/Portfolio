"""
Streamlit UI for Text Embeddings Explorer
Interactive tool for exploring and visualizing text embeddings
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embedder import Embedder, EmbeddingResult
from similarity import SimilarityCalculator
from visualizer import EmbeddingVisualizer
from utils import TokenCounter, format_number, truncate_text

# Page config
st.set_page_config(
    page_title="Text Embeddings Explorer",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'embedder' not in st.session_state:
        st.session_state.embedder = None
    if 'last_embedding' not in st.session_state:
        st.session_state.last_embedding = None
    if 'embedding_history' not in st.session_state:
        st.session_state.embedding_history = []


def main():
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header"> Text Embeddings Explorer</h1>', 
                unsafe_allow_html=True)
    st.markdown("*Explore and visualize how text is converted to numerical vectors*")
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header(" Configuration")
        
        # Model selection
        st.subheader(" Model Selection")
        model_options = list(Embedder.AVAILABLE_MODELS.keys())
        selected_model = st.selectbox(
            "Choose Embedding Model",
            model_options,
            help="Different models have different trade-offs between speed and quality"
        )
        
        # Show model info
        model_info = Embedder.AVAILABLE_MODELS[selected_model]
        st.info(f"""
        **{model_info['name']}**
        - Dimensions: {model_info['dimensions']}
        - Speed: {model_info['speed']}
        - {model_info['description']}
        """)
        
        # Load model button
        if st.button(" Load Model", type="primary"):
            with st.spinner(f"Loading {selected_model}..."):
                st.session_state.embedder = Embedder(selected_model)
            st.success("Model loaded!")
        
        st.markdown("---")
        
        # Visualization settings
        st.subheader(" Visualization Settings")
        reduction_method = st.selectbox(
            "Dimensionality Reduction",
            ['umap', 'tsne', 'pca'],
            help="Method for reducing to 2D/3D"
        )
        
        plot_dimension = st.radio(
            "Plot Dimension",
            ['2D', '3D'],
            horizontal=True
        )
        
        st.markdown("---")
        
        # Statistics
        st.subheader(" Session Stats")
        st.metric("Embeddings Generated", len(st.session_state.embedding_history))
        if st.session_state.embedder:
            st.metric("Current Model", selected_model)
            st.metric("Dimensions", model_info['dimensions'])
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Generate Embeddings",
        " Compare Similarity",
        " Visualize",
        " Batch Process",
        " Learn"
    ])
    
    # Tab 1: Generate Embeddings
    with tab1:
        st.header("Generate Text Embeddings")
        
        if st.session_state.embedder is None:
            st.warning(" Please load a model from the sidebar first!")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Text")
            input_text = st.text_area(
                "Enter your text here:",
                height=150,
                placeholder="Type or paste any text to generate its embedding..."
            )
            
            if st.button(" Generate Embedding", type="primary"):
                if input_text.strip():
                    with st.spinner("Generating embedding..."):
                        # Generate embedding
                        result = st.session_state.embedder.embed(input_text)
                        st.session_state.last_embedding = result
                        st.session_state.embedding_history.append({
                            'text': truncate_text(input_text, 50),
                            'full_text': input_text,
                            'embedding': result.embeddings,
                            'model': result.model_name
                        })
                    
                    st.success(" Embedding generated successfully!")
                else:
                    st.error("Please enter some text!")
        
        with col2:
            st.subheader("Token Analysis")
            if input_text:
                counter = TokenCounter()
                tokens = counter.count_tokens(input_text)
                cost = counter.estimate_cost(input_text, model='grok')
                
                st.metric("Token Count", f"{tokens:,}")
                st.metric("Characters", f"{len(input_text):,}")
                st.metric("Est. Cost (Grok)", f"${cost:.6f}")
        
        # Display results
        if st.session_state.last_embedding:
            st.markdown("---")
            st.subheader(" Embedding Results")
            
            result = st.session_state.last_embedding
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dimensions", result.dimensions)
            with col2:
                st.metric("Model", result.model_name)
            with col3:
                norm = np.linalg.norm(result.embeddings)
                st.metric("Vector Norm", f"{norm:.4f}")
            
            # Statistics
            stats = st.session_state.embedder.get_embedding_stats(result.embeddings)
            
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.markdown("**Statistics:**")
                st.write(f"Mean: {stats['mean']:.4f}")
                st.write(f"Std Dev: {stats['std']:.4f}")
            
            with stats_col2:
                st.markdown("**Range:**")
                st.write(f"Min: {stats['min']:.4f}")
                st.write(f"Max: {stats['max']:.4f}")
            
            # Visualization
            viz = EmbeddingVisualizer()
            fig_dist = viz.plot_embedding_distribution(
                result.embeddings,
                title="Embedding Value Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Show first few values
            with st.expander(" View Embedding Vector"):
                st.write("First 20 values:")
                st.code(str(result.embeddings[:20]))
                
                if st.button(" Download Full Embedding"):
                    st.download_button(
                        "Download as NumPy file",
                        data=result.embeddings.tobytes(),
                        file_name="embedding.npy",
                        mime="application/octet-stream"
                    )
    
    # Tab 2: Compare Similarity
    with tab2:
        st.header("Compare Text Similarity")
        
        if st.session_state.embedder is None:
            st.warning(" Please load a model from the sidebar first!")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Text 1")
            text1 = st.text_area("Enter first text:", height=100, key="text1")
        
        with col2:
            st.subheader("Text 2")
            text2 = st.text_area("Enter second text:", height=100, key="text2")
        
        if st.button("🔍 Compare Similarity", type="primary"):
            if text1.strip() and text2.strip():
                with st.spinner("Calculating similarity..."):
                    # Generate embeddings
                    emb1 = st.session_state.embedder.embed(text1).embeddings
                    emb2 = st.session_state.embedder.embed(text2).embeddings
                    
                    # Calculate all metrics
                    calc = SimilarityCalculator()
                    metrics = calc.calculate_all_metrics(emb1, emb2)
                
                st.success(" Similarity calculated!")
                
                st.markdown("---")
                st.subheader(" Similarity Metrics")
                
                # Display metrics
                metric_cols = st.columns(len(metrics))
                for i, (metric, score) in enumerate(metrics.items()):
                    with metric_cols[i]:
                        info = SimilarityCalculator.AVAILABLE_METRICS[metric]
                        st.metric(
                            info['name'],
                            f"{score:.4f}",
                            help=f"{info['description']}\nRange: {info['range']}"
                        )
                
                # Cosine similarity interpretation
                cosine_score = metrics['cosine']
                interpretation = calc.interpret_cosine_similarity(cosine_score)
                
                # Color based on similarity
                if cosine_score >= 0.7:
                    st.success(f" **{interpretation}**")
                elif cosine_score >= 0.4:
                    st.info(f" **{interpretation}**")
                else:
                    st.warning(f" **{interpretation}**")
                
                # Visualization
                st.markdown("---")
                st.subheader(" Visual Comparison")
                
                comparison_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Score': list(metrics.values())
                })
                
                st.bar_chart(comparison_df.set_index('Metric'))
            else:
                st.error("Please enter both texts!")
    
    # Tab 3: Visualize
    with tab3:
        st.header("Visualize Embeddings")
        
        if not st.session_state.embedding_history:
            st.info(" Generate some embeddings first to visualize them!")
            return
        
        st.write(f"You have **{len(st.session_state.embedding_history)}** embeddings in history")
        
        if len(st.session_state.embedding_history) >= 2:
            # Prepare data
            embeddings = np.array([
                item['embedding'] for item in st.session_state.embedding_history
            ])
            texts = [
                item['text'] for item in st.session_state.embedding_history
            ]
            
            viz = EmbeddingVisualizer()
            
            # Create visualization
            with st.spinner("Creating visualization..."):
                if plot_dimension == '2D':
                    fig = viz.plot_2d(
                        embeddings,
                        texts=texts,
                        method=reduction_method,
                        title=f"Embedding Visualization ({reduction_method.upper()})"
                    )
                else:
                    fig = viz.plot_3d(
                        embeddings,
                        texts=texts,
                        method=reduction_method,
                        title=f"Embedding Visualization ({reduction_method.upper()})"
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Similarity matrix
            if len(st.session_state.embedding_history) >= 3:
                st.markdown("---")
                st.subheader(" Similarity Heatmap")
                
                calc = SimilarityCalculator()
                sim_matrix = calc.create_similarity_matrix(embeddings)
                
                fig_heatmap = viz.plot_similarity_heatmap(
                    sim_matrix,
                    labels=texts,
                    title="Pairwise Similarity Matrix"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("Need at least 2 embeddings for visualization!")
        
        # Clear history
        if st.button(" Clear History", type="secondary"):
            st.session_state.embedding_history = []
            st.session_state.last_embedding = None
            st.rerun()
    
    # Tab 4: Batch Process
    with tab4:
        st.header("Batch Process Multiple Texts")
        
        if st.session_state.embedder is None:
            st.warning(" Please load a model from the sidebar first!")
            return
        
        st.markdown("Process multiple texts at once. Enter one text per line:")
        
        batch_input = st.text_area(
            "Enter texts (one per line):",
            height=200,
            placeholder="First text\nSecond text\nThird text\n..."
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "Or upload a text file (one text per line)",
            type=['txt']
        )
        
        if st.button(" Process Batch", type="primary"):
            texts = []
            
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                texts = [line.strip() for line in content.split('\n') if line.strip()]
            elif batch_input.strip():
                texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            if texts:
                st.write(f"Processing **{len(texts)}** texts...")
                
                with st.spinner("Generating embeddings..."):
                    result = st.session_state.embedder.embed(texts, show_progress=True)
                    
                    # Add to history
                    for text, emb in zip(texts, result.embeddings):
                        st.session_state.embedding_history.append({
                            'text': truncate_text(text, 50),
                            'full_text': text,
                            'embedding': emb,
                            'model': result.model_name
                        })
                
                st.success(f" Generated {len(texts)} embeddings!")
                
                # Statistics
                counter = TokenCounter()
                token_stats = counter.batch_token_count(texts)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Texts", len(texts))
                with col2:
                    st.metric("Total Tokens", f"{token_stats['total_tokens']:,}")
                with col3:
                    st.metric("Avg Tokens/Text", f"{token_stats['avg_tokens']:.0f}")
                with col4:
                    total_cost = counter.estimate_cost(
                        ' '.join(texts), 
                        model='grok'
                    )
                    st.metric("Est. Cost", f"${total_cost:.6f}")
                
                # Show results
                st.markdown("---")
                st.subheader(" Processed Texts")
                
                df = pd.DataFrame({
                    'Text': [truncate_text(t, 60) for t in texts],
                    'Tokens': token_stats['per_text'],
                    'Embedding Shape': [f"({result.dimensions},)"] * len(texts)
                })
                
                st.dataframe(df, use_container_width=True)
            else:
                st.error("Please enter some texts or upload a file!")
    
    # Tab 5: Learn
    with tab5:
        st.header(" Learn About Embeddings")
        
        st.markdown("""
        ### What are Text Embeddings?
        
        Text embeddings are dense vector representations of text that capture semantic meaning. 
        They allow computers to understand and process natural language by converting words and 
        sentences into numbers.
        
        ### Why Do They Matter?
        
        - **Semantic Search**: Find similar content based on meaning, not just keywords
        - **RAG Systems**: Power retrieval in Retrieval-Augmented Generation
        - **Clustering**: Group similar documents automatically
        - **Recommendations**: Find related items based on content
        # Before embedding, clean the text
        text = remove_urls(text)
        text = remove_html(text)
        text = fix_spacing(text)
        # Then embed
        embedding = model.encode(text)
        ```

        ---

        ## 🎓 The Modern LLMOps Philosophy
        ```
        Old NLP (2000s):
        Text → Clean → Tokenize → Stem → Remove stops → Features → Model

        Modern LLMOps (2020s):
        Text → (Basic cleaning) → Embed → Done!
        ```

        **Why the change?**
        1. **Context matters** - "not good" ≠ "good"
        2. **Transformers are smart** - They learn relationships
        3. **End-to-end learning** - Model figures it out
        4. **Better results** - Proven in practice

        ---

        ## Summary - What You Need to Know

        ### **For Your LLMOps Journey:**

        **DO USE:**
        - Text cleaning (remove HTML, URLs, weird characters)
        - Embeddings (modern, context-aware)
        - NER (when you need metadata)
        - Chunking (smart splitting)

        **DON'T USE (Usually):**
        - Stemming
        - Lemmatization  
        - Stop word removal
        - Excessive preprocessing

        **Exception:** Traditional NLP is still useful for:
        - Keyword extraction
        - Hybrid search systems
        - Metadata generation
        - Language-specific rules

        ---

        ##  Quick Decision Guide

        **Should I preprocess my text?**
        ```
        Is your model a Transformer (BERT, GPT, etc.)?
            YES → Minimal preprocessing (just cleaning)
            NO  → Traditional NLP (stem, lemmatize, etc.)

        Are you using embeddings?
            YES → No stemming/lemmatization needed
            NO  → Consider traditional methods

        Do you need keyword search?
            YES → Stemming might help
            NO  → Skip it

        Do you need to extract entities?
            YES → Use NER
            NO  → Skip it
        ### Key Concepts
        """)
        
        with st.expander(" Dimensions"):
            st.markdown("""
            Embeddings are high-dimensional vectors (typically 384 to 768 dimensions). 
            Each dimension captures different aspects of meaning. More dimensions generally 
            means more nuanced representations, but also slower processing.
            """)
        
        with st.expander(" Similarity Metrics"):
            st.markdown("""
            **Cosine Similarity**: Measures the angle between vectors (0 to 1, higher is more similar)
            - Best for: General text similarity
            - Range: -1 to 1 (typically 0 to 1 for text)
            
            **Euclidean Distance**: Straight-line distance between points (lower is more similar)
            - Best for: When magnitude matters
            - Range: 0 to ∞
            
            **Manhattan Distance**: Sum of absolute differences (lower is more similar)
            - Best for: High-dimensional spaces
            - Range: 0 to ∞
            """)
        
        with st.expander(" Dimensionality Reduction"):
            st.markdown("""
            Since embeddings have hundreds of dimensions, we use techniques to visualize them in 2D/3D:
            
            **UMAP**: Preserves global structure well, fast
            **t-SNE**: Great for revealing clusters, slower
            **PCA**: Fastest, linear reduction
            """)
        
        with st.expander(" Model Selection"):
            st.markdown("""
            Different models have different trade-offs:
            
            - **all-MiniLM-L6-v2**: Fast, 384 dimensions, good for most tasks
            - **all-mpnet-base-v2**: Higher quality, 768 dimensions, slower
            - **paraphrase-MiniLM-L3-v2**: Fastest, 384 dimensions, smaller model
            
            Choose based on your needs: speed vs. quality
            """)
        
        st.markdown("---")
        st.markdown("""
        ###  Further Reading
        
        - [Sentence Transformers Documentation](https://www.sbert.net/)
        - [Understanding Embeddings](https://platform.openai.com/docs/guides/embeddings)
        - [Vector Similarity Explained](https://www.pinecone.io/learn/vector-similarity/)
        """)


if __name__ == "__main__":
    main()