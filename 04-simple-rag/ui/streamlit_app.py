"""
Streamlit UI for Simple RAG Bot
Upload documents and ask questions
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import os

sys.path.append(str(Path(__file__).parent.parent / "src"))

from document_loader import DocumentLoader
from rag_engine import RAGEngine

st.set_page_config(
    page_title="Simple RAG Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #2b313e;
        border-left: 3px solid #667eea;
    }
    .assistant-message {
        background-color: #1e1e1e;
        border-left: 3px solid #764ba2;
    }
    .source-box {
        background-color: #2b313e;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False


def main():
    init_session_state()
    
    st.markdown('<h1 class="main-header">📚 Simple RAG Bot</h1>', 
                unsafe_allow_html=True)
    st.markdown("*Upload documents and ask questions*")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Configuration")
        
        if st.button("Initialize RAG System", type="primary"):
            with st.spinner("Initializing..."):
                st.session_state.rag_engine = RAGEngine()
            st.success("RAG system ready")
        
        st.markdown("---")
        
        st.subheader("API Configuration")
        api_key = st.text_input(
            "Grok API Key (optional)",
            type="password",
            help="Leave empty to use mock responses"
        )
        
        use_mock = st.checkbox(
            "Use mock LLM (no API needed)",
            value=not bool(api_key)
        )
        
        st.markdown("---")
        
        st.subheader("Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf', 'html'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Documents"):
            if not st.session_state.rag_engine:
                st.error("Please initialize RAG system first")
                return
            
            with st.spinner("Processing documents..."):
                loader = DocumentLoader()
                documents = []
                
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_file.name).suffix
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        doc = loader.load(tmp_path)
                        
                        if doc.content and doc.content.strip():
                            documents.append(doc.to_dict())
                        else:
                            st.warning(f"Skipped {uploaded_file.name}: No text content found")
                    
                    except Exception as e:
                        st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                    
                    finally:
                        os.unlink(tmp_path)
                
                if not documents:
                    st.error("No valid documents to process. Please check your files.")
                    return
                
                try:
                    st.session_state.rag_engine.add_documents(documents)
                    st.session_state.documents_loaded = True
                    st.success(f"Processed {len(documents)} documents")
                except Exception as e:
                    st.error(f"Error adding documents: {str(e)}")
                    st.info("Make sure documents contain text content")
        
        if st.session_state.rag_engine:
            stats = st.session_state.rag_engine.get_statistics()
            
            st.markdown("---")
            st.subheader("Statistics")
            st.metric("Chunks in DB", stats['total_chunks'])
            st.metric("Chunk Size", stats['chunk_size'])
        
        if st.button("Clear Database"):
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_database()
                st.session_state.chat_history = []
                st.session_state.documents_loaded = False
                st.success("Database cleared")
    
    tab1, tab2, tab3 = st.tabs(["Chat", "Retrieved Context", "About"])
    
    with tab1:
        st.header("Ask Questions")
        
        if not st.session_state.rag_engine:
            st.warning("Please initialize RAG system from sidebar")
            return
        
        if not st.session_state.documents_loaded:
            st.info("Upload and process documents to start asking questions")
            return
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<strong>You:</strong><br>{message["content"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message">'
                    f'<strong>Assistant:</strong><br>{message["content"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        question = st.text_input(
            "Your question:",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        
        with col1:
            if st.button("Ask", type="primary"):
                if question.strip():
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': question
                    })
                    
                    with st.spinner("Thinking..."):
                        provider = 'mock' if use_mock else 'grok'
                        result = st.session_state.rag_engine.query(
                            question,
                            top_k=3,
                            llm_provider=provider,
                            api_key=api_key if not use_mock else None
                        )
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': result['answer'],
                        'context': result['retrieved_chunks']
                    })
                    
                    st.rerun()
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    with tab2:
        st.header("Retrieved Context")
        
        if not st.session_state.chat_history:
            st.info("Ask a question to see retrieved context")
            return
        
        last_message = None
        for msg in reversed(st.session_state.chat_history):
            if msg['role'] == 'assistant' and 'context' in msg:
                last_message = msg
                break
        
        if last_message:
            st.subheader("Chunks Used to Generate Answer")
            
            for i, chunk in enumerate(last_message['context'], 1):
                st.markdown(
                    f'<div class="source-box">'
                    f'<strong>Chunk {i}:</strong><br>{chunk}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No context available")
    
    with tab3:
        st.header("About Simple RAG Bot")
        
        st.markdown("""
        ### What is RAG?
        
        Retrieval-Augmented Generation combines:
        - **Retrieval**: Finding relevant information from your documents
        - **Generation**: Using an LLM to generate answers based on that information
        
        ### How It Works
        
        1. **Upload Documents**: PDF, TXT, or HTML files
        2. **Processing**: Documents are chunked and embedded
        3. **Storage**: Embeddings stored in ChromaDB (local vector database)
        4. **Query**: You ask a question
        5. **Retrieval**: System finds relevant chunks
        6. **Generation**: LLM generates answer based on retrieved context
        
        ### Features
        
        - Supports multiple file formats (PDF, TXT, HTML)
        - Local vector database (ChromaDB)
        - Semantic search with embeddings
        - Optional Grok API integration
        - Mock mode for testing without API
        
        ### Mock Mode
        
        Enable "Use mock LLM" to test without an API key. The system will:
        - Still retrieve relevant chunks
        - Show mock responses instead of real LLM answers
        - Useful for testing and learning
        
        ### Technical Details
        
        - **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
        - **Vector DB**: ChromaDB (local, persistent)
        - **LLM**: Grok API (optional)
        - **Chunking**: 500 words with 50 word overlap
        """)


if __name__ == "__main__":
    main()