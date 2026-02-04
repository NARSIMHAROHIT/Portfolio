"""
RAG Engine - Complete Retrieval-Augmented Generation system
Combines document loading, chunking, embedding, retrieval, and generation
"""

from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Complete RAG system
    
    Pipeline:
    1. Load documents
    2. Chunk documents
    3. Embed chunks
    4. Store in vector DB
    5. Retrieve relevant chunks for query
    6. Generate answer with LLM
    """
    
    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 collection_name: str = 'rag_documents',
                 persist_directory: str = './chroma_db'):
        """
        Initialize RAG engine
        
        Args:
            embedding_model: Model for embeddings
            collection_name: ChromaDB collection name
            persist_directory: Where to store the database
        """
        logger.info("Initializing RAG engine...")
        
        self.embedding_model = SentenceTransformer(embedding_model)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        logger.info("RAG engine initialized")
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the RAG system
        
        Args:
            documents: List of document dicts with 'content' and 'metadata'
        """
        if not documents:
            logger.warning("No documents provided")
            return
        
        logger.info(f"Processing {len(documents)} documents...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        chunk_id = 0
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            
            if not content or not content.strip():
                logger.warning(f"Document {doc_idx} has no content, skipping")
                continue
            
            chunks = self._chunk_text(content)
            
            if not chunks:
                logger.warning(f"Document {doc_idx} produced no chunks, skipping")
                continue
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                
                metadata = doc.get('metadata', {}).copy()
                metadata['chunk_index'] = chunk_idx
                metadata['document_index'] = doc_idx
                all_metadatas.append(metadata)
                
                all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")
                chunk_id += 1
        
        if not all_chunks:
            logger.error("No chunks created from documents. Check document content.")
            raise ValueError("No valid content found in documents")
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            all_chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info("Storing in vector database...")
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        logger.info(f"Successfully added {len(all_chunks)} chunks to database")
    
    def query(self,
              question: str,
              top_k: int = 3,
              llm_provider: str = 'grok',
              api_key: Optional[str] = None) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            llm_provider: LLM to use ('grok' or 'mock')
            api_key: API key for LLM
            
        Returns:
            Dict with answer, sources, and retrieved chunks
        """
        logger.info(f"Processing query: {question}")
        
        query_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True
        )
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        retrieved_chunks = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        context = "\n\n".join(retrieved_chunks)
        
        if llm_provider == 'mock' or not api_key:
            answer = self._mock_llm_response(question, context)
        else:
            answer = self._call_llm(question, context, llm_provider, api_key)
        
        return {
            'answer': answer,
            'retrieved_chunks': retrieved_chunks,
            'metadatas': metadatas,
            'distances': distances,
            'context': context
        }
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text with overlap
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        words = text.split()
        
        if not words:
            return []
        
        if len(words) <= self.chunk_size:
            return [text.strip()]
        
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def _call_llm(self,
                  question: str,
                  context: str,
                  provider: str,
                  api_key: str) -> str:
        """
        Call LLM API
        
        Args:
            question: User question
            context: Retrieved context
            provider: LLM provider
            api_key: API key
            
        Returns:
            Generated answer
        """
        prompt = f"""Answer the question based on the context below. If you cannot answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            if provider == 'grok':
                response = requests.post(
                    'https://api.x.ai/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'grok-beta',
                        'messages': [
                            {'role': 'user', 'content': prompt}
                        ],
                        'max_tokens': 500,
                        'temperature': 0.7
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    logger.error(f"LLM API error: {response.status_code}")
                    return f"Error calling LLM: {response.status_code}"
            
            else:
                return "Unsupported LLM provider"
        
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"Error: {str(e)}"
    
    def _mock_llm_response(self, question: str, context: str) -> str:
        """
        Mock LLM response for testing without API
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Mock answer
        """
        return f"[MOCK RESPONSE] Based on the provided context, here's what I found relevant to your question: {question}\n\nThe context contains information that could help answer this. In a real implementation with an LLM API, this would be a proper generated answer."
    
    def get_statistics(self) -> Dict:
        """
        Get RAG system statistics
        
        Returns:
            Statistics dictionary
        """
        count = self.collection.count()
        
        return {
            'total_chunks': count,
            'collection_name': self.collection.name,
            'embedding_model': 'all-MiniLM-L6-v2',
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
    
    def clear_database(self):
        """Clear all documents from the database"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Database cleared")


if __name__ == "__main__":
    rag = RAGEngine()
    
    documents = [
        {
            'content': 'Machine learning is a subset of artificial intelligence.',
            'metadata': {'source': 'intro.txt'}
        },
        {
            'content': 'Deep learning uses neural networks with multiple layers.',
            'metadata': {'source': 'deep_learning.txt'}
        }
    ]
    
    rag.add_documents(documents)
    
    result = rag.query("What is machine learning?", llm_provider='mock')
    
    print(f"Answer: {result['answer']}")
    print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks")