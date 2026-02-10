"""
Multi-Source RAG Engine with collection management and citations
Tracks sources, provides citations, manages multiple collections
"""

from typing import List, Dict, Optional, Set
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSourceRAG:
    """
    RAG system with multi-source support and citations
    
    Features:
    - Multiple collections
    - Source tracking
    - Citation generation
    - Metadata filtering
    """
    
    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 persist_directory: str = './chroma_db'):
        """
        Initialize multi-source RAG
        
        Args:
            embedding_model: Model for embeddings
            persist_directory: Database location
        """
        logger.info("Initializing Multi-Source RAG...")
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        logger.info("Multi-Source RAG initialized")
    
    def create_collection(self, name: str) -> None:
        """
        Create a new collection
        
        Args:
            name: Collection name
        """
        try:
            self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created collection: {name}")
        except Exception as e:
            logger.warning(f"Collection {name} might already exist: {e}")
    
    def get_collection(self, name: str):
        """Get or create collection"""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, 
                     documents: List[Dict],
                     collection_name: str = 'default'):
        """
        Add documents to collection
        
        Args:
            documents: List of document dicts with content and metadata
            collection_name: Target collection
        """
        if not documents:
            logger.warning("No documents provided")
            return
        
        logger.info(f"Processing {len(documents)} documents for collection '{collection_name}'...")
        
        collection = self.get_collection(collection_name)
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            
            if not content or not content.strip():
                logger.warning(f"Document {doc_idx} has no content, skipping")
                continue
            
            chunks = self._chunk_text(content)
            
            if not chunks:
                logger.warning(f"Document {doc_idx} produced no chunks, skipping")
                continue
            
            base_metadata = doc.get('metadata', {})
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                
                metadata = base_metadata.copy()
                metadata['chunk_index'] = chunk_idx
                metadata['document_index'] = doc_idx
                metadata['collection'] = collection_name
                
                all_metadatas.append(metadata)
                
                chunk_id = f"{collection_name}_{doc_idx}_{chunk_idx}"
                all_ids.append(chunk_id)
        
        if not all_chunks:
            logger.error("No chunks created from documents")
            raise ValueError("No valid content found in documents")
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            all_chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Storing in collection '{collection_name}'...")
        collection.add(
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        logger.info(f"Successfully added {len(all_chunks)} chunks to {collection_name}")
    
    def query(self,
              question: str,
              collections: Optional[List[str]] = None,
              top_k: int = 3,
              filters: Optional[Dict] = None,
              api_key: str = None) -> Dict:
        """
        Query with source attribution
        
        Args:
            question: User question
            collections: List of collections to search (None = all)
            top_k: Number of results per collection
            filters: Metadata filters
            api_key: Groq API key
            
        Returns:
            Answer with citations and sources
        """
        logger.info(f"Processing query: {question}")
        
        if collections is None:
            collections = [c.name for c in self.client.list_collections()]
        
        all_chunks = []
        all_metadatas = []
        all_distances = []
        
        query_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True
        )
        
        for coll_name in collections:
            try:
                collection = self.client.get_collection(coll_name)
                
                where_filter = self._build_filter(filters) if filters else None
                
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    where=where_filter
                )
                
                if results['documents'][0]:
                    all_chunks.extend(results['documents'][0])
                    all_metadatas.extend(results['metadatas'][0])
                    all_distances.extend(results['distances'][0])
            
            except Exception as e:
                logger.warning(f"Error querying collection {coll_name}: {e}")
        
        if not all_chunks:
            return {
                'answer': "No relevant information found in any collection.",
                'sources': [],
                'chunks': [],
                'metadatas': []
            }
        
        sorted_indices = sorted(range(len(all_distances)), key=lambda i: all_distances[i])
        top_indices = sorted_indices[:top_k]
        
        retrieved_chunks = [all_chunks[i] for i in top_indices]
        retrieved_metadata = [all_metadatas[i] for i in top_indices]
        
        context = "\n\n".join([
            f"[Source: {self._format_source(meta)}]\n{chunk}"
            for chunk, meta in zip(retrieved_chunks, retrieved_metadata)
        ])
        
        if not api_key:
            raise ValueError("API key is required")
        
        answer = self._call_llm(question, context, api_key)
        
        sources = self._extract_sources(retrieved_metadata)
        
        return {
            'answer': answer,
            'sources': sources,
            'chunks': retrieved_chunks,
            'metadatas': retrieved_metadata,
            'context': context
        }
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text with overlap"""
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
    
    def _call_llm(self, question: str, context: str, api_key: str) -> str:
        """Call Groq API with source-aware prompt"""
        logger.info("Calling Groq API...")
        
        system_prompt = """You are a helpful assistant that answers questions based on provided context. 
        IMPORTANT: When you use information from a source, reference it in your answer like this: (Source: filename).
        If the context doesn't contain enough information, say so clearly."""
        
        user_prompt = f"""Context with sources:
{context}

Question: {question}

Please answer the question based on the context above. Include source references in your answer."""
        
        try:
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'llama-3.3-70b-versatile',
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    'max_tokens': 500,
                    'temperature': 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                logger.error(f"API error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return f"Error calling LLM (status {response.status_code}): {response.text[:200]}"
        
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"Error: {str(e)}"
    
    def _build_filter(self, filters: Dict) -> Dict:
        """Build ChromaDB where filter"""
        where = {}
        for key, value in filters.items():
            where[key] = value
        return where
    
    def _format_source(self, metadata: Dict) -> str:
        """Format source citation"""
        parts = []
        
        if 'filename' in metadata:
            parts.append(metadata['filename'])
        
        if 'page_number' in metadata:
            parts.append(f"page {metadata['page_number']}")
        
        if 'row_number' in metadata:
            parts.append(f"row {metadata['row_number']}")
        
        return ", ".join(parts) if parts else "Unknown source"
    
    def _extract_sources(self, metadatas: List[Dict]) -> List[Dict]:
        """Extract unique sources"""
        sources = []
        seen = set()
        
        for meta in metadatas:
            source_key = (
                meta.get('filename', 'Unknown'),
                meta.get('page_number'),
                meta.get('row_number')
            )
            
            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    'filename': meta.get('filename', 'Unknown'),
                    'page': meta.get('page_number'),
                    'row': meta.get('row_number'),
                    'type': meta.get('file_type', 'unknown'),
                    'collection': meta.get('collection', 'default')
                })
        
        return sources
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        return [c.name for c in self.client.list_collections()]
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics for a collection"""
        try:
            collection = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'count': collection.count(),
                'exists': True
            }
        except:
            return {
                'name': collection_name,
                'count': 0,
                'exists': False
            }
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")


if __name__ == "__main__":
    rag = MultiSourceRAG()
    
    documents = [
        {
            'content': 'Machine learning is a subset of AI.',
            'metadata': {'filename': 'intro.txt', 'page_number': 1}
        }
    ]
    
    rag.add_documents(documents, 'test_collection')
    
    print(f"Collections: {rag.list_collections()}")