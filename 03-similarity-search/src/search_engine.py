"""
Core SearchEngine class for similarity-based search
Supports vector search, keyword search, and hybrid approaches
"""

from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for a single search result"""
    text: str
    score: float
    rank: int
    doc_id: int
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'score': self.score,
            'rank': self.rank,
            'doc_id': self.doc_id,
            'metadata': self.metadata or {}
        }


@dataclass
class SearchResults:
    """Container for search results"""
    query: str
    results: List[SearchResult]
    method: str
    total_docs: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'method': self.method,
            'total_docs': self.total_docs,
            'num_results': len(self.results)
        }


class SearchEngine:
    """
    Similarity-based search engine
    
    Search methods:
    - semantic: Vector similarity using embeddings
    - keyword: TF-IDF based keyword matching
    - hybrid: Combination of semantic and keyword
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize search engine
        
        Args:
            embedding_model: Model for generating embeddings
        """
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        self.documents = []
        self.embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.metadata = []
        
        logger.info("Search engine initialized")
    
    def index_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Index documents for searching
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        logger.info(f"Indexing {len(documents)} documents...")
        
        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        self.embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build TF-IDF index for keyword search
        logger.info("Building TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        logger.info(f"Indexed {len(documents)} documents successfully")
    
    def search(self,
               query: str,
               method: str = 'semantic',
               top_k: int = 5,
               metric: str = 'cosine',
               filters: Optional[Dict] = None) -> SearchResults:
        """
        Search for similar documents
        
        Args:
            query: Search query
            method: Search method ('semantic', 'keyword', 'hybrid')
            top_k: Number of results to return
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            filters: Optional metadata filters
            
        Returns:
            SearchResults object
        """
        if not self.documents:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        if method == 'semantic':
            scores = self._semantic_search(query, metric)
        elif method == 'keyword':
            scores = self._keyword_search(query)
        elif method == 'hybrid':
            scores = self._hybrid_search(query, metric)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        # Apply filters if provided
        if filters:
            scores = self._apply_filters(scores, filters)
        
        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Create result objects
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:  # Only include results with positive scores
                results.append(SearchResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    doc_id=int(idx),
                    metadata=self.metadata[idx]
                ))
        
        return SearchResults(
            query=query,
            results=results,
            method=method,
            total_docs=len(self.documents)
        )
    
    def _semantic_search(self, query: str, metric: str = 'cosine') -> np.ndarray:
        """
        Semantic search using embeddings
        
        Args:
            query: Search query
            metric: Similarity metric
            
        Returns:
            Array of similarity scores
        """
        # Embed query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Calculate similarities
        if metric == 'cosine':
            scores = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embeddings
            )[0]
        elif metric == 'euclidean':
            distances = euclidean_distances(
                query_embedding.reshape(1, -1),
                self.embeddings
            )[0]
            # Convert distances to similarities (lower distance = higher similarity)
            scores = 1 / (1 + distances)
        elif metric == 'dot':
            scores = np.dot(self.embeddings, query_embedding)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return scores
    
    def _keyword_search(self, query: str) -> np.ndarray:
        """
        Keyword search using TF-IDF
        
        Args:
            query: Search query
            
        Returns:
            Array of TF-IDF scores
        """
        # Transform query
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        return scores
    
    def _hybrid_search(self,
                      query: str,
                      metric: str = 'cosine',
                      semantic_weight: float = 0.7) -> np.ndarray:
        """
        Hybrid search combining semantic and keyword
        
        Args:
            query: Search query
            metric: Similarity metric for semantic search
            semantic_weight: Weight for semantic scores (0-1)
            
        Returns:
            Array of combined scores
        """
        # Get both types of scores
        semantic_scores = self._semantic_search(query, metric)
        keyword_scores = self._keyword_search(query)
        
        # Normalize scores to 0-1 range
        semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
        keyword_scores = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min() + 1e-8)
        
        # Combine with weights
        keyword_weight = 1 - semantic_weight
        combined_scores = (semantic_weight * semantic_scores) + (keyword_weight * keyword_scores)
        
        return combined_scores
    
    def _apply_filters(self, scores: np.ndarray, filters: Dict) -> np.ndarray:
        """
        Apply metadata filters to search results
        
        Args:
            scores: Array of scores
            filters: Dictionary of filter criteria
            
        Returns:
            Filtered scores (non-matching set to 0)
        """
        filtered_scores = scores.copy()
        
        for idx, meta in enumerate(self.metadata):
            # Check if document matches all filters
            matches = all(
                meta.get(key) == value
                for key, value in filters.items()
            )
            
            if not matches:
                filtered_scores[idx] = 0
        
        return filtered_scores
    
    def batch_search(self,
                    queries: List[str],
                    method: str = 'semantic',
                    top_k: int = 5) -> List[SearchResults]:
        """
        Search multiple queries
        
        Args:
            queries: List of queries
            method: Search method
            top_k: Results per query
            
        Returns:
            List of SearchResults
        """
        results = []
        
        for query in queries:
            result = self.search(query, method=method, top_k=top_k)
            results.append(result)
        
        return results
    
    def find_similar_documents(self,
                              doc_id: int,
                              top_k: int = 5,
                              exclude_self: bool = True) -> SearchResults:
        """
        Find documents similar to a given document
        
        Args:
            doc_id: Index of the document
            top_k: Number of similar documents
            exclude_self: Whether to exclude the document itself
            
        Returns:
            SearchResults object
        """
        if doc_id >= len(self.documents):
            raise ValueError(f"Document ID {doc_id} out of range")
        
        # Calculate similarities
        doc_embedding = self.embeddings[doc_id]
        scores = cosine_similarity(
            doc_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        # Exclude self if requested
        if exclude_self:
            scores[doc_id] = -1
        
        # Get top-k
        top_indices = np.argsort(scores)[-(top_k + (1 if exclude_self else 0)):][::-1]
        
        if exclude_self:
            top_indices = top_indices[top_indices != doc_id][:top_k]
        
        # Create results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:
                results.append(SearchResult(
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    doc_id=int(idx),
                    metadata=self.metadata[idx]
                ))
        
        return SearchResults(
            query=f"Similar to document {doc_id}",
            results=results,
            method='semantic_similarity',
            total_docs=len(self.documents)
        )
    
    def get_statistics(self) -> Dict:
        """
        Get search engine statistics
        
        Returns:
            Dictionary of statistics
        """
        if not self.documents:
            return {'indexed': False}
        
        return {
            'indexed': True,
            'num_documents': len(self.documents),
            'embedding_dimensions': self.embeddings.shape[1],
            'avg_doc_length': np.mean([len(doc) for doc in self.documents]),
            'has_metadata': bool(any(self.metadata)),
            'tfidf_features': self.tfidf_matrix.shape[1]
        }
    
    @staticmethod
    def get_available_methods() -> Dict[str, str]:
        """Get available search methods"""
        return {
            'semantic': 'Vector similarity using embeddings',
            'keyword': 'TF-IDF based keyword matching',
            'hybrid': 'Combination of semantic and keyword search'
        }


if __name__ == "__main__":
    # Example usage
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret images",
        "Reinforcement learning trains agents through rewards",
        "The weather is sunny today",
        "I love eating pizza for dinner"
    ]
    
    # Initialize search engine
    engine = SearchEngine()
    
    # Index documents
    engine.index_documents(documents)
    
    # Search
    query = "What is AI and machine learning?"
    
    print("\n" + "="*60)
    print("SEMANTIC SEARCH")
    print("="*60)
    results = engine.search(query, method='semantic', top_k=3)
    
    for result in results.results:
        print(f"\nRank {result.rank} (Score: {result.score:.4f}):")
        print(f"  {result.text}")
    
    print("\n" + "="*60)
    print("KEYWORD SEARCH")
    print("="*60)
    results = engine.search(query, method='keyword', top_k=3)
    
    for result in results.results:
        print(f"\nRank {result.rank} (Score: {result.score:.4f}):")
        print(f"  {result.text}")
    
    print("\n" + "="*60)
    print("HYBRID SEARCH")
    print("="*60)
    results = engine.search(query, method='hybrid', top_k=3)
    
    for result in results.results:
        print(f"\nRank {result.rank} (Score: {result.score:.4f}):")
        print(f"  {result.text}")
    
    # Statistics
    stats = engine.get_statistics()
    print("\n" + "="*60)
    print("SEARCH ENGINE STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"  {key}: {value}")