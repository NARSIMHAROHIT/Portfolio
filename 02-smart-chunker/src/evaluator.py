"""
Evaluation metrics for comparing chunking strategies
Measures coherence, consistency, and retrieval quality
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkingEvaluator:
    """
    Evaluate chunking quality using various metrics
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize evaluator
        
        Args:
            embedding_model: Model for computing embeddings
        """
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
    
    def evaluate_chunking(self, chunks: List[str]) -> Dict[str, float]:
        """
        Evaluate chunking quality with multiple metrics
        
        Args:
            chunks: List of chunk texts
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {
            'coherence': self.calculate_coherence(chunks),
            'consistency': self.calculate_consistency(chunks),
            'size_variance': self.calculate_size_variance(chunks),
            'overlap_ratio': self.calculate_overlap_ratio(chunks)
        }
        
        return metrics
    
    def calculate_coherence(self, chunks: List[str]) -> float:
        """
        Measure internal coherence of chunks
        
        High coherence = sentences within chunk are semantically similar
        
        Args:
            chunks: List of chunk texts
            
        Returns:
            Average coherence score (0-1)
        """
        if not chunks:
            return 0.0
        
        coherence_scores = []
        
        for chunk in chunks:
            # Split chunk into sentences
            sentences = self._split_sentences(chunk)
            
            if len(sentences) < 2:
                continue
            
            # Embed sentences
            embeddings = self.model.encode(sentences, show_progress_bar=False)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings)
            
            # Average similarity (excluding diagonal)
            n = len(similarities)
            if n > 1:
                total_sim = (similarities.sum() - n) / (n * (n - 1))
                coherence_scores.append(total_sim)
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        logger.info(f"Coherence score: {avg_coherence:.3f}")
        return float(avg_coherence)
    
    def calculate_consistency(self, chunks: List[str]) -> float:
        """
        Measure consistency of chunk sizes
        
        Lower variance = more consistent
        
        Args:
            chunks: List of chunk texts
            
        Returns:
            Consistency score (0-1), higher is more consistent
        """
        if not chunks:
            return 0.0
        
        sizes = [len(chunk) for chunk in chunks]
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Coefficient of variation (normalized std dev)
        cv = std_size / mean_size if mean_size > 0 else 0
        
        # Convert to 0-1 score (lower CV = higher consistency)
        consistency = 1 / (1 + cv)
        
        logger.info(f"Consistency score: {consistency:.3f} (CV: {cv:.3f})")
        return float(consistency)
    
    def calculate_size_variance(self, chunks: List[str]) -> Dict[str, float]:
        """
        Calculate size distribution statistics
        
        Args:
            chunks: List of chunk texts
            
        Returns:
            Dictionary with size statistics
        """
        if not chunks:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        sizes = [len(chunk) for chunk in chunks]
        
        stats = {
            'mean': float(np.mean(sizes)),
            'std': float(np.std(sizes)),
            'min': float(np.min(sizes)),
            'max': float(np.max(sizes)),
            'median': float(np.median(sizes))
        }
        
        return stats
    
    def calculate_overlap_ratio(self, chunks: List[str]) -> float:
        """
        Calculate overlap between consecutive chunks
        
        Args:
            chunks: List of chunk texts
            
        Returns:
            Average overlap ratio (0-1)
        """
        if len(chunks) < 2:
            return 0.0
        
        overlaps = []
        
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            
            # Find overlap
            overlap = self._find_overlap(chunk1, chunk2)
            overlap_ratio = len(overlap) / min(len(chunk1), len(chunk2))
            overlaps.append(overlap_ratio)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0.0
        
        logger.info(f"Average overlap: {avg_overlap:.3f}")
        return float(avg_overlap)
    
    def compare_strategies(self, 
                          results_dict: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Compare multiple chunking strategies
        
        Args:
            results_dict: Dict mapping strategy name to list of chunks
            
        Returns:
            Comparison results for each strategy
        """
        comparison = {}
        
        for strategy, chunks in results_dict.items():
            logger.info(f"Evaluating strategy: {strategy}")
            
            metrics = self.evaluate_chunking(chunks)
            size_stats = self.calculate_size_variance(chunks)
            
            comparison[strategy] = {
                **metrics,
                'size_stats': size_stats,
                'num_chunks': len(chunks)
            }
        
        return comparison
    
    def calculate_retrieval_quality(self,
                                   chunks: List[str],
                                   queries: List[str],
                                   ground_truth: Dict[str, List[int]]) -> float:
        """
        Evaluate retrieval quality using test queries
        
        Args:
            chunks: List of chunk texts
            queries: Test queries
            ground_truth: Dict mapping query to relevant chunk indices
            
        Returns:
            Average precision score
        """
        if not queries or not ground_truth:
            return 0.0
        
        # Embed chunks
        logger.info(f"Embedding {len(chunks)} chunks for retrieval evaluation")
        chunk_embeddings = self.model.encode(chunks, show_progress_bar=False)
        
        precision_scores = []
        
        for query in queries:
            # Embed query
            query_embedding = self.model.encode(query, show_progress_bar=False)
            
            # Calculate similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                chunk_embeddings
            )[0]
            
            # Get top 3 chunks
            top_k = 3
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Calculate precision
            if query in ground_truth:
                relevant = set(ground_truth[query])
                retrieved = set(top_indices)
                
                precision = len(relevant & retrieved) / len(retrieved)
                precision_scores.append(precision)
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0.0
        
        logger.info(f"Retrieval precision: {avg_precision:.3f}")
        return float(avg_precision)
    
    def visualize_chunk_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for visualization
        
        Args:
            chunks: List of chunk texts
            
        Returns:
            2D array of embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_overlap(self, text1: str, text2: str, min_length: int = 10) -> str:
        """
        Find overlapping text between two strings
        
        Args:
            text1: First text
            text2: Second text
            min_length: Minimum overlap length to consider
            
        Returns:
            Overlapping text
        """
        max_overlap = ""
        
        # Check suffixes of text1 against prefixes of text2
        for i in range(len(text1), min_length - 1, -1):
            suffix = text1[-i:]
            if text2.startswith(suffix):
                if len(suffix) > len(max_overlap):
                    max_overlap = suffix
                break
        
        return max_overlap
    
    @staticmethod
    def get_metric_descriptions() -> Dict[str, str]:
        """Get descriptions of evaluation metrics"""
        return {
            'coherence': 'Internal semantic similarity within chunks (0-1, higher is better)',
            'consistency': 'Uniformity of chunk sizes (0-1, higher is better)',
            'size_variance': 'Statistics about chunk size distribution',
            'overlap_ratio': 'Overlap between consecutive chunks (0-1)',
            'retrieval_quality': 'Precision of retrieving relevant chunks (0-1, higher is better)'
        }


# Example usage
if __name__ == "__main__":
    # Sample chunks from different strategies
    fixed_chunks = [
        "Machine learning is a subset of artificial intelligence. It involves training",
        " algorithms on data. These algorithms can then make predictions or decisions",
        " without being explicitly programmed. There are several types of machine"
    ]
    
    semantic_chunks = [
        "Machine learning is a subset of artificial intelligence. It involves training algorithms on data.",
        "These algorithms can then make predictions or decisions without being explicitly programmed.",
        "There are several types of machine learning. Supervised learning uses labeled data."
    ]
    
    # Initialize evaluator
    evaluator = ChunkingEvaluator()
    
    # Compare strategies
    results = {
        'fixed': fixed_chunks,
        'semantic': semantic_chunks
    }
    
    comparison = evaluator.compare_strategies(results)
    
    print("\n" + "="*60)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*60)
    
    for strategy, metrics in comparison.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Coherence: {metrics['coherence']:.3f}")
        print(f"  Consistency: {metrics['consistency']:.3f}")
        print(f"  Overlap: {metrics['overlap_ratio']:.3f}")
        print(f"  Num chunks: {metrics['num_chunks']}")
        print(f"  Avg size: {metrics['size_stats']['mean']:.1f} chars")