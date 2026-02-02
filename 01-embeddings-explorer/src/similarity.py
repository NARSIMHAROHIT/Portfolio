"""
Similarity calculation utilities for comparing embeddings
Supports multiple distance metrics and batch comparisons
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cityblock
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    Calculate similarity between embeddings using various metrics
    
    Supported metrics:
    - cosine: Cosine similarity (default, range: -1 to 1, higher is more similar)
    - euclidean: Euclidean distance (lower is more similar)
    - manhattan: Manhattan/City-block distance (lower is more similar)
    - dot_product: Dot product (higher is more similar)
    """
    
    AVAILABLE_METRICS = {
        'cosine': {
            'name': 'Cosine Similarity',
            'range': '[-1, 1]',
            'interpretation': 'Higher is more similar',
            'description': 'Measures angle between vectors, ignores magnitude'
        },
        'euclidean': {
            'name': 'Euclidean Distance',
            'range': '[0, ∞)',
            'interpretation': 'Lower is more similar',
            'description': 'Straight-line distance between points'
        },
        'manhattan': {
            'name': 'Manhattan Distance',
            'range': '[0, ∞)',
            'interpretation': 'Lower is more similar',
            'description': 'Sum of absolute differences along each dimension'
        },
        'dot_product': {
            'name': 'Dot Product',
            'range': '(-∞, ∞)',
            'interpretation': 'Higher is more similar',
            'description': 'Inner product of two vectors'
        }
    }
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1 (higher is more similar)
        """
        # Reshape for sklearn if 1D
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Distance (lower is more similar)
        """
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        distance = euclidean_distances(emb1, emb2)[0][0]
        return float(distance)
    
    @staticmethod
    def manhattan_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate Manhattan distance between two embeddings
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Distance (lower is more similar)
        """
        distance = cityblock(emb1.flatten(), emb2.flatten())
        return float(distance)
    
    @staticmethod
    def dot_product(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate dot product between two embeddings
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Dot product (higher is more similar)
        """
        product = np.dot(emb1.flatten(), emb2.flatten())
        return float(product)
    
    @classmethod
    def calculate_similarity(cls, emb1: np.ndarray, emb2: np.ndarray, 
                           metric: str = 'cosine') -> float:
        """
        Calculate similarity using specified metric
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            metric: Similarity metric to use
            
        Returns:
            Similarity/distance score
        """
        if metric not in cls.AVAILABLE_METRICS:
            raise ValueError(
                f"Metric '{metric}' not available. "
                f"Choose from: {list(cls.AVAILABLE_METRICS.keys())}"
            )
        
        if metric == 'cosine':
            return cls.cosine_similarity(emb1, emb2)
        elif metric == 'euclidean':
            return cls.euclidean_distance(emb1, emb2)
        elif metric == 'manhattan':
            return cls.manhattan_distance(emb1, emb2)
        elif metric == 'dot_product':
            return cls.dot_product(emb1, emb2)
    
    @classmethod
    def calculate_all_metrics(cls, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """
        Calculate all available similarity metrics
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Dictionary of metric names and scores
        """
        results = {}
        for metric in cls.AVAILABLE_METRICS.keys():
            results[metric] = cls.calculate_similarity(emb1, emb2, metric)
        return results
    
    @staticmethod
    def create_similarity_matrix(embeddings: np.ndarray, 
                                metric: str = 'cosine') -> np.ndarray:
        """
        Create pairwise similarity matrix for multiple embeddings
        
        Args:
            embeddings: 2D array of embeddings (n_samples, n_features)
            metric: Similarity metric to use
            
        Returns:
            Symmetric matrix of pairwise similarities (n_samples, n_samples)
        """
        n = len(embeddings)
        matrix = np.zeros((n, n))
        
        calc = SimilarityCalculator()
        
        for i in range(n):
            for j in range(i, n):
                similarity = calc.calculate_similarity(
                    embeddings[i], 
                    embeddings[j], 
                    metric
                )
                matrix[i, j] = similarity
                matrix[j, i] = similarity  # Symmetric
        
        return matrix
    
    @staticmethod
    def find_most_similar(query_embedding: np.ndarray, 
                         embeddings: np.ndarray,
                         texts: List[str],
                         top_k: int = 5,
                         metric: str = 'cosine') -> List[Tuple[str, float]]:
        """
        Find most similar texts to a query
        
        Args:
            query_embedding: Query embedding vector
            embeddings: Array of embeddings to search
            texts: Corresponding text labels
            top_k: Number of top results to return
            metric: Similarity metric to use
            
        Returns:
            List of (text, similarity_score) tuples, sorted by similarity
        """
        calc = SimilarityCalculator()
        similarities = []
        
        for i, emb in enumerate(embeddings):
            score = calc.calculate_similarity(query_embedding, emb, metric)
            similarities.append((texts[i], score))
        
        # Sort based on metric (cosine and dot_product: descending, others: ascending)
        reverse = metric in ['cosine', 'dot_product']
        similarities.sort(key=lambda x: x[1], reverse=reverse)
        
        return similarities[:top_k]
    
    @staticmethod
    def interpret_cosine_similarity(score: float) -> str:
        """
        Provide human-readable interpretation of cosine similarity score
        
        Args:
            score: Cosine similarity score
            
        Returns:
            Interpretation string
        """
        if score >= 0.9:
            return "Very High Similarity - Nearly identical meaning"
        elif score >= 0.7:
            return "High Similarity - Strong semantic relationship"
        elif score >= 0.5:
            return "Moderate Similarity - Related topics"
        elif score >= 0.3:
            return "Low Similarity - Weak relationship"
        else:
            return "Very Low Similarity - Unrelated content"
    
    @classmethod
    def get_metric_info(cls, metric: Optional[str] = None) -> Dict:
        """
        Get information about available metrics
        
        Args:
            metric: Specific metric to get info for, or None for all
            
        Returns:
            Dictionary of metric information
        """
        if metric:
            return cls.AVAILABLE_METRICS.get(metric, {})
        return cls.AVAILABLE_METRICS


# Example usage
if __name__ == "__main__":
    # Create sample embeddings
    np.random.seed(42)
    emb1 = np.random.randn(384)  # Random 384-dimensional vector
    emb2 = np.random.randn(384)
    emb3 = emb1 + np.random.randn(384) * 0.1  # Similar to emb1
    
    calc = SimilarityCalculator()
    
    # Compare two embeddings
    print("Comparing embeddings:")
    print(f"Cosine similarity: {calc.cosine_similarity(emb1, emb2):.4f}")
    print(f"Euclidean distance: {calc.euclidean_distance(emb1, emb2):.4f}")
    print(f"Manhattan distance: {calc.manhattan_distance(emb1, emb2):.4f}")
    print(f"Dot product: {calc.dot_product(emb1, emb2):.4f}")
    
    # All metrics at once
    print("\nAll metrics for similar embeddings:")
    all_metrics = calc.calculate_all_metrics(emb1, emb3)
    for metric, score in all_metrics.items():
        print(f"  {metric}: {score:.4f}")
    
    # Similarity matrix
    embeddings = np.array([emb1, emb2, emb3])
    matrix = calc.create_similarity_matrix(embeddings)
    print("\nSimilarity matrix:")
    print(matrix)
    
    # Find most similar
    texts = ["Text 1", "Text 2", "Text 3"]
    similar = calc.find_most_similar(emb1, embeddings, texts, top_k=3)
    print("\nMost similar to Text 1:")
    for text, score in similar:
        print(f"  {text}: {score:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    print(calc.interpret_cosine_similarity(0.95))
    print(calc.interpret_cosine_similarity(0.65))
    print(calc.interpret_cosine_similarity(0.25))