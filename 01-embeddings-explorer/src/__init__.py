"""
Text Embeddings Explorer
Core modules for generating, comparing, and visualizing text embeddings
"""

from .embedder import Embedder, EmbeddingResult
from .similarity import SimilarityCalculator
from .visualizer import EmbeddingVisualizer

__version__ = "1.0.0"
__all__ = ['Embedder', 'EmbeddingResult', 'SimilarityCalculator', 'EmbeddingVisualizer']