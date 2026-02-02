"""
Smart Chunker - Text chunking with multiple strategies
"""

from .chunker import Chunker, Chunk, ChunkingResult
from .evaluator import ChunkingEvaluator
from .visualizer import ChunkingVisualizer

__version__ = "1.0.0"
__all__ = ['Chunker', 'Chunk', 'ChunkingResult', 'ChunkingEvaluator', 'ChunkingVisualizer']
