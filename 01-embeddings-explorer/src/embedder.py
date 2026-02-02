"""
Core Embedder class for generating text embeddings
Supports multiple models and batch processing
"""

from typing import List, Optional, Union, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Container for embedding results"""
    text: Union[str, List[str]]
    embeddings: np.ndarray
    model_name: str
    dimensions: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'text': self.text,
            'embeddings': self.embeddings.tolist(),
            'model_name': self.model_name,
            'dimensions': self.dimensions
        }


class Embedder:
    """
    Text embedding generator supporting multiple models
    
    Available models:
    - all-MiniLM-L6-v2: Fast, 384 dimensions, good for most tasks
    - all-mpnet-base-v2: Accurate, 768 dimensions, best quality
    - paraphrase-MiniLM-L3-v2: Tiny, 384 dimensions, fastest
    """
    
    AVAILABLE_MODELS = {
        'all-MiniLM-L6-v2': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimensions': 384,
            'description': 'Fast and efficient, good for most tasks',
            'speed': 'Fast',
        },
        'all-mpnet-base-v2': {
            'name': 'sentence-transformers/all-mpnet-base-v2',
            'dimensions': 768,
            'description': 'Highest quality embeddings',
            'speed': 'Medium',
        },
        'paraphrase-MiniLM-L3-v2': {
            'name': 'sentence-transformers/paraphrase-MiniLM-L3-v2',
            'dimensions': 384,
            'description': 'Smallest and fastest',
            'speed': 'Very Fast',
        },
    }
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedder with a specific model
        
        Args:
            model_name: Name of the model to use (key from AVAILABLE_MODELS)
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model_name}' not available. "
                f"Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.model_info = self.AVAILABLE_MODELS[model_name]
        
        logger.info(f"Loading model: {self.model_info['name']}")
        self.model = SentenceTransformer(self.model_info['name'])
        logger.info(f"Model loaded successfully! Dimensions: {self.model_info['dimensions']}")
    
    def embed(self, text: Union[str, List[str]], show_progress: bool = False) -> EmbeddingResult:
        """
        Generate embeddings for text(s)
        
        Args:
            text: Single string or list of strings to embed
            show_progress: Show progress bar for batch processing
            
        Returns:
            EmbeddingResult containing the embeddings and metadata
        """
        # Convert single string to list for uniform processing
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        logger.info(f"Generating embeddings for {len(texts)} text(s)")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # If single text, return single embedding
        if is_single:
            embeddings = embeddings[0]
        
        result = EmbeddingResult(
            text=text,
            embeddings=embeddings,
            model_name=self.model_name,
            dimensions=self.model_info['dimensions']
        )
        
        logger.info("Embeddings generated successfully")
        return result
    
    def get_embedding_stats(self, embedding: np.ndarray) -> Dict:
        """
        Calculate statistics for an embedding vector
        
        Args:
            embedding: Numpy array of the embedding
            
        Returns:
            Dictionary of statistics
        """
        return {
            'dimensions': len(embedding),
            'mean': float(np.mean(embedding)),
            'std': float(np.std(embedding)),
            'min': float(np.min(embedding)),
            'max': float(np.max(embedding)),
            'norm': float(np.linalg.norm(embedding)),
        }
    
    @classmethod
    def get_model_info(cls, model_name: Optional[str] = None) -> Dict:
        """
        Get information about available models
        
        Args:
            model_name: Specific model to get info for, or None for all models
            
        Returns:
            Dictionary of model information
        """
        if model_name:
            return cls.AVAILABLE_MODELS.get(model_name, {})
        return cls.AVAILABLE_MODELS
    
    def embed_batch_from_file(self, file_path: Union[str, Path], 
                             batch_size: int = 32) -> EmbeddingResult:
        """
        Read texts from file and generate embeddings in batches
        
        Args:
            file_path: Path to text file (one text per line)
            batch_size: Number of texts to process at once
            
        Returns:
            EmbeddingResult for all texts in the file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Reading texts from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Found {len(texts)} texts in file")
        return self.embed(texts, show_progress=True)
    
    def save_embeddings(self, result: EmbeddingResult, output_path: Union[str, Path]):
        """
        Save embeddings to a numpy file
        
        Args:
            result: EmbeddingResult to save
            output_path: Path to save the embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, result.embeddings)
        logger.info(f"Embeddings saved to {output_path}")
    
    def load_embeddings(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load embeddings from a numpy file
        
        Args:
            file_path: Path to the embeddings file
            
        Returns:
            Numpy array of embeddings
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        embeddings = np.load(file_path)
        logger.info(f"Loaded embeddings from {file_path}")
        logger.info(f"Shape: {embeddings.shape}")
        
        return embeddings


# Example usage
if __name__ == "__main__":
    # Initialize embedder
    embedder = Embedder('all-MiniLM-L6-v2')
    
    # Single text embedding
    text = "This is a test sentence for embedding."
    result = embedder.embed(text)
    
    print(f"\nText: {result.text}")
    print(f"Model: {result.model_name}")
    print(f"Dimensions: {result.dimensions}")
    print(f"Embedding shape: {result.embeddings.shape}")
    print(f"First 5 values: {result.embeddings[:5]}")
    
    # Get statistics
    stats = embedder.get_embedding_stats(result.embeddings)
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Batch embedding
    texts = [
        "I love programming in Python",
        "Machine learning is fascinating",
        "Natural language processing is cool"
    ]
    
    batch_result = embedder.embed(texts)
    print(f"\nBatch embeddings shape: {batch_result.embeddings.shape}")
    
    # Model info
    print(f"\nAvailable models:")
    for name, info in Embedder.get_model_info().items():
        print(f"  {name}: {info['dimensions']}d, {info['speed']}")