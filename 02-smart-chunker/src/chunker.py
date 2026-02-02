"""
Core Chunker class with multiple chunking strategies
Supports fixed-size, recursive, semantic, and overlap-based chunking
"""

from typing import List, Optional, Dict, Callable
from dataclasses import dataclass
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Container for a single chunk"""
    text: str
    start_idx: int
    end_idx: int
    chunk_id: int
    metadata: Dict = None
    
    def __len__(self):
        return len(self.text)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'chunk_id': self.chunk_id,
            'length': len(self.text),
            'metadata': self.metadata or {}
        }


@dataclass
class ChunkingResult:
    """Container for chunking results"""
    chunks: List[Chunk]
    strategy: str
    total_chars: int
    num_chunks: int
    avg_chunk_size: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'chunks': [c.to_dict() for c in self.chunks],
            'strategy': self.strategy,
            'total_chars': self.total_chars,
            'num_chunks': self.num_chunks,
            'avg_chunk_size': self.avg_chunk_size
        }


class Chunker:
    """
    Text chunking with multiple strategies
    
    Strategies:
    - fixed: Fixed-size chunks
    - recursive: Split by separators (paragraphs, sentences, words)
    - semantic: Split based on semantic similarity
    - sliding: Fixed-size with overlap
    """
    
    def __init__(self, embedding_model: Optional[str] = None):
        """
        Initialize chunker
        
        Args:
            embedding_model: Model for semantic chunking (optional)
        """
        self.embedding_model = None
        if embedding_model:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
    
    def chunk(self, 
              text: str,
              strategy: str = 'recursive',
              chunk_size: int = 500,
              chunk_overlap: int = 50,
              **kwargs) -> ChunkingResult:
        """
        Chunk text using specified strategy
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy ('fixed', 'recursive', 'semantic', 'sliding')
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks (for sliding window)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            ChunkingResult with chunks and metadata
        """
        if strategy == 'fixed':
            chunks = self._fixed_size_chunking(text, chunk_size)
        elif strategy == 'recursive':
            chunks = self._recursive_chunking(text, chunk_size, chunk_overlap)
        elif strategy == 'semantic':
            if not self.embedding_model:
                raise ValueError("Semantic chunking requires embedding model")
            chunks = self._semantic_chunking(text, chunk_size, **kwargs)
        elif strategy == 'sliding':
            chunks = self._sliding_window_chunking(text, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Calculate statistics
        total_chars = len(text)
        num_chunks = len(chunks)
        avg_size = total_chars / num_chunks if num_chunks > 0 else 0
        
        return ChunkingResult(
            chunks=chunks,
            strategy=strategy,
            total_chars=total_chars,
            num_chunks=num_chunks,
            avg_chunk_size=avg_size
        )
    
    def _fixed_size_chunking(self, text: str, chunk_size: int) -> List[Chunk]:
        """
        Split text into fixed-size chunks
        
        Simple but can break sentences/words
        """
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=i,
                end_idx=min(i + chunk_size, len(text)),
                chunk_id=len(chunks)
            ))
        
        logger.info(f"Fixed-size chunking: {len(chunks)} chunks created")
        return chunks
    
    def _recursive_chunking(self, 
                           text: str, 
                           chunk_size: int,
                           chunk_overlap: int) -> List[Chunk]:
        """
        Split text recursively by separators
        
        Tries separators in order: \n\n, \n, '. ', ' '
        """
        # Separators in order of preference
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_text(text: str, separators: List[str]) -> List[str]:
            """Recursively split text"""
            if not separators:
                return [text]
            
            separator = separators[0]
            splits = text.split(separator) if separator else list(text)
            
            # Rejoin separator
            if separator:
                splits = [s + separator for s in splits[:-1]] + [splits[-1]]
            
            # Process splits
            final_chunks = []
            current_chunk = ""
            
            for split in splits:
                if len(current_chunk) + len(split) <= chunk_size:
                    current_chunk += split
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk)
                    
                    # If split itself is too large, recurse
                    if len(split) > chunk_size:
                        sub_chunks = split_text(split, separators[1:])
                        final_chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = split
            
            if current_chunk:
                final_chunks.append(current_chunk)
            
            return final_chunks
        
        # Get text chunks
        text_chunks = split_text(text, separators)
        
        # Add overlap
        if chunk_overlap > 0:
            text_chunks = self._add_overlap(text_chunks, chunk_overlap)
        
        # Convert to Chunk objects
        chunks = []
        current_pos = 0
        
        for chunk_text in text_chunks:
            # Find actual position in original text
            start_idx = text.find(chunk_text, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text),
                chunk_id=len(chunks)
            ))
            
            current_pos = start_idx + len(chunk_text)
        
        logger.info(f"Recursive chunking: {len(chunks)} chunks created")
        return chunks
    
    def _semantic_chunking(self,
                          text: str,
                          chunk_size: int,
                          similarity_threshold: float = 0.5) -> List[Chunk]:
        """
        Split text based on semantic similarity
        
        Groups sentences with similar meanings together
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [Chunk(text=text, start_idx=0, end_idx=len(text), chunk_id=0)]
        
        # Embed sentences
        logger.info(f"Embedding {len(sentences)} sentences for semantic chunking")
        embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
        
        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            sim = sim / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
            similarities.append(sim)
        
        # Find split points (where similarity drops)
        split_points = [0]
        current_chunk_size = len(sentences[0])
        
        for i, sim in enumerate(similarities):
            current_chunk_size += len(sentences[i + 1])
            
            # Split if similarity is low OR chunk is too large
            if sim < similarity_threshold or current_chunk_size > chunk_size:
                split_points.append(i + 1)
                current_chunk_size = len(sentences[i + 1])
        
        split_points.append(len(sentences))
        
        # Create chunks
        chunks = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            
            chunk_sentences = sentences[start:end]
            chunk_text = ' '.join(chunk_sentences)
            
            # Find position in original text
            start_idx = text.find(chunk_sentences[0])
            end_idx = start_idx + len(chunk_text)
            
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_id=len(chunks),
                metadata={'sentences': len(chunk_sentences)}
            ))
        
        logger.info(f"Semantic chunking: {len(chunks)} chunks created")
        return chunks
    
    def _sliding_window_chunking(self,
                                 text: str,
                                 chunk_size: int,
                                 chunk_overlap: int) -> List[Chunk]:
        """
        Create overlapping chunks (sliding window)
        
        Each chunk overlaps with the next by chunk_overlap characters
        """
        chunks = []
        step = chunk_size - chunk_overlap
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            
            # Skip very small final chunks
            if len(chunk_text) < chunk_size * 0.3 and chunks:
                # Merge with previous chunk
                chunks[-1].text += chunk_text
                chunks[-1].end_idx = i + len(chunk_text)
            else:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=i,
                    end_idx=min(i + chunk_size, len(text)),
                    chunk_id=len(chunks),
                    metadata={'overlap': chunk_overlap}
                ))
            
            # Stop if we've reached the end
            if i + chunk_size >= len(text):
                break
        
        logger.info(f"Sliding window chunking: {len(chunks)} chunks created")
        return chunks
    
    def _add_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """Add overlap between consecutive chunks"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            # Take last 'overlap' chars from previous chunk
            prev_overlap = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else chunks[i - 1]
            overlapped.append(prev_overlap + chunks[i])
        
        return overlapped
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter (can be improved with NLTK/spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def get_available_strategies() -> Dict[str, str]:
        """Get available chunking strategies and descriptions"""
        return {
            'fixed': 'Fixed-size chunks (simple, can break sentences)',
            'recursive': 'Split by separators (respects structure)',
            'semantic': 'Split by meaning (best quality, slower)',
            'sliding': 'Overlapping chunks (prevents info loss)'
        }


# Example usage
if __name__ == "__main__":
    # Sample text
    text = """
    Machine learning is a subset of artificial intelligence. It involves training algorithms on data.
    These algorithms can then make predictions or decisions without being explicitly programmed.
    
    There are several types of machine learning. Supervised learning uses labeled data.
    Unsupervised learning finds patterns in unlabeled data. Reinforcement learning learns through trial and error.
    
    Deep learning is a subset of machine learning. It uses neural networks with multiple layers.
    These networks can learn complex patterns. They are particularly good at tasks like image recognition.
    """
    
    # Initialize chunker
    chunker = Chunker(embedding_model='all-MiniLM-L6-v2')
    
    # Try different strategies
    strategies = ['fixed', 'recursive', 'semantic', 'sliding']
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()}")
        print('='*60)
        
        result = chunker.chunk(
            text,
            strategy=strategy,
            chunk_size=200,
            chunk_overlap=50
        )
        
        print(f"Total chunks: {result.num_chunks}")
        print(f"Avg chunk size: {result.avg_chunk_size:.1f} chars")
        print(f"\nChunks:")
        
        for i, chunk in enumerate(result.chunks):
            print(f"\nChunk {i + 1} ({len(chunk)} chars):")
            print(f"  {chunk.text[:100]}...")