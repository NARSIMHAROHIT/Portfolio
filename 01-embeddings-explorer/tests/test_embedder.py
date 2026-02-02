"""
Unit tests for the Embedder class
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embedder import Embedder, EmbeddingResult


class TestEmbedder:
    """Test suite for Embedder class"""
    
    def test_initialization(self):
        """Test that embedder initializes correctly"""
        embedder = Embedder('all-MiniLM-L6-v2')
        assert embedder.model_name == 'all-MiniLM-L6-v2'
        assert embedder.model is not None
    
    def test_invalid_model(self):
        """Test that invalid model raises error"""
        with pytest.raises(ValueError):
            Embedder('invalid-model-name')
    
    def test_single_text_embedding(self):
        """Test embedding a single text"""
        embedder = Embedder('all-MiniLM-L6-v2')
        text = "This is a test sentence"
        result = embedder.embed(text)
        
        assert isinstance(result, EmbeddingResult)
        assert result.text == text
        assert result.model_name == 'all-MiniLM-L6-v2'
        assert result.dimensions == 384
        assert result.embeddings.shape == (384,)
    
    def test_batch_embedding(self):
        """Test embedding multiple texts"""
        embedder = Embedder('all-MiniLM-L6-v2')
        texts = ["First text", "Second text", "Third text"]
        result = embedder.embed(texts)
        
        assert isinstance(result, EmbeddingResult)
        assert result.text == texts
        assert result.embeddings.shape == (3, 384)
    
    def test_embedding_stats(self):
        """Test that statistics are calculated correctly"""
        embedder = Embedder('all-MiniLM-L6-v2')
        text = "Test sentence"
        result = embedder.embed(text)
        
        stats = embedder.get_embedding_stats(result.embeddings)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'norm' in stats
        assert 'dimensions' in stats
        
        assert stats['dimensions'] == 384
        assert isinstance(stats['mean'], float)
    
    def test_similar_texts_have_similar_embeddings(self):
        """Test that similar texts produce similar embeddings"""
        embedder = Embedder('all-MiniLM-L6-v2')
        
        text1 = "I love programming"
        text2 = "I enjoy coding"
        text3 = "The weather is nice today"
        
        emb1 = embedder.embed(text1).embeddings
        emb2 = embedder.embed(text2).embeddings
        emb3 = embedder.embed(text3).embeddings
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        sim_12 = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        sim_13 = cosine_similarity(emb1.reshape(1, -1), emb3.reshape(1, -1))[0][0]
        
        # Similar texts should be more similar than unrelated texts
        assert sim_12 > sim_13
    
    def test_model_info(self):
        """Test getting model information"""
        info = Embedder.get_model_info()
        
        assert isinstance(info, dict)
        assert 'all-MiniLM-L6-v2' in info
        assert 'all-mpnet-base-v2' in info
        
        minilm_info = Embedder.get_model_info('all-MiniLM-L6-v2')
        assert minilm_info['dimensions'] == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])