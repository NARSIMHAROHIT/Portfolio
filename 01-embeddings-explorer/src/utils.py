"""
Utility functions for the Embeddings Explorer project
"""

import tiktoken
from typing import List, Dict
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Count tokens for different models and estimate costs
    """
    
    # Approximate cost per 1M tokens (as of 2026)
    COSTS = {
        'grok': {
            'input': 5.00,  # $ per 1M tokens
            'output': 15.00
        },
        'gpt-4': {
            'input': 30.00,
            'output': 60.00
        },
        'gpt-3.5-turbo': {
            'input': 0.50,
            'output': 1.50
        }
    }
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in text using tiktoken
        
        Args:
            text: Input text
            model: Model name for tokenizer
            
        Returns:
            Number of tokens
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (used by GPT-4 and newer models)
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        return len(tokens)
    
    @classmethod
    def estimate_cost(cls, text: str, model: str = "grok", 
                     token_type: str = "input") -> float:
        """
        Estimate API cost for processing text
        
        Args:
            text: Input text
            model: Model to use
            token_type: 'input' or 'output'
            
        Returns:
            Estimated cost in USD
        """
        tokens = cls.count_tokens(text)
        
        if model not in cls.COSTS:
            logger.warning(f"Unknown model {model}, using 'grok' pricing")
            model = "grok"
        
        cost_per_token = cls.COSTS[model][token_type] / 1_000_000
        return tokens * cost_per_token
    
    @classmethod
    def batch_token_count(cls, texts: List[str], model: str = "gpt-3.5-turbo") -> Dict:
        """
        Count tokens for multiple texts
        
        Args:
            texts: List of texts
            model: Model name
            
        Returns:
            Dictionary with token statistics
        """
        token_counts = [cls.count_tokens(text, model) for text in texts]
        
        return {
            'total_tokens': sum(token_counts),
            'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'texts_count': len(texts),
            'per_text': token_counts
        }


def save_json(data: dict, filepath: Path):
    """
    Save data to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Data saved to {filepath}")


def load_json(filepath: Path) -> dict:
    """
    Load data from JSON file
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Data loaded from {filepath}")
    return data


def load_texts_from_file(filepath: Path) -> List[str]:
    """
    Load texts from a file (one per line)
    
    Args:
        filepath: Input file path
        
    Returns:
        List of texts
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(texts)} texts from {filepath}")
    return texts


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format number with commas and decimals
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{num:,.{decimals}f}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# Example usage
if __name__ == "__main__":
    counter = TokenCounter()
    
    # Single text
    text = "This is a test sentence for token counting."
    tokens = counter.count_tokens(text)
    cost = counter.estimate_cost(text)
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Estimated cost (Grok): ${cost:.6f}")
    
    # Batch
    texts = [
        "First sentence",
        "Second sentence is a bit longer",
        "Third sentence is the longest of them all"
    ]
    
    batch_stats = counter.batch_token_count(texts)
    print(f"\nBatch statistics:")
    for key, value in batch_stats.items():
        if key != 'per_text':
            print(f"  {key}: {value}")
    
    # Save and load
    test_data = {'embeddings': [1, 2, 3], 'model': 'test'}
    save_json(test_data, Path('test.json'))
    loaded = load_json(Path('test.json'))
    print(f"\nSaved and loaded: {loaded}")
    
    # Clean up
    Path('test.json').unlink()