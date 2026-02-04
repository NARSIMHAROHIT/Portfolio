"""
Advanced ranking and re-ranking for search results
Includes multiple ranking strategies and result post-processing
"""

import numpy as np
from typing import List, Dict, Callable
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RankingFeatures:
    """Features used for ranking"""
    similarity_score: float
    query_term_coverage: float
    document_length: int
    position_in_results: int
    metadata_match: float = 0.0


class SearchRanker:
    """
    Advanced ranking for search results
    
    Ranking strategies:
    - score: Simple similarity score
    - bm25: BM25 ranking algorithm
    - reciprocal_rank_fusion: Combine multiple rankings
    - learning_to_rank: Feature-based ranking
    """
    
    def __init__(self):
        """Initialize ranker"""
        pass
    
    def rank_results(self,
                    results: List[Dict],
                    query: str,
                    strategy: str = 'score',
                    **kwargs) -> List[Dict]:
        """
        Rank search results
        
        Args:
            results: List of search results
            query: Original query
            strategy: Ranking strategy
            **kwargs: Additional parameters
            
        Returns:
            Re-ranked results
        """
        if strategy == 'score':
            return self._rank_by_score(results)
        elif strategy == 'diversity':
            return self.rank_by_diversity(results, **kwargs)
        elif strategy == 'freshness':
            return self._rank_by_freshness(results)
        elif strategy == 'popularity':
            return self._rank_by_popularity(results)
        elif strategy == 'custom':
            return self._rank_by_custom(results, kwargs.get('ranking_function'))
        else:
            raise ValueError(f"Unknown ranking strategy: {strategy}")
    
    def _rank_by_score(self, results: List[Dict]) -> List[Dict]:
        """
        Rank by similarity score (default)
        
        Args:
            results: Search results
            
        Returns:
            Sorted results
        """
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    def rank_by_diversity(self,
                         results: List[Dict],
                         diversity_weight: float = 0.3) -> List[Dict]:
        """
        Rank with diversity consideration (avoid similar results)
        
        Args:
            results: Search results
            diversity_weight: Weight for diversity (0-1)
            
        Returns:
            Re-ranked results promoting diversity
        """
        if len(results) <= 1:
            return results
        
        # Start with highest scoring result
        ranked = [results[0]]
        remaining = results[1:]
        
        while remaining:
            # Calculate diversity score for each remaining result
            diversity_scores = []
            
            for candidate in remaining:
                # Calculate average similarity to already selected results
                similarities = [
                    self._text_similarity(candidate.get('text', ''), r.get('text', ''))
                    for r in ranked
                ]
                avg_similarity = np.mean(similarities)
                
                # Combine original score with diversity
                original_score = candidate.get('score', 0)
                diversity_score = (1 - diversity_weight) * original_score + diversity_weight * (1 - avg_similarity)
                
                diversity_scores.append(diversity_score)
            
            # Select result with highest combined score
            best_idx = np.argmax(diversity_scores)
            ranked.append(remaining.pop(best_idx))
        
        # Update ranks
        for i, result in enumerate(ranked, 1):
            result['rank'] = i
        
        return ranked
    
    def _rank_by_freshness(self, results: List[Dict]) -> List[Dict]:
        """
        Rank by recency (requires 'timestamp' in metadata)
        
        Args:
            results: Search results
            
        Returns:
            Sorted by freshness
        """
        def get_timestamp(result):
            metadata = result.get('metadata', {})
            return metadata.get('timestamp', 0)
        
        return sorted(results, key=get_timestamp, reverse=True)
    
    def _rank_by_popularity(self, results: List[Dict]) -> List[Dict]:
        """
        Rank by popularity (requires 'views' or 'clicks' in metadata)
        
        Args:
            results: Search results
            
        Returns:
            Sorted by popularity
        """
        def get_popularity(result):
            metadata = result.get('metadata', {})
            return metadata.get('views', 0) + metadata.get('clicks', 0)
        
        return sorted(results, key=get_popularity, reverse=True)
    
    def _rank_by_custom(self,
                       results: List[Dict],
                       ranking_function: Callable) -> List[Dict]:
        """
        Rank using custom function
        
        Args:
            results: Search results
            ranking_function: Custom ranking function
            
        Returns:
            Sorted results
        """
        if ranking_function is None:
            raise ValueError("Custom ranking requires a ranking_function")
        
        return sorted(results, key=ranking_function, reverse=True)
    
    def reciprocal_rank_fusion(self,
                               result_lists: List[List[Dict]],
                               k: int = 60) -> List[Dict]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion
        
        Args:
            result_lists: List of result lists from different methods
            k: RRF constant (default 60)
            
        Returns:
            Fused results
        """
        # Collect all unique documents
        all_docs = {}
        
        for result_list in result_lists:
            for rank, result in enumerate(result_list, 1):
                doc_id = result.get('doc_id')
                
                if doc_id not in all_docs:
                    all_docs[doc_id] = {
                        'doc': result,
                        'rrf_score': 0
                    }
                
                # Add RRF score: 1 / (k + rank)
                all_docs[doc_id]['rrf_score'] += 1 / (k + rank)
        
        # Sort by RRF score
        fused = sorted(
            all_docs.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        # Extract documents and update ranks
        results = []
        for rank, item in enumerate(fused, 1):
            doc = item['doc'].copy()
            doc['rank'] = rank
            doc['rrf_score'] = item['rrf_score']
            results.append(doc)
        
        return results
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity (Jaccard)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def calculate_metrics(self,
                         results: List[Dict],
                         relevant_docs: List[int]) -> Dict[str, float]:
        """
        Calculate ranking metrics
        
        Args:
            results: Search results
            relevant_docs: List of relevant document IDs
            
        Returns:
            Dictionary of metrics
        """
        if not results or not relevant_docs:
            return {}
        
        # Extract retrieved document IDs
        retrieved = [r.get('doc_id') for r in results]
        relevant_set = set(relevant_docs)
        
        # Precision@K
        def precision_at_k(k):
            if k > len(retrieved):
                k = len(retrieved)
            retrieved_at_k = set(retrieved[:k])
            return len(retrieved_at_k & relevant_set) / k if k > 0 else 0
        
        # Recall@K
        def recall_at_k(k):
            if k > len(retrieved):
                k = len(retrieved)
            retrieved_at_k = set(retrieved[:k])
            return len(retrieved_at_k & relevant_set) / len(relevant_set)
        
        # Mean Reciprocal Rank
        mrr = 0
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                mrr = 1 / i
                break
        
        # Normalized Discounted Cumulative Gain
        def dcg(k):
            score = 0
            for i in range(min(k, len(retrieved))):
                relevance = 1 if retrieved[i] in relevant_set else 0
                score += relevance / np.log2(i + 2)
            return score
        
        def ndcg(k):
            actual_dcg = dcg(k)
            ideal_dcg = dcg(min(k, len(relevant_docs)))
            return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        return {
            'precision@1': precision_at_k(1),
            'precision@3': precision_at_k(3),
            'precision@5': precision_at_k(5),
            'recall@5': recall_at_k(5),
            'recall@10': recall_at_k(10),
            'mrr': mrr,
            'ndcg@5': ndcg(5),
            'ndcg@10': ndcg(10)
        }
    
    @staticmethod
    def get_available_strategies() -> Dict[str, str]:
        """Get available ranking strategies"""
        return {
            'score': 'Rank by similarity score (default)',
            'diversity': 'Promote diverse results',
            'freshness': 'Rank by recency (needs timestamp)',
            'popularity': 'Rank by views/clicks',
            'custom': 'Use custom ranking function'
        }


class ResultFilter:
    """
    Filter and post-process search results
    """
    
    @staticmethod
    def filter_by_threshold(results: List[Dict], threshold: float) -> List[Dict]:
        """
        Filter results below similarity threshold
        
        Args:
            results: Search results
            threshold: Minimum score
            
        Returns:
            Filtered results
        """
        return [r for r in results if r.get('score', 0) >= threshold]
    
    @staticmethod
    def filter_by_metadata(results: List[Dict], filters: Dict) -> List[Dict]:
        """
        Filter by metadata criteria
        
        Args:
            results: Search results
            filters: Metadata filters
            
        Returns:
            Filtered results
        """
        filtered = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Check if all filters match
            matches = all(
                metadata.get(key) == value
                for key, value in filters.items()
            )
            
            if matches:
                filtered.append(result)
        
        return filtered
    
    @staticmethod
    def deduplicate(results: List[Dict], similarity_threshold: float = 0.9) -> List[Dict]:
        """
        Remove duplicate or very similar results
        
        Args:
            results: Search results
            similarity_threshold: Similarity threshold for deduplication
            
        Returns:
            Deduplicated results
        """
        if not results:
            return results
        
        unique = [results[0]]
        
        for result in results[1:]:
            is_duplicate = False
            
            for unique_result in unique:
                # Simple word-based similarity
                words1 = set(result.get('text', '').lower().split())
                words2 = set(unique_result.get('text', '').lower().split())
                
                if words1 and words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique.append(result)
        
        return unique


if __name__ == "__main__":
    # Example usage
    results = [
        {'text': 'Machine learning basics', 'score': 0.95, 'doc_id': 0},
        {'text': 'Deep learning intro', 'score': 0.88, 'doc_id': 1},
        {'text': 'ML fundamentals', 'score': 0.92, 'doc_id': 2},
        {'text': 'Neural networks', 'score': 0.85, 'doc_id': 3}
    ]
    
    ranker = SearchRanker()
    
    # Rank by diversity
    diverse_results = ranker.rank_by_diversity(results, diversity_weight=0.5)
    
    print("DIVERSITY RANKING:")
    for r in diverse_results:
        print(f"  Rank {r['rank']}: {r['text']} (score: {r['score']:.2f})")
    
    # Calculate metrics
    metrics = ranker.calculate_metrics(results, relevant_docs=[0, 2])
    
    print("\nMETRICS:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")