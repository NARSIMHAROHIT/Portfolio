"""
Visualization utilities for embeddings
Includes dimensionality reduction and interactive plots
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional, Dict, Tuple
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """
    Visualize high-dimensional embeddings in 2D/3D space
    
    Reduction methods:
    - UMAP: Best for preserving global structure
    - t-SNE: Best for revealing clusters
    - PCA: Fast, linear dimensionality reduction
    """
    
    REDUCTION_METHODS = {
        'umap': {
            'name': 'UMAP',
            'description': 'Uniform Manifold Approximation and Projection',
            'best_for': 'Preserving global structure',
            'speed': 'Fast'
        },
        'tsne': {
            'name': 't-SNE',
            'description': 't-Distributed Stochastic Neighbor Embedding',
            'best_for': 'Revealing clusters',
            'speed': 'Slow'
        },
        'pca': {
            'name': 'PCA',
            'description': 'Principal Component Analysis',
            'best_for': 'Linear relationships',
            'speed': 'Very Fast'
        }
    }
    
    @staticmethod
    def reduce_dimensions(embeddings: np.ndarray,
                         n_components: int = 2,
                         method: str = 'umap',
                         random_state: int = 42) -> np.ndarray:
        """
        Reduce embeddings to lower dimensions
        
        Args:
            embeddings: High-dimensional embeddings (n_samples, n_features)
            n_components: Target dimensions (2 or 3)
            method: Reduction method ('umap', 'tsne', or 'pca')
            random_state: Random seed for reproducibility
            
        Returns:
            Reduced embeddings (n_samples, n_components)
        """
        logger.info(f"Reducing {embeddings.shape} to {n_components}D using {method.upper()}")
        
        if method == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=15,
                min_dist=0.1
            )
        elif method == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=min(30, len(embeddings) - 1)
            )
        elif method == 'pca':
            reducer = PCA(
                n_components=n_components,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced = reducer.fit_transform(embeddings)
        logger.info(f"Reduction complete: {reduced.shape}")
        
        return reduced
    
    @staticmethod
    def plot_2d(embeddings: np.ndarray,
                texts: Optional[List[str]] = None,
                labels: Optional[List[str]] = None,
                title: str = "Text Embeddings in 2D",
                method: str = 'umap') -> go.Figure:
        """
        Create 2D scatter plot of embeddings
        
        Args:
            embeddings: High-dimensional embeddings
            texts: Text labels for hover info
            labels: Category labels for coloring
            title: Plot title
            method: Dimensionality reduction method
            
        Returns:
            Plotly figure object
        """
        # Reduce to 2D if needed
        if embeddings.shape[1] > 2:
            reduced = EmbeddingVisualizer.reduce_dimensions(
                embeddings, n_components=2, method=method
            )
        else:
            reduced = embeddings
        
        # Prepare data
        data = {
            'x': reduced[:, 0],
            'y': reduced[:, 1],
        }
        
        if texts:
            data['text'] = texts
        else:
            data['text'] = [f"Point {i}" for i in range(len(reduced))]
        
        if labels:
            data['label'] = labels
        
        # Create plot
        if labels:
            fig = px.scatter(
                data,
                x='x',
                y='y',
                color='label',
                hover_data=['text'],
                title=title,
                labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'}
            )
        else:
            fig = px.scatter(
                data,
                x='x',
                y='y',
                hover_data=['text'],
                title=title,
                labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'}
            )
        
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
        fig.update_layout(
            template='plotly_dark',
            hovermode='closest',
            width=800,
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_3d(embeddings: np.ndarray,
                texts: Optional[List[str]] = None,
                labels: Optional[List[str]] = None,
                title: str = "Text Embeddings in 3D",
                method: str = 'umap') -> go.Figure:
        """
        Create 3D scatter plot of embeddings
        
        Args:
            embeddings: High-dimensional embeddings
            texts: Text labels for hover info
            labels: Category labels for coloring
            title: Plot title
            method: Dimensionality reduction method
            
        Returns:
            Plotly figure object
        """
        # Reduce to 3D if needed
        if embeddings.shape[1] > 3:
            reduced = EmbeddingVisualizer.reduce_dimensions(
                embeddings, n_components=3, method=method
            )
        else:
            reduced = embeddings
        
        # Prepare hover text
        if texts:
            hover_text = texts
        else:
            hover_text = [f"Point {i}" for i in range(len(reduced))]
        
        # Create trace
        if labels:
            unique_labels = list(set(labels))
            traces = []
            
            for label in unique_labels:
                mask = [l == label for l in labels]
                traces.append(go.Scatter3d(
                    x=reduced[mask, 0],
                    y=reduced[mask, 1],
                    z=reduced[mask, 2],
                    mode='markers',
                    name=label,
                    text=[t for t, m in zip(hover_text, mask) if m],
                    hovertemplate='<b>%{text}</b><br>' +
                                 f'{method.upper()}1: %{{x:.3f}}<br>' +
                                 f'{method.upper()}2: %{{y:.3f}}<br>' +
                                 f'{method.upper()}3: %{{z:.3f}}',
                    marker=dict(size=8, line=dict(width=0.5, color='white'))
                ))
            
            fig = go.Figure(data=traces)
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=reduced[:, 0],
                y=reduced[:, 1],
                z=reduced[:, 2],
                mode='markers',
                text=hover_text,
                hovertemplate='<b>%{text}</b><br>' +
                             f'{method.upper()}1: %{{x:.3f}}<br>' +
                             f'{method.upper()}2: %{{y:.3f}}<br>' +
                             f'{method.upper()}3: %{{z:.3f}}',
                marker=dict(
                    size=8,
                    color=reduced[:, 2],
                    colorscale='Viridis',
                    showscale=True,
                    line=dict(width=0.5, color='white')
                )
            )])
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            scene=dict(
                xaxis_title=f'{method.upper()} 1',
                yaxis_title=f'{method.upper()} 2',
                zaxis_title=f'{method.upper()} 3'
            ),
            width=900,
            height=700
        )
        
        return fig
    
    @staticmethod
    def plot_similarity_heatmap(similarity_matrix: np.ndarray,
                               labels: Optional[List[str]] = None,
                               title: str = "Similarity Heatmap") -> go.Figure:
        """
        Create heatmap of similarity matrix
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            labels: Text labels for axes
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if labels is None:
            labels = [f"Text {i+1}" for i in range(len(similarity_matrix))]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            text=similarity_matrix,
            texttemplate='%{text:.3f}',
            textfont={"size": 10},
            colorbar=dict(title="Similarity")
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            xaxis_title="Texts",
            yaxis_title="Texts",
            width=800,
            height=800
        )
        
        return fig
    
    @staticmethod
    def plot_embedding_distribution(embedding: np.ndarray,
                                   title: str = "Embedding Distribution") -> go.Figure:
        """
        Plot distribution of values in an embedding vector
        
        Args:
            embedding: Single embedding vector
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=embedding,
            nbinsx=50,
            name='Distribution',
            marker_color='lightblue'
        ))
        
        # Add mean line
        mean_val = np.mean(embedding)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.3f}"
        )
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            xaxis_title="Value",
            yaxis_title="Frequency",
            showlegend=True,
            width=800,
            height=400
        )
        
        return fig
    
    @staticmethod
    def compare_reduction_methods(embeddings: np.ndarray,
                                 texts: List[str],
                                 methods: List[str] = ['umap', 'tsne', 'pca']) -> Dict[str, go.Figure]:
        """
        Compare different dimensionality reduction methods side by side
        
        Args:
            embeddings: High-dimensional embeddings
            texts: Text labels
            methods: List of methods to compare
            
        Returns:
            Dictionary of method names to figures
        """
        figures = {}
        
        for method in methods:
            fig = EmbeddingVisualizer.plot_2d(
                embeddings,
                texts=texts,
                title=f"Embeddings - {method.upper()}",
                method=method
            )
            figures[method] = fig
        
        return figures


# Example usage
if __name__ == "__main__":
    # Create sample embeddings
    np.random.seed(42)
    n_samples = 100
    n_features = 384
    
    # Create 3 clusters
    cluster1 = np.random.randn(30, n_features)
    cluster2 = np.random.randn(30, n_features) + 5
    cluster3 = np.random.randn(40, n_features) - 3
    
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    texts = [f"Text {i+1}" for i in range(n_samples)]
    labels = ['Cluster A'] * 30 + ['Cluster B'] * 30 + ['Cluster C'] * 40
    
    viz = EmbeddingVisualizer()
    
    # 2D plot
    print("Creating 2D plot...")
    fig_2d = viz.plot_2d(embeddings, texts=texts, labels=labels)
    fig_2d.write_html("embedding_2d.html")
    print("Saved to embedding_2d.html")
    
    # 3D plot
    print("Creating 3D plot...")
    fig_3d = viz.plot_3d(embeddings, texts=texts, labels=labels)
    fig_3d.write_html("embedding_3d.html")
    print("Saved to embedding_3d.html")
    
    # Similarity heatmap
    from similarity import SimilarityCalculator
    print("Creating similarity heatmap...")
    sample_embeddings = embeddings[:10]
    sample_texts = texts[:10]
    similarity_matrix = SimilarityCalculator.create_similarity_matrix(sample_embeddings)
    fig_heatmap = viz.plot_similarity_heatmap(similarity_matrix, labels=sample_texts)
    fig_heatmap.write_html("similarity_heatmap.html")
    print("Saved to similarity_heatmap.html")
    
    # Distribution plot
    print("Creating distribution plot...")
    fig_dist = viz.plot_embedding_distribution(embeddings[0])
    fig_dist.write_html("embedding_distribution.html")
    print("Saved to embedding_distribution.html")
    
    print("\nAll visualizations created successfully!")