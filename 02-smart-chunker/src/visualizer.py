"""
Visualization utilities for chunking analysis
Creates plots to compare chunking strategies
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict
import umap
from sklearn.decomposition import PCA


class ChunkingVisualizer:
    """
    Visualize chunking results and comparisons
    """
    
    @staticmethod
    def plot_chunk_sizes(results_dict: Dict[str, List[str]], 
                        title: str = "Chunk Size Distribution") -> go.Figure:
        """
        Plot chunk size distribution for multiple strategies
        
        Args:
            results_dict: Dict mapping strategy name to list of chunks
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for strategy, chunks in results_dict.items():
            sizes = [len(chunk) for chunk in chunks]
            
            fig.add_trace(go.Box(
                y=sizes,
                name=strategy.title(),
                boxmean='sd'
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Chunk Size (characters)",
            xaxis_title="Strategy",
            template='plotly_dark',
            showlegend=True,
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_chunk_distribution(chunks: List[str],
                               strategy: str = "Strategy") -> go.Figure:
        """
        Plot histogram of chunk sizes
        
        Args:
            chunks: List of chunk texts
            strategy: Strategy name for title
            
        Returns:
            Plotly figure
        """
        sizes = [len(chunk) for chunk in chunks]
        
        fig = go.Figure(data=[go.Histogram(
            x=sizes,
            nbinsx=30,
            marker_color='lightblue'
        )])
        
        # Add mean line
        mean_size = np.mean(sizes)
        fig.add_vline(
            x=mean_size,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_size:.0f}"
        )
        
        fig.update_layout(
            title=f"Chunk Size Distribution - {strategy}",
            xaxis_title="Size (characters)",
            yaxis_title="Frequency",
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_metrics_comparison(comparison: Dict[str, Dict]) -> go.Figure:
        """
        Compare evaluation metrics across strategies
        
        Args:
            comparison: Results from ChunkingEvaluator.compare_strategies()
            
        Returns:
            Plotly figure
        """
        strategies = list(comparison.keys())
        metrics = ['coherence', 'consistency', 'overlap_ratio']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [comparison[s][metric] for s in strategies]
            
            fig.add_trace(go.Bar(
                name=metric.title(),
                x=strategies,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Chunking Metrics Comparison",
            xaxis_title="Strategy",
            yaxis_title="Score",
            barmode='group',
            template='plotly_dark',
            height=500,
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    @staticmethod
    def plot_chunk_embeddings(embeddings: np.ndarray,
                             chunks: List[str],
                             method: str = 'umap',
                             title: str = "Chunk Embeddings") -> go.Figure:
        """
        Visualize chunk embeddings in 2D
        
        Args:
            embeddings: Array of chunk embeddings
            chunks: List of chunk texts (for hover)
            method: Dimensionality reduction method ('umap' or 'pca')
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Reduce dimensions
        if method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        else:  # pca
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        
        # Truncate chunk text for hover
        hover_texts = [
            chunk[:100] + "..." if len(chunk) > 100 else chunk
            for chunk in chunks
        ]
        
        fig = go.Figure(data=go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=np.arange(len(chunks)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Chunk ID")
            ),
            text=hover_texts,
            hovertemplate='<b>Chunk %{marker.color}</b><br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{title} ({method.upper()})",
            xaxis_title=f"{method.upper()} 1",
            yaxis_title=f"{method.upper()} 2",
            template='plotly_dark',
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_strategy_comparison(results_dict: Dict[str, List[str]],
                                 embeddings_dict: Dict[str, np.ndarray]) -> go.Figure:
        """
        Side-by-side comparison of multiple strategies
        
        Args:
            results_dict: Dict mapping strategy to chunks
            embeddings_dict: Dict mapping strategy to embeddings
            
        Returns:
            Plotly figure with subplots
        """
        n_strategies = len(results_dict)
        
        fig = make_subplots(
            rows=1,
            cols=n_strategies,
            subplot_titles=list(results_dict.keys())
        )
        
        for i, (strategy, chunks) in enumerate(results_dict.items(), 1):
            embeddings = embeddings_dict[strategy]
            
            # Reduce to 2D
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced = reducer.fit_transform(embeddings)
            
            fig.add_trace(
                go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=np.arange(len(chunks)),
                        colorscale='Viridis'
                    ),
                    showlegend=False
                ),
                row=1,
                col=i
            )
        
        fig.update_layout(
            title_text="Strategy Comparison - Chunk Embeddings",
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_chunk_timeline(chunks: List[str],
                           show_text: bool = True) -> go.Figure:
        """
        Show chunks as a timeline/sequence
        
        Args:
            chunks: List of chunk texts
            show_text: Whether to show chunk text
            
        Returns:
            Plotly figure
        """
        sizes = [len(chunk) for chunk in chunks]
        positions = [0] + list(np.cumsum(sizes))
        
        fig = go.Figure()
        
        for i, (start, end) in enumerate(zip(positions[:-1], positions[1:])):
            # Chunk bar
            fig.add_trace(go.Scatter(
                x=[start, end],
                y=[i, i],
                mode='lines',
                line=dict(width=20, color=f'rgb({i*30%255}, {i*50%255}, {200})'),
                hovertext=f"Chunk {i+1}: {len(chunks[i])} chars",
                showlegend=False
            ))
            
            # Add chunk text if requested
            if show_text and len(chunks) <= 10:
                text = chunks[i][:50] + "..." if len(chunks[i]) > 50 else chunks[i]
                fig.add_annotation(
                    x=(start + end) / 2,
                    y=i,
                    text=text,
                    showarrow=False,
                    font=dict(size=8)
                )
        
        fig.update_layout(
            title="Chunk Timeline",
            xaxis_title="Position in Document (characters)",
            yaxis_title="Chunk Number",
            template='plotly_dark',
            height=max(400, len(chunks) * 30),
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_comparison_dashboard(results_dict: Dict[str, List[str]],
                                   comparison_metrics: Dict[str, Dict]) -> go.Figure:
        """
        Create comprehensive comparison dashboard
        
        Args:
            results_dict: Dict mapping strategy to chunks
            comparison_metrics: Evaluation metrics from ChunkingEvaluator
            
        Returns:
            Plotly figure with multiple subplots
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Chunk Size Distribution",
                "Metrics Comparison",
                "Number of Chunks",
                "Size Statistics"
            ],
            specs=[
                [{"type": "box"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # 1. Chunk size distribution
        for strategy, chunks in results_dict.items():
            sizes = [len(chunk) for chunk in chunks]
            fig.add_trace(
                go.Box(y=sizes, name=strategy, showlegend=False),
                row=1, col=1
            )
        
        # 2. Metrics comparison
        strategies = list(results_dict.keys())
        coherence = [comparison_metrics[s]['coherence'] for s in strategies]
        consistency = [comparison_metrics[s]['consistency'] for s in strategies]
        
        fig.add_trace(
            go.Bar(x=strategies, y=coherence, name="Coherence"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=strategies, y=consistency, name="Consistency"),
            row=1, col=2
        )
        
        # 3. Number of chunks
        num_chunks = [len(chunks) for chunks in results_dict.values()]
        fig.add_trace(
            go.Bar(x=strategies, y=num_chunks, showlegend=False,
                  marker_color='lightblue'),
            row=2, col=1
        )
        
        # 4. Statistics table
        headers = ["Strategy", "Chunks", "Avg Size", "Coherence"]
        cells = [
            strategies,
            [len(results_dict[s]) for s in strategies],
            [f"{comparison_metrics[s]['size_stats']['mean']:.0f}" for s in strategies],
            [f"{comparison_metrics[s]['coherence']:.3f}" for s in strategies]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=headers, fill_color='darkblue'),
                cells=dict(values=cells, fill_color='darkslategray')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Chunking Strategy Comparison Dashboard",
            template='plotly_dark',
            height=800,
            showlegend=True
        )
        
        return fig


# Example usage
if __name__ == "__main__":
    # Sample data
    fixed_chunks = ["chunk " * 50 for _ in range(10)]
    semantic_chunks = ["chunk " * 40 for _ in range(8)]
    
    results = {
        'fixed': fixed_chunks,
        'semantic': semantic_chunks
    }
    
    viz = ChunkingVisualizer()
    
    # Create visualizations
    fig1 = viz.plot_chunk_sizes(results)
    fig1.write_html("chunk_sizes.html")
    
    fig2 = viz.plot_chunk_distribution(fixed_chunks, "Fixed")
    fig2.write_html("distribution.html")
    
    print("Visualizations created!")