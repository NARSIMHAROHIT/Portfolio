"""
CLI for Smart Chunker
Command-line interface for text chunking
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
from pathlib import Path
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chunker import Chunker
from evaluator import ChunkingEvaluator

app = typer.Typer(
    help="✂️ Smart Chunker - Compare text chunking strategies",
    add_completion=False
)
console = Console()


@app.command()
def chunk(
    input_file: Path = typer.Argument(..., help="Input text file"),
    strategy: str = typer.Option("recursive", "--strategy", "-s", help="Chunking strategy"),
    chunk_size: int = typer.Option(400, "--size", help="Target chunk size"),
    overlap: int = typer.Option(50, "--overlap", help="Chunk overlap"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file (.json)"),
):
    """
    Chunk a text file using specified strategy
    """
    # Read input
    console.print(f"[blue]Reading {input_file}...[/blue]")
    text = input_file.read_text(encoding='utf-8')
    
    # Initialize chunker
    with console.status("[green]Loading models..."):
        chunker = Chunker(embedding_model='all-MiniLM-L6-v2' if strategy == 'semantic' else None)
    
    # Chunk
    console.print(f"[blue]Chunking with {strategy} strategy...[/blue]")
    result = chunker.chunk(
        text,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    # Display results
    console.print(f"\n[green]✓[/green] Chunking complete!")
    console.print(f"  Strategy: {strategy}")
    console.print(f"  Chunks: {result.num_chunks}")
    console.print(f"  Avg size: {result.avg_chunk_size:.0f} chars")
    
    # Show sample chunks
    console.print(f"\n[bold]Sample Chunks:[/bold]")
    for i, chunk in enumerate(result.chunks[:3]):
        console.print(f"\n[cyan]Chunk {i+1}[/cyan] ({len(chunk)} chars):")
        preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
        console.print(f"  {preview}")
    
    if len(result.chunks) > 3:
        console.print(f"\n  ... and {len(result.chunks) - 3} more chunks")
    
    # Save if requested
    if output:
        data = result.to_dict()
        output.write_text(json.dumps(data, indent=2))
        console.print(f"\n[green]✓[/green] Saved to {output}")


@app.command()
def compare(
    input_file: Path = typer.Argument(..., help="Input text file"),
    chunk_size: int = typer.Option(400, "--size", help="Target chunk size"),
    overlap: int = typer.Option(50, "--overlap", help="Chunk overlap"),
    output: Path = typer.Option(None, "--output", "-o", help="Output comparison (.json)"),
):
    """
    Compare multiple chunking strategies
    """
    # Read input
    console.print(f"[blue]Reading {input_file}...[/blue]")
    text = input_file.read_text(encoding='utf-8')
    
    # Initialize
    with console.status("[green]Loading models..."):
        chunker = Chunker(embedding_model='all-MiniLM-L6-v2')
        evaluator = ChunkingEvaluator('all-MiniLM-L6-v2')
    
    # Strategies to compare
    strategies = ['fixed', 'recursive', 'semantic', 'sliding']
    
    console.print("\n[bold]Chunking with all strategies...[/bold]\n")
    
    results = {}
    chunks_dict = {}
    
    for strategy in track(strategies, description="Processing"):
        result = chunker.chunk(
            text,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        results[strategy] = result
        chunks_dict[strategy] = [c.text for c in result.chunks]
    
    # Evaluate
    console.print("\n[bold]Evaluating strategies...[/bold]\n")
    comparison = evaluator.compare_strategies(chunks_dict)
    
    # Display comparison table
    table = Table(title="Strategy Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Chunks", justify="right", style="magenta")
    table.add_column("Avg Size", justify="right", style="yellow")
    table.add_column("Coherence", justify="right", style="green")
    table.add_column("Consistency", justify="right", style="green")
    
    for strategy in strategies:
        metrics = comparison[strategy]
        table.add_row(
            strategy.title(),
            str(metrics['num_chunks']),
            f"{metrics['size_stats']['mean']:.0f}",
            f"{metrics['coherence']:.3f}",
            f"{metrics['consistency']:.3f}"
        )
    
    console.print(table)
    
    # Recommendations
    coherence_scores = {s: comparison[s]['coherence'] for s in strategies}
    best_coherence = max(coherence_scores, key=coherence_scores.get)
    
    console.print(f"\n[green]✓ Best coherence:[/green] {best_coherence.title()} ({coherence_scores[best_coherence]:.3f})")
    
    # Save if requested
    if output:
        save_data = {
            'results': {s: r.to_dict() for s, r in results.items()},
            'comparison': comparison
        }
        output.write_text(json.dumps(save_data, indent=2))
        console.print(f"\n[green]✓[/green] Comparison saved to {output}")


@app.command()
def evaluate(
    input_file: Path = typer.Argument(..., help="Input text file"),
    strategy: str = typer.Option("recursive", "--strategy", "-s", help="Strategy to evaluate"),
    chunk_size: int = typer.Option(400, "--size", help="Target chunk size"),
):
    """
    Evaluate a specific chunking strategy
    """
    # Read input
    text = input_file.read_text(encoding='utf-8')
    
    # Initialize
    with console.status("[green]Loading models..."):
        chunker = Chunker(embedding_model='all-MiniLM-L6-v2')
        evaluator = ChunkingEvaluator('all-MiniLM-L6-v2')
    
    # Chunk
    console.print(f"[blue]Chunking with {strategy}...[/blue]")
    result = chunker.chunk(text, strategy=strategy, chunk_size=chunk_size)
    chunks = [c.text for c in result.chunks]
    
    # Evaluate
    console.print(f"[blue]Evaluating quality...[/blue]")
    metrics = evaluator.evaluate_chunking(chunks)
    size_stats = evaluator.calculate_size_variance(chunks)
    
    # Display results
    console.print(f"\n[bold]Evaluation Results - {strategy.title()}[/bold]\n")
    
    # Metrics table
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Description", style="yellow")
    
    descriptions = evaluator.get_metric_descriptions()
    
    for metric, score in metrics.items():
        table.add_row(
            metric.title(),
            f"{score:.3f}",
            descriptions.get(metric, "")
        )
    
    console.print(table)
    
    # Size statistics
    console.print(f"\n[bold]Size Statistics:[/bold]")
    console.print(f"  Mean: {size_stats['mean']:.0f} chars")
    console.print(f"  Std Dev: {size_stats['std']:.0f}")
    console.print(f"  Min: {size_stats['min']:.0f}")
    console.print(f"  Max: {size_stats['max']:.0f}")
    console.print(f"  Total chunks: {len(chunks)}")


@app.command()
def strategies():
    """
    List available chunking strategies
    """
    table = Table(title="Available Chunking Strategies")
    table.add_column("Strategy", style="cyan")
    table.add_column("Description", style="yellow")
    
    for name, desc in Chunker.get_available_strategies().items():
        table.add_row(name.title(), desc)
    
    console.print(table)


@app.command()
def info():
    """
    Show project information
    """
    console.print("[bold]Smart Chunker[/bold]")
    console.print("Version: 1.0.0")
    console.print("Project 2 of LLMOps Learning Journey")
    console.print("\nCompare and evaluate text chunking strategies")


if __name__ == "__main__":
    app()