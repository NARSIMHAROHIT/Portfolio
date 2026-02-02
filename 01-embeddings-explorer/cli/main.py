"""
CLI for Text Embeddings Explorer
Command-line interface for generating and analyzing embeddings
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
from pathlib import Path
import sys
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embedder import Embedder
from similarity import SimilarityCalculator
from utils import TokenCounter, save_json, load_texts_from_file

app = typer.Typer(
    help="Text Embeddings Explorer - Generate and analyze text embeddings",
    add_completion=False
)
console = Console()


@app.command()
def embed(
    text: str = typer.Argument(..., help="Text to embed"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", "-m", help="Model to use"),
    output: Path = typer.Option(None, "--output", "-o", help="Save embedding to file"),
    show_stats: bool = typer.Option(True, "--stats/--no-stats", help="Show statistics"),
):
    """
    Generate embedding for a single text
    """
    console.print(f"[bold blue]Generating embedding for:[/bold blue] {text[:100]}...")
    
    # Load model
    with console.status("[bold green]Loading model..."):
        embedder = Embedder(model)
    
    # Generate embedding
    result = embedder.embed(text)
    
    console.print(f"[green]✓[/green] Embedding generated!")
    console.print(f"  Model: {result.model_name}")
    console.print(f"  Dimensions: {result.dimensions}")
    
    if show_stats:
        stats = embedder.get_embedding_stats(result.embeddings)
        
        table = Table(title="Embedding Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in stats.items():
            table.add_row(key.title(), f"{value:.4f}")
        
        console.print(table)
    
    # Token analysis
    counter = TokenCounter()
    tokens = counter.count_tokens(text)
    cost = counter.estimate_cost(text)
    
    console.print(f"\n[bold]Token Analysis:[/bold]")
    console.print(f"  Tokens: {tokens:,}")
    console.print(f"  Est. Cost: ${cost:.6f}")
    
    # Save if requested
    if output:
        if output.suffix == '.npy':
            np.save(output, result.embeddings)
        elif output.suffix == '.json':
            data = result.to_dict()
            save_json(data, output)
        else:
            console.print("[red]Error: Output must be .npy or .json[/red]")
            raise typer.Exit(1)
        
        console.print(f"[green]✓[/green] Saved to {output}")


@app.command()
def similarity(
    text1: str = typer.Argument(..., help="First text"),
    text2: str = typer.Argument(..., help="Second text"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", "-m", help="Model to use"),
    metric: str = typer.Option("cosine", "--metric", help="Similarity metric"),
):
    """
    Calculate similarity between two texts
    """
    console.print("[bold blue]Calculating similarity...[/bold blue]")
    
    # Load model
    with console.status("[bold green]Loading model..."):
        embedder = Embedder(model)
    
    # Generate embeddings
    emb1 = embedder.embed(text1).embeddings
    emb2 = embedder.embed(text2).embeddings
    
    # Calculate similarity
    calc = SimilarityCalculator()
    
    if metric == "all":
        metrics = calc.calculate_all_metrics(emb1, emb2)
        
        table = Table(title="Similarity Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Interpretation", style="yellow")
        
        for m, score in metrics.items():
            info = calc.AVAILABLE_METRICS[m]
            table.add_row(info['name'], f"{score:.4f}", info['interpretation'])
        
        console.print(table)
        
        # Special interpretation for cosine
        cosine_score = metrics['cosine']
        interpretation = calc.interpret_cosine_similarity(cosine_score)
        console.print(f"\n[bold]Cosine Similarity:[/bold] {interpretation}")
    else:
        score = calc.calculate_similarity(emb1, emb2, metric)
        info = calc.AVAILABLE_METRICS[metric]
        
        console.print(f"\n[bold]{info['name']}:[/bold] {score:.4f}")
        console.print(f"Interpretation: {info['interpretation']}")
        
        if metric == 'cosine':
            interpretation = calc.interpret_cosine_similarity(score)
            console.print(f"[yellow]{interpretation}[/yellow]")


@app.command()
def batch(
    input_file: Path = typer.Argument(..., help="File with texts (one per line)"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file (.npy or .json)"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", "-m", help="Model to use"),
):
    """
    Process multiple texts from a file
    """
    # Load texts
    console.print(f"[bold blue]Loading texts from {input_file}...[/bold blue]")
    texts = load_texts_from_file(input_file)
    console.print(f"[green]✓[/green] Loaded {len(texts)} texts")
    
    # Load model
    with console.status("[bold green]Loading model..."):
        embedder = Embedder(model)
    
    # Generate embeddings
    console.print("[bold blue]Generating embeddings...[/bold blue]")
    
    all_embeddings = []
    for text in track(texts, description="Processing"):
        result = embedder.embed(text)
        all_embeddings.append(result.embeddings)
    
    embeddings = np.array(all_embeddings)
    
    console.print(f"[green]✓[/green] Generated {len(embeddings)} embeddings")
    console.print(f"  Shape: {embeddings.shape}")
    
    # Token statistics
    counter = TokenCounter()
    token_stats = counter.batch_token_count(texts)
    
    table = Table(title="Batch Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Texts", str(token_stats['texts_count']))
    table.add_row("Total Tokens", f"{token_stats['total_tokens']:,}")
    table.add_row("Avg Tokens", f"{token_stats['avg_tokens']:.1f}")
    table.add_row("Min Tokens", str(token_stats['min_tokens']))
    table.add_row("Max Tokens", str(token_stats['max_tokens']))
    
    console.print(table)
    
    # Save if requested
    if output_file:
        if output_file.suffix == '.npy':
            np.save(output_file, embeddings)
        elif output_file.suffix == '.json':
            data = {
                'texts': texts,
                'embeddings': embeddings.tolist(),
                'model': model,
                'dimensions': embeddings.shape[1],
                'count': len(embeddings)
            }
            save_json(data, output_file)
        else:
            console.print("[red]Error: Output must be .npy or .json[/red]")
            raise typer.Exit(1)
        
        console.print(f"[green]✓[/green] Saved to {output_file}")


@app.command()
def info(
    model: str = typer.Option(None, "--model", "-m", help="Specific model to show info for")
):
    """
    Show information about available models
    """
    if model:
        info_dict = Embedder.get_model_info(model)
        if not info_dict:
            console.print(f"[red]Model '{model}' not found[/red]")
            raise typer.Exit(1)
        
        console.print(f"\n[bold]Model: {model}[/bold]")
        console.print(f"  Name: {info_dict['name']}")
        console.print(f"  Dimensions: {info_dict['dimensions']}")
        console.print(f"  Speed: {info_dict['speed']}")
        console.print(f"  Description: {info_dict['description']}")
    else:
        table = Table(title="Available Models")
        table.add_column("Model", style="cyan")
        table.add_column("Dimensions", style="magenta")
        table.add_column("Speed", style="yellow")
        table.add_column("Description", style="green")
        
        for name, info_dict in Embedder.get_model_info().items():
            table.add_row(
                name,
                str(info_dict['dimensions']),
                info_dict['speed'],
                info_dict['description']
            )
        
        console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Query text"),
    file: Path = typer.Argument(..., help="File with texts to search"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", "-m", help="Model to use"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of results to return"),
):
    """
    Search for most similar texts to a query
    """
    console.print(f"[bold blue]Searching for:[/bold blue] {query}")
    
    # Load texts
    texts = load_texts_from_file(file)
    console.print(f"[green]✓[/green] Loaded {len(texts)} texts to search")
    
    # Load model
    with console.status("[bold green]Loading model..."):
        embedder = Embedder(model)
    
    # Generate embeddings
    console.print("[bold blue]Generating embeddings...[/bold blue]")
    query_emb = embedder.embed(query).embeddings
    
    all_embeddings = []
    for text in track(texts, description="Processing"):
        result = embedder.embed(text)
        all_embeddings.append(result.embeddings)
    
    embeddings = np.array(all_embeddings)
    
    # Find most similar
    calc = SimilarityCalculator()
    results = calc.find_most_similar(query_emb, embeddings, texts, top_k=top_k)
    
    # Display results
    table = Table(title=f"Top {top_k} Most Similar Texts")
    table.add_column("#", style="cyan")
    table.add_column("Text", style="green")
    table.add_column("Similarity", style="magenta")
    
    for i, (text, score) in enumerate(results, 1):
        # Truncate long texts
        display_text = text if len(text) <= 60 else text[:57] + "..."
        table.add_row(str(i), display_text, f"{score:.4f}")
    
    console.print(table)


@app.command()
def version():
    """
    Show version information
    """
    console.print("[bold]Text Embeddings Explorer[/bold]")
    console.print("Version: 1.0.0")
    console.print("Project 1 of LLMOps Learning Journey")


if __name__ == "__main__":
    app()