"""
CLI for Similarity Search Engine
Command-line interface for searching documents
"""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent / "src"))

from search_engine import SearchEngine

app = typer.Typer(help="🔍 Similarity Search Engine CLI", add_completion=False)
console = Console()


@app.command()
def index(
    input_file: Path = typer.Argument(..., help="File with documents (one per line)"),
    output: Path = typer.Option("index.json", "--output", "-o", help="Save index"),
):
    """
    Index documents for searching
    """
    console.print(f"[blue]Reading documents from {input_file}...[/blue]")
    
    documents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(line)
    
    console.print(f"[green]Loaded {len(documents)} documents[/green]")
    
    with console.status("[green]Initializing search engine..."):
        engine = SearchEngine()
    
    console.print("[blue]Indexing documents...[/blue]")
    engine.index_documents(documents)
    
    console.print(f"[green]Indexed successfully![/green]")
    
    stats = engine.get_statistics()
    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"  Documents: {stats['num_documents']}")
    console.print(f"  Embedding dims: {stats['embedding_dimensions']}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    docs_file: Path = typer.Argument(..., help="Documents file"),
    method: str = typer.Option("semantic", "--method", "-m", help="Search method"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of results"),
):
    """
    Search for similar documents
    """
    console.print(f"[blue]Loading documents...[/blue]")
    
    documents = []
    with open(docs_file, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
    
    with console.status("[green]Initializing..."):
        engine = SearchEngine()
        engine.index_documents(documents)
    
    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"[bold]Method:[/bold] {method}\n")
    
    results = engine.search(query, method=method, top_k=top_k)
    
    table = Table(title=f"Search Results ({len(results.results)})")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Score", style="magenta", width=10)
    table.add_column("Document", style="white")
    
    for result in results.results:
        doc_preview = result.text[:80] + "..." if len(result.text) > 80 else result.text
        table.add_row(
            str(result.rank),
            f"{result.score:.4f}",
            doc_preview
        )
    
    console.print(table)


@app.command()
def compare(
    query: str = typer.Argument(..., help="Search query"),
    docs_file: Path = typer.Argument(..., help="Documents file"),
    top_k: int = typer.Option(3, "--top", "-k", help="Results per method"),
):
    """
    Compare all search methods
    """
    console.print(f"[blue]Loading documents...[/blue]")
    
    documents = []
    with open(docs_file, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
    
    with console.status("[green]Initializing..."):
        engine = SearchEngine()
        engine.index_documents(documents)
    
    console.print(f"\n[bold]Query:[/bold] {query}\n")
    
    methods = ['semantic', 'keyword', 'hybrid']
    
    for method in methods:
        console.print(f"\n[bold cyan]{method.upper()} SEARCH[/bold cyan]")
        console.print("-" * 60)
        
        results = engine.search(query, method=method, top_k=top_k)
        
        for result in results.results:
            console.print(f"\n[yellow]{result.rank}. Score: {result.score:.4f}[/yellow]")
            preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
            console.print(f"   {preview}")


@app.command()
def similar(
    doc_id: int = typer.Argument(..., help="Document ID"),
    docs_file: Path = typer.Argument(..., help="Documents file"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of similar docs"),
):
    """
    Find documents similar to a given document
    """
    console.print(f"[blue]Loading documents...[/blue]")
    
    documents = []
    with open(docs_file, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
    
    if doc_id >= len(documents):
        console.print(f"[red]Error: Document ID {doc_id} out of range (0-{len(documents)-1})[/red]")
        raise typer.Exit(1)
    
    with console.status("[green]Initializing..."):
        engine = SearchEngine()
        engine.index_documents(documents)
    
    console.print(f"\n[bold]Source Document (ID {doc_id}):[/bold]")
    console.print(f"  {documents[doc_id]}\n")
    
    results = engine.find_similar_documents(doc_id, top_k=top_k)
    
    console.print(f"[bold]Similar Documents:[/bold]\n")
    
    for result in results.results:
        console.print(f"[yellow]{result.rank}. Score: {result.score:.4f}[/yellow]")
        preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
        console.print(f"   {preview}\n")


@app.command()
def methods():
    """
    List available search methods
    """
    table = Table(title="Available Search Methods")
    table.add_column("Method", style="cyan")
    table.add_column("Description", style="yellow")
    
    for name, desc in SearchEngine.get_available_methods().items():
        table.add_row(name.title(), desc)
    
    console.print(table)


@app.command()
def info():
    """
    Show project information
    """
    console.print("[bold]Similarity Search Engine[/bold]")
    console.print("Version: 1.0.0")
    console.print("Project 3 of LLMOps Learning Journey")
    console.print("\nSearch documents using semantic similarity")


if __name__ == "__main__":
    app()