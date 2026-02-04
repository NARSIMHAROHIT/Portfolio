"""
CLI for Simple RAG Bot
Command-line interface for document processing and querying
"""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from document_loader import DocumentLoader
from rag_engine import RAGEngine

app = typer.Typer(help="Simple RAG Bot CLI", add_completion=False)
console = Console()


@app.command()
def add(
    file_path: Path = typer.Argument(..., help="Document to add"),
    persist_dir: str = typer.Option("./chroma_db", help="Database directory"),
):
    """
    Add a document to the RAG system
    """
    console.print(f"[blue]Loading document: {file_path}[/blue]")
    
    loader = DocumentLoader()
    doc = loader.load(str(file_path))
    
    console.print(f"[green]Loaded: {doc.metadata['filename']}[/green]")
    console.print(f"  Size: {len(doc.content)} characters")
    
    with console.status("[green]Initializing RAG engine..."):
        rag = RAGEngine(persist_directory=persist_dir)
    
    console.print("[blue]Processing and storing document...[/blue]")
    rag.add_documents([doc.to_dict()])
    
    stats = rag.get_statistics()
    console.print(f"[green]Document added successfully[/green]")
    console.print(f"  Total chunks in DB: {stats['total_chunks']}")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    persist_dir: str = typer.Option("./chroma_db", help="Database directory"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of chunks to retrieve"),
    api_key: str = typer.Option(None, "--api-key", help="Groq API key"),
):
    """
    Query the RAG system
    """
    with console.status("[green]Initializing RAG engine..."):
        rag = RAGEngine(persist_directory=persist_dir)
    
    stats = rag.get_statistics()
    
    if stats['total_chunks'] == 0:
        console.print("[red]No documents in database. Add documents first.[/red]")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(f"[dim]Searching {stats['total_chunks']} chunks...[/dim]\n")
    
    provider = 'mock' if not api_key else 'groq'
    
    result = rag.query(
        question,
        top_k=top_k,
        llm_provider=provider,
        api_key=api_key
    )
    
    console.print("[bold cyan]Answer:[/bold cyan]")
    console.print(result['answer'])
    
    console.print(f"\n[bold]Retrieved Chunks ({len(result['retrieved_chunks'])}):[/bold]")
    for i, chunk in enumerate(result['retrieved_chunks'], 1):
        console.print(f"\n[yellow]Chunk {i}:[/yellow]")
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        console.print(f"  {preview}")


@app.command()
def add_dir(
    directory: Path = typer.Argument(..., help="Directory with documents"),
    persist_dir: str = typer.Option("./chroma_db", help="Database directory"),
):
    """
    Add all documents from a directory
    """
    console.print(f"[blue]Loading documents from: {directory}[/blue]")
    
    loader = DocumentLoader()
    documents = loader.load_directory(str(directory))
    
    console.print(f"[green]Loaded {len(documents)} documents[/green]")
    
    with console.status("[green]Initializing RAG engine..."):
        rag = RAGEngine(persist_directory=persist_dir)
    
    console.print("[blue]Processing and storing documents...[/blue]")
    rag.add_documents([doc.to_dict() for doc in documents])
    
    stats = rag.get_statistics()
    console.print(f"[green]All documents added successfully[/green]")
    console.print(f"  Total chunks in DB: {stats['total_chunks']}")


@app.command()
def stats(
    persist_dir: str = typer.Option("./chroma_db", help="Database directory"),
):
    """
    Show RAG system statistics
    """
    with console.status("[green]Loading RAG engine..."):
        rag = RAGEngine(persist_directory=persist_dir)
    
    stats = rag.get_statistics()
    
    table = Table(title="RAG System Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Chunks", str(stats['total_chunks']))
    table.add_row("Collection Name", stats['collection_name'])
    table.add_row("Embedding Model", stats['embedding_model'])
    table.add_row("Chunk Size", str(stats['chunk_size']))
    table.add_row("Chunk Overlap", str(stats['chunk_overlap']))
    
    console.print(table)


@app.command()
def clear(
    persist_dir: str = typer.Option("./chroma_db", help="Database directory"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """
    Clear all documents from the database
    """
    if not confirm:
        confirm = typer.confirm("Are you sure you want to clear the database?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)
    
    with console.status("[green]Clearing database..."):
        rag = RAGEngine(persist_directory=persist_dir)
        rag.clear_database()
    
    console.print("[green]Database cleared successfully[/green]")


@app.command()
def info():
    """
    Show information about the RAG system
    """
    console.print("[bold]Simple RAG Bot[/bold]")
    console.print("Version: 1.0.0")
    console.print("Project 4 of LLMOps Learning Journey")
    console.print("\nFeatures:")
    console.print("  - PDF, TXT, HTML document support")
    console.print("  - ChromaDB vector storage")
    console.print("  - Semantic search with embeddings")
    console.print("  - Groq API integration (optional)")


if __name__ == "__main__":
    app()