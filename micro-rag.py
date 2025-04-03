#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import ollama
import typer
from halo import Halo
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Initialize Typer app
app = typer.Typer(help="A minimal RAG-based conversational AI chat tool.")
console = Console()

def setup_models(chat_model_name: str, embed_model_name: str, ollama_url: str) -> Tuple[Ollama, OllamaEmbedding]:
    """Set up Ollama chat and embedding models."""
    with Halo(text=f"Setting up chat model: {chat_model_name}", spinner="dots") as spinner:
        try:
            response = ollama.pull(chat_model_name)
            assert response["status"] == "success"
            
            chat_model = Ollama(
                model=chat_model_name,
                base_url=ollama_url,
                request_timeout=90,
            )
            Settings.llm = chat_model
            spinner.succeed(f"Chat model {chat_model_name} loaded successfully")
        except Exception as e:
            spinner.fail(f"Failed to load chat model: {e}")
            sys.exit(1)
    
    with Halo(text=f"Setting up embedding model: {embed_model_name}", spinner="dots") as spinner:
        try:
            response = ollama.pull(embed_model_name)
            assert response["status"] == "success"
            
            embed_model = OllamaEmbedding(
                model_name=embed_model_name,
                base_url=ollama_url,
            )
            Settings.embed_model = embed_model
            spinner.succeed(f"Embedding model {embed_model_name} loaded successfully")
        except Exception as e:
            spinner.fail(f"Failed to load embedding model: {e}")
            sys.exit(1)
    
    return chat_model, embed_model

def load_documents(document_dir: str) -> List:
    """Load documents from the specified directory."""
    try:
        # Get list of files first
        doc_file_paths = list(Path(document_dir).glob('**/*.*'))
        if not doc_file_paths:
            console.print(f"[bold red]Error:[/bold red] No documents found in {document_dir}")
            sys.exit(1)
            
        # Initialize progress bar for document loading
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"[green]Loading documents from {document_dir}", total=len(doc_file_paths))
            
            # Create a custom callback for progress updates
            def progress_callback(i, path):
                progress.update(task, advance=1, description=f"[green]Loading document: {Path(path).name}")
                return True
            
            # Load documents with progress updates
            documents = SimpleDirectoryReader(
                input_files=doc_file_paths,
                #callback=progress_callback
            ).load_data()
            
        console.print(f"[bold green]✓[/bold green] Loaded {len(documents)} document(s)")
        return documents
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to load documents: {e}")
        sys.exit(1)

def build_index(documents: List, chunk_size: int = 512, chunk_overlap: int = 50) -> VectorStoreIndex:
    """Build a vector index from the documents."""
    try:
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        # Set up the progress bar for index building
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TimeElapsedColumn(),
        ) as progress:
            # Add main indexing task
            index_task = progress.add_task("[yellow]Building vector index", total=1000)
            
            # We'll simulate progress since we don't have direct access to the indexing steps
            def update_progress():
                steps = 50
                for i in range(steps):
                    progress.update(index_task, advance=1000/steps)
                    time.sleep(0.1)
            
            # Create a background task to update progress while indexing happens
            # This is a visual approximation since we don't have access to actual progress
            import threading
            progress_thread = threading.Thread(target=update_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            # Build the index
            index = VectorStoreIndex.from_documents(
                documents=documents,
            )
            
            # Ensure progress reaches 100%
            progress.update(index_task, completed=1000)
            
        console.print("[bold green]✓[/bold green] Vector index built successfully")
        return index
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to build vector index: {e}")
        sys.exit(1)

def chat_loop(chat_engine):
    """Run the chat loop for user interaction."""
    console.print("\n[bold green]RAG Chat initialized. Type 'exit' or 'quit' to end the session.[/bold green]\n")
    
    chat_history = []
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            
            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold green]Goodbye![/bold green]")
                break
            
            with Halo(text="Thinking...", spinner="dots") as spinner:
                response = chat_engine.stream_chat(user_input)
                spinner.stop()
            
            console.print("\n[bold green]AI[/bold green]:", end=" ")
            
            # Print response stream
            full_response = ""
            for token in response.response_gen:
                full_response += token
                console.print(token, end="")
                sys.stdout.flush()
            
            # Add to chat history
            chat_history.append((user_input, full_response))
            
            # Print source information
            if hasattr(response, 'get_formatted_sources') and response.get_formatted_sources():
                sources = response.get_formatted_sources()
                console.print("\n\n[dim italic]Sources:[/dim italic]")
                console.print(Markdown(sources))
            
            console.print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Chat session interrupted.[/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error: {str(e)}[/bold red]")

@app.command()
def chat(
    documents_dir: str = typer.Argument(..., help="Directory containing the documents to chat with"),
    chat_model: str = typer.Option("olmo2:7b", help="Ollama chat model to use"),
    embed_model: str = typer.Option("nomic-embed-text", help="Ollama embedding model to use"),
    ollama_host: str = typer.Option(None, help="Ollama host URL (default: http://localhost:11434)"),
    chunk_size: int = typer.Option(512, help="Size of document chunks"),
    chunk_overlap: int = typer.Option(50, help="Overlap between document chunks")
):
    """Start a RAG-based chat with your documents."""
    # Display header
    console.print("[bold]Minimal RAG Chat Tool[/bold]")
    console.print(f"Documents: {documents_dir}")
    console.print(f"Chat Model: {chat_model}")
    console.print(f"Embedding Model: {embed_model}")
    console.print("-" * 50)
    
    # Set Ollama URL
    ollama_url = ollama_host or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    console.print(f"Using Ollama at: {ollama_url}")
    
    # Setup models
    setup_models(chat_model, embed_model, ollama_url)
    
    # Load documents
    documents = load_documents(documents_dir)
    
    # Build index
    index = build_index(documents, chunk_size, chunk_overlap)
    
    # Create chat engine
    chat_engine = index.as_chat_engine(streaming=True)
    
    # Start chat loop
    chat_loop(chat_engine)

if __name__ == "__main__":
    app() 