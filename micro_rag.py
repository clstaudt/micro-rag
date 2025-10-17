#!/usr/bin/env python3

import logging
import os
import sys
import warnings
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

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("llama_index").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Initialize Typer app
app = typer.Typer(help="A minimal RAG-based conversational AI chat tool.")
console = Console()

def get_embedding_model_context_length(model_name: str, ollama_url: str) -> int:
    """Get the context length for an embedding model from Ollama."""
    try:
        # Query the model info from Ollama
        client = ollama.Client(host=ollama_url)
        model_info = client.show(model_name)
        
        # Try to extract context length from model parameters
        if "model_info" in model_info and isinstance(model_info["model_info"], dict):
            params = model_info["model_info"]
            # Common parameter names for context length
            for key in ["context_length", "max_position_embeddings", "n_ctx"]:
                if key in params:
                    return int(params[key])
        
        # Default context lengths for known models
        known_models = {
            "mxbai-embed-large": 512,
            "nomic-embed-text": 8192,
            "all-minilm": 512,
            "snowflake-arctic-embed": 512,
        }
        
        for known_model, context_length in known_models.items():
            if known_model in model_name:
                return context_length
        
        # Conservative default if unknown
        return 512
    except Exception:
        # Conservative default on error
        return 512

def setup_models(chat_model_name: str, embed_model_name: str, ollama_url: str) -> Tuple[Ollama, OllamaEmbedding]:
    """Set up Ollama chat and embedding models."""
    with Halo(text=f"Setting up chat model: {chat_model_name}", spinner="dots") as spinner:
        try:
            response = ollama.pull(chat_model_name)
            assert response["status"] == "success"
            
            chat_model = Ollama(
                model=chat_model_name,
                base_url=ollama_url,
                request_timeout=120.0,
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
                request_timeout=120.0,
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
            
            # Load documents
            documents = SimpleDirectoryReader(
                input_dir=document_dir
            ).load_data()
            
            # Complete the progress bar
            progress.update(task, completed=len(doc_file_paths))
            
        console.print(f"[bold green]✓[/bold green] Loaded {len(documents)} document(s)")
        return documents
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to load documents: {e}")
        sys.exit(1)

def build_index(documents: List, chunk_size: int = 512, chunk_overlap: int = 50) -> VectorStoreIndex:
    """Build a vector index from the documents."""
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    
    try:
        with Halo(text="Building vector index...", spinner="dots") as spinner:
            index = VectorStoreIndex.from_documents(documents=documents)
            spinner.succeed("Vector index built successfully")
            return index
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to build vector index: {e}")
        console.print("[yellow]Tip:[/yellow] Try restarting the Ollama service or reducing chunk_size")
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
    embed_model: str = typer.Option("mxbai-embed-large", help="Ollama embedding model to use"),
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
    
    # Get and validate chunk size against embedding model context length
    embedding_context_length = get_embedding_model_context_length(embed_model, ollama_url)
    console.print(f"Embedding model context length: {embedding_context_length} tokens")
    
    if chunk_size > embedding_context_length:
        console.print(f"[bold yellow]Warning:[/bold yellow] Chunk size ({chunk_size}) exceeds embedding model context length ({embedding_context_length})")
        chunk_size = embedding_context_length
        console.print(f"[bold green]✓[/bold green] Adjusted chunk size to {chunk_size}")
    
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
