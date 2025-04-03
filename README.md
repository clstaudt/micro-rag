<div align="center">
  <img src="img/logo.png" alt="Micro RAG Logo" width="150"/>
</div>

# Micro RAG

A minimalist command-line tool for Retrieval-Augmented Generation (RAG) chat with your documents using local LLMs via Ollama.

## Features

- Chat with your documents using RAG technology
- Uses local LLMs via Ollama
- Simple and intuitive command-line interface
- Supports various document formats
- Streaming responses with source citations

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/minimal-rag.git
cd minimal-rag

# Install dependencies
pip install -r requirements.txt

# Make the script executable
chmod +x minimal-rag.py
```

## Usage

Basic usage:

```bash
python minimal-rag.py /path/to/documents
```

With custom models:

```bash
python minimal-rag.py /path/to/documents --chat-model "llama3:latest" --embed-model "nomic-embed-text"
```

Full options:

```bash
python minimal-rag.py --help
```

### Command-line Arguments

- `documents_dir`: Directory containing the documents to chat with (required)
- `--chat-model`: Ollama chat model to use (default: "orca-mini:13b")
- `--embed-model`: Ollama embedding model to use (default: "nomic-embed-text")
- `--ollama-host`: Ollama host URL (default: http://localhost:11434 or OLLAMA_HOST env variable)
- `--chunk-size`: Size of document chunks (default: 512)
- `--chunk-overlap`: Overlap between document chunks (default: 50)

## Requirements

- Python 3.8+
- Ollama running locally or remotely
- Available models in Ollama for chat and embeddings

## Notes

- The first run will download the models if they aren't already available in Ollama
- Type 'exit' or 'quit' to end the chat session
- Press Ctrl+C to interrupt the chat

## License

MIT
