import pytest
from typer.testing import CliRunner
from micro_rag import app

runner = CliRunner()

# Patch the functions called within the 'chat' command
@pytest.fixture(autouse=True)
def patch_internal_functions(mocker):
    mocker.patch('micro_rag.setup_models', return_value=(None, None))
    mocker.patch('micro_rag.load_documents', return_value=[])
    mocker.patch('micro_rag.build_index', return_value=None)
    mocker.patch('micro_rag.chat_loop', return_value=None)

def test_default_args(mocker):
    """Test that the default arguments trigger the command successfully."""
    # Mocks are automatically applied by the fixture
    mock_setup = mocker.patch('micro_rag.setup_models')
    mock_load = mocker.patch('micro_rag.load_documents')
    mock_build = mocker.patch('micro_rag.build_index')
    mock_loop = mocker.patch('micro_rag.chat_loop')

    result = runner.invoke(app, ["/path/to/docs"])
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}\nOutput:\n{result.stdout}"

    # Basic check: Ensure the core functions were called, implying the command ran
    mock_setup.assert_called_once()
    mock_load.assert_called_once_with("/path/to/docs")
    mock_build.assert_called_once()
    mock_loop.assert_called_once()

    # Check if setup_models was called with default model names and inferred host
    args, kwargs = mock_setup.call_args
    assert args[0] == "olmo2:7b" # Default chat model
    assert args[1] == "nomic-embed-text" # Default embed model
    # We don't assert the host here as it depends on env vars, test ollama_host_precedence covers host logic

def test_custom_args(mocker):
    """Test that custom command-line arguments trigger the command successfully."""
    # Mocks are automatically applied by the fixture
    mock_setup = mocker.patch('micro_rag.setup_models')
    mock_load = mocker.patch('micro_rag.load_documents')
    mock_build = mocker.patch('micro_rag.build_index')
    mock_loop = mocker.patch('micro_rag.chat_loop')

    result = runner.invoke(app, [
        "/another/path",
        "--chat-model", "my-chat-model",
        "--embed-model", "my-embed-model",
        "--ollama-host", "http://custom:1234",
        "--chunk-size", "1024",
        "--chunk-overlap", "100"
    ])
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}\nOutput:\n{result.stdout}"

    # Basic check: Ensure the core functions were called
    mock_setup.assert_called_once()
    mock_load.assert_called_once_with("/another/path")
    mock_build.assert_called_once()
    mock_loop.assert_called_once()

    # Check if setup_models was called with custom args
    args, kwargs = mock_setup.call_args
    assert args[0] == "my-chat-model"
    assert args[1] == "my-embed-model"
    assert args[2] == "http://custom:1234" # Custom host

    # Check if build_index was called with custom chunk args
    args, kwargs = mock_build.call_args
    # chunk_size and chunk_overlap are passed as positional args
    # args[0] is the documents list (mocked as [])
    assert args[1] == 1024 # chunk_size is the second positional arg
    assert args[2] == 100  # chunk_overlap is the third positional arg


def test_missing_docs_dir():
    """Test that the command fails if the documents directory is missing."""
    # No need to mock here as Typer should handle the missing argument error
    result = runner.invoke(app)
    assert result.exit_code != 0  # Expecting failure
    assert "Missing argument 'DOCUMENTS_DIR'" in result.stdout

def test_help_option():
    """Test that the --help option works."""
     # No need to mock here as Typer should handle the help option
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Corrected assertion to match Typer's default output
    assert "Usage: chat [OPTIONS] DOCUMENTS_DIR" in result.stdout
    assert "Start a RAG-based chat with your documents." in result.stdout 