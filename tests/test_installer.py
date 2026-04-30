"""Tests for the installer module."""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch


class TestInstaller:
    """Tests for the installer module."""
    
    def test_check_env_var_missing(self):
        """Test checking for missing environment variable."""
        from scripts.installer import check_env_var
        
        with patch.dict(os.environ, {}, clear=True):
            result = check_env_var("FAKE_VAR")
            assert result is None
    
    def test_check_env_var_present(self):
        """Test checking for present environment variable."""
        from scripts.installer import check_env_var
        
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = check_env_var("TEST_VAR")
            assert result == "test_value"
    
    def test_prompt_for_api_key(self):
        """Test prompting for API key."""
        from scripts.installer import prompt_for_api_key
        
        with patch('builtins.input', return_value="fake-key-123"):
            result = prompt_for_api_key("OpenAI")
            assert result == "fake-key-123"
    
    def test_create_venv(self):
        """Test creating virtual environment."""
        from scripts.installer import create_venv
        
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = os.path.join(tmpdir, ".venv")
            with patch('subprocess.run') as mock_run:
                create_venv(venv_path)
                
                mock_run.assert_called_once()
                call_args = mock_run.call_args
                # Check that uv venv command was called with correct path
                assert call_args[0][0] == ["uv", "venv", venv_path]
    
    def test_install_dependencies(self):
        """Test installing dependencies."""
        from scripts.installer import install_dependencies
        
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = os.path.join(tmpdir, ".venv")
            # Create fake activate script
            activate_path = os.path.join(venv_path, "Scripts", "activate.bat") if os.name == 'nt' else os.path.join(venv_path, "bin", "activate")
            os.makedirs(os.path.dirname(activate_path), exist_ok=True)
            with open(activate_path, 'w') as f:
                f.write("")
            
            with patch('subprocess.run') as mock_run:
                install_dependencies(venv_path)
                
                # Should call uv pip install
                mock_run.assert_called()
    
    def test_create_env_file(self):
        """Test creating .env file."""
        from scripts.installer import create_env_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            config = {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key"
            }
            
            create_env_file(env_path, config)
            
            assert os.path.exists(env_path)
            with open(env_path, 'r') as f:
                content = f.read()
                assert "LLM_PROVIDER=openai" in content
                assert "OPENAI_API_KEY=test-key" in content
    
    def test_check_ollama_available(self):
        """Test checking if Ollama is available."""
        from scripts.installer import check_ollama_available
        
        with patch('subprocess.run', side_effect=FileNotFoundError):
            result = check_ollama_available()
            assert result is False
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            result = check_ollama_available()
            assert result is False
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            result = check_ollama_available()
            assert result is True
    
    def test_download_ollama_model(self):
        """Test downloading Ollama model."""
        from scripts.installer import download_ollama_model
        
        with patch('subprocess.run') as mock_run:
            download_ollama_model("llama3.2")
            
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "ollama" in call_args
            assert "pull" in call_args
            assert "llama3.2" in call_args
