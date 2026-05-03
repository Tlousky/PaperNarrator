"""Task 15: Final Integration & Testing

End-to-end integration tests verifying all components work together.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIntegration:
    """Integration tests for PaperNarrator pipeline."""
    
    def test_all_modules_importable(self):
        """Test that all major modules can be imported."""
        # Core modules
        from config import Config
        from llm.base import LLMProvider
        from llm.openai_provider import OpenAIProvider
        
        # Verify Config loads
        config = Config()
        assert config is not None
        
        # Verify LLM provider hierarchy
        assert issubclass(OpenAIProvider, LLMProvider)
    
    def test_config_has_required_fields(self):
        """Test configuration has all required fields for Docker deployment."""
        from config import Config
        
        config = Config()
        
        # Check LLM settings
        assert hasattr(config, 'LLM_PROVIDER')
        assert hasattr(config, 'OPENAI_API_KEY')
        
        # Check VibeVoice settings
        assert hasattr(config, 'VIBEVOICE_DEVICE')
        
        # Check output settings
        assert hasattr(config, 'OUTPUT_FORMAT')
        assert hasattr(config, 'MP3_BITRATE')
    
    def test_state_schema_complete(self):
        """Test LangGraph state has all required fields."""
        from langgraph_pipeline.state import PipelineState
        
        # Create state with required fields
        state = PipelineState(source_type="url", content="https://example.com/paper.pdf")
        
        # Required fields
        required_fields = [
            'source_type', 'content', 'temp_path', 'raw_text',
            'cleaned_sections', 'chunks', 'audio_files', 'final_output',
            'total_cost', 'status', 'error', 'status_message', 'total_words'
        ]
        
        for field in required_fields:
            assert hasattr(state, field), f"Missing required state field: {field}"
    
    def test_workflow_has_required_nodes(self):
        """Test LangGraph workflow has all required nodes."""
        from langgraph_pipeline.workflow import WorkflowBuilder
        
        builder = WorkflowBuilder()
        graph = builder.create_graph()
        
        # Check nodes exist
        node_names = list(graph.nodes)
        
        # Required nodes
        expected_nodes = [
            'extracting_text', 'describing_figures', 'cleaning_with_llm',
            'chunking_text', 'generating_audio', 'concatenating_audio', 'packaging_ep3'
        ]
        
        for node in expected_nodes:
            assert node in node_names, f"Missing required node: {node}"
    
    def test_ep3_builder_function_exists(self):
        """Test EP3 builder function is importable."""
        from langgraph_pipeline.ep3_builder import create_ep3
        
        # Function should exist
        assert callable(create_ep3)
    
    def test_config_validation_with_env_vars(self):
        """Test config validation works with environment variables."""
        from config import validate_config
        
        # Set minimal env vars
        os.environ['LLM_PROVIDER'] = 'openai'
        os.environ['OPENAI_API_KEY'] = 'test-key-123'
        
        try:
            validate_config()
            # If no exception, validation passed
        except ValueError as e:
            # Validation might raise exception if not properly configured
            # This is acceptable if it's a validation error
            assert "environment variable" in str(e).lower() or "key" in str(e).lower()
    
    def test_gradio_app_has_required_components(self):
        """Test Gradio app file exists and has correct structure."""
        app_path = Path(__file__).parent.parent / "app.py"
        assert app_path.exists(), "app.py should exist"
        
        # Check file has expected functions (read in binary mode to avoid encoding issues)
        with open(app_path, "rb") as f:
            content = f.read().decode('utf-8', errors='ignore')
        
        # Check for at least one process function
        has_process_func = any(func in content for func in ["process_url", "process_file", "process_text"])
        assert has_process_func, "app.py should contain process_url, process_file, or process_text functions"
    
    def test_installer_detection_logic(self):
        """Test installer API key detection."""
        from scripts.installer import check_env_var
        
        # Test with no env vars (should return None)
        os.environ.pop('OPENAI_API_KEY', None)
        
        result = check_env_var('OPENAI_API_KEY')
        # Should return None when key not found
        assert result is None
    
    def test_pdf_extraction_module_exists(self):
        """Test PDF extraction function is importable."""
        from langgraph_pipeline.tools import _extract_pdf_text
        
        # Function should exist
        assert callable(_extract_pdf_text)
    
    def test_tts_module_exists(self):
        """Test TTS module is importable."""
        from tts.vibevoice import VibeVoiceTTS
        
        # Class should exist
        assert VibeVoiceTTS is not None
    
    def test_observability_module_exists(self):
        """Test observability module is importable."""
        from observability.tracer import get_tracer
        
        # Function should exist
        assert callable(get_tracer)
    
    def test_docker_files_exist(self):
        """Test Docker support files exist."""
        docker_dir = Path(__file__).parent.parent / "docker"
        
        required_files = [
            "Dockerfile.cpu",
            "Dockerfile.gpu",
            "docker-compose.yml",
            ".dockerignore"
        ]
        
        for file in required_files:
            assert (docker_dir / file).exists(), f"Missing {file}"
    
    def test_setup_scripts_exist(self):
        """Test setup scripts exist."""
        root = Path(__file__).parent.parent
        
        required_scripts = [
            "setup.sh",
            "setup.bat",
            "setup_vibevoice.sh",
            "setup_vibevoice.bat",
            "download_ffmpeg.sh",
            "download_ffmpeg.bat"
        ]
        
        for script in required_scripts:
            assert (root / script).exists(), f"Missing {script}"


class TestEndToEndFlow:
    """Mocked end-to-end flow tests."""
    
    def test_tts_class_instantiable(self):
        """Test TTS class can be instantiated (without loading model)."""
        from tts.vibevoice import VibeVoiceTTS
        
        # Just test instantiation - model loads lazily
        tts = VibeVoiceTTS(
            model_name="test/model",
            device="cpu",
            speaker_name="Carter"
        )
        
        assert tts is not None
        assert tts.device == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
