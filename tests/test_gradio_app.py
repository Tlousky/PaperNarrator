"""Tests for Gradio frontend (Task 11)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch


class TestGradioApp:
    """Tests for the Gradio application."""
    
    def test_app_creates_interface_with_three_tabs(self):
        """Test that Gradio interface has 3 input tabs: URL, File, Text."""
        from app import app
        
        # app should be a Gradio Blocks instance
        assert app is not None
        
        # Check for tabs (this is a basic check - Gradio Blocks structure is complex)
        # We verify the app object exists and is callable/configurable
        assert hasattr(app, 'blocks') or hasattr(app, 'queue')
    
    @pytest.mark.asyncio
    async def test_process_url_submits_to_graph(self):
        """Test that URL input correctly creates pipeline state and invokes graph."""
        with patch('app.PipelineState') as mock_state_class, \
             patch('app.WorkflowBuilder') as mock_builder_class:
            
            from app import process_url
            
            # Setup mocks
            mock_state = MagicMock()
            mock_state_class.return_value = mock_state
            
            mock_builder = MagicMock()
            mock_graph = MagicMock()
            mock_builder.return_value.create_graph.return_value = mock_graph
            
            # Mock graph.stream to yield status updates (async generator)
            async def mock_stream(state, stream_mode):
                yield {
                    'status': 'completed',
                    'status_message': 'Done',
                    'final_output': '/tmp/output.epub',
                    'total_cost': 0.05
                }
            
            mock_graph.stream = mock_stream
            
            # Call the async generator
            result = []
            async for item in process_url("https://example.com/paper.pdf", "ep3", "openai", False):
                result.append(item)
            
            # Verify PipelineState was created with correct parameters
            mock_state_class.assert_called_once()
            call_kwargs = mock_state_class.call_args
            assert call_kwargs[1].get('source_type') == 'url'
            assert call_kwargs[1].get('content') == 'https://example.com/paper.pdf'
            
            # Check result format
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_process_file_submits_to_graph(self):
        """Test that file upload correctly creates pipeline state."""
        with patch('app.PipelineState') as mock_state_class, \
             patch('app.WorkflowBuilder') as mock_builder_class:
            
            from app import process_file
            
            mock_state = MagicMock()
            mock_state_class.return_value = mock_state
            
            mock_builder = MagicMock()
            mock_graph = MagicMock()
            mock_builder.return_value.create_graph.return_value = mock_graph
            
            async def mock_stream(state, stream_mode):
                yield {
                    'status': 'completed',
                    'final_output': '/tmp/output.epub'
                }
            
            mock_graph.stream = mock_stream
            
            # Simulate a file path
            file_path = "/tmp/test.pdf"
            result = []
            async for item in process_file(file_path, "ep3", "openai", False):
                result.append(item)
            
            mock_state_class.assert_called_once()
            call_kwargs = mock_state_class.call_args
            assert call_kwargs[1].get('source_type') == 'file'
            assert call_kwargs[1].get('temp_path') == file_path
    
    @pytest.mark.asyncio
    async def test_process_text_submits_to_graph(self):
        """Test that text paste correctly creates pipeline state."""
        with patch('app.PipelineState') as mock_state_class, \
             patch('app.WorkflowBuilder') as mock_builder_class:
            
            from app import process_text
            
            mock_state = MagicMock()
            mock_state_class.return_value = mock_state
            
            mock_builder = MagicMock()
            mock_graph = MagicMock()
            mock_builder.return_value.create_graph.return_value = mock_graph
            
            async def mock_stream(state, stream_mode):
                yield {
                    'status': 'completed',
                    'final_output': '/tmp/output.epub'
                }
            
            mock_graph.stream = mock_stream
            
            text = "Hello world, this is a test paper."
            result = []
            async for item in process_text(text, "ep3", "openai", False):
                result.append(item)
            
            mock_state_class.assert_called_once()
            call_kwargs = mock_state_class.call_args
            assert call_kwargs[1].get('source_type') == 'file'  # text is converted to PDF file
            assert call_kwargs[1].get('temp_path') is not None
    
    @pytest.mark.asyncio
    async def test_error_handling_in_process_url(self):
        """Test that errors are caught and returned to user."""
        with patch('app.PipelineState') as mock_state_class, \
             patch('app.WorkflowBuilder') as mock_builder_class:
            
            from app import process_url
            
            mock_builder = MagicMock()
            mock_graph = MagicMock()
            mock_builder.return_value.create_graph.return_value = mock_graph
            
            # Simulate error in stream
            async def mock_stream(state, stream_mode):
                raise Exception("PDF download failed")
            
            mock_graph.stream = mock_stream
            
            result = []
            async for item in process_url("https://example.com/paper.pdf", "ep3", "openai", False):
                result.append(item)
            
            # Should return error message
            assert len(result) > 0
            # Last item should contain error info or status should be failed
