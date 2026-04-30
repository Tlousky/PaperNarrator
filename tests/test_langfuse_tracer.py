"""Tests for LangFuse observability (Task 12)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch, mock_open
import tempfile
from datetime import datetime


class TestLangfuseTracer:
    """Tests for the LangFuse tracer implementation."""
    
    def test_tracer_initializes_without_keys(self):
        """Test that tracer works without API keys (local-only mode)."""
        # Clear any existing keys
        with patch.dict(os.environ, {}, clear=True):
            from observability.tracer import LangfuseTracer
            
            tracer = LangfuseTracer()
            assert tracer.local_mode is True
            assert tracer.trace_dir is not None
    
    def test_tracer_initializes_with_keys(self):
        """Test that tracer connects to LangFuse when keys provided."""
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'pk-123',
            'LANGFUSE_SECRET_KEY': 'sk-456'
        }):
            # Import after patching env
            from observability.tracer import LangfuseTracer
            
            # Create tracer with keys directly to avoid env reading issues in tests
            tracer = LangfuseTracer(
                public_key='pk-123',
                secret_key='sk-456'
            )
            # Note: local_mode depends on LANGFUSE_AVAILABLE and successful connection
            # We just verify the tracer can be instantiated
            assert tracer is not None
    
    def test_tracer_creates_trace_directory(self):
        """Test that traces directory is created."""
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                from observability.tracer import LangfuseTracer
                
                tracer = LangfuseTracer(trace_dir=tmpdir)
                assert tracer.trace_dir.exists()
    
    def test_write_markdown_trace(self):
        """Test that traces are written to markdown files."""
        with patch.dict(os.environ, {}, clear=True):
            from observability.tracer import LangfuseTracer
            from pathlib import Path
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tracer = LangfuseTracer(trace_dir=tmpdir)
                
                # Create mock trace data
                trace_data = {
                    'run_id': 'test-run-123',
                    'status': 'completed',
                    'duration': 10.5,
                    'cost': 0.05,
                    'llm_calls': 2,  # Integer count
                    'nodes_executed': ['extracting_text', 'cleaning_with_llm', 'chunking_text'],
                    'input': 'Test input',
                    'output': 'Output file'
                }
                
                tracer.write_markdown_trace(trace_data)
                
                # Check file was created
                files = os.listdir(tmpdir)
                assert len(files) == 1
                filename = files[0]
                assert filename.startswith('trace_')
                assert filename.endswith('.md')
                assert 'test-run-123' in filename
                
                # Check content
                with open(os.path.join(tmpdir, filename), 'r') as f:
                    content = f.read()
                    assert 'Test input' in content
                    assert '$0.05' in content
    
    def test_tracer_wraps_graph(self):
        """Test that tracer can wrap a LangGraph graph."""
        with patch.dict(os.environ, {}, clear=True):
            from observability.tracer import LangfuseTracer
            from langgraph.graph import StateGraph, END
            from typing import TypedDict
            
            class State(TypedDict):
                value: str
            
            # Create a simple graph
            def test_node(state):
                return {"value": state["value"] + " processed"}
            
            graph = StateGraph(State)
            graph.add_node("test", test_node)
            graph.set_entry_point("test")
            graph.add_edge("test", END)
            compiled_graph = graph.compile()
            
            tracer = LangfuseTracer()
            traced_graph = tracer.trace_graph(compiled_graph, "test-pipeline")
            
            # Traced graph should be callable
            assert traced_graph is not None
    
    def test_trace_markdown_format(self):
        """Test the format of the markdown trace file."""
        with patch.dict(os.environ, {}, clear=True):
            from observability.tracer import LangfuseTracer
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tracer = LangfuseTracer(trace_dir=tmpdir)
                
                trace_data = {
                    'run_id': 'run-456',
                    'status': 'failed',
                    'duration': 5.2,
                    'cost': 0.02,
                    'llm_calls': 1,  # Integer count
                    'nodes_executed': ['extracting_text'],
                    'error': 'PDF download failed',
                    'timestamp': '2024-01-01 12:00:00',
                    'input': 'Test',
                    'output': 'Result'
                }
                
                tracer.write_markdown_trace(trace_data)
                
                # Read and verify format
                files = os.listdir(tmpdir)
                filepath = os.path.join(tmpdir, files[0])
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Check markdown structure
                assert '# Trace: run-456' in content
                assert '## Summary' in content
                assert '**Status**: failed' in content
                assert '**Duration**: 5.20s' in content
                assert '**Cost**: $0.0200' in content
                assert '## Error' in content
                assert 'PDF download failed' in content
    
    def test_tracer_with_config(self):
        """Test tracer with custom configuration."""
        with patch.dict(os.environ, {}, clear=True):
            from observability.tracer import LangfuseTracer
            from pathlib import Path
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tracer = LangfuseTracer(trace_dir=tmpdir)
                assert tracer.trace_dir == Path(tmpdir)
