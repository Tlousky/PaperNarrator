"""LangFuse observability and markdown trace export.

Provides:
- Integration with LangFuse cloud tracing (when API keys provided)
- Local markdown trace export to ./traces/{timestamp}_{run_id}.md
- Fallback to local-only mode when API keys not configured
"""

import os
import sys
from datetime import datetime
from typing import Optional, Any, Dict
from pathlib import Path


# Only import LangFuse if available
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


class LangfuseTracer:
    """
    Tracer for LangGraph pipelines with LangFuse integration and markdown export.
    
    Features:
    - Cloud tracing via LangFuse (optional, requires API keys)
    - Local markdown trace files in ./traces/
    - Captures LLM calls, costs, execution times, node traces
    """
    
    def __init__(
        self,
        trace_dir: Optional[str] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None
    ):
        """
        Initialize the tracer.
        
        Args:
            trace_dir: Directory for markdown traces (default: ./traces)
            public_key: LangFuse public key (from env or arg)
            secret_key: LangFuse secret key (from env or arg)
            host: LangFuse host URL (default: cloud)
        """
        self.trace_dir = Path(trace_dir or "./traces")
        # Only mkdir if it doesn't exist (don't fail if exists)
        if not self.trace_dir.exists():
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        # Get credentials
        self.public_key = public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
        self.secret_key = secret_key or os.getenv('LANGFUSE_SECRET_KEY')
        self.host = host or os.getenv('LANGFUSE_HOST')
        
        # Determine mode
        self.local_mode = not (self.public_key and self.secret_key)
        
        # Initialize LangFuse client if keys provided
        self.langfuse_client = None
        if not self.local_mode and LANGFUSE_AVAILABLE:
            try:
                self.langfuse_client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host
                )
                print(f"[LangfuseTracer] Connected to LangFuse at {self.host or 'https://cloud.langfuse.com'}")
            except Exception as e:
                print(f"[LangfuseTracer] Failed to connect to LangFuse: {e}. Using local-only mode.")
                self.local_mode = True
        else:
            print("[LangfuseTracer] Running in local-only mode (no API keys or LangFuse not installed)")
    
    def trace_graph(self, graph, graph_name: str = "paper-narrator"):
        """
        Wrap a LangGraph graph with tracing.
        
        Args:
            graph: Compiled LangGraph StateGraph
            graph_name: Name for the trace
            
        Returns:
            Traced graph (wrapped) that logs to both LangFuse and markdown
        """
        # Create a wrapper that adds tracing
        original_invoke = graph.invoke
        
        async def traced_invoke(input_data, config=None):
            run_id = config.get('run_id', None) if config else None
            if not run_id:
                import uuid
                run_id = str(uuid.uuid4())[:8]
            
            trace_data = {
                'run_id': run_id,
                'graph_name': graph_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'running',
                'input': str(input_data)[:500],  # Truncate for safety
                'nodes_executed': [],
                'llm_calls': [],
                'cost': 0.0,
                'duration': 0.0,
                'error': None
            }
            
            # Start LangFuse trace if available
            langfuse_trace = None
            if not self.local_mode and self.langfuse_client:
                langfuse_trace = self.langfuse_client.trace(
                    name=graph_name,
                    input=input_data,
                    id=run_id
                )
            
            try:
                start_time = datetime.now()
                
                # Execute graph
                result = await original_invoke(input_data, config)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                trace_data['status'] = 'completed'
                trace_data['duration'] = duration
                trace_data['output'] = str(result)[:500]
                
                # Update with actual state data if available
                if isinstance(result, dict):
                    trace_data['cost'] = result.get('total_cost', 0.0)
                    trace_data['nodes_executed'] = self._extract_nodes_executed(result)
                    if 'error' in result and result['error']:
                        trace_data['status'] = 'failed'
                        trace_data['error'] = result['error']
                
                # Write markdown trace
                self.write_markdown_trace(trace_data)
                
                # Update LangFuse trace
                if langfuse_trace:
                    langfuse_trace.update(
                        output=result,
                        status=trace_data['status'],
                        usage={
                            'cost': trace_data['cost'],
                            'units': 'USD'
                        }
                    )
                
                return result
                
            except Exception as e:
                trace_data['status'] = 'failed'
                trace_data['error'] = str(e)
                trace_data['duration'] = (datetime.now() - start_time).total_seconds()
                
                # Write error trace
                self.write_markdown_trace(trace_data)
                
                # Update LangFuse trace
                if langfuse_trace:
                    langfuse_trace.update(
                        status='failed',
                        output={'error': str(e)}
                    )
                
                raise
        
        # Patch the invoke method
        graph.invoke = traced_invoke
        return graph
    
    def write_markdown_trace(self, trace_data: Dict[str, Any]):
        """
        Write trace data to a markdown file.
        
        Args:
            trace_data: Dictionary containing trace information
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = trace_data.get('run_id', 'unknown')
        filename = f"trace_{timestamp}_{run_id}.md"
        filepath = self.trace_dir / filename
        
        # Build markdown content
        md_lines = [
            f"# Trace: {run_id}",
            "",
            "## Summary",
            f"- **Status**: {trace_data.get('status', 'unknown')}",
            f"- **Duration**: {trace_data.get('duration', 0):.2f}s",
            f"- **Cost**: ${trace_data.get('cost', 0):.4f}",
            f"- **Timestamp**: {trace_data.get('timestamp', datetime.now().isoformat())}",
            f"- **Graph**: {trace_data.get('graph_name', 'unknown')}",
            "",
            "## Execution Details",
            f"- **Nodes Executed**: {', '.join(trace_data.get('nodes_executed', []))}",
            f"- **LLM Calls**: {trace_data.get('llm_calls', 0)}",  # Can be int or list
            "",
            "## Input",
            "```",
            trace_data.get('input', 'N/A')[:1000],  # Limit input size
            "```",
            "",
            "## Output",
            "```",
            trace_data.get('output', 'N/A')[:1000],  # Limit output size
            "```",
            "",
        ]
        
        # Add error section if failed
        if trace_data.get('error'):
            md_lines.extend([
                "## Error",
                "```",
                trace_data['error'],
                "```",
                "",
            ])
        
        # Add LLM calls section
        llm_calls = trace_data.get('llm_calls', 0)
        # Handle both int (count) and list (details)
        if isinstance(llm_calls, int):
            llm_calls_list = []
        else:
            llm_calls_list = llm_calls
        
        if llm_calls_list:
            md_lines.append("## LLM Calls")
            for i, call in enumerate(llm_calls_list, 1):
                md_lines.append(f"### Call {i}")
                md_lines.append(f"- Provider: {call.get('provider', 'unknown')}")
                md_lines.append(f"- Model: {call.get('model', 'unknown')}")
                md_lines.append(f"- Cost: ${call.get('cost', 0):.6f}")
                md_lines.append(f"- Tokens: {call.get('input_tokens', 0)} in, {call.get('output_tokens', 0)} out")
                md_lines.append("")
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        
        print(f"[LangfuseTracer] Trace written to: {filepath}")
    
    def _extract_nodes_executed(self, state: dict) -> list:
        """Extract list of executed nodes from state (heuristic)."""
        # This is a simplified extraction - in production you'd want to track this during execution
        nodes = []
        if state.get('raw_text'):
            nodes.append('extracting_text')
        if state.get('cleaned_sections'):
            nodes.append('cleaning_with_llm')
        if state.get('chunks'):
            nodes.append('chunking_text')
        if state.get('audio_files'):
            nodes.append('generating_audio')
        if state.get('final_output'):
            nodes.append('concatenating_audio')
            if state.get('final_output', '').endswith('.epub'):
                nodes.append('packaging_ep3')
        return nodes


# Singleton instance
_tracer_instance: Optional[LangfuseTracer] = None

def get_tracer() -> LangfuseTracer:
    """Get or create the global tracer instance."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = LangfuseTracer()
    return _tracer_instance
