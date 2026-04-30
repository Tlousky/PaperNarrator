"""Gradio frontend for PaperNarrator.

Provides a web interface with:
- 3 input tabs: URL, File Upload, Text Paste
- Streaming status updates via chatbot
- File download for final output (EP3/MP3/WAV)
- Cost tracking display
"""

import os
import tempfile
import gradio as gr
from typing import Generator

from langgraph_pipeline.state import PipelineState, PipelineStatus
from langgraph_pipeline.workflow import WorkflowBuilder


# Global graph instance (created once for efficiency)
_graph = None

def get_graph():
    """Get or create the LangGraph graph."""
    global _graph
    if _graph is None:
        config = None  # Will load from environment
        builder = WorkflowBuilder(config=config)
        _graph = builder.create_graph()
    return _graph


async def process_url(
    url: str, 
    output_format: str,
    llm_provider: str,
    vlm_enabled: bool
) -> Generator:
    """
    Process a URL input.
    
    Args:
        url: PDF URL to download and process
        output_format: 'ep3', 'mp3', or 'wav'
        llm_provider: 'openai', 'gemini', 'anthropic', or 'ollama'
        vlm_enabled: Whether to enable vision language model for figures
        
    Yields:
        Tuples of (chat_message, file, cost_display, status_text)
    """
    try:
        # Update config
        os.environ['OUTPUT_FORMAT'] = output_format
        os.environ['LLM_PROVIDER'] = llm_provider
        os.environ['VLM_ENABLED'] = str(vlm_enabled).lower()
        
        # Recreate graph with new config
        config = None  # Loads from env
        builder = WorkflowBuilder(config=config)
        graph = builder.create_graph()
        
        # Create state
        state = PipelineState(
            source_type="url",
            content=url,
            temp_path=None
        )
        
        # Run graph with streaming
        result = None
        for chunk in graph.stream(state, stream_mode="values"):
            status_msg = chunk.get('status_message', '')
            status = chunk.get('status', PipelineStatus.PENDING)
            
            # Yield status update
            yield (
                f"{status_msg}",  # Chat message
                None,              # No file yet
                f"Cost: ${chunk.get('total_cost', 0):.4f}",  # Cost
                str(status)        # Status
            )
            result = chunk
        
        # Yield final result
        final_file = result.get('final_output')
        total_cost = result.get('total_cost', 0)
        
        if result.get('status') == PipelineStatus.FAILED:
            error_msg = result.get('error', 'Unknown error')
            yield (
                f"❌ Error: {error_msg}",
                None,
                f"Cost: ${total_cost:.4f}",
                "failed"
            )
        else:
            yield (
                f"✅ Complete! Generated {output_format.upper()} audiobook.",
                final_file,
                f"Cost: ${total_cost:.4f}",
                "completed"
            )
            
    except Exception as e:
        yield (
            f"❌ Error: {str(e)}",
            None,
            "Cost: $0.00",
            "failed"
        )


async def process_file(
    file: str,  # Gradio file upload gives us path
    output_format: str,
    llm_provider: str,
    vlm_enabled: bool
) -> Generator:
    """
    Process an uploaded file.
    
    Args:
        file: Path to uploaded PDF file
        output_format: 'ep3', 'mp3', or 'wav'
        llm_provider: LLM provider
        vlm_enabled: Enable VLM
        
    Yields:
        Same as process_url
    """
    try:
        os.environ['OUTPUT_FORMAT'] = output_format
        os.environ['LLM_PROVIDER'] = llm_provider
        os.environ['VLM_ENABLED'] = str(vlm_enabled).lower()
        
        config = None
        builder = WorkflowBuilder(config=config)
        graph = builder.create_graph()
        
        state = PipelineState(
            source_type="file",
            content="",
            temp_path=file
        )
        
        result = None
        for chunk in graph.stream(state, stream_mode="values"):
            status_msg = chunk.get('status_message', '')
            status = chunk.get('status', PipelineStatus.PENDING)
            
            yield (
                f"{status_msg}",
                None,
                f"Cost: ${chunk.get('total_cost', 0):.4f}",
                str(status)
            )
            result = chunk
        
        final_file = result.get('final_output')
        total_cost = result.get('total_cost', 0)
        
        if result.get('status') == PipelineStatus.FAILED:
            error_msg = result.get('error', 'Unknown error')
            yield (
                f"❌ Error: {error_msg}",
                None,
                f"Cost: ${total_cost:.4f}",
                "failed"
            )
        else:
            yield (
                f"✅ Complete! Generated {output_format.upper()} audiobook.",
                final_file,
                f"Cost: ${total_cost:.4f}",
                "completed"
            )
            
    except Exception as e:
        yield (
            f"❌ Error: {str(e)}",
            None,
            "Cost: $0.00",
            "failed"
        )


async def process_text(
    text: str,
    output_format: str,
    llm_provider: str,
    vlm_enabled: bool
) -> Generator:
    """
    Process pasted text (for testing or when PDF not available).
    
    Note: This creates a temporary PDF from text for processing.
    In production, you might want to add a direct text processing path.
    """
    try:
        # For text input, we'll create a temporary PDF
        # Using reportlab or similar to convert text to PDF
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_pdf.close()
        
        # Create simple PDF from text
        c = canvas.Canvas(temp_pdf.name, pagesize=letter)
        width, height = letter
        
        # Wrap text and add to PDF
        paragraphs = text.split('\n\n')
        y = height - 50
        for para in paragraphs:
            text_obj = c.beginText(50, y)
            text_obj.setFont("Helvetica", 12)
            for line in para.split('\n'):
                text_obj.textLine(line)
            c.drawText(text_obj)
            y -= 30
            if y < 50:
                c.showPage()
                y = height - 50
        
        c.save()
        
        # Process as file (manually iterate to avoid yield from in async)
        async for result in process_file(temp_pdf.name, output_format, llm_provider, vlm_enabled):
            yield result
        
        # Cleanup
        os.unlink(temp_pdf.name)
            
    except Exception as e:
        yield (
            f"❌ Error: {str(e)}",
            None,
            "Cost: $0.00",
            "failed"
        )


# Create the Gradio interface
with gr.Blocks(title="PaperNarrator - AI Audiobook Generator") as app:
    gr.Markdown("# 🎧 PaperNarrator")
    gr.Markdown("**Convert scientific papers into narrated audiobooks with AI-powered text cleaning**")
    
    with gr.Tabs():
        # Tab 1: URL Input
        with gr.Tab("🔗 URL Input"):
            url_input = gr.Textbox(
                label="PDF URL",
                placeholder="https://arxiv.org/pdf/2301.1234.pdf",
                type="text"
            )
            submit_url = gr.Button("Generate Audiobook", variant="primary")
        
        # Tab 2: File Upload
        with gr.Tab("📁 File Upload"):
            file_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )
            submit_file = gr.Button("Generate Audiobook", variant="primary")
        
        # Tab 3: Text Input
        with gr.Tab("📝 Paste Text"):
            text_input = gr.Textbox(
                label="Paper Text",
                placeholder="Paste your paper text here (Abstract, Introduction, Methods, etc.)",
                lines=10,
                type="text"
            )
            submit_text = gr.Button("Generate Audiobook", variant="primary")
    
    # Common controls
    gr.Markdown("### Settings")
    with gr.Row():
        output_format = gr.Radio(
            choices=["ep3", "mp3", "wav"],
            value="ep3",
            label="Output Format",
            info="EP3 (audiobook with chapters), MP3, or WAV"
        )
        llm_provider = gr.Radio(
            choices=["openai", "gemini", "anthropic", "ollama"],
            value="openai",
            label="LLM Provider",
            info="OpenAI (default), Google Gemini, Anthropic, or Local Ollama"
        )
        vlm_enabled = gr.Checkbox(
            label="Enable VLM (Vision)",
            value=False,
            info="Describe figures using Vision Language Model"
        )
    
    # Output area
    gr.Markdown("### Output")
    with gr.Row():
        with gr.Column(scale=2):
            chat_output = gr.Chatbot(
                label="Processing Status",
                height=300,
                show_label=True
            )
        with gr.Column(scale=1):
            file_output = gr.File(
                label="Download Audiobook",
                interactive=False
            )
            cost_display = gr.Textbox(
                label="Total Cost",
                value="Cost: $0.00",
                interactive=False
            )
            status_display = gr.Textbox(
                label="Status",
                value="pending",
                interactive=False,
                visible=False
            )
    
    # Define events
    submit_url.click(
        fn=process_url,
        inputs=[url_input, output_format, llm_provider, vlm_enabled],
        outputs=[chat_output, file_output, cost_display, status_display]
    )
    
    submit_file.click(
        fn=process_file,
        inputs=[file_input, output_format, llm_provider, vlm_enabled],
        outputs=[chat_output, file_output, cost_display, status_display]
    )
    
    submit_text.click(
        fn=process_text,
        inputs=[text_input, output_format, llm_provider, vlm_enabled],
        outputs=[chat_output, file_output, cost_display, status_display]
    )


if __name__ == "__main__":
    app.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
