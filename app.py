"""Gradio frontend for PaperNarrator.

Provides a web interface with:
- 3 input tabs: URL, File Upload, Text Paste
- Streaming status updates via chatbot
- File download for final output (M4B/MP3/WAV)
- Cost tracking display
"""

import os
import tempfile
import logging
import gradio as gr
from typing import Generator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradio.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.info("--- App Starting ---")

from langgraph_pipeline.state import PipelineState, PipelineStatus
from langgraph_pipeline.workflow import WorkflowBuilder


# Global graph instance (created once for efficiency)
_graph = None

def get_graph():
    """Get or create the LangGraph graph."""
    global _graph
    if _graph is None:
        logger.info("Creating new LangGraph graph instance")
        logger.info(f"Environment: LLM_PROVIDER={os.getenv('LLM_PROVIDER')}, OPENAI_API_KEY={os.getenv('OPENAI_API_KEY', '')[:5]}...")
        config = None  # Will load from environment
        builder = WorkflowBuilder(config=config)
        _graph = builder.create_graph()
        logger.info("Graph created successfully")
    return _graph


import uuid

# App version for tracing
APP_VERSION = "1.0.0"

def get_session_id():
    """Generate a new session ID."""
    return str(uuid.uuid4())

async def process_url(
    url: str, 
    output_format: str,
    llm_provider: str,
    vlm_enabled: bool,
    voice_profile: str,
    session_id: str
) -> Generator:
    """
    Process a URL input.
    """
    logger.info(f"Starting URL processing: {url[:50]}... session={session_id}")
    history = [{"role": "user", "content": f"Process URL: {url}"}]
    try:
        # Update config
        os.environ['OUTPUT_FORMAT'] = output_format
        os.environ['LLM_PROVIDER'] = llm_provider
        os.environ['VLM_ENABLED'] = str(vlm_enabled).lower()
        
        # Create state with session info
        state = PipelineState(
            source_type="url",
            content=url,
            temp_path=None,
            voice_profile=voice_profile,
            session_id=session_id,
            version=APP_VERSION,
            tags=["gradio-app", "url-input"]
        )
        
        # Recreate graph with new config
        config = None  # Loads from env
        builder = WorkflowBuilder(config=config)
        graph = builder.create_graph()
        
        # Run graph with streaming (async)
        result = None
        async for chunk in graph.astream(state, stream_mode="values"):
            status_msg = chunk.get('status_message', '')
            status = chunk.get('status', PipelineStatus.PENDING)
            
            # Yield status update
            history.append({"role": "assistant", "content": status_msg})
            yield (
                history,  # Chat history
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
            history.append({"role": "assistant", "content": f"❌ Error: {error_msg}"})
            yield (
                history,
                None,
                f"Cost: ${total_cost:.4f}",
                "failed"
            )
        else:
            history.append({"role": "assistant", "content": f"✅ Complete! Generated {output_format.upper()} audiobook."})
            yield (
                history,
                final_file,
                f"Cost: ${total_cost:.4f}",
                "completed"
            )
            
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}", exc_info=True)
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        yield (
            history,
            None,
            "Cost: $0.00",
            "failed"
        )


async def process_file(
    file: str,
    output_format: str,
    llm_provider: str,
    vlm_enabled: bool,
    voice_profile: str,
    session_id: str
) -> Generator:
    """
    Process an uploaded file.
    """
    logger.info(f"Starting file processing session={session_id}")
    history = [{"role": "user", "content": f"Process uploaded file"}]
    try:
        os.environ['OUTPUT_FORMAT'] = output_format
        os.environ['LLM_PROVIDER'] = llm_provider
        os.environ['VLM_ENABLED'] = str(vlm_enabled).lower()
        
        state = PipelineState(
            source_type="file",
            content="",
            temp_path=file,
            voice_profile=voice_profile,
            session_id=session_id,
            version=APP_VERSION,
            tags=["gradio-app", "file-upload"]
        )
        
        config = None
        builder = WorkflowBuilder(config=config)
        graph = builder.create_graph()
        
        result = None
        async for chunk in graph.astream(state, stream_mode="values"):
            status_msg = chunk.get('status_message', '')
            status = chunk.get('status', PipelineStatus.PENDING)
            
            history.append({"role": "assistant", "content": status_msg})
            yield (
                history,
                None,
                f"Cost: ${chunk.get('total_cost', 0):.4f}",
                str(status)
            )
            result = chunk
        
        final_file = result.get('final_output')
        total_cost = result.get('total_cost', 0)
        
        if result.get('status') == PipelineStatus.FAILED:
            error_msg = result.get('error', 'Unknown error')
            history.append({"role": "assistant", "content": f"❌ Error: {error_msg}"})
            yield (
                history,
                None,
                f"Cost: ${total_cost:.4f}",
                "failed"
            )
        else:
            history.append({"role": "assistant", "content": f"✅ Complete! Generated {output_format.upper()} audiobook."})
            yield (
                history,
                final_file,
                f"Cost: ${total_cost:.4f}",
                "completed"
            )
            
    except Exception as e:
        logger.error(f"Error processing file {file}: {str(e)}", exc_info=True)
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        yield (
            history,
            None,
            "Cost: $0.00",
            "failed"
        )


async def process_text(
    text: str,
    output_format: str,
    llm_provider: str,
    vlm_enabled: bool,
    voice_profile: str,
    session_id: str
) -> Generator:
    """
    Process pasted text.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_pdf.close()
        
        c = canvas.Canvas(temp_pdf.name, pagesize=letter)
        width, height = letter
        
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
        
        async for result in process_file(temp_pdf.name, output_format, llm_provider, vlm_enabled, voice_profile, session_id):
            yield result
        
        os.unlink(temp_pdf.name)
            
    except Exception as e:
        logger.error(f"Error processing text input: {str(e)}", exc_info=True)
        history = [{"role": "user", "content": "Process pasted text"}, {"role": "assistant", "content": f"❌ Error: {str(e)}"}]
        yield (
            history,
            None,
            "Cost: $0.00",
            "failed"
        )


# Create the Gradio interface
with gr.Blocks(title="PaperNarrator - AI Audiobook Generator") as app:
    # Session state (initialized on app load)
    session_id = gr.State()
    
    gr.Markdown("# 🎧 PaperNarrator")
    gr.Markdown("**Convert scientific papers into narrated audiobooks with AI-powered text cleaning**")
    
    with gr.Tabs():
        with gr.Tab("🔗 URL Input"):
            url_input = gr.Textbox(
                label="PDF URL",
                placeholder="https://arxiv.org/pdf/2301.1234.pdf",
                type="text"
            )
            submit_url = gr.Button("Generate Audiobook", variant="primary")
        
        with gr.Tab("📁 File Upload"):
            file_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )
            submit_file = gr.Button("Generate Audiobook", variant="primary")
        
        with gr.Tab("📝 Paste Text"):
            text_input = gr.Textbox(
                label="Paper Text",
                placeholder="Paste your paper text here (Abstract, Introduction, Methods, etc.)",
                lines=10,
                type="text"
            )
            submit_text = gr.Button("Generate Audiobook", variant="primary")
    
    gr.Markdown("### Settings")
    with gr.Row():
        output_format = gr.Radio(
            choices=["m4b", "mp3", "wav"],
            value="m4b",
            label="Output Format",
            info="M4B (audiobook with chapters), MP3, or WAV"
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
        voice_profile = gr.Dropdown(
            choices=["Carter", "Davis", "Emma"],
            value="Emma",
            label="Voice Profile",
            info="Select the voice for the audiobook"
        )
    
    gr.Markdown("### Output")
    with gr.Row():
        with gr.Column(scale=2):
            chat_output = gr.Chatbot(
                label="Processing Status",
                height=300,
                show_label=True,
                type="messages"
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
    app.load(get_session_id, outputs=[session_id])
    
    submit_url.click(
        fn=process_url,
        inputs=[url_input, output_format, llm_provider, vlm_enabled, voice_profile, session_id],
        outputs=[chat_output, file_output, cost_display, status_display]
    )
    
    submit_file.click(
        fn=process_file,
        inputs=[file_input, output_format, llm_provider, vlm_enabled, voice_profile, session_id],
        outputs=[chat_output, file_output, cost_display, status_display]
    )
    
    submit_text.click(
        fn=process_text,
        inputs=[text_input, output_format, llm_provider, vlm_enabled, voice_profile, session_id],
        outputs=[chat_output, file_output, cost_display, status_display]
    )


if __name__ == "__main__":
    app.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
