import pytest
import os
from pathlib import Path
from dotenv import load_dotenv
from langgraph_pipeline.state import PipelineState, PipelineStatus
from langgraph_pipeline.workflow import WorkflowBuilder
from langgraph_pipeline.tools import _remove_citations

load_dotenv()

EXAMPLES_DIR = Path("./examples")
EXAMPLE_PDFS = [
    EXAMPLES_DIR / "AutoHarness_improving_LLM_agents_by_automatically_.pdf",
    EXAMPLES_DIR / "2603.25723v1.pdf",
]

async def run_processing_pipeline(pdf_path: Path, session_id: str = "test-session-123"):
    """Run the pipeline using the compiled graph to exercise tracing."""
    builder = WorkflowBuilder()
    graph = builder.create_graph()
    
    state = PipelineState(
        source_type="file",
        content="",
        temp_path=str(pdf_path),
        voice_profile="Emma",
        session_id=session_id,
        tags=["pytest", "pdf-processing"]
    )
    
    print(f"  [START] Running graph with session_id: {session_id}")
    
    # Run the graph until completion (or until chunking is done)
    # We can use astream to see progress
    final_state = None
    async for event in graph.astream(state, stream_mode="updates"):
        # The last event in 'updates' mode will contain the state after the last node
        # but actually astream yields dictionaries of updates.
        # To get the final state, it's easier to use ainvoke if we don't need streaming.
        pass
    
    # Actually, let's use ainvoke to get the final state easily
    result = await graph.ainvoke(state)
    
    # Save chunks for verification
    if result.get("chunks"):
        print(f"  [SAVE] Saving {len(result['chunks'])} chunks to {result.get('run_dir')}/text_chunks")
        for chunk in result["chunks"]:
            chunk_path = os.path.join(result["run_dir"], "text_chunks", f"{chunk.chunk_id}.txt")
            os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk.text)
    
    # Return as PipelineState object
    return PipelineState(**result) if isinstance(result, dict) else result

@pytest.mark.asyncio
async def test_remove_citations_unit():
    """Unit test for the regex citation removal logic."""
    text = "This is a sentence [1]. This is another [2-5, 10]. (Smith, 2024) argued that (Smith & Jones, 2024) was right. NASA (2024) says ¹."
    cleaned = await _remove_citations(text)
    
    # Brackets should be gone
    assert "[1]" not in cleaned
    assert "[2-5, 10]" not in cleaned
    # Parenthetical citations with names and years should be gone
    assert "(Smith, 2024)" not in cleaned
    assert "(Smith & Jones, 2024)" not in cleaned
    # Superscripts should be gone
    assert "¹" not in cleaned
    # Narrative citations (like NASA (2024)) might keep the name but regex might remove the year if parenthetical
    # In my current regex: r"\((?:[A-Za-z0-9\s&,.\-\"\']+(?:\s+et\.?\s+al\.?)?\,?\s*)?\d{4}\)"
    # NASA (2024) -> "NASA "
    assert "(2024)" not in cleaned
    assert "NASA" in cleaned

@pytest.mark.asyncio
@pytest.mark.parametrize("pdf_path", EXAMPLE_PDFS)
async def test_pdf_text_processing(pdf_path):
    """Test text extraction, cleaning and chunking for a PDF."""
    if not pdf_path.exists():
        pytest.skip(f"PDF not found: {pdf_path}")
        
    print(f"\nTesting Text Processing for: {pdf_path.name}")
    result = await run_processing_pipeline(pdf_path)
    
    assert result.status != PipelineStatus.FAILED, f"Pipeline failed: {result.error}"
    assert result.cleaned_sections is not None
    assert len(result.cleaned_sections) > 0
    assert result.chunks is not None
    assert len(result.chunks) > 0
    
    print(f"  [SUCCESS] {len(result.cleaned_sections)} sections, {len(result.chunks)} chunks.")
    
    # Verify section isolation in chunks
    for chunk in result.chunks:
        assert len(chunk.section_names) == 1, f"Chunk {chunk.chunk_id} contains multiple sections: {chunk.section_names}"
        assert chunk.word_count > 0
        assert len(chunk.text) > 0
        
        # Verify citation removal (basic check)
        assert not any(c in chunk.text for c in ["[1]", "[2]", "[10]"]), f"Found citation in chunk {chunk.chunk_id}"
        # Check for APA style markers (regex should have caught these)
        import re
        assert not re.search(r"\(\w+,\s+\d{4}\)", chunk.text), f"Found parenthetical citation in chunk {chunk.chunk_id}"
        
    # Check if we have both Abstract and Introduction (typical for these papers)
    section_titles = [s.title.lower() for s in result.cleaned_sections]
    assert any("abstract" in t for t in section_titles), "Abstract section missing"
    assert any("introduction" in t for t in section_titles), "Introduction section missing"
