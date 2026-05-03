"""End-to-End tests for PaperNarrator pipeline.

These tests run the full pipeline WITHOUT mocks:
  - Real VibeVoice-Realtime-0.5B model inference
  - Real LLM text cleaning (uses local llama.cpp server from .env)
  - Real PDF processing using the two example PDFs

Run with:
    pytest tests/test_e2e_pdfs.py -v -m e2e

Prerequisites:
  1. Run download_vibevoice.py to download the 0.5B model + voices
  2. Have the local LLM server running (configured in .env)
  3. Example PDFs must exist in examples/

Outputs are saved to ./outputs/e2e/ for inspection.
"""

import os
import asyncio
import zipfile
import pytest
from pathlib import Path

# Mark all tests in this file as e2e
pytestmark = pytest.mark.e2e

# Paths
MODELS_DIR = Path("./models/microsoft/VibeVoice-Realtime-0.5B")
EXAMPLES_DIR = Path("./examples")
OUTPUT_DIR = Path("./outputs/e2e")

EXAMPLE_PDFS = [
    EXAMPLES_DIR / "AutoHarness_improving_LLM_agents_by_automatically_.pdf",
    EXAMPLES_DIR / "2603.25723v1.pdf",
]


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring real models and LLM server")


@pytest.fixture(scope="session", autouse=True)
def setup_output_dir():
    """Create output directory before E2E tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _check_prerequisites():
    """Check that models and example PDFs exist."""
    missing = []
    if not MODELS_DIR.exists():
        missing.append(f"Model not found: {MODELS_DIR} — run download_vibevoice.py")
    for pdf in EXAMPLE_PDFS:
        if not pdf.exists():
            missing.append(f"Example PDF not found: {pdf}")
    return missing


# Skip the whole module if prerequisites are missing
_missing = _check_prerequisites()
if _missing:
    pytest.skip(
        f"E2E prerequisites not met:\n" + "\n".join(_missing),
        allow_module_level=True
    )


@pytest.fixture(scope="module")
def workflow():
    """Create a single workflow graph (expensive to build)."""
    from dotenv import load_dotenv
    load_dotenv()

    from langgraph_pipeline.workflow import WorkflowBuilder
    builder = WorkflowBuilder()
    return builder.create_graph()


async def _run_pipeline(graph, pdf_path: Path, voice: str = "Emma") -> dict:
    """Run the full pipeline on a PDF and return the final state dict."""
    from langgraph_pipeline.state import PipelineState, PipelineStatus

    state = PipelineState(
        source_type="file",
        content="",
        temp_path=str(pdf_path),
        voice_profile=voice,
    )

    result = None
    async for chunk in graph.astream(state, stream_mode="values"):
        result = chunk
        status_msg = chunk.get("status_message", "")
        print(f"  [{chunk.get('status', '')}] {status_msg}")

    return result


class TestE2EAutoHarnessPDF:
    """E2E test for AutoHarness PDF → M4B audiobook."""

    PDF = EXAMPLE_PDFS[0]

    async def test_full_pipeline_generates_m4b(self, workflow):
        """Test that the AutoHarness PDF produces a valid M4B audiobook."""
        print(f"\nProcessing: {self.PDF.name}")
        result = await _run_pipeline(workflow, self.PDF, voice="Emma")

        assert result is not None
        from langgraph_pipeline.state import PipelineStatus
        assert result.get("status") == PipelineStatus.COMPLETED, \
            f"Pipeline failed: {result.get('error')}"

        m4b_path = result.get("final_output")
        assert m4b_path is not None, "No final_output in result"
        assert m4b_path.endswith(".m4b"), f"Expected .m4b, got: {m4b_path}"
        assert os.path.exists(m4b_path), f"M4B file not found: {m4b_path}"
        assert os.path.getsize(m4b_path) > 0, "M4B file is empty"

        # Copy to output dir for inspection
        import shutil
        dest = OUTPUT_DIR / f"autoharness_{self.PDF.stem}.m4b"
        shutil.copy2(m4b_path, dest)
        print(f"  [SUCCESS] M4B saved to: {dest}")

    async def test_full_pipeline_generates_mp3(self, workflow):
        """Test that the AutoHarness PDF produces a valid MP3 file."""
        import os
        os.environ["OUTPUT_FORMAT"] = "mp3"

        from langgraph_pipeline.workflow import WorkflowBuilder
        builder = WorkflowBuilder()
        mp3_graph = builder.create_graph()

        result = await _run_pipeline(mp3_graph, self.PDF, voice="Carter")

        os.environ["OUTPUT_FORMAT"] = "m4b"  # Restore default

        from langgraph_pipeline.state import PipelineStatus
        assert result.get("status") == PipelineStatus.COMPLETED, \
            f"Pipeline failed: {result.get('error')}"

        mp3_path = result.get("final_output")
        assert mp3_path is not None
        assert mp3_path.endswith(".mp3"), f"Expected .mp3, got: {mp3_path}"
        assert os.path.exists(mp3_path)
        assert os.path.getsize(mp3_path) > 0, "MP3 file is empty"

        # Copy to output dir
        import shutil
        dest = OUTPUT_DIR / f"autoharness_{self.PDF.stem}.mp3"
        shutil.copy2(mp3_path, dest)
        print(f"  [SUCCESS] MP3 saved to: {dest}")


class TestE2EArxivPDF:
    """E2E test for the ArXiv PDF (2603.25723v1) → M4B audiobook."""

    PDF = EXAMPLE_PDFS[1]

    async def test_full_pipeline_generates_m4b(self, workflow):
        """Test that the ArXiv PDF produces a valid M4B audiobook."""
        print(f"\nProcessing: {self.PDF.name}")
        result = await _run_pipeline(workflow, self.PDF, voice="Davis")

        assert result is not None
        from langgraph_pipeline.state import PipelineStatus
        assert result.get("status") == PipelineStatus.COMPLETED, \
            f"Pipeline failed: {result.get('error')}"

        m4b_path = result.get("final_output")
        assert m4b_path is not None
        assert m4b_path.endswith(".m4b")
        assert os.path.exists(m4b_path)
        assert os.path.getsize(m4b_path) > 0

        import shutil
        dest = OUTPUT_DIR / f"arxiv_{self.PDF.stem}.m4b"
        shutil.copy2(m4b_path, dest)
        print(f"  [SUCCESS] M4B saved to: {dest}")


class TestE2EVoiceProfiles:
    """E2E test that all three voice profiles produce valid output."""

    PDF = EXAMPLE_PDFS[0]  # Use the smaller PDF for speed

    @pytest.mark.parametrize("voice", ["Carter", "Davis", "Emma"])
    async def test_voice_profile(self, workflow, voice):
        """Test that each voice profile produces valid audio."""
        print(f"\nTesting voice: {voice}")
        result = await _run_pipeline(workflow, self.PDF, voice=voice)

        from langgraph_pipeline.state import PipelineStatus
        assert result.get("status") == PipelineStatus.COMPLETED, \
            f"Pipeline failed for voice {voice}: {result.get('error')}"

        m4b_path = result.get("final_output")
        assert m4b_path is not None
        assert os.path.exists(m4b_path)
        assert os.path.getsize(m4b_path) > 0

        import shutil
        dest = OUTPUT_DIR / f"voice_{voice.lower()}.m4b"
        shutil.copy2(m4b_path, dest)
        print(f"  [SUCCESS] {voice} M4B saved to: {dest}")
