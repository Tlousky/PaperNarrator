"""Tools for LangGraph pipeline (Task 5-7)."""

import re
from typing import List, Dict
from langchain_core.tools import tool


async def _extract_pdf_text(path: str) -> str:
    """
    Extract text from PDF using PyMuPDF.
    
    Args:
        path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    import fitz  # PyMuPDF
    
    try:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            # Using blocks mode is more robust against character-spacing distortion
            blocks = page.get_text("blocks")
            for b in blocks:
                text += b[4] + "\n"
        doc.close()
        return text
    except Exception as e:
        raise Exception(f"Failed to open file '{path}'.") from e


extract_pdf_text = tool(_extract_pdf_text)


async def _extract_figures(path: str) -> List[Dict]:
    """
    Extract figures from PDF using PyMuPDF.
    
    Args:
        path: Path to PDF file
        
    Returns:
        List of figure dictionaries with image data and metadata
    """
    import fitz  # PyMuPDF
    from PIL import Image
    import io
    import base64
    
    doc = fitz.open(path)
    figures = []
    
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to base64 for storage
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                figures.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "width": image.width,
                    "height": image.height,
                    "image_data": img_base64,
                    "description": f"Figure {img_index + 1} on page {page_num + 1}"
                })
            except Exception as e:
                print(f"Error extracting image {xref}: {e}")
    
    doc.close()
    return figures


extract_figures = tool(_extract_figures)


async def _extract_sections(text: str) -> List[Dict]:
    """
    Extract standard scientific paper sections using regex patterns.
    
    Args:
        text: Raw extracted text from PDF
        
    Returns:
        List of section dictionaries with title, content, and word_count
    """
    # Define section patterns (case insensitive)
    # We look for the header at the start of a line or preceded by newlines
    header_prefix = r"(?:\n\s*|^)"
    patterns = {
        "Abstract": rf"{header_prefix}\babstract\b.*?(?={header_prefix}\b(introduction|methods|material|results|discussion|conclusion|acknowledgement|reference|literature|author|funding|supplement|appendix|keyword)\b|$)",
        "Introduction": rf"{header_prefix}\bintroduction\b.*?(?={header_prefix}\b(methods|material|results|discussion|conclusion|acknowledgement|reference|literature|author|funding|supplement|appendix|keyword)\b|$)",
        "Methods": rf"{header_prefix}\b(methods?|methodology|material\s+and\s+methods?|experimental\s+methods?|procedures?)\b.*?(?={header_prefix}\b(results?|discussion|conclusion|acknowledgement|reference|literature|author|funding|supplement|appendix|keyword)\b|$)",
        "Results": rf"{header_prefix}\bresults?\b.*?(?={header_prefix}\b(discussion|conclusion|acknowledgement|reference|literature|author|funding|supplement|appendix|keyword)\b|$)",
        "Discussion": rf"{header_prefix}\bdiscussion\b.*?(?={header_prefix}\b(conclusion|acknowledgement|reference|literature|author|funding|supplement|appendix|keyword)\b|$)",
        "Conclusion": rf"{header_prefix}\bconclusion\b.*?(?={header_prefix}\b(acknowledgement|reference|literature|author|funding|supplement|appendix|keyword)\b|$)",
    }
    
    sections = []
    
    for section_name, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(0).strip()
            # Remove section title from content
            content = re.sub(rf"^{section_name}\s*", "", content, flags=re.MULTILINE | re.IGNORECASE).strip()
            word_count = len(content.split())
            
            sections.append({
                "title": section_name,
                "content": content,
                "word_count": word_count,
                "start_pos": match.start(),
                "end_pos": match.end()
            })
    
    # Sort by position in text to preserve document order
    sections.sort(key=lambda x: x["start_pos"])
    
    # Fallback for plain text without scientific headers
    if not sections:
        sections.append({
            "title": "General Content",
            "content": text.strip(),
            "word_count": len(text.split()),
            "start_pos": 0,
            "end_pos": len(text)
        })
    
    return sections


extract_sections = tool(_extract_sections)


async def _remove_citations(text: str) -> str:
    """
    Remove citation markers from text.
    
    Removes:
    - [1], [2-5], [10]
    - (Smith, 2023), (Smith et al., 2023)
    - Superscript numbers ^1, ^2
    
    Args:
        text: Text with citations
        
    Returns:
        Text without citations
    """
    # Remove [1], [2-5], [10-15], [1, 7–9]
    text = re.sub(r"\[\d+(?:[\s,–-]+\d+)*\]", "", text)
    # Remove (Smith, 2023), (Smith & Jones, 2023), (Smith et al., 2023)
    # Also handles (NASA, 2024), ("Title", 2024)
    text = re.sub(r"\((?:[A-Za-z0-9\s&,.\-\"\']+(?:\s+et\.?\s+al\.?)?\,?\s*)?\d{4}\)", "", text)
    # Remove superscript numbers (literal ^1 or Unicode ¹)
    text = re.sub(r"[\^¹²³⁴⁵⁶⁷⁸⁹⁰]+[0-9]*", "", text)
    return text


remove_citations = tool(_remove_citations)


async def _remove_metadata(text: str) -> str:
    """
    Remove metadata and non-content sections.
    
    Removes:
    - Keywords section
    - References/Bibliography
    - Acknowledgements
    - Funding statements
    - Author affiliations
    
    Args:
        text: Text with metadata
        
    Returns:
        Cleaned text
    """
    # Remove Keywords line
    text = re.sub(r"^keywords?:\s*.*?\n", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Remove arXiv identifiers (e.g. arXiv:2603.03329v1 [cs.CL] 10 Feb 2026)
    text = re.sub(r"arXiv:\d+\.\d+(?:v\d+)?\s+\[[a-zA-Z.-]+\]\s+\d+\s+\w+\s+\d+", "", text, flags=re.IGNORECASE)
    # Remove common PDF headers/footers
    text = re.sub(r"(?:Downloaded from|Published in|Copyright|All rights reserved).*?\n", "", text, flags=re.IGNORECASE)
    # Remove References/Bibliography section
    text = re.sub(r"(references?|bibliography|literature\s+cited)\s*:?.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove Acknowledgements
    text = re.sub(r"acknowledgement(?:s)?\s*:?.*?(?=\b(abstract|introduction|methods|results|discussion|conclusion)\b|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove Funding
    text = re.sub(r"funding\s+statement?.*?(?=\n\w|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    return text.strip()


remove_metadata = tool(_remove_metadata)


async def _smooth_for_tts(text: str) -> str:
    """
    Smooth text for TTS pronunciation.
    
    Expands:
    - "et al." -> "and colleagues"
    - "Fig." -> "Figure"
    - "Table" -> "Table"
    - "i.e." -> "that is"
    - "e.g." -> "for example"
    
    Args:
        text: Text to smooth
        
    Returns:
        Text smoothed for TTS
    """
    # Expand common abbreviations
    replacements = [
        (r"\bFig\.\s*(\d+)", r"Figure \1"),
        (r"\bfig\.\s*(\d+)", r"Figure \1"),
        (r"\bTable\s*(\d+)", r"Table \1"),
        (r"\btable\s*(\d+)", r"Table \1"),
        (r"\betc\.\b", "and so on"),
        (r"et\s+al\.(?=\s|$)", "and colleagues"),
        (r"\bi\.e\.\b", "that is"),
        (r"\be\.g\.\b", "for example"),
        (r"\bvs\.\b", "versus"),
        (r"\bvs\b", "versus"),
        (r"\b%\b", "percent"),
    ]
    
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    return text


smooth_for_tts = tool(_smooth_for_tts)