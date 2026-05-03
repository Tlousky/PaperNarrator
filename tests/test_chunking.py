import pytest
import re
import math
from langgraph_pipeline.state import PaperSection, TextChunk

def chunk_text_logic(cleaned_sections, MAX_WORDS=4000):
    """Extracted logic from workflow.py for testing."""
    all_chunks = []
    
    for section in cleaned_sections:
        section_title = section.title
        section_text = section.content
        section_word_count = section.word_count
        
        if section_word_count == 0:
            continue
        
        # 1. Split section into sentences (looking for . ! ? followed by space and Capital, or end of line)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$', section_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue
        
        # 2. Determine how many chunks this section needs
        num_chunks = math.ceil(section_word_count / MAX_WORDS)
        target_words_per_chunk = math.ceil(section_word_count / num_chunks)
        
        # 3. Distribute sentences into chunks
        section_chunks = []
        current_sentences = []
        current_word_count = 0
        
        for sent in sentences:
            sent_words = len(sent.split())
            
            # If adding this sentence exceeds MAX_WORDS, we MUST save current chunk
            # OR if we have reached the target and it's not the last chunk
            if (current_word_count + sent_words > MAX_WORDS) or \
               (current_word_count >= target_words_per_chunk and len(section_chunks) < num_chunks - 1):
                
                if current_sentences:
                    section_chunks.append(TextChunk(
                        text=" ".join(current_sentences),
                        word_count=current_word_count,
                        section_names=[section_title],
                        chunk_id=f"{section_title}_chunk_{len(section_chunks)+1}"
                    ))
                    current_sentences = []
                    current_word_count = 0
            
            current_sentences.append(sent)
            current_word_count += sent_words
        
        # Save the last chunk of the section
        if current_sentences:
            section_chunks.append(TextChunk(
                text=" ".join(current_sentences),
                word_count=current_word_count,
                section_names=[section_title],
                chunk_id=f"{section_title}_chunk_{len(section_chunks)+1}"
            ))
        
        all_chunks.extend(section_chunks)
    
    return all_chunks

def test_chunking_section_isolation():
    """Verify that sections are never mixed in a single chunk."""
    sections = [
        PaperSection(title="Abstract", content="This is the abstract.", word_count=4),
        PaperSection(title="Introduction", content="This is the introduction.", word_count=4)
    ]
    chunks = chunk_text_logic(sections, MAX_WORDS=100)
    
    assert len(chunks) == 2
    assert chunks[0].section_names == ["Abstract"]
    assert chunks[1].section_names == ["Introduction"]
    assert "introduction" not in chunks[0].text.lower()

def test_chunking_sentence_awareness():
    """Verify that sentences are never split."""
    content = "Sentence one. Sentence two. Sentence three."
    sections = [PaperSection(title="Section", content=content, word_count=6)]
    
    # Target 2 words per chunk (total 6) -> 3 chunks
    chunks = chunk_text_logic(sections, MAX_WORDS=2)
    
    for chunk in chunks:
        # Each chunk should contain at least one full sentence
        assert "Sentence" in chunk.text
        # No sentence should be cut
        assert chunk.text.endswith(".")

def test_balanced_chunking():
    """Verify that chunks are roughly equal in size."""
    # 10 words, MAX_WORDS=6 -> should split 5 and 5, not 6 and 4
    content = "One two three four five. Six seven eight nine ten."
    sections = [PaperSection(title="Section", content=content, word_count=10)]
    
    chunks = chunk_text_logic(sections, MAX_WORDS=6)
    assert len(chunks) == 2
    assert chunks[0].word_count == 5
    assert chunks[1].word_count == 5

def test_mid_sentence_split_edge_case():
    """Test the case where a single sentence is very long."""
    long_sentence = "This is a very long sentence that exceeds the word limit " * 10 # ~100 words
    sections = [PaperSection(title="Section", content=long_sentence, word_count=110)]
    
    chunks = chunk_text_logic(sections, MAX_WORDS=50)
    assert len(chunks) == 1 
    assert chunks[0].word_count == 110 # It kept it whole!

def test_problematic_chunking():
    """Test with the specific text that failed in the user's run."""
    abstract_end = "show that using a smaller model to synthesize a custom code harness (or entire policy) can outperform a much larger model, while also being more cost effective."
    intro_start = "1 Introduction. Large language models (LLMs) have demonstrated remarkable capabilities..."
    
    sections = [
        PaperSection(title="Abstract", content=abstract_end, word_count=len(abstract_end.split())),
        PaperSection(title="Introduction", content=intro_start, word_count=len(intro_start.split()))
    ]
    
    # In the user's run, they were combined in chunk 2.
    # Our balanced logic should keep them in separate chunks because it iterates by section.
    chunks = chunk_text_logic(sections, MAX_WORDS=4000)
    
    assert len(chunks) >= 2
    assert "Introduction" not in chunks[0].text
    assert chunks[1].section_names == ["Introduction"]
    assert chunks[1].text.startswith("1 Introduction")