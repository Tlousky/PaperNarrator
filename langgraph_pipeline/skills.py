"""Specialized LLM instructions (Skills) for PaperNarrator cleaning."""

# Citation Removal Skill
# Based on docs/reference/citations.md and user requirements
CITATION_REMOVAL_SKILL = """
SKILL: CITATION REMOVAL
Your goal is to remove in-text academic citations to make the text flow better for an audiobook, while preserving narrative flow and 'direct mentions' of researchers.

RULES:
1. REMOVE parenthetical citations (e.g., APA style):
   - "(Smith, 2024)" -> [Removed]
   - "(Smith & Jones, 2024)" -> [Removed]
   - "(Smith et al., 2024)" -> [Removed]
   - "(NASA, 2024)" -> [Removed]
   - "(Adams, 2020; Zepf, 2024)" -> [Removed]
   - "(Title of Work, 2024)" -> [Removed]

2. REMOVE MLA-style citations:
   - "(Smith 42)" -> [Removed]
   - "(Smith and Jones 15)" -> [Removed]
   - "(Smith et al. 110)" -> [Removed]

3. REMOVE Vancouver/IEEE brackets and superscripts:
   - "[1]", "[2, 3]", "[4-7]" -> [Removed]
   - "¹", "²", "³" -> [Removed]

4. HANDLE NARRATIVE CITATIONS (Direct Mentions):
   - Definition: When authors are mentioned directly in the sentence, e.g., "Smith (2024) found..."
   - ACTION: Keep the author names (Smith) but REMOVE the associated parenthetical/bracketed info ((2024)).
   - Goal: The resulting sentence should be grammatically correct and natural to read.
   - Example 1: "Smith (2024) argued that..." -> "Smith argued that..."
   - Example 2: "As noted by Smith and Jones (2024)..." -> "As noted by Smith and Jones..."
   - Example 3: "According to NASA (2024)..." -> "According to NASA..."
   - Example 4: "In Smith et al. (2024)..." -> "In Smith and colleagues..." (Convert 'et al.' to 'and colleagues' for better TTS).
   - Example 5: "Smith [1] demonstrated..." -> "Smith demonstrated..."

5. DO NOT REMOVE:
   - References to Figures or Tables (e.g., "Figure 1", "Table 2").
   - Tools or Software versions (e.g., "Python 3.12").
   - Mathematical expressions or specific values (e.g., "p < 0.05").

6. CLEANUP & FLOW:
   - Ensure that after removal, there are no double spaces, double punctuation, or orphaned commas.
   - Example: "This is true (Smith, 2024), and that is false." -> "This is true, and that is false."
   - Example: "Research shows (Smith, 2024). Next sentence." -> "Research shows. Next sentence."
"""

# Figure Cleaning Skill
FIGURE_CLEANING_SKILL = """
SKILL: FIGURE & DIAGRAM CLEANING
Your goal is to remove all non-narrative elements that originate from figures, charts, diagrams, and tables.

RULES:
1. REMOVE Figure labels and captions:
   - "Figure 1: Title of figure." -> [Removed]
   - "Fig. 2. Graph showing..." -> [Removed]
   - Any floating text that looks like a caption or label.

2. REMOVE Numeric Data from charts/plots:
   - Isolated numbers like "0.1", "0.4", "0.2", "0.5", "0.6", "0.3" often appear in sequence when a chart's axis labels are extracted.
   - REMOVE these if they are not part of a natural sentence.

3. REMOVE Diagram internal markers:
   - "Refiner", "New Code", "Old Code", "Rollout", "Critic", "Evaluator", "Envs", "New H"
   - These are often labels from a box-and-arrow diagram incorrectly interleaved into the paragraph.

4. REMOVE Table artifacts:
   - Column headers or fragmented table rows that don't form sentences.

5. DO NOT REMOVE:
   - Narrative mentions of figures in the text, e.g., "As shown in Figure 1..." -> [KEEP]
   - Measurements that are part of a sentence, e.g., "The accuracy was 0.6." -> [KEEP]
"""
