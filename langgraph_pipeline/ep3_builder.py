"""EP3 (EPUB3) audiobook builder with Media Overlays and chapter navigation.

Creates EPUB3 files with:
- XHTML content documents (one per section)
- SMIL synchronization files
- OPF package file
- Navigation document (nav.xhtml)
"""

import os
import zipfile
import uuid
import shutil
from pathlib import Path
from typing import List, Dict


def create_ep3(
    audio_path: str,
    cleaned_text: str,
    sections: List[Dict[str, str]],
    output_path: str
) -> str:
    """
    Create an EP3 audiobook from audio and text sections.
    
    Args:
        audio_path: Path to concatenated WAV file
        cleaned_text: Full cleaned text content
        sections: List of dicts with 'title' and 'content' keys
        output_path: Output path for .epub file
        
    Returns:
        Path to created EP3 file
    """
    # Generate unique ID
    unique_id = str(uuid.uuid4())
    
    # Create temporary directory for EPUB structure
    temp_dir = Path(output_path).parent / f"ep3_temp_{unique_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create EPUB directory structure
        oebps_dir = temp_dir / "OEBPS"
        oebps_dir.mkdir()
        (oebps_dir / "css").mkdir()
        (oebps_dir / "audio").mkdir()
        (oebps_dir / "chapters").mkdir()
        (oebps_dir / "SMIL").mkdir()
        
        # Copy and rename audio file
        audio_filename = "audio.wav"
        shutil.copy2(audio_path, oebps_dir / "audio" / audio_filename)
        
        # Create CSS file
        css_content = """body {
    font-family: sans-serif;
    line-height: 1.6;
    margin: 1em;
}
section {
    margin-bottom: 2em;
}
h1 {
    color: #333;
}
"""
        with open(oebps_dir / "css" / "style.css", "w", encoding="utf-8") as f:
            f.write(css_content)
        
        # Create content documents for each section
        chapter_files = []
        for i, section in enumerate(sections, 1):
            title = section.get("title", f"Section {i}")
            content = section.get("content", "")
            
            filename = f"chapter_{i:03d}.xhtml"
            chapter_files.append(filename)
            
            xhtml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>{title}</title>
    <link rel="stylesheet" type="text/css" href="css/style.css"/>
</head>
<body>
    <section epub:type="{title.lower()}">
        <h1>{title}</h1>
        <p>{content}</p>
    </section>
</body>
</html>
"""
            with open(oebps_dir / "chapters" / filename, "w", encoding="utf-8") as f:
                f.write(xhtml_content)
        
        # Create navigation document (nav.xhtml) - REQUIRED for EPUB3
        nav_items = "".join(
            f'                <li><a href="chapters/{f}">{title}</a></li>\n'
            for f, s in zip(chapter_files, sections)
            for title in [s.get("title", "Unknown")]
        )
        
        nav_xhtml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>Navigation</title>
    <link rel="stylesheet" type="text/css" href="css/style.css"/>
</head>
<body>
    <nav epub:type="toc" id="toc">
        <h1>Table of Contents</h1>
        <ol>
{nav_items}
        </ol>
    </nav>
    <nav epub:type="page-list" id="pagelist">
        <h1>Page List</h1>
        <ol>
{nav_items}
        </ol>
    </nav>
</body>
</html>
"""
        with open(oebps_dir / "nav.xhtml", "w", encoding="utf-8") as f:
            f.write(nav_xhtml)
        
        # Create SMIL file for media overlay synchronization
        smil_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<smil xmlns="http://www.w3.org/ns/smil" xmlns:epub="http://www.idpf.org/2007/ops">
    <head>
        <layout>
            <root-container>
                <region id="text" width="100%" height="100%"/>
            </root-container>
        </layout>
    </head>
    <body>
        <par>
            <audio src="audio/{audio_filename}" clip-begin="0s"/>  
            <text src="chapters/chapter_001.xhtml" region="text"/>
        </par>
    </body>
</smil>
"""
        with open(oebps_dir / "SMIL" / "overlay.smil", "w", encoding="utf-8") as f:
            f.write(smil_content)
        
        # Create OPF package file
        manifest_items = []
        manifest_items.append(f'    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>')
        manifest_items.append(f'    <item id="css" href="css/style.css" media-type="text/css"/>')
        manifest_items.append(f'    <item id="audio" href="audio/{audio_filename}" media-type="audio/wav"/>')
        manifest_items.append(f'    <item id="smil" href="SMIL/overlay.smil" media-type="application/smil+xml"/>')
        
        for i, filename in enumerate(chapter_files, 1):
            manifest_items.append(f'    <item id="chapter{i}" href="chapters/{filename}" media-type="application/xhtml+xml"/>')
        
        spine_items = '    <itemref idref="nav"/>\n' + "\n".join(
            f'    <itemref idref="chapter{i}"/>\n'
            for i in range(1, len(chapter_files) + 1)
        )
        
        opf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0" unique-identifier="BookId">
    <metadata>
        <dc:identifier id="BookId">{unique_id}</dc:identifier>
        <dc:title>Paper Narrator Audiobook</dc:title>
        <dc:language>en</dc:language>
        <meta property="dcterms:modified">{unique_id}</meta>
    </metadata>
    <manifest>
{chr(10).join(manifest_items)}
    </manifest>
    <spine toc="nav">
{spine_items}
    </spine>
</package>
"""
        with open(oebps_dir / "content.opf", "w", encoding="utf-8") as f:
            f.write(opf_content)
        
        # Create META-INF/container.xml
        container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>
"""
        meta_inf_dir = temp_dir / "META-INF"
        meta_inf_dir.mkdir()
        with open(meta_inf_dir / "container.xml", "w", encoding="utf-8") as f:
            f.write(container_xml)
        
        # Create ZIP file (EPUB is a ZIP)
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as epub:
            # Add META-INF/container.xml first (required)
            epub.write(
                meta_inf_dir / "container.xml",
                "META-INF/container.xml"
            )
            
            # Add OEBPS files
            for root, dirs, files in os.walk(oebps_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    epub.write(file_path, arcname)
        
        return output_path
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
