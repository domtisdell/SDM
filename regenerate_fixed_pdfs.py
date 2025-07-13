#!/usr/bin/env python3
"""
Re-generate only the fixed WSA Production Flow Hierarchy PDFs
"""

import os
import glob
import re
import base64
import subprocess
from pathlib import Path
import tempfile
import shutil

# Import libraries
from weasyprint import HTML, CSS
import markdown


# Custom CSS for beautiful PDF styling
CUSTOM_CSS = """
@page {
    size: A4;
    margin: 2.5cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 10px;
        color: #666;
    }
}

body {
    font-family: 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    line-height: 1.8;
    color: #2c3e50;
    max-width: 100%;
    font-size: 11pt;
}

/* Header styling */
h1, h2, h3, h4, h5, h6 {
    color: #1e3a5f;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    font-weight: 600;
    page-break-after: avoid;
}

h1 {
    font-size: 32px;
    border-bottom: 4px solid #3498db;
    padding-bottom: 12px;
    margin-top: 0;
}

h2 {
    font-size: 26px;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 8px;
    color: #2c5282;
}

h3 {
    font-size: 22px;
    color: #2d5382;
}

h4 {
    font-size: 18px;
    color: #34495e;
}

p {
    margin: 1.2em 0;
    text-align: justify;
    line-height: 1.8;
}

/* Code blocks */
code {
    background-color: #f5f5f5;
    padding: 2px 4px;
    border-radius: 3px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.9em;
}

pre {
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    padding: 0;
}

/* Diagram styling */
.mermaid-diagram {
    text-align: center;
    margin: 2em 0;
    page-break-inside: avoid;
}

.mermaid-diagram img {
    max-width: 100%;
    height: auto;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 10px;
    background: white;
}

/* Beautiful tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 2em 0;
    page-break-inside: avoid;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-radius: 8px;
    overflow: hidden;
}

table th {
    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
    color: white;
    padding: 14px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 10pt;
    letter-spacing: 0.5px;
}

table td {
    padding: 12px 16px;
    border-bottom: 1px solid #ecf0f1;
    font-size: 10pt;
}

table tr:nth-child(even) {
    background-color: #f8f9fa;
}

table tr:hover {
    background-color: #e8f4fd;
}

table tr:last-child td {
    border-bottom: none;
}

/* Blockquotes */
blockquote {
    border-left: 5px solid #3498db;
    padding: 15px 20px;
    margin: 1.5em 0;
    background-color: #f8f9fa;
    border-radius: 0 5px 5px 0;
    color: #555;
    font-style: italic;
}

/* Lists */
ul, ol {
    margin: 1.2em 0;
    padding-left: 35px;
}

li {
    margin: 0.6em 0;
    line-height: 1.7;
}

/* Links */
a {
    color: #3498db;
    text-decoration: none;
    border-bottom: 1px dotted #3498db;
}

a:hover {
    color: #2980b9;
    border-bottom-style: solid;
}

/* Emphasis */
strong {
    font-weight: 600;
    color: #1e3a5f;
}

em {
    font-style: italic;
    color: #555;
}

/* Horizontal rules */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #3498db, transparent);
    margin: 3em 0;
}

/* Page breaks */
.page-break {
    page-break-after: always;
}

/* Report header */
.report-header {
    text-align: center;
    padding: 40px 0;
    border-bottom: 2px solid #ecf0f1;
    margin-bottom: 30px;
}

.report-header h1 {
    border: none;
    margin-bottom: 10px;
}

.report-header .date {
    color: #7f8c8d;
    font-size: 14px;
}

/* Footer styling */
.report-footer {
    margin-top: 50px;
    padding-top: 20px;
    border-top: 2px solid #ecf0f1;
    text-align: center;
    color: #7f8c8d;
    font-size: 10pt;
}
"""


def render_mermaid_with_cli(mermaid_code, output_path):
    """Render mermaid diagram using mermaid-cli (mmdc)"""
    try:
        # Create temporary mermaid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as f:
            f.write(mermaid_code)
            mermaid_file = f.name
        
        # Call mmdc to render the diagram
        cmd = [
            'mmdc',
            '-i', mermaid_file,
            '-o', output_path,
            '-b', 'white',
            '-w', '1200',
            '-H', '800',
            '--theme', 'default'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp file
        os.unlink(mermaid_file)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return True
        else:
            if result.stderr:
                print(f"    Mermaid CLI error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"    Error with mermaid-cli: {e}")
        return False


def process_mermaid_diagrams(md_content, temp_dir):
    """Process markdown content to convert mermaid diagrams to images"""
    # Pattern to find mermaid code blocks
    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    
    diagram_counter = 0
    
    def replace_mermaid(match):
        nonlocal diagram_counter
        diagram_counter += 1
        
        mermaid_code = match.group(1)
        
        # Generate image filename
        image_filename = f'diagram_{diagram_counter}.png'
        image_path = os.path.join(temp_dir, image_filename)
        
        # Try to render with mermaid-cli
        success = render_mermaid_with_cli(mermaid_code, image_path)
        
        if success and os.path.exists(image_path):
            # Read image and convert to base64
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode()
            
            # Return HTML img tag with embedded image
            return f'''
<div class="mermaid-diagram">
    <img src="data:image/png;base64,{img_base64}" alt="WSA Production Flow Hierarchy Diagram"/>
</div>
'''
        else:
            # Fallback - should not happen now that we fixed the HTML entities
            return f'''
<div class="mermaid-diagram">
    <div class="diagram-notice">
        <strong>Diagram {diagram_counter}: Flow Diagram</strong><br>
        <em>Note: Diagram rendering failed. Please check the mermaid code.</em>
    </div>
</div>
'''
    
    # Replace all mermaid blocks with rendered images
    processed_content = re.sub(mermaid_pattern, replace_mermaid, md_content, flags=re.DOTALL)
    
    return processed_content


def enhance_html_structure(html_content, title):
    """Enhance HTML with additional structure for better PDF rendering"""
    
    # Add report header
    report_header = f"""
    <div class="report-header">
        <h1>{title}</h1>
        <div class="date">Steel Demand Model - Forecast Report</div>
    </div>
    """
    
    # Add report footer
    report_footer = """
    <div class="report-footer">
        Generated by SDM Forecasting System
    </div>
    """
    
    # Wrap content in a container
    enhanced_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
    </head>
    <body>
        {report_header}
        <div class="content">
            {html_content}
        </div>
        {report_footer}
    </body>
    </html>
    """
    
    return enhanced_html


def convert_md_to_pdf_with_diagrams(md_file_path, output_dir):
    """Convert a single markdown file to PDF with rendered diagrams"""
    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Read markdown file
        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Process mermaid diagrams
        print(f"  Processing mermaid diagrams...")
        md_content = process_mermaid_diagrams(md_content, temp_dir)
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['extra', 'tables', 'toc', 'attr_list', 'def_list']
        )
        
        # Get clean title
        title = Path(md_file_path).stem.replace('_', ' ')
        
        # Enhance HTML structure
        full_html = enhance_html_structure(html_content, title)
        
        # Generate PDF filename
        pdf_filename = Path(md_file_path).stem + '.pdf'
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        # Convert to PDF with custom styling
        HTML(string=full_html).write_pdf(
            pdf_path,
            stylesheets=[CSS(string=CUSTOM_CSS)]
        )
        
        return pdf_path
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def main():
    """Main function to regenerate only the fixed Production Flow Hierarchy PDFs"""
    # Create output directory
    output_dir = '/home/dominic/wsl_coding/projects/SDM/forecasts/pdf_final'
    
    # Only process the fixed Production Flow Hierarchy files
    md_files = [
        '/home/dominic/wsl_coding/projects/SDM/forecasts/track_a_20250714_061713/wsa_steel_taxonomy_analysis/WSA_Production_Flow_Hierarchy_2025.md',
        '/home/dominic/wsl_coding/projects/SDM/forecasts/track_a_20250714_061713/wsa_steel_taxonomy_analysis/WSA_Production_Flow_Hierarchy_2035.md',
        '/home/dominic/wsl_coding/projects/SDM/forecasts/track_a_20250714_061713/wsa_steel_taxonomy_analysis/WSA_Production_Flow_Hierarchy_2050.md'
    ]
    
    print(f"Re-generating {len(md_files)} fixed Production Flow Hierarchy PDFs")
    
    # Convert each file
    successful = 0
    failed = 0
    
    for md_file in md_files:
        try:
            print(f"\nConverting: {os.path.basename(md_file)}")
            pdf_path = convert_md_to_pdf_with_diagrams(md_file, output_dir)
            print(f"  ✓ Created: {os.path.basename(pdf_path)}")
            successful += 1
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\nPDF regeneration complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Fixed PDFs saved to: {output_dir}")


if __name__ == "__main__":
    main()