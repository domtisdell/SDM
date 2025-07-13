#!/usr/bin/env python3
"""
Convert Markdown files to beautiful PDFs with proper styling
"""

import os
import glob
import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import re

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
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 100%;
}

h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    font-weight: 600;
    page-break-after: avoid;
}

h1 {
    font-size: 28px;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}

h2 {
    font-size: 24px;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 8px;
}

h3 {
    font-size: 20px;
}

h4 {
    font-size: 18px;
}

p {
    margin: 1em 0;
    text-align: justify;
}

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

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.5em 0;
    page-break-inside: avoid;
}

table th {
    background-color: #3498db;
    color: white;
    padding: 12px;
    text-align: left;
    font-weight: 600;
}

table td {
    padding: 10px;
    border-bottom: 1px solid #ecf0f1;
}

table tr:nth-child(even) {
    background-color: #f9f9f9;
}

table tr:hover {
    background-color: #f5f5f5;
}

blockquote {
    border-left: 4px solid #3498db;
    padding-left: 20px;
    margin-left: 0;
    color: #666;
    font-style: italic;
}

ul, ol {
    margin: 1em 0;
    padding-left: 30px;
}

li {
    margin: 0.5em 0;
}

a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

strong {
    font-weight: 600;
    color: #2c3e50;
}

em {
    font-style: italic;
    color: #555;
}

hr {
    border: none;
    border-top: 2px solid #ecf0f1;
    margin: 2em 0;
}

/* Mermaid diagram styling */
.mermaid {
    text-align: center;
    margin: 1.5em 0;
    page-break-inside: avoid;
}

/* Special styling for key metrics */
.metric {
    background-color: #ecf0f1;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

/* Page breaks */
.page-break {
    page-break-after: always;
}

/* Cover page styling */
.cover-page {
    text-align: center;
    padding-top: 30%;
    page-break-after: always;
}

.cover-page h1 {
    font-size: 36px;
    border: none;
    margin-bottom: 20px;
}

.cover-page .subtitle {
    font-size: 18px;
    color: #666;
    margin-top: 10px;
}
"""

def preprocess_markdown(md_content):
    """Preprocess markdown content to handle mermaid diagrams"""
    # Convert mermaid blocks to simple code blocks with a note
    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    
    def replace_mermaid(match):
        diagram_content = match.group(1)
        return f'```\n[Mermaid Diagram]\n{diagram_content}\n```\n*Note: This is a Mermaid diagram. View in a Mermaid-compatible viewer for visual representation.*'
    
    content = re.sub(mermaid_pattern, replace_mermaid, md_content, flags=re.DOTALL)
    
    return content

def convert_md_to_pdf(md_file_path, output_dir):
    """Convert a single markdown file to PDF"""
    # Read markdown file
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Preprocess markdown
    md_content = preprocess_markdown(md_content)
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables', 'toc']
    )
    
    # Add HTML structure
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{Path(md_file_path).stem}</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate PDF filename
    pdf_filename = Path(md_file_path).stem + '.pdf'
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    # Convert to PDF
    HTML(string=full_html).write_pdf(
        pdf_path,
        stylesheets=[CSS(string=CUSTOM_CSS)]
    )
    
    return pdf_path

def main():
    """Main function to convert all markdown files"""
    # Create output directory
    output_dir = '/home/dominic/wsl_coding/projects/SDM/outputs/track_a/pdf_reports'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all markdown files in track A outputs
    md_files = []
    md_files.extend(glob.glob('/home/dominic/wsl_coding/projects/SDM/outputs/track_a/wsa_steel_taxonomy_analysis/*.md'))
    md_files.extend(glob.glob('/home/dominic/wsl_coding/projects/SDM/forecasts/track_a_*/wsa_steel_taxonomy_analysis/*.md'))
    
    # Remove duplicates
    md_files = list(set(md_files))
    
    print(f"Found {len(md_files)} markdown files to convert")
    
    # Convert each file
    successful = 0
    failed = 0
    
    for md_file in md_files:
        try:
            print(f"Converting: {os.path.basename(md_file)}")
            pdf_path = convert_md_to_pdf(md_file, output_dir)
            print(f"  ✓ Created: {os.path.basename(pdf_path)}")
            successful += 1
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            failed += 1
    
    print(f"\nConversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  PDFs saved to: {output_dir}")

if __name__ == "__main__":
    main()