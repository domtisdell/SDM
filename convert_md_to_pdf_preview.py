#!/usr/bin/env python3
"""
Convert Markdown files to beautiful PDFs - Preview Version
This version excludes code blocks and focuses on content preview
"""

import os
import glob
import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import re

# Enhanced CSS for beautiful PDF styling (preview version)
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

/* Header styling with gradient effect */
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
    background: linear-gradient(135deg, #1e3a5f 0%, #3498db 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
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

/* Preview notice box */
.preview-notice {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 5px solid #3498db;
    padding: 15px 20px;
    margin: 20px 0;
    border-radius: 5px;
    font-style: italic;
    color: #555;
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
    transition: background-color 0.3s ease;
}

table tr:last-child td {
    border-bottom: none;
}

/* Enhanced blockquotes */
blockquote {
    border-left: 5px solid #3498db;
    padding: 15px 20px;
    margin: 1.5em 0;
    background-color: #f8f9fa;
    border-radius: 0 5px 5px 0;
    color: #555;
    font-style: italic;
}

/* Lists with better spacing */
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

/* Special sections */
.summary-box {
    background: linear-gradient(135deg, #ecf8ff 0%, #dbeafe 100%);
    border: 1px solid #3498db;
    border-radius: 8px;
    padding: 20px;
    margin: 2em 0;
}

.metric-box {
    background-color: #f0f8ff;
    border-left: 4px solid #3498db;
    padding: 15px;
    margin: 15px 0;
    border-radius: 0 5px 5px 0;
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

def preprocess_markdown_preview(md_content):
    """Preprocess markdown content to create a preview version"""
    
    # Add preview notice at the beginning
    preview_notice = """
<div class="preview-notice">
<strong>Preview Version</strong><br>
This is a preview version of the report with code blocks and technical details removed for better readability.
</div>

---

"""
    
    # Remove code blocks (both inline and fenced)
    # Remove fenced code blocks
    md_content = re.sub(r'```[\s\S]*?```', '[Code block removed for preview]', md_content)
    
    # Remove inline code but keep the text
    md_content = re.sub(r'`([^`]+)`', r'\1', md_content)
    
    # Convert mermaid blocks to descriptive text
    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    
    def replace_mermaid(match):
        return '[Diagram: Visual representation of the data flow and relationships]'
    
    md_content = re.sub(mermaid_pattern, replace_mermaid, md_content, flags=re.DOTALL)
    
    # Add section formatting for better visual hierarchy
    # Make sure headers have proper spacing
    md_content = re.sub(r'(#+\s+[^\n]+)', r'\n\n\1\n', md_content)
    
    # Add the preview notice
    md_content = preview_notice + md_content
    
    return md_content

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
        Generated by SDM Forecasting System | Preview Version
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

def convert_md_to_pdf_preview(md_file_path, output_dir):
    """Convert a single markdown file to PDF preview"""
    # Read markdown file
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Preprocess markdown for preview
    md_content = preprocess_markdown_preview(md_content)
    
    # Convert markdown to HTML with extensions
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'tables', 'toc', 'attr_list', 'def_list']
    )
    
    # Get clean title
    title = Path(md_file_path).stem.replace('_', ' ')
    
    # Enhance HTML structure
    full_html = enhance_html_structure(html_content, title)
    
    # Generate PDF filename with preview suffix
    pdf_filename = Path(md_file_path).stem + '_preview.pdf'
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    # Convert to PDF with custom styling
    HTML(string=full_html).write_pdf(
        pdf_path,
        stylesheets=[CSS(string=CUSTOM_CSS)]
    )
    
    return pdf_path

def main():
    """Main function to convert all markdown files in forecasts directory"""
    # Create output directory for forecast previews
    output_dir = '/home/dominic/wsl_coding/projects/SDM/forecasts/pdf_previews'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all markdown files in forecasts directory
    md_files = glob.glob('/home/dominic/wsl_coding/projects/SDM/forecasts/**/*.md', recursive=True)
    
    print(f"Found {len(md_files)} markdown files in forecasts directory")
    
    # Convert each file
    successful = 0
    failed = 0
    
    for md_file in md_files:
        try:
            print(f"Converting: {os.path.relpath(md_file, '/home/dominic/wsl_coding/projects/SDM/forecasts/')}")
            pdf_path = convert_md_to_pdf_preview(md_file, output_dir)
            print(f"  ✓ Created: {os.path.basename(pdf_path)}")
            successful += 1
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            failed += 1
    
    print(f"\nPreview PDF conversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Preview PDFs saved to: {output_dir}")

if __name__ == "__main__":
    main()