#!/usr/bin/env python3
"""
Convert Markdown files to beautiful PDFs with fully rendered Mermaid diagrams
Fixed version with proper mermaid initialization
"""

import os
import glob
import re
import base64
import asyncio
from pathlib import Path
import tempfile
import shutil

# Import libraries
from pyppeteer import launch
from weasyprint import HTML, CSS
import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor


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

# Fixed Mermaid HTML template for rendering
MERMAID_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {
            background: white;
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        #graph-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
        }
        .mermaid {
            display: block;
        }
    </style>
</head>
<body>
    <div id="graph-container">
        <pre class="mermaid">
{mermaid_code}
        </pre>
    </div>
    <script>
        mermaid.initialize({ 
            startOnLoad: true,
            theme: 'default',
            themeVariables: {
                primaryColor: '#3498db',
                primaryTextColor: '#fff',
                primaryBorderColor: '#2980b9',
                lineColor: '#5a5a5a',
                secondaryColor: '#ecf0f1',
                tertiaryColor: '#e8f4fd'
            }
        });
    </script>
</body>
</html>
"""


async def render_mermaid_to_image(mermaid_code, output_path):
    """Render mermaid diagram to PNG image using pyppeteer"""
    try:
        # Create HTML with mermaid code
        html_content = MERMAID_HTML_TEMPLATE.format(mermaid_code=mermaid_code)
        
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_html_path = f.name
        
        # Launch headless browser
        browser = await launch(
            headless=True, 
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        page = await browser.newPage()
        
        # Set viewport
        await page.setViewport({'width': 1400, 'height': 900})
        
        # Load the HTML file
        await page.goto(f'file://{temp_html_path}', {'waitUntil': 'networkidle0'})
        
        # Wait for mermaid to render
        await page.waitForSelector('.mermaid[data-processed="true"]', {'timeout': 15000})
        
        # Wait a bit more for complete rendering
        await page.waitFor(1000)
        
        # Get the bounding box of the diagram
        element = await page.querySelector('#graph-container')
        if element:
            box = await element.boundingBox()
            
            # Take screenshot of just the diagram
            await page.screenshot({
                'path': output_path,
                'clip': {
                    'x': box['x'],
                    'y': box['y'],
                    'width': box['width'],
                    'height': box['height']
                }
            })
        else:
            # Fallback to full page screenshot
            await page.screenshot({'path': output_path})
        
        await browser.close()
        
        # Clean up temp file
        os.unlink(temp_html_path)
        
        return True
    except Exception as e:
        print(f"    Warning: Could not render mermaid diagram: {e}")
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
        
        # Render mermaid to image (run async function)
        success = asyncio.run(render_mermaid_to_image(mermaid_code, image_path))
        
        if success and os.path.exists(image_path):
            # Read image and convert to base64
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode()
            
            # Return HTML img tag with embedded image
            return f'<div class="mermaid-diagram"><img src="data:image/png;base64,{img_base64}" alt="Diagram {diagram_counter}"/></div>'
        else:
            # Fallback - show the mermaid code as a code block
            return f'''<div class="mermaid-diagram">
<pre><code>Mermaid Diagram {diagram_counter}:
{mermaid_code}</code></pre>
<p><em>[Note: Diagram rendering failed. Above shows the mermaid code.]</em></p>
</div>'''
    
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
        pdf_filename = Path(md_file_path).stem + '_diagrams.pdf'
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
    """Main function to convert all markdown files with diagrams"""
    # Create output directory
    output_dir = '/home/dominic/wsl_coding/projects/SDM/forecasts/pdf_with_diagrams_fixed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all markdown files in forecasts directory
    md_files = glob.glob('/home/dominic/wsl_coding/projects/SDM/forecasts/**/*.md', recursive=True)
    
    print(f"Found {len(md_files)} markdown files in forecasts directory")
    print("Rendering mermaid diagrams to images...")
    
    # Convert each file
    successful = 0
    failed = 0
    
    for md_file in md_files:
        try:
            print(f"\nConverting: {os.path.relpath(md_file, '/home/dominic/wsl_coding/projects/SDM/forecasts/')}")
            pdf_path = convert_md_to_pdf_with_diagrams(md_file, output_dir)
            print(f"  ✓ Created: {os.path.basename(pdf_path)}")
            successful += 1
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\nPDF conversion with diagrams complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  PDFs saved to: {output_dir}")


if __name__ == "__main__":
    # Run main conversion
    main()