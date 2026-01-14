#!/usr/bin/env python3
"""
Convert Jupyter notebooks to Markdown with Plotly figures as static images.
This keeps documentation fast and lightweight.
"""
import json
import sys
import base64
from pathlib import Path


def convert_notebook_to_markdown(notebook_path: Path, output_path: Path, images_dir: Path):
    """Convert a notebook to markdown with Plotly outputs as static images."""
    try:
        import plotly.io as pio
        import plotly.graph_objects as go
    except ImportError:
        pio = None
        print("Warning: plotly not available, Plotly outputs will be skipped", file=sys.stderr)
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    md_lines = []
    fig_count = 0
    
    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type')
        
        if cell_type == 'markdown':
            # Add markdown content directly
            source = ''.join(cell.get('source', []))
            md_lines.append(source)
            md_lines.append('\n\n')
            
        elif cell_type == 'code':
            # Add code block
            source = ''.join(cell.get('source', []))
            if source.strip():
                md_lines.append('```python\n')
                md_lines.append(source)
                if not source.endswith('\n'):
                    md_lines.append('\n')
                md_lines.append('```\n\n')
            
            # Process outputs
            for output in cell.get('outputs', []):
                output_type = output.get('output_type')
                
                if output_type == 'stream':
                    # Text output (print statements)
                    text = ''.join(output.get('text', []))
                    if text.strip():
                        md_lines.append('```\n')
                        md_lines.append(text)
                        if not text.endswith('\n'):
                            md_lines.append('\n')
                        md_lines.append('```\n\n')
                
                elif output_type in ('display_data', 'execute_result'):
                    data = output.get('data', {})
                    
                    # Check for Plotly JSON - convert to static image
                    if 'application/vnd.plotly.v1+json' in data and pio:
                        plotly_json = data['application/vnd.plotly.v1+json']
                        try:
                            fig = go.Figure(plotly_json)
                            fig_count += 1
                            
                            # Save as PNG
                            img_name = f"{notebook_path.stem}_fig{fig_count}.png"
                            img_path = images_dir / img_name
                            
                            # Export to PNG (requires kaleido)
                            fig.write_image(str(img_path), width=900, height=700, scale=2)
                            
                            # Add image reference to markdown
                            rel_path = f"images/{img_name}"
                            md_lines.append(f'![Figure {fig_count}]({rel_path})\n\n')
                            
                            print(f"  Exported figure {fig_count}: {img_name}")
                        except Exception as e:
                            print(f"Warning: Could not export Plotly figure in {notebook_path.name}: {e}", file=sys.stderr)
                            # Fallback: try to add as text
                            md_lines.append(f'*[Figure {fig_count} - export failed: {e}]*\n\n')
                    
                    # Check for existing image data (PNG)
                    elif 'image/png' in data:
                        fig_count += 1
                        img_data = data['image/png']
                        img_name = f"{notebook_path.stem}_fig{fig_count}.png"
                        img_path = images_dir / img_name
                        
                        # Decode and save
                        with open(img_path, 'wb') as f:
                            f.write(base64.b64decode(img_data))
                        
                        rel_path = f"images/{img_name}"
                        md_lines.append(f'![Figure {fig_count}]({rel_path})\n\n')
                    
                    # Check for plain text
                    elif 'text/plain' in data:
                        text = ''.join(data['text/plain'])
                        # Skip unhelpful outputs like object representations
                        if text.strip() and not text.startswith('<') and not text.startswith('Figure'):
                            md_lines.append('```\n')
                            md_lines.append(text)
                            if not text.endswith('\n'):
                                md_lines.append('\n')
                            md_lines.append('```\n\n')
                
                elif output_type == 'error':
                    # Skip error outputs
                    pass
    
    # Write markdown file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(md_lines))
    
    print(f"Converted {notebook_path.name} -> {output_path.name} ({fig_count} figures)")


if __name__ == '__main__':
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    examples_dir = Path(__file__).parent.parent / 'docs' / 'examples'
    images_dir = examples_dir / 'images'
    
    # Create directories
    examples_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Notebooks to convert (excluding logo.ipynb)
    notebooks = [
        'compute_geodesics.ipynb',
        'meshes.ipynb', 
        'projection_hypersurfaces.ipynb',
        'samplers.ipynb'
    ]
    
    for nb_name in notebooks:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            print(f"\nConverting {nb_name}...")
            md_name = nb_path.stem + '.md'
            md_path = examples_dir / md_name
            convert_notebook_to_markdown(nb_path, md_path, images_dir)
        else:
            print(f"Warning: {nb_path} not found", file=sys.stderr)

