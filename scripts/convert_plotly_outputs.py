#!/usr/bin/env python3
"""
Convert Plotly JSON outputs in notebooks to HTML for proper rendering in MkDocs.
This script processes notebooks and converts application/vnd.plotly.v1+json outputs
to text/html so they display correctly.
"""
import json
import sys
from pathlib import Path

def convert_plotly_to_html(notebook_path: Path):
    """Convert Plotly JSON outputs in a notebook to HTML."""
    try:
        import plotly.io as pio
    except ImportError:
        print("Warning: plotly not available, skipping Plotly conversion", file=sys.stderr)
        return False
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    # Process each cell
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if 'data' in output:
                    data = output['data']
                    # Check for Plotly JSON output
                    if 'application/vnd.plotly.v1+json' in data:
                        plotly_json = data['application/vnd.plotly.v1+json']
                        try:
                            # Convert to HTML
                            html_str = pio.to_html(
                                plotly_json, 
                                include_plotlyjs='cdn',
                                div_id=f"plotly-{abs(hash(str(plotly_json)))}"
                            )
                            # Replace JSON output with HTML
                            data['text/html'] = [html_str]
                            # Keep the JSON for reference but HTML takes precedence
                            modified = True
                        except Exception as e:
                            print(f"Warning: Could not convert Plotly output in {notebook_path.name}: {e}", file=sys.stderr)
    
    # Write back if modified
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Converted Plotly outputs in {notebook_path.name}")
        return True
    return False

if __name__ == '__main__':
    examples_dir = Path(__file__).parent.parent / 'docs' / 'examples'
    
    # Convert each notebook
    for nb_path in examples_dir.glob('*.ipynb'):
        convert_plotly_to_html(nb_path)

