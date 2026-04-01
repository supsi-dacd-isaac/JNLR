#!/usr/bin/env python3
"""
Convert Jupyter notebooks to Markdown with interactive Plotly figures.

Plotly figures are exported as standalone HTML files (Plotly.js loaded from CDN)
and embedded via lazy-loading iframes.  The browser caches Plotly.js after the
first figure, so subsequent figures load almost instantly.
"""
import json
import sys
import base64
import copy
from pathlib import Path


def _optimize_plotly_data(fig_dict, decimals=4):
    """Round float arrays in Plotly traces to reduce file size.

    Typical savings: 30-50 % on dense 3-D mesh/scatter data.
    """
    for trace in fig_dict.get("data", []):
        for key in list(trace.keys()):
            val = trace[key]
            if isinstance(val, list) and len(val) > 0:
                first_real = next((v for v in val if v is not None), None)
                if isinstance(first_real, float):
                    trace[key] = [
                        round(v, decimals) if isinstance(v, float) else v
                        for v in val
                    ]
    return fig_dict


def _export_plotly_html(fig_dict, html_path, *, decimals=4):
    """Export a Plotly figure dict as a lightweight standalone HTML file."""
    import plotly.graph_objects as go

    fig_dict = copy.deepcopy(fig_dict)
    _optimize_plotly_data(fig_dict, decimals=decimals)

    layout = fig_dict.setdefault("layout", {})
    layout["autosize"] = True

    fig = go.Figure(fig_dict)
    html = fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["sendDataToCloud"],
        },
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path.stat().st_size


def convert_notebook_to_markdown(
    notebook_path: Path,
    output_path: Path,
    images_dir: Path,
    plots_dir: Path,
):
    """Convert a notebook to markdown with interactive Plotly figures."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    md_lines: list[str] = []
    fig_count = 0

    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type")

        if cell_type == "markdown":
            source = "".join(cell.get("source", []))
            md_lines.append(source)
            md_lines.append("\n\n")

        elif cell_type == "code":
            source = "".join(cell.get("source", []))
            if source.strip():
                md_lines.append("```python\n")
                md_lines.append(source)
                if not source.endswith("\n"):
                    md_lines.append("\n")
                md_lines.append("```\n\n")

            for output in cell.get("outputs", []):
                output_type = output.get("output_type")

                if output_type == "stream":
                    text = "".join(output.get("text", []))
                    if text.strip():
                        md_lines.append("```\n")
                        md_lines.append(text)
                        if not text.endswith("\n"):
                            md_lines.append("\n")
                        md_lines.append("```\n\n")

                elif output_type in ("display_data", "execute_result"):
                    data = output.get("data", {})

                    # ── Interactive Plotly figure ───────────────────────
                    if "application/vnd.plotly.v1+json" in data:
                        fig_dict = data["application/vnd.plotly.v1+json"]
                        fig_count += 1

                        html_name = f"{notebook_path.stem}_fig{fig_count}.html"
                        html_path = plots_dir / html_name

                        size = _export_plotly_html(fig_dict, html_path)
                        print(
                            f"  Figure {fig_count}: {html_name} "
                            f"({size / 1024:.0f} KB)"
                        )

                        height = fig_dict.get("layout", {}).get("height", 700)

                        # MkDocs uses directory URLs by default, so
                        # examples/page/index.html -> ../plots/file.html
                        md_lines.append(
                            f'<div class="plotly-figure">\n'
                            f'  <iframe src="../plots/{html_name}" '
                            f'loading="lazy" '
                            f'style="width:100%;height:{height}px;'
                            f"border:none;border-radius:8px;"
                            f'box-shadow:0 2px 8px rgba(0,0,0,0.08);">'
                            f"</iframe>\n"
                            f"</div>\n\n"
                        )

                    # ── Raster image (PNG) ─────────────────────────────
                    elif "image/png" in data:
                        fig_count += 1
                        img_data = data["image/png"]
                        img_name = f"{notebook_path.stem}_fig{fig_count}.png"
                        img_path = images_dir / img_name

                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(img_data))

                        rel_path = f"images/{img_name}"
                        md_lines.append(f"![Figure {fig_count}]({rel_path})\n\n")

                    # ── Plain text ─────────────────────────────────────
                    elif "text/plain" in data:
                        text = "".join(data["text/plain"])
                        if (
                            text.strip()
                            and not text.startswith("<")
                            and not text.startswith("Figure")
                        ):
                            md_lines.append("```\n")
                            md_lines.append(text)
                            if not text.endswith("\n"):
                                md_lines.append("\n")
                            md_lines.append("```\n\n")

                elif output_type == "error":
                    pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))

    print(f"Converted {notebook_path.name} -> {output_path.name} ({fig_count} figures)")


if __name__ == "__main__":
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    examples_dir = Path(__file__).parent.parent / "docs" / "examples"
    images_dir = examples_dir / "images"
    plots_dir = examples_dir / "plots"

    for d in (examples_dir, images_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Notebooks to convert (logo.ipynb excluded on purpose)
    notebooks = [
        "compute_geodesics.ipynb",
        "meshes.ipynb",
        "projection_minimal_example.ipynb",
        "projection_hypersurfaces.ipynb",
        "samplers.ipynb",
        "should.ipynb",
    ]

    for nb_name in notebooks:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            print(f"\nConverting {nb_name}...")
            md_name = nb_path.stem + ".md"
            md_path = examples_dir / md_name
            convert_notebook_to_markdown(nb_path, md_path, images_dir, plots_dir)
        else:
            print(f"Warning: {nb_path} not found", file=sys.stderr)
