#!/usr/bin/env python3
"""
Convert Jupyter notebook to Python script with markdown cells as comments.
"""

import json
import sys
from pathlib import Path


def convert_notebook_to_script(notebook_path: str, output_path: str):
    """
    Convert a Jupyter notebook to a Python script.

    Parameters:
    -----------
    notebook_path : str
        Path to the input .ipynb file
    output_path : str
        Path to save the output .py file
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    script_lines = []
    script_lines.append("#!/usr/bin/env python3")
    script_lines.append('"""')
    script_lines.append(f"Converted from: {Path(notebook_path).name}")
    script_lines.append('"""')
    script_lines.append("")

    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])

        if cell_type == 'markdown':
            # Convert markdown to comments
            script_lines.append("")
            script_lines.append("# " + "=" * 78)
            for line in source:
                # Remove trailing newlines and add # prefix
                line = line.rstrip('\n')
                if line.strip():
                    script_lines.append(f"# {line}")
                else:
                    script_lines.append("#")
            script_lines.append("# " + "=" * 78)
            script_lines.append("")

        elif cell_type == 'code':
            # Add code as-is
            for line in source:
                script_lines.append(line.rstrip('\n'))
            script_lines.append("")

    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_lines))

    print(f"✅ Converted {notebook_path} -> {output_path}")
    print(f"   Total lines: {len(script_lines)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_notebook.py <input.ipynb> <output.py>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    output_path = sys.argv[2]

    convert_notebook_to_script(notebook_path, output_path)
