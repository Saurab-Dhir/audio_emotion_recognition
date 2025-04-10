import json
import os

# Path to the notebook
notebook_path = 'notebooks/02_Feature_Extraction_Visualization.ipynb'

# Load the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Fix the style in the cell (assuming it's the second cell, index 1)
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        # Find the cell with the style.use line and fix it
        source = cell['source']
        for j, line in enumerate(source):
            if "plt.style.use('seaborn-whitegrid')" in line:
                source[j] = line.replace("seaborn-whitegrid", "seaborn-v0_8-whitegrid")
                print(f"Fixed style in cell {i+1}")
            
            # Also fix any indentation issues
            if line.startswith("    plt.style.use"):
                source[j] = line.lstrip()
                print(f"Fixed indentation in cell {i+1}")

# Save the notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook {notebook_path} has been updated.") 