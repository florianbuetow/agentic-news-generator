# Notebook Restructuring TODO

## Goal
Separate narrative text (markdown) from computation (code) to improve readability.

## Principles
1. **Markdown for narrative**: All section headers, questions, explanations go in markdown cells
2. **Merge, don't duplicate**: When code cells print narrative, MERGE that content into the PREVIOUS markdown cell
3. **Code for computation**: No narrative in code cells except processing status
4. **Status prints acceptable**: Keep "Generating..." and "✓ Saved" messages - these are processing feedback
5. **Warnings acceptable**: Keep "⚠ Skipping..." messages - these are runtime status
6. **Separate calculation from display**: When possible, split into distinct cells

## Changes to Make

### Cell 1 (markdown) - Introduction
**ADD** analysis goals upfront:
```markdown
## Analysis Goals

This analysis addresses three key questions:

**GOAL 1: TEST PATTERN ANALYSIS**
- Question: How do different models compare on individual tests?
- Approach: Identify which test scripts are most commonly misclassified across models

**GOAL 2: CHARACTERISTIC ANALYSIS**
- Question: What makes models accurate? What makes models fast?
- Approach: Analyze relationships between model parameters, size, quantization, and performance

**GOAL 3: BEST MODEL SELECTION**
- Question: Which model is best (most accurate, fastest to run)?
- Approach: Use Pareto frontier analysis to identify optimal speed/accuracy trade-offs
```

### Cell 2 (code) - Imports + Config
**SPLIT INTO 3 CELLS:**

**New Cell 2a (code)** - Imports only:
```python
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
import os

from src.config import Config

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
```

**New Cell 2b (code)** - Configuration (NO debug prints):
```python
# This notebook is located in: notebooks/shellscript_analyzer/
# We need to navigate up to the project root
NOTEBOOK_LOCATION = Path("notebooks/shellscript_analyzer")
PROJECT_ROOT = Path.cwd().parent.parent

# Load configuration
_config = Config(PROJECT_ROOT / "config" / "config.yaml")

# Configuration constants
REPORTS_DIR = PROJECT_ROOT / _config.getReportsDir()
OUTPUT_DIR = PROJECT_ROOT / NOTEBOOK_LOCATION / "gfx"
CSV_PATTERN = "*.csv"
JPEG_DPI = 300
IMAGE_PREFIX = "shell_script_classification_"
FIGSIZE_LARGE = (16, 10)
FIGSIZE_WIDE = (20, 8)
FIGSIZE_XLARGE = (24, 12)

# Modern Color Palette
COLOR_PALETTE = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#ec4899",
    "success": "#10b981",
    "error": "#ef4444",
    "warning": "#f59e0b",
    "info": "#3b82f6",
    "false_positive": "#f87171",
    "false_negative": "#60a5fa",
    "heatmap_sequential": ["#ede9fe", "#c7d2fe", "#a5b4fc", "#818cf8", "#6366f1", "#4f46e5"],
    "heatmap_diverging": ["#ef4444", "#fca5a5", "#fef3c7", "#a7f3d0", "#34d399", "#10b981"],
    "categorical": ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#3b82f6", "#06b6d4", "#14b8a6"],
}

# Set seaborn style
sns.set_theme(style="whitegrid")
sns.set_palette(COLOR_PALETTE["categorical"])
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = JPEG_DPI
plt.rcParams["font.size"] = 10
```

**New Cell 2c (markdown)** - Explain configuration:
```markdown
## Section 1: Configuration

The notebook uses the following directory structure:
- **Notebook location**: `notebooks/shellscript_analyzer/`
- **Project root**: Automatically detected (two levels up from notebook location)
- **Reports directory**: `reports/` (loaded from `config.yaml`)
- **Output directory**: `notebooks/shellscript_analyzer/gfx/`

All visualizations will be saved as high-resolution JPEG images (300 DPI) with the prefix `shell_script_classification_`.
```

### Cell 8 (code) - Setup
**REMOVE** all print statements with "=" decorations:
```python
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

### Cell 10 (code) - Load Data
**REMOVE** narrative print headers:
```python
model_data = load_all_models()
model_specs = load_model_specifications()
```

### Cell 12 (code) - Metrics
**SPLIT INTO 2 CELLS:**

**New Cell 12a (code)** - Calculation only:
```python
metrics_df = calculate_metrics(model_data, model_specs)
```

**New Cell 12b (code)** - Display only:
```python
print("Performance Summary (Ranked by F1 Score)")
print("=" * 80)
print(metrics_df.to_string(index=False))
```

### Cell 15 (code) - Visualization 1: Confusion Matrices
**REMOVE** these lines from the beginning:
```python
print("\n" + "=" * 80)
print("ERROR ANALYSIS: CONFUSION MATRICES")
print("=" * 80)
print("Question: What types of errors do models make?")
print()
```

**KEEP** only:
```python
print("Generating Visualization 1: Confusion Matrices...")
# ... rest of visualization code ...
print(f"  ✓ Saved: {output_path.name}")
```

### Cell 17 (code) - Visualization 2: Misclassification Analysis
**REMOVE** these lines:
```python
print("\n" + "=" * 80)
print("GOAL 1: TEST PATTERN ANALYSIS")
print("=" * 80)
print("Question: How do different models compare on individual tests?")
print()
```

**KEEP** only:
```python
print("Generating Visualization 2: Misclassification Analysis...")
# ... rest of code ...
print(f"  ✓ Saved: {output_path.name}")
```

### Cell 19 (code) - Visualization 3: Heatmaps
**REMOVE**:
```python
print("\nGenerating Visualization 3: Model-Test Agreement Heatmaps...")
```

**ADD** inside the `create_heatmap()` function at the beginning:
```python
def create_heatmap(test_subset: pd.DataFrame, title: str, filename: str) -> None:
    """Helper function to create a single heatmap"""
    print(f"  Generating {title}...")
    # ... rest of function ...
```

### Cell 22 (code) - Visualization 4: Efficiency Scatter
**REMOVE** these lines:
```python
print("\n" + "=" * 80)
print("GOAL 3: BEST MODEL SELECTION (SPEED VS ACCURACY TRADE-OFF)")
print("=" * 80)
print("Question: Which model is best (most accurate, fastest to run)?")
print()
```

**ADD** x-axis formatting to show full numbers (not scientific notation):
```python
# After ax.set_ylim(bottom=0, top=1.0)
# Set x-axis to show full numbers without scientific notation
from matplotlib.ticker import ScalarFormatter, MaxNLocator
if df["Time_s"].max() / df["Time_s"].min() > 10:
    ax.set_xscale("log")
    ax.set_xlabel("Execution Time (seconds, log scale)", fontweight="bold")
# Format x-axis to show full numbers
ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=False))
formatter = ScalarFormatter(useOffset=False)
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)
```

### Cell 24 (code) - Visualization 5: Performance vs Characteristics
**REMOVE** these lines:
```python
print("\n" + "=" * 80)
print("GOAL 2: CHARACTERISTIC ANALYSIS")
print("=" * 80)
print("\nWhat Makes Models Accurate?")
print("-" * 80)
print("Question: How does parameter count impact performance?")
print()
```

**KEEP** only:
```python
print("Generating Visualization 5a: Performance vs Model Parameters...")
# ... code ...
print("Generating Visualization 5b: Performance vs Model Size...")
# ... code ...
print(f"  ✓ Saved: {output_path.name}")
```

### Cell 26 (code) - Visualization 6: Speed vs Characteristics
**REMOVE** these lines:
```python
print("\nWhat Makes Models Fast?")
print("-" * 80)
print("Question: How do size and parameter count individually impact speed?")
print()
```

**KEEP** only:
```python
print("Generating Visualization 6a: Speed vs Model Parameters...")
# ... code ...
print("Generating Visualization 6b: Speed vs Model Size...")
# ... code ...
print(f"  ✓ Saved: {output_path.name}")
```

**ADD** y-axis formatting (already in code, verify it's there):
```python
# Set y-axis to show whole numbers without scientific notation
from matplotlib.ticker import ScalarFormatter, MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=False))
formatter = ScalarFormatter(useOffset=False)
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)
ax.set_ylim(bottom=0)
```

## Implementation Strategy

**CRITICAL**: Make changes ONE CELL AT A TIME and validate after each change:

1. Work on ONE cell
2. Use NotebookEdit to update that cell
3. Validate JSON: `python3 -m json.tool notebook.ipynb > /dev/null`
4. Validate syntax: `uv run python -c "import nbformat; nb = nbformat.read(...); compile cells"`
5. Move to next cell

**DO NOT** try to edit multiple cells in parallel or use automation scripts.

## Validation Checklist

After ALL changes:
- [ ] JSON structure valid
- [ ] All Python cells compile
- [ ] Run notebook: `Kernel > Restart & Run All`
- [ ] All visualizations generated in `gfx/`
- [ ] No functional changes to analysis
- [ ] Narrative flow clear when reading markdown only

## Notes

- The notebook currently has 28 cells
- After restructuring, it will have ~35-36 cells (due to cell splits)
- All changes are cosmetic - no functional changes to analysis logic
- Original has checkpoint at `.ipynb_checkpoints/shellscript_analyzer-checkpoint.ipynb`
