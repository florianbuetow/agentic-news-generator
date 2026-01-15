# Model Performance Visualizations

This directory contains performance analysis visualizations for shell script classification models.

**Location:** `notebooks/gfx/`
**Filename prefix:** `shell_script_classification_*`

All visualizations use a consistent, modern color palette inspired by contemporary design systems for better visual coherence and accessibility.

## Analysis Goals

This analysis answers three specific questions in order:

### Goal 1: Pattern Recognition in Tests
**Question**: How do different models compare on individual tests?
**Need**: Per-test comparison to spot patterns across models

### Goal 2: Characteristic Analysis
**Questions**:
1. How does parameter count impact performance?
2. How do size and parameter count individually impact speed?

### Goal 3: Model Selection
**Question**: Which model is best (most accurate, fastest to run)?
**Need**: Speed vs accuracy trade-off visualization

## Generated Visualizations

### Metrics Summary (Console Output)
Comprehensive table printed to console showing all performance metrics for each model:
- Accuracy, Precision, Recall, F1 Score
- TP, TN, FP, FN counts
- Total execution time (in seconds)
- Model parameters (billions), size (GB), quantization
- Ranked by F1 Score (best to worst)

---

## Goal 1: Test Pattern Analysis

### Viz 03: Misclassification Analysis (`shell_script_classification_03_misclassification_by_test.jpg`)
Vertical stacked plots showing misclassification percentages:
- **Top**: False Positives (bad scripts predicted as PASS) in light red
- **Bottom**: False Negatives (good scripts predicted as FAIL) in light blue
- Top 5 most misclassified tests are labeled
- Y-axis shows percentage of models (0-100%)

**Answers**: Which tests are hardest? What mistakes do models make most often?

### Viz 04a: Heatmap - FAIL Tests (`shell_script_classification_04a_heatmap_fail_tests.jpg`)
Heatmap showing model performance on FAIL tests (bad_*.sh):
- X-axis: Test IDs (0-49)
- Y-axis: Model names
- Green = Correct (predicted FAIL), Red = Incorrect (predicted PASS)
- Square cells for optimal readability

**Answers**: Do all models struggle with the same bad scripts?

### Viz 04b: Heatmap - PASS Tests (`shell_script_classification_04b_heatmap_pass_tests.jpg`)
Heatmap showing model performance on PASS tests (good_*.sh):
- X-axis: Test IDs (50-99)
- Y-axis: Model names
- Green = Correct (predicted PASS), Red = Incorrect (predicted FAIL)
- Square cells for optimal readability

**Answers**: Which good scripts are consistently classified correctly?

---

## Goal 2: Characteristic Analysis

### Part 1: What Makes Models Accurate?

#### Viz 10a: Performance vs Model Parameters (`shell_script_classification_10a_performance_vs_params.jpg`)
Scatter plot showing relationship between parameters and performance:
- **X-axis**: Model Parameters (billions)
- **Y-axis**: F1 Score
- **Point color**: Quantization method (4bit=blue, 6bit=purple, 8bit=pink)
- **Annotations**: Model names

**Answers**: Do more parameters = better accuracy?

#### Viz 10b: Performance vs Model Size (`shell_script_classification_10b_performance_vs_size.jpg`)
Scatter plot showing relationship between disk size and performance:
- **X-axis**: Model Size (GB)
- **Y-axis**: F1 Score
- **Point color**: Quantization method
- **Annotations**: Model names

**Answers**: Does disk size correlate with accuracy?

### Part 2: What Makes Models Fast?

#### Viz 11a: Speed vs Model Parameters (`shell_script_classification_11a_speed_vs_params.jpg`)
Scatter plot showing relationship between parameters and speed:
- **X-axis**: Model Parameters (billions)
- **Y-axis**: Execution Time (seconds, log scale if range > 10x)
- **Point color**: Quantization method
- **Annotations**: Model names

**Answers**: Do more parameters = slower execution?

#### Viz 11b: Speed vs Model Size (`shell_script_classification_11b_speed_vs_size.jpg`)
Scatter plot showing relationship between disk size and speed:
- **X-axis**: Model Size (GB)
- **Y-axis**: Execution Time (seconds, log scale if range > 10x)
- **Point color**: Quantization method
- **Annotations**: Model names

**Answers**: Does disk size impact execution speed?

---

## Goal 3: Best Model Selection

### Viz 08: Efficiency Scatter - Speed vs Accuracy (`shell_script_classification_08_efficiency_scatter.jpg`)
Scatter plot showing trade-off between execution speed and accuracy:
- **X-axis**: Execution time in seconds (log scale if range > 10x)
- **Y-axis**: F1 Score (0-1)
- **Pareto front** highlighted with star markers and dashed line
  - Pareto-optimal models: No other model is both faster AND more accurate
  - These represent the "efficient frontier" of model choices
- **Quadrant lines**: Divide models into fast/slow × accurate/less-accurate regions
- **Color palette**: Consistent categorical colors for each model

**Answers**: Which model should I pick based on my speed/accuracy priorities?

**Key insight:** A model is on the Pareto front if any faster model has lower accuracy, and any more accurate model is slower.

---

## Error Analysis

### Viz 02: Confusion Matrices Grid (`shell_script_classification_02_confusion_matrices_grid.jpg`)
Grid of 2x2 confusion matrices for each model showing:
- True/False Positives and Negatives
- Counts and percentages
- Model name with F1 score
- Uses modern indigo color gradient for consistent branding

**Answers**: What types of errors does my chosen model make? How many false positives (dangerous) vs false negatives (annoying)?

---

## How to Regenerate

Run the analysis script:
```bash
uv run python notebooks/shell_script_classifier_llm_analysis.py
```

The script will:
1. Scan `reports/` for CSV files
2. Load model specifications from `notebooks/benchmark_models.json`
3. Calculate all metrics
4. Generate all visualizations in `notebooks/gfx/`
5. Overwrite existing JPEG files with prefix `shell_script_classification_`

## CSV Format Requirements

The script expects CSV files in the following format:
```
Script,Expected,Predicted,Match,Error
```

- **Script**: Path to test script
- **Expected**: PASS or FAIL
- **Predicted**: PASS, FAIL, or ERROR
- **Match**: Y or N
- **Error**: Error details (optional)

Old 4-column format is not supported and must be converted.

## Model Specifications Format

The script expects a JSON file at `notebooks/benchmark_models.json`:
```json
{
  "models": [
    {
      "model_name": "qwen3-30b-a3b-thinking-2507-mlx@6bit",
      "params_billions": 30.0,
      "size_gb": 24.82,
      "quantization": "6bit",
      "arch": "qwen",
      "publisher": "qwen"
    }
  ]
}
```

**Quantization formats** are normalized (Q4_K_M, Q4_K_S → 4bit; Q6_K → 6bit; Q8_0 → 8bit).

## Understanding the Metrics

### Classification Context
- **PASS** = Positive class (no violations, clean script)
- **FAIL** = Negative class (has violations)

### Confusion Matrix Values
- **TP (True Positive)**: Expected PASS, Predicted PASS ✓
- **TN (True Negative)**: Expected FAIL, Predicted FAIL ✓
- **FP (False Positive)**: Expected FAIL, Predicted PASS ✗ (more concerning!)
- **FN (False Negative)**: Expected PASS, Predicted FAIL ✗

### Performance Metrics
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) - how many predicted PASS are actually PASS
- **Recall**: TP / (TP + FN) - how many actual PASS were predicted as PASS
- **F1 Score**: Harmonic mean of precision and recall (used for ranking)

### Pareto Front Explained

The **Pareto front** (also called Pareto frontier or efficient frontier) is a concept from multi-objective optimization that identifies the set of "best" solutions when optimizing for multiple competing objectives simultaneously.

In the context of model selection with speed vs accuracy trade-offs:
- **Pareto Optimal Model**: A model where you cannot improve one objective (speed OR accuracy) without making the other objective worse. In other words, no other model is BOTH faster AND more accurate.
- **Dominated Model**: A model where another model exists that is both faster AND more accurate. These models are strictly inferior choices.
- **Pareto Front**: The set of all Pareto-optimal models, representing the "efficient frontier" of speed/accuracy combinations. These are the only rational choices when trading off speed vs accuracy.

#### Why is this useful?

When selecting a model, you should typically only consider models on the Pareto front, because any model NOT on the front has a strictly better alternative. The Pareto front helps you:
1. Eliminate dominated models from consideration
2. Understand the true speed/accuracy trade-off curve
3. Make informed decisions about which trade-off point suits your use case

#### Example

Consider three models:
- **Model A**: 90% accuracy, 10 seconds (Pareto optimal)
- **Model B**: 95% accuracy, 20 seconds (Pareto optimal)
- **Model C**: 92% accuracy, 25 seconds (Dominated by B - slower AND less accurate)

Models A and B are on the Pareto front because:
- A is faster but less accurate than B (trade-off choice)
- B is more accurate but slower than A (trade-off choice)

Model C is dominated because Model B is both faster AND more accurate, making C a strictly inferior choice.

### Why F1 Score for Ranking?
F1 Score balances both precision and recall. While false positives (predicting PASS for bad scripts) are more concerning, F1 ensures models perform well at both:
1. Identifying clean scripts (precision)
2. Catching violations (recall)

## Notes

- All visualizations are 300 DPI high-resolution JPEGs
- Colorblind-friendly palettes used where possible
- Script is idempotent - safe to run multiple times
- Test scripts: 50 bad_*.sh (FAIL) + 50 good_*.sh (PASS) = 100 total
- **Quantization colors (consistent)**:
  - 4bit: Blue (#3b82f6)
  - 6bit: Purple (#8b5cf6)
  - 8bit: Pink (#ec4899)
