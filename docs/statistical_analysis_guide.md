# Statistical Analysis of Model Performance

This document explains how to perform statistical analysis on model performance results.

## Overview

The statistical analysis functionality provides a rigorous evaluation of model performance differences, allowing you to determine if observed differences between models are statistically significant or due to random variation.

## Running Statistical Analysis

After performing cross-validation, you can run the statistical analysis with:

```bash
python main.py --statistical-analysis --output-dir results
```

This will:
1. Load cross-validation results from the specified results directory
2. Perform paired t-tests between model performances
3. Calculate bootstrap confidence intervals for performance metrics
4. Generate visualizations comparing model performances with confidence intervals
5. Create tables showing significance test results

## Output Files

The analysis produces the following outputs in the `results/statistical_analysis` directory:

1. **model_comparison_with_ci.png**: Bar chart showing model performance with 95% confidence intervals
2. **significance_test_results_accuracy.csv**: Table of t-test results for accuracy metrics
3. **significance_test_results_f1.csv**: Table of t-test results for F1-score metrics

## Interpreting Results

### Significance Tests

The significance test tables include:
- **Model Comparison**: The pair of models being compared
- **t-statistic**: The t-test statistic (larger absolute values indicate stronger evidence)
- **p-value**: The probability of observing the difference by chance
- **Significant**: Whether the difference is statistically significant (p < 0.05)
- **Better Model**: Which model performs better (if difference is significant)

### Confidence Intervals

The model comparison visualization shows:
- Mean performance for each model
- 95% confidence intervals showing the range of plausible values
- Non-overlapping confidence intervals suggest significant differences

## Prerequisites

Before running statistical analysis, you must perform cross-validation using:

```bash
python main.py --cross-validate --n-folds 5 --output-dir results
```

This generates the required `cv_results.pkl` file that the statistical analysis uses as input.

## Implementation Details

The statistical analysis uses:
- **Paired t-tests**: To account for fold-to-fold correlation in performance
- **Bootstrap resampling**: For robust estimation of confidence intervals
- **Significance level**: Default α = 0.05 (95% confidence) 