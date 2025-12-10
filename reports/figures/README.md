# Figures Directory

This directory contains visualization outputs from supervised learning experiments.

## Expected Figures

### MAR Classification Figures
- `mar_model_comparison.png` - Bar chart comparing all models
- `mar_roc_curves.png` - ROC curves for all models
- `mar_confusion_matrix_*.png` - Confusion matrix for each model
- `mar_feature_importance.png` - Feature importance visualization

### Species Classification Figures
- `species_model_comparison.png` - Bar chart comparing all models
- `species_confusion_matrix_*.png` - Confusion matrix for each model
- `species_per_class_metrics.png` - Per-class performance metrics

## Figure Specifications

All figures are saved with:
- DPI: 300 (high resolution)
- Format: PNG
- Bbox: tight (no unnecessary whitespace)

## Usage

Figures are generated automatically when running:
```bash
jupyter notebook notebooks/03_supervised_learning.ipynb
jupyter notebook notebooks/04_model_comparison.ipynb
```
