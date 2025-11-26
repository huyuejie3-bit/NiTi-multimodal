# Superelastic Degradation in Niti Alloys Prediction- Multimodal Fusion Based on Feature Decoupling and Integration

## Project Overview

This project implements a DL-based material property prediction model, which combines image features and numerical parameters through a multimodal fusion method. 
The model extracts matrix features by separating channels and performs  fusion through a feature synergy module to achieve high-precision prediction of the superelastic degradation of shape memory alloys.

## Project Structure

The project is organized using a rigorous research-grade directory structure:

```
project/
├── analysis/              # Analysis results and code
│   ├── shap/              # SHAP explainability analysis results
│   └── visualization/     # Other analysis visualizations
├── checkpoints/           # Saved model checkpoints
│   └── */                 # Checkpoints from each training run
├── data/                  # Raw and processed data
├── research/              # Research-related files
│   ├── figures/           # Publication-quality figures and visualizations
│   └── papers/            # Related papers and references
├── scripts/               # Standalone scripts
│   ├── analysis/          # Analysis scripts
│   ├── evaluation/        # Evaluation scripts
│   └── utils/             # Utility scripts
├── src/                   # Source code
│   ├── data/              # Data processing code
│   ├── models/            # Model definitions
│   ├── utils/             # Utility functions
│   └── visualization/     # Visualization code
└── tensorboard_logs/      # TensorBoard logs
```

## Core Functionality

1.Multimodal data fusion: Integrates image features and numerical parameters for prediction.
2.Deep learning model: Extracts matrix features in a channel-wise manner and then fuses them with numerical features.
3.Model interpretability: Uses SHAP analysis to explain the model’s decision process.
4.Visualization analysis: Provides rich visualization tools to analyze model performance and prediction results.

## Instructions for Use

### Environment Configuration

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python main.py --mode train --epochs 150 --batch_size 32
```

### Test

```bash
python main.py --mode test --checkpoint ./checkpoints/best_model.pth
```

### SHAP Analysis

```bash
python scripts/analysis/analyze_shap.py --checkpoint ./checkpoints/best_model.pth --num_samples 980
```

### Custom SHAP Visualization

```bash
python scripts/analysis/create_custom_shap_plot.py
```

## TensorBoard Visualization

Launch TensorBoard to view the training process and analysis results:

```bash
python -m tensorboard.main --logdir=tensorboard_logs/runs --port=6006
```

## reference

- SHAP: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- Transformer Model: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## team

Research Team on Predicting Superelastic Degradation in Niti Alloys
- If memory issues arise, you can reduce image resolution or decrease model depth. 
