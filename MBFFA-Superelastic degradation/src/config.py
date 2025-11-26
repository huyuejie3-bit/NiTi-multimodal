"""
Configuration file, defines model parameters and training configuration
"""

# Data configuration
DATA_CONFIG = {
    'num_images': 3,         # Number of input images
    'num_parameters': 3,     # Number of input numerical parameters
    'num_outputs': 1,        # Number of predicted outputs
    'image_size': 336,       # Input image size
    'normalize': True,       # Whether to normalize the images
}

# Model configuration
MODEL_CONFIG = {
    'backbone': 'resnet18',  # CNN backbone network to use, options: 'resnet18', 'resnet34', 'resnet50'
    'pretrained': True,      # Whether to use pretrained weights
    'feature_dim': 512,      # Feature dimension
    'dropout': 0.5,          # Dropout ratio
    'use_attention': True,   # Whether to use attention mechanism, can use Transformer
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 1,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'lr_scheduler': 'cosine', # Learning rate scheduler, options: 'step', 'cosine', 'plateau'
    'early_stopping': 10,     # Early stopping patience
    'save_dir': './checkpoints',
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'save_dir': './results',
    'plot_loss': True,
    'plot_accuracy': True,
    'plot_confusion_matrix': True,
    'plot_heatmap': True,    # Whether to plot material heatmaps
} 