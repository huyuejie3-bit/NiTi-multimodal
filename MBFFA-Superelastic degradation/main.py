"""
Superelastic Degradation Prediction Model Main Program
"""
# 在 model = MaterialCNN(config) 之后添加：
# 仅关闭严格检查（可能仍有部分警告）

import os
import argparse
import torch
import numpy as np
import random
from datetime import datetime
import glob
import shutil

from src.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, VISUALIZATION_CONFIG
from src.data.dataset import MaterialDataset, get_data_loaders
from src.models.cnn_model import MaterialCNN
from src.utils.trainer import Trainer
from src.visualization.visualizer import MaterialVisualizer

def set_seed(seed):
    """
    Set random seed to ensure reproducibility of results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Material Performance Prediction Model Training and Testing')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'predict'],
                        help='Running mode: train, test, or predict')
    
    # Data related
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory path')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of input images, defaults to value in config file')
    parser.add_argument('--num_params', type=int, default=None,
                        help='Number of input numerical parameters, defaults to value in config file')
    parser.add_argument('--num_outputs', type=int, default=None,
                        help='Number of output performance metrics, defaults to value in config file')
    
    # Model related
    parser.add_argument('--backbone', type=str, default=None,
                        help='CNN backbone network, defaults to value in config file')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained weights')
    parser.add_argument('--no_attention', action='store_true',
                        help='Do not use attention mechanism')
    
    # Training related
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size, defaults to value in config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs, defaults to value in config file')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate, defaults to value in config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu), defaults to auto-select')
    
    # Path related
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to load model checkpoint')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save directory, defaults to value in config file')
    
    return parser.parse_args()

def update_config(args):
    """
    Update configuration using command line arguments
    """
    config = {
        'data_config': DATA_CONFIG.copy(),
        'model_config': MODEL_CONFIG.copy(),
        'train_config': TRAIN_CONFIG.copy(),
        'visualization_config': VISUALIZATION_CONFIG.copy()
    }
    
    # Update data configuration
    if args.num_images is not None:
        config['data_config']['num_images'] = args.num_images
    if args.num_params is not None:
        config['data_config']['num_parameters'] = args.num_params
    if args.num_outputs is not None:
        config['data_config']['num_outputs'] = args.num_outputs

    # Update model configuration
    if args.backbone is not None:
        config['model_config']['backbone'] = args.backbone
    if args.no_pretrained:
        config['model_config']['pretrained'] = False
    if args.no_attention:
        config['model_config']['use_attention'] = False
    
    # Update training configuration
    if args.batch_size is not None:
        config['train_config']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['train_config']['num_epochs'] = args.epochs
    if args.lr is not None:
        config['train_config']['learning_rate'] = args.lr
    if args.save_dir is not None:
        config['train_config']['save_dir'] = args.save_dir
    
    # Generate default save directory (if not specified)
    if config['train_config']['save_dir'] == './checkpoints':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config['train_config']['save_dir'] = f'./checkpoints/{timestamp}'
    
    # Ensure visualization directory matches training directory
    config['visualization_config']['save_dir'] = os.path.join(
        config['train_config']['save_dir'], 'visualizations'
    )
    
    return config

def load_data(config):
    """
    Load data
    """
    # Load image paths
    num_images = config['data_config']['num_images']
    image_paths = []
    
    # Get number of all samples
    sample_pattern = os.path.join('data', 'sample_*_0.jpg')
    total_samples = len(glob.glob(sample_pattern))
    
    # Collect all images for each sample
    for i in range(total_samples):
        sample_images = []
        for j in range(num_images):
            img_path = os.path.join('data', f'sample_{i}_{j}.jpg')
            sample_images.append(img_path)
        image_paths.append(sample_images)
    
    # Load numerical parameters and target values
    params = np.load('data/parameters.npy')
    targets = np.load('data/targets.npy')
    
    return image_paths, params, targets

def main():
    """
    Main function
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Update configuration
    config = update_config(args)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create necessary directories
    os.makedirs(config['train_config']['save_dir'], exist_ok=True)
    os.makedirs(config['visualization_config']['save_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    image_paths, params, targets = load_data(config)
    print(f"Data loading complete, total {len(image_paths)} samples")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        image_paths, params, targets, config['train_config']
    )
    print(f"Data loaders created, training set size: {len(train_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = MaterialCNN(config)
    # model = torch.jit.script(model)
    model.to(device)
    # print(f"Model creation complete, using backbone: {config['model_config']['backbone']}")
    print(f"Model creation complete, using custom CNN backbone")
    
    # Load checkpoint (if specified)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Execute operations based on mode
    if args.mode == 'train':
        print("Starting model training...")
        trainer.train(train_loader, val_loader)
        
        print("Training complete, starting testing...")
        metrics, predictions, targets = trainer.test(test_loader)
        
        # Visualize results
        print("Generating visualization results...")
        visualizer = MaterialVisualizer(model, config, device)
        visualizer.visualize_predictions(predictions, targets)
        
        # Generate feature importance visualization
        visualizer.visualize_feature_importance(test_loader)
        
        # Generate SHAP values visualization
        print("Generating SHAP explanatory analysis...")
        try:
            # Use a smaller number of samples to reduce memory usage
            visualizer.visualize_shap_values(test_loader, num_samples=3, save_name="shap_values_params.png")
        except Exception as e:
            print(f"SHAP analysis generation failed, error: {str(e)}")
        
        # Generate heatmap visualization
        print("Generating convolution layer heatmaps...")
        # Get a test sample
        test_images, test_params, _ = next(iter(test_loader))
        # Generate heatmap for each image
        for i in range(min(3, test_images.size(1))):  # Show heatmaps for at most 3 images
            image = test_images[0, i].to(device)  # Take the first sample's i-th image
            # Ensure image has 3 channels
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            # Generate model output
            # model_output = model(
            #     image.unsqueeze(0).unsqueeze(0),
            #     test_params[0:1].to(device)
            # )
            visualizer.visualize_heatmap(
                image, 
                test_params[0:1].to(device),
                save_name=f'heatmap_{i+1}.png'
            )
        
        # Copy visualization results to tensorboard_logs for easy viewing
        print("Copying visualization results to tensorboard_logs...")
        # Use a safer path for visualization images
        tensorboard_image_dir = os.path.join('tensorboard_logs', 'images')
        os.makedirs(tensorboard_image_dir, exist_ok=True)
        
        # Get visualization directory
        vis_dir = config['visualization_config']['save_dir']
        
        # Copy all visualization files
        for file_name in os.listdir(vis_dir):
            if file_name.endswith('.png'):
                src_file = os.path.join(vis_dir, file_name)
                dst_file = os.path.join(tensorboard_image_dir, file_name)
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"  Copied {file_name} to {tensorboard_image_dir}")
                except Exception as e:
                    print(f"  Warning: Failed to copy {file_name}: {str(e)}")
        
        # Also copy training history plot
        history_plot = os.path.join(config['train_config']['save_dir'], 'training_history.png')
        if os.path.exists(history_plot):
            dst_file = os.path.join(tensorboard_image_dir, 'training_history.png')
            try:
                shutil.copy2(history_plot, dst_file)
                print(f"  Copied training_history.png to {tensorboard_image_dir}")
            except Exception as e:
                print(f"  Warning: Failed to copy training_history.png: {str(e)}")
        
    elif args.mode == 'test':
        print("Testing model...")
        metrics, predictions, targets = trainer.test(test_loader)
        
        # Visualize results
        print("Generating visualization results...")
        visualizer = MaterialVisualizer(model, config, device)
        visualizer.visualize_predictions(predictions, targets)
        
        # Generate SHAP values visualization
        print("Generating SHAP explanatory analysis...")
        try:
            # Use a smaller number of samples to reduce memory usage
            visualizer.visualize_shap_values(test_loader, num_samples=3, save_name="shap_values_params.png")
        except Exception as e:
            print(f"SHAP analysis generation failed, error: {str(e)}")
        
        # Generate heatmap visualization
        print("Generating convolution layer heatmaps...")
        test_images, test_params, _ = next(iter(test_loader))
        for i in range(min(3, test_images.size(1))):
            image = test_images[0, i].to(device)
            # Ensure image has 3 channels
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            # # Generate model output
            # model_output = model(
            #     image.unsqueeze(0).unsqueeze(0),
            #     test_params[0:1].to(device)
            # )
            visualizer.visualize_heatmap(
                image,
                test_params[0:1].to(device),
                save_name=f'heatmap_{i+1}.png'
            )
    
    elif args.mode == 'predict':
        print("Performing prediction...")
        model.eval()
        
        # Example: Predict for the first few samples of the test set
        with torch.no_grad():
            for batch in test_loader:
                images, params, targets = batch
                images = images.to(device)
                params = params.to(device)
                
                # Forward propagation
                outputs = model(images, params)
                
                # Output prediction results
                for i in range(min(5, len(outputs))):
                    print(f"Sample {i+1}:")
                    print(f"  Predicted value: {outputs[i].cpu().numpy()}")
                    print(f"  Actual value: {targets[i].numpy()}")
                    print()
                
                break  # Process only one batch
    
    print("Program execution complete")

if __name__ == "__main__":
    main() 