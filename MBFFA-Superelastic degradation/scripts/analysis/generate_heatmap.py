"""
Heatmap Visualization Tool for Superelastic Degradation Prediction Model
-------------------------------------------------------------------
This script generates heatmap visualizations for specific input images
to help understand which regions of the image are important for the model's prediction.

Features:
1. Select specific sample and image by index
2. Customize target layer for Grad-CAM analysis
3. Save heatmaps to specified directory
4. Support for batch processing multiple images

pycharm-terminal:
python scripts/analysis/generate_heatmap.py --checkpoint checkpoints/20250330_130512/best_model.pth --sample 0 --image 1

Author: Research Team
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import logging
import glob

# Add project root to path for proper imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.cnn_model import MaterialCNN
from src.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, VISUALIZATION_CONFIG
from src.data.dataset import MaterialDataset, get_data_loaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('analysis', 'visualization', 'heatmap.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Heatmap_Visualization")

def load_data(config):
    """
    Load image paths, parameters, and targets from the data directory.
    
    Args:
        config (dict): Configuration dictionary with data settings
        
    Returns:
        tuple: (image_paths, parameters, targets)
    """
    # Load image paths
    num_images = config['data_config']['num_images']
    image_paths = []
    
    # Get total number of samples
    sample_pattern = os.path.join('data', 'sample_*_0.jpg')
    total_samples = len(glob.glob(sample_pattern))
    logger.info(f"Found {total_samples} total samples")
    
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

def get_activation_maps(model, image_tensor, params_tensor, target_layer_name=None):
    """
    Get activation maps from target layer using Grad-CAM
    
    Args:
        model: Model to analyze
        image_tensor: Input image tensor
        params_tensor: Input parameters tensor
        target_layer_name: Name of target layer (if None, will use last convolutional layer)
        
    Returns:
        tuple: (activation_map, grad_cam)
    """
    model.eval()
    
    # Find appropriate target layer
    target_layer = None
    if target_layer_name:
        # Try to find layer by name
        for name, module in model.named_modules():
            if target_layer_name in name:
                target_layer = module
                logger.info(f"Found specified layer: {name}")
                break
    
    if target_layer is None:
        # Default: Use last convolutional layer
        for i, layer in reversed(list(enumerate(model.features))):
            # Check if it's a convolutional layer or residual block
            if hasattr(layer, 'conv') or hasattr(layer, 'conv1') or hasattr(layer, 'convs'):
                target_layer = layer
                logger.info(f"Using layer {i} for Grad-CAM")
                break
    
    # If still not found, use the last layer of features
    if target_layer is None:
        target_layer = model.features[-1]
        logger.info("Using last layer of features for Grad-CAM")
    
    # Save features and gradients
    feature_maps = []
    gradients = []
    
    def save_feature_maps(module, input, output):
        feature_maps.append(output)
    
    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register forward and backward hooks
    handle_forward = target_layer.register_forward_hook(save_feature_maps)
    handle_backward = target_layer.register_backward_hook(save_gradients)
    
    try:
        # Forward propagation
        output = model(image_tensor, params_tensor)
        
        # For classification models, use argmax; for regression, use the output directly
        if output.shape[1] > 1:  # Classification
            output_index = torch.argmax(output)
            target = output[0, output_index]
        else:  # Regression
            target = output[0, 0]  # Use the single regression output
        
        # Backward propagation
        model.zero_grad()
        target.backward()
        
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        
        # Calculate weights
        feature_map = feature_maps[0].detach().cpu().numpy()[0]
        gradient = gradients[0].detach().cpu().numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradient, axis=(1, 2))
        
        # Generate CAM
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]
        
        # ReLU activation
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return feature_map, cam
        
    except Exception as e:
        logger.error(f"Failed to generate heatmap: {str(e)}")
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        return None, np.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=np.float32)

def apply_heatmap(img, heatmap, alpha=0.5):
    """
    Apply heatmap to original image
    
    Args:
        img: Original image (numpy array, RGB format)
        heatmap: Heatmap (numpy array)
        alpha: Transparency of heatmap
        
    Returns:
        numpy array: Image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to JET color mapping
    heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay on original image
    img_with_heatmap = alpha * heatmap_colored + (1-alpha) * img
    
    # Normalize
    img_with_heatmap = img_with_heatmap / img_with_heatmap.max()
    
    return img_with_heatmap

def visualize_single_heatmap(model, image, params, device, save_dir, target_layer_name=None, save_name='heatmap.png', alpha=0.5):
    """
    Generate and visualize heatmap for a single image
    
    Args:
        model: Model to analyze
        image: Image tensor with shape [C, H, W]
        params: Parameters tensor with shape [N]
        device: Device to run on
        save_dir: Directory to save results
        target_layer_name: Name of target layer (if None, will use last convolutional layer)
        save_name: Name of output file
        alpha: Transparency of heatmap overlay
    """
    # Ensure the image has 3 channels
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    
    # Ensure image dimension is correct [1, C, H, W]
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    # Move to device
    image = image.to(device)
    params = params.to(device)
    
    # De-normalize
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    # De-normalize and convert to numpy array
    img_np = inv_normalize(image[0]).cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = np.clip(img_np, 0, 1)
    
    # Generate activation maps and heatmap
    _, heatmap = get_activation_maps(model, image, params, target_layer_name)
    
    # Apply heatmap to original image
    img_with_heatmap = apply_heatmap(img_np, heatmap, alpha)
    
    # Plot original image, heatmap, and overlay
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Activation Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_with_heatmap)
    plt.title('Heatmap Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    # Save figure
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap visualization to {save_path}")
    
    # Also copy to research/figures for paper use
    research_dir = os.path.join('research', 'figures')
    os.makedirs(research_dir, exist_ok=True)
    try:
        import shutil
        shutil.copy2(save_path, os.path.join(research_dir, save_name))
        logger.info(f"Copied heatmap to {os.path.join(research_dir, save_name)}")
    except Exception as e:
        logger.warning(f"Failed to copy heatmap to research figures: {str(e)}")

def generate_heatmaps(checkpoint_path, sample_index=0, image_index=None, target_layer=None, alpha=0.5, output_dir=None):
    """
    Generate heatmap visualizations for selected images
    
    Args:
        checkpoint_path: Path to model checkpoint
        sample_index: Index of sample to analyze
        image_index: Index of image within sample to analyze (None for all images)
        target_layer: Name of target layer for Grad-CAM
        alpha: Transparency of heatmap overlay
        output_dir: Directory to save results (default: analysis/visualization)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create config
    config = {
        'data_config': DATA_CONFIG,
        'model_config': MODEL_CONFIG,
        'train_config': TRAIN_CONFIG,
        'visualization_config': VISUALIZATION_CONFIG
    }
    
    # Create model
    model = MaterialCNN(config)
    model.to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Set save directory
    if output_dir:
        save_dir = output_dir
    else:
        save_dir = os.path.join('analysis', 'visualization')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    image_paths, params, targets = load_data(config)
    
    # Validate sample index
    if sample_index < 0 or sample_index >= len(image_paths):
        logger.error(f"Invalid sample index {sample_index}. Available samples: 0-{len(image_paths)-1}")
        return
    
    # Get sample data
    sample_image_paths = image_paths[sample_index]
    sample_params = params[sample_index]
    sample_target = targets[sample_index]
    
    logger.info(f"Analyzing sample {sample_index} with target value: {sample_target}")
    
    # Convert params to tensor
    params_tensor = torch.tensor(sample_params, dtype=torch.float32).unsqueeze(0)
    
    # Determine which images to process
    if image_index is not None:
        # Validate image index
        if image_index < 0 or image_index >= len(sample_image_paths):
            logger.error(f"Invalid image index {image_index}. Available images: 0-{len(sample_image_paths)-1}")
            return
        
        # Process single specified image
        image_indices = [image_index]
    else:
        # Process all images in sample
        image_indices = range(len(sample_image_paths))
    
    # Process each image
    for idx in image_indices:
        logger.info(f"Processing image {idx} from sample {sample_index}")
        
        # Load image
        image_path = sample_image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image from {image_path}")
            continue
        
        # Convert to RGB and preprocess
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image)
        
        # Generate heatmap
        save_name = f"heatmap_sample{sample_index}_image{idx}.png"
        visualize_single_heatmap(
            model, 
            image_tensor, 
            params_tensor, 
            device, 
            save_dir, 
            target_layer, 
            save_name,
            alpha
        )
    
    logger.info(f"Heatmap generation complete. Results saved to {save_dir}")
    
    # Also copy to tensorboard_logs for visualization
    tensorboard_image_dir = os.path.join('tensorboard_logs', 'images')
    os.makedirs(tensorboard_image_dir, exist_ok=True)
    
    try:
        import shutil
        for file_name in os.listdir(save_dir):
            if file_name.startswith("heatmap_") and file_name.endswith(".png"):
                src_file = os.path.join(save_dir, file_name)
                dst_file = os.path.join(tensorboard_image_dir, file_name)
                try:
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"Copied {file_name} to {tensorboard_image_dir}")
                except Exception as e:
                    logger.warning(f"Failed to copy {file_name} to tensorboard: {str(e)}")
    except Exception as e:
        logger.error(f"Error copying files to tensorboard: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Heatmap Visualization Tool for Material Performance Prediction')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--sample', type=int, default=0,
                        help='Index of sample to analyze')
    parser.add_argument('--image', type=int, default=None,
                        help='Index of image within sample to analyze (default: analyze all images)')
    parser.add_argument('--layer', type=str, default=None,
                        help='Name of target layer for Grad-CAM (default: last convolutional layer)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency of heatmap overlay (0-1)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: analysis/visualization)')
    
    args = parser.parse_args()
    generate_heatmaps(
        args.checkpoint, 
        args.sample, 
        args.image, 
        args.layer, 
        args.alpha, 
        args.output_dir
    ) 