"""
SHAP Analysis Script for Superelastic Degradation Prediction Model
----------------------------------------------------------------
This script conducts SHAP (SHapley Additive exPlanations) analysis on a trained
superelastic degradation prediction model to evaluate feature importance and dependencies.

The analysis includes:
1. Feature extraction from the model
2. SHAP value calculation using KernelExplainer
3. Visualization of feature importance (summary plot)
4. Comparison of image features vs numerical parameters importance
5. Dependence plots for the most important features

The script ensures all 12 features (9 image features and 3 numerical parameters)
are properly analyzed and visualized.

Author: Research Team
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import argparse
import glob
import logging
from torch.utils.data import DataLoader
import sys

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
        logging.FileHandler(os.path.join('analysis', 'shap', 'shap_analysis.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SHAP_Analysis")

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

def extract_features(model, data_loader, device, max_samples=2000, batch_size=4):
    """
    Extract features from the model for SHAP analysis.
    
    Args:
        model (torch.nn.Module): Trained model
        data_loader (DataLoader): Data loader for feature extraction
        device (torch.device): Device to run extraction on
        max_samples (int): Maximum number of samples to extract
        batch_size (int): Batch size for extraction
        
    Returns:
        tuple: (all_features, all_targets, feature_names)
    """
    logger.info(f"Extracting features (max {max_samples} samples)...")
    all_features = []
    all_targets = []
    current_samples = 0
    
    with torch.no_grad():
        for images, params, targets in data_loader:
            if current_samples >= max_samples:
                break
                
            images = images.to(device)
            params = params.to(device)
            
            # Process only a subset of each batch to manage memory
            samples_from_batch = min(batch_size, max_samples - current_samples)
            
            for j in range(samples_from_batch):
                # Extract individual sample
                img = images[j:j+1]
                param = params[j:j+1]
                target = targets[j:j+1]
                
                # Extract image features
                image_features = []
                for k in range(img.size(1)):  # Iterate over each image
                    single_img = img[:, k]
                    x = model.features(single_img)
                    x = x.view(1, -1)  # Flatten
                    x = model.feature_reducer(x)
                    image_features.append(x)
                
                # Concatenate all image features
                image_features = torch.cat(image_features, dim=1)
                
                # Concatenate image features and parameters
                combined_features = torch.cat([image_features, param], dim=1)
                
                all_features.append(combined_features.cpu().numpy())
                all_targets.append(target.numpy())
                current_samples += 1
                
                # Output feature dimensions for the first sample
                if j == 0 and current_samples == 1:
                    logger.info(f"Image features shape: {image_features.shape}")
                    logger.info(f"Parameters shape: {param.shape}")
                    logger.info(f"Combined features shape: {combined_features.shape}")
                
                if current_samples >= max_samples:
                    break
    
    # Stack features and targets
    all_features = np.vstack(all_features)
    all_targets = np.vstack(all_targets)
    
    logger.info(f"Collected {len(all_features)} feature samples with dimensions {all_features.shape}")
    
    # Create feature names
    num_images = model.config['data_config']['num_images']
    num_params = model.config['data_config']['num_parameters']
    
    feature_names = []
    # Add names for each image feature
    for i in range(num_images):
        for j in range(3):  # Each image has 3 extracted features
            feature_names.append(f'Img{i+1}_Feat{j+1}')
    
    # Add names for each numerical parameter
    for i in range(num_params):
        feature_names.append(f'Param{i+1}')
    
    logger.info(f"Feature names: {feature_names}")
    
    # Ensure feature names match feature dimensions
    assert len(feature_names) == all_features.shape[1], \
        f"Feature names count ({len(feature_names)}) doesn't match feature dimensions ({all_features.shape[1]})"
    
    return all_features, all_targets, feature_names

def create_model_predictor(model, device):
    """
    Create a wrapper function for model prediction during SHAP analysis.
    
    Args:
        model (torch.nn.Module): Trained model
        device (torch.device): Device to run prediction on
        
    Returns:
        function: Prediction function for SHAP explainer
    """
    def model_predict(features_data):
        # Convert to tensor if numpy array
        if isinstance(features_data, np.ndarray):
            features_data = torch.tensor(features_data, dtype=torch.float32).to(device)
        
        # Ensure batch dimension
        if len(features_data.shape) == 1:
            features_data = features_data.unsqueeze(0)
        
        outputs = []
        
        # Process each feature sample in batches to avoid memory issues
        with torch.no_grad():
            batch_size = 5  # Small batch processing
            for i in range(0, len(features_data), batch_size):
                batch = features_data[i:i+batch_size]
                
                # Process each sample
                for feature in batch:
                    # Adjust to 2D tensor
                    feature = feature.view(1, -1)
                    
                    # Pass through model's attention and subsequent layers
                    # attended_features = model.attention(feature)
                    # transformed_features = model.transformer_fusion(attended_features)
                    fused_features = model.feature_synergy(feature)
                    fused_features = model.prediction_head(fused_features)
                    output = model.output_layer(fused_features)
                    
                    outputs.append(output.cpu().numpy())
        
        # Return stacked results
        return np.vstack(outputs)
    
    return model_predict

def generate_shap_plots(shap_values, test_samples, feature_names, save_dir, img_features_count):
    """
    Generate and save SHAP analysis plots.
    
    Args:
        shap_values (numpy.ndarray): SHAP values
        test_samples (numpy.ndarray): Feature samples used for testing
        feature_names (list): Names of features
        save_dir (str): Directory to save plots
        img_features_count (int): Number of image features
    """
    # Process SHAP values based on type
    if isinstance(shap_values, list):
        if len(shap_values) > 0:
            shap_array = np.abs(shap_values[0])
        else:
            logger.warning("Empty SHAP values list")
            shap_array = np.zeros((len(test_samples), len(feature_names)))
    else:
        shap_array = np.abs(shap_values)
    
    # 1. SHAP Summary Plot
    logger.info("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    try:
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap.summary_plot(shap_values[0], test_samples, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, test_samples, feature_names=feature_names, show=False)
        
        plt.title('Multimodal Feature SHAP Importance Analysis (All 12 Features)')
        plt.tight_layout()
        summary_plot_path = os.path.join(save_dir, 'shap_values_all.png')
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {summary_plot_path}")
    except Exception as e:
        logger.error(f"Error generating summary plot: {str(e)}")
    finally:
        plt.close()
    
    # 2. Modality Importance Pie Chart
    logger.info("Calculating feature type importance...")
    img_importance = np.sum(shap_array[:, :img_features_count])
    param_importance = np.sum(shap_array[:, img_features_count:])
    
    logger.info(f"Image features importance: {img_importance:.6f}, Numerical parameters importance: {param_importance:.6f}")
    
    plt.figure(figsize=(10, 8))
    plt.pie(
        [img_importance, param_importance],
        labels=['Image Features', 'Numerical Parameters'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#ff9999', '#66b3ff']
    )
    plt.title('Importance Ratio: Image Features vs Numerical Parameters')
    plt.axis('equal')
    plt.tight_layout()
    pie_chart_path = os.path.join(save_dir, 'modal_importance_pie.png')
    plt.savefig(pie_chart_path, dpi=300)
    logger.info(f"Saved modality importance pie chart to {pie_chart_path}")
    plt.close()
    
    # 3. Dependence Plot for Most Important Feature
    logger.info("Generating SHAP dependence plot for most important feature...")
    
    # Check for non-zero SHAP values
    feature_importance = np.mean(np.abs(shap_array), axis=0)
    if np.max(feature_importance) > 0:
        # Find the most important feature index
        most_important_idx = np.argmax(feature_importance)
        feature_name = feature_names[most_important_idx]
        
        logger.info(f"Most important feature: {feature_name}, index: {most_important_idx}")
        
        try:
            plt.figure(figsize=(10, 6))
            
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap.dependence_plot(most_important_idx, shap_values[0], test_samples, 
                                     feature_names=feature_names, show=False)
            else:
                shap.dependence_plot(most_important_idx, shap_values, test_samples, 
                                     feature_names=feature_names, show=False)
            
            plt.title(f'SHAP Dependence Plot for {feature_name}')
            plt.tight_layout()
            dependence_plot_path = os.path.join(save_dir, f'shap_dependence_{feature_name}.png')
            plt.savefig(dependence_plot_path, dpi=300)
            logger.info(f"Saved dependence plot to {dependence_plot_path}")
        except Exception as e:
            logger.error(f"Error generating dependence plot: {str(e)}")
        finally:
            plt.close()
    else:
        logger.warning("No significant feature importance found, skipping dependence plot")

def analyze_shap(checkpoint_path, num_samples=980, batch_size=4):
    """
    Conduct SHAP analysis on the model, ensuring all 12 features are displayed.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        num_samples (int): Number of samples for analysis
        batch_size (int): Batch size for data loading
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
    model.config = config  # Store config in model for later use
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    logger.info("Loading data...")
    image_paths, params, targets = load_data(config)
    logger.info(f"Data loading complete, total {len(image_paths)} samples")
    
    # Create data loaders
    train_config = config['train_config'].copy()
    train_config['batch_size'] = batch_size
    _, _, test_loader = get_data_loaders(
        image_paths, params, targets, train_config
    )
    
    # Create save directories
    save_dir = os.path.join('analysis', 'shap')
    os.makedirs(save_dir, exist_ok=True)
    research_figures_dir = os.path.join('research', 'figures')
    os.makedirs(research_figures_dir, exist_ok=True)
    
    # Extract features
    all_features, all_targets, feature_names = extract_features(
        model, test_loader, device, max_samples=num_samples, batch_size=batch_size
    )
    
    # Create model predictor
    model_predict = create_model_predictor(model, device)
    
    # Compute SHAP values
    logger.info("Running SHAP analysis with KernelExplainer...")
    
    # Use fewer background samples to reduce computational burden
    n_background = min(100, len(all_features))
    background_data = all_features[:n_background]
    
    # Limit explanation samples
    n_explain = min(num_samples, len(all_features))
    test_samples = all_features[:n_explain]
    
    # Compute SHAP values with reduced nsamples parameter
    logger.info("Computing SHAP values...")
    explainer = shap.KernelExplainer(model_predict, background_data, link="identity")
    shap_values = explainer.shap_values(test_samples, nsamples=50)
    
    # Generate and save plots
    generate_shap_plots(
        shap_values, 
        test_samples, 
        feature_names, 
        save_dir, 
        config['data_config']['num_images'] * 3  # Number of image features
    )
    
    logger.info(f"SHAP analysis complete, results saved in {save_dir} directory")
    
    # Copy results to research/figures for paper
    try:
        import shutil
        for file_name in os.listdir(save_dir):
            if file_name.endswith('.png'):
                src_file = os.path.join(save_dir, file_name)
                dst_file = os.path.join(research_figures_dir, file_name)
                try:
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"Copied {file_name} to {research_figures_dir}")
                except Exception as e:
                    logger.warning(f"Failed to copy {file_name} to research figures: {str(e)}")
    except Exception as e:
        logger.error(f"Error copying files to research figures: {str(e)}")
    
    # Also copy to tensorboard_logs for visualization
    tensorboard_image_dir = os.path.join('tensorboard_logs', 'images')
    os.makedirs(tensorboard_image_dir, exist_ok=True)
    
    try:
        import shutil
        for file_name in os.listdir(save_dir):
            if file_name.endswith('.png'):
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
    parser = argparse.ArgumentParser(description='SHAP Analysis Tool for Material Performance Prediction')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/enhanced_shap/best_model.pth',
                        help='Model checkpoint path')
    parser.add_argument('--num_samples', type=int, default=980,
                        help='Number of samples for analysis')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for data loading')
    
    args = parser.parse_args()
    analyze_shap(args.checkpoint, args.num_samples, args.batch_size) 