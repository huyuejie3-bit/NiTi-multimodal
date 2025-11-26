"""
Dataset module, used for loading and preprocessing material images and numerical parameters
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MaterialDataset(Dataset):
    """
    Superelastic Degradation Prediction Dataset Class
    Supports multiple images and numerical parameters as input
    """
    def __init__(self, image_paths, numerical_params, targets=None, config=None, transform=None, is_train=True):
        """
        Initialize dataset
        
        Parameters:
            image_paths: List of image paths, each element can be a single path or a list of paths
            numerical_params: List of numerical parameters
            targets: Target performance values
            config: Configuration parameters
            transform: Image transformation
            is_train: Whether in training mode
        """
        self.image_paths = image_paths
        self.numerical_params = numerical_params
        self.targets = targets
        self.is_train = is_train
        self.config = config
        
        # Setup default transformation
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((336, 336)),
                transforms.Grayscale(num_output_channels=3),  # Ensure 3 output channels
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Process image input (supports multiple images)
        if isinstance(self.image_paths[idx], list):
            # Multiple images
            images = []
            for img_path in self.image_paths[idx]:
                img = Image.open(img_path).convert('RGB')  # Ensure loading as RGB
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            # Stack multiple images
            image_tensor = torch.stack(images, dim=0)
        else:
            # Single image
            img = Image.open(self.image_paths[idx]).convert('RGB')  # Ensure loading as RGB
            if self.transform:
                image_tensor = self.transform(img).unsqueeze(0)
        
        # Process numerical parameters
        numerical_tensor = torch.tensor(self.numerical_params[idx], dtype=torch.float32)
        
        # In training mode, return target values; in test mode, don't return
        if self.is_train and self.targets is not None:
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image_tensor, numerical_tensor, target_tensor
        else:
            return image_tensor, numerical_tensor

def get_data_loaders(image_paths, numerical_params, targets, config, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Create training, validation, and test data loaders
    """
    # Dataset size
    dataset_size = len(image_paths)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    # Calculate split points
    train_end = int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create datasets
    train_dataset = MaterialDataset(
        [image_paths[i] for i in train_indices],
        [numerical_params[i] for i in train_indices],
        [targets[i] for i in train_indices],
        config, is_train=True
    )
    
    val_dataset = MaterialDataset(
        [image_paths[i] for i in val_indices],
        [numerical_params[i] for i in val_indices],
        [targets[i] for i in val_indices],
        config, is_train=True
    )
    
    test_dataset = MaterialDataset(
        [image_paths[i] for i in test_indices],
        [numerical_params[i] for i in test_indices],
        [targets[i] for i in test_indices],
        config, is_train=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    return train_loader, val_loader, test_loader 