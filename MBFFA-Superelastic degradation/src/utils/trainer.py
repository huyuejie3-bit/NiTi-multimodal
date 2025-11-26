"""
Training module, used for training, validation and testing models
"""

import os
import time
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from torch.utils.tensorboard import SummaryWriter
import torchvision
import tempfile

def calculate_metrics(predictions, targets):
    """
    Calculate various evaluation metrics
    
    Parameters:
        predictions: Predicted values
        targets: True values
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {}
    
    # 1. Regression metrics
    # Root Mean Squared Error (RMSE)
    metrics['rmse'] = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Mean Absolute Error (MAE)
    metrics['mae'] = np.mean(np.abs(predictions - targets))
    
    # Coefficient of Determination (R²)
    metrics['r2'] = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets, axis=0)) ** 2)
    
    # Mean Squared Error (MSE)
    metrics['mse'] = mean_squared_error(targets, predictions)
    
    # Mean Absolute Percentage Error (MAPE)
    mask = targets != 0  # Avoid division by zero
    metrics['mape'] = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    
    try:
        # Mean Squared Logarithmic Error (MSLE) - Only for positive values
        if np.all(predictions > 0) and np.all(targets > 0):
            metrics['msle'] = mean_squared_log_error(targets, predictions)
    except:
        metrics['msle'] = np.nan

    # 2. Classification metrics (Converting regression problem to classification problem)
    try:
        # Using median as threshold
        threshold = np.median(targets)
        y_true_binary = (targets > threshold).astype(int)
        y_pred_binary = (predictions > threshold).astype(int)
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        
        # Precision
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, average='binary')
        
        # Recall
        metrics['recall'] = recall_score(y_true_binary, y_pred_binary, average='binary')
        
        # F1 Score
        metrics['f1'] = f1_score(y_true_binary, y_pred_binary, average='binary')
        
        # AUC-ROC
        metrics['auc_roc'] = roc_auc_score(y_true_binary, predictions)
    except:
        metrics['accuracy'] = np.nan
        metrics['precision'] = np.nan
        metrics['recall'] = np.nan
        metrics['f1'] = np.nan
        metrics['auc_roc'] = np.nan
    
    return metrics

class Trainer:
    """
    Model Trainer
    Manages training, validation and testing processes
    Records training losses, accuracies and other metrics
    """
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer
        
        Parameters:
            model: Model to train
            config: Training configuration
            device: Training device
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config['train_config']['learning_rate'],
            weight_decay=config['train_config']['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler_type = config['train_config'].get('lr_scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=config['train_config']['num_epochs']
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer, 
                step_size=30, 
                gamma=0.1
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': []
        }
        
        # Set current epoch to 0
        self.current_epoch = 0
        
        # Create save directory and ensure it's an absolute path
        self.save_dir = os.path.abspath(config['train_config']['save_dir'])
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup TensorBoard - create a flag to track if it was successful
        self.tensorboard_enabled = False
        try:
            # Use a safer approach for TensorBoard directory
            # Use a different location for TensorBoard logs to avoid path issues
            self.tensorboard_dir = os.path.join('tensorboard_logs', 'runs', 
                                               os.path.basename(self.save_dir))
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            
            # Try to initialize the SummaryWriter with a safer path
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
            self.tensorboard_enabled = True
            print(f"TensorBoard logs will be saved to: {self.tensorboard_dir}")
        except Exception as e:
            print(f"Warning: TensorBoard initialization failed: {e}")
            # Fallback to a temporary directory if needed
            try:
                temp_dir = tempfile.mkdtemp(prefix="tb_logs_")
                self.tensorboard_dir = temp_dir
                self.writer = SummaryWriter(log_dir=temp_dir)
                self.tensorboard_enabled = True
                print(f"Using temporary directory for TensorBoard: {temp_dir}")
            except:
                self.writer = None
                print("TensorBoard disabled due to initialization errors")
            
        # Early stopping settings
        self.early_stopping_patience = config['train_config'].get('early_stopping', 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _log_metrics_to_tensorboard(self, metrics, step, prefix='train'):
        """
        Log metrics to TensorBoard
        """
        if not self.tensorboard_enabled or self.writer is None:
            return
            
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            if not np.isnan(metric_value):  # Log only non-NaN values
                try:
                    self.writer.add_scalar(f'{prefix}/{metric_name}', metric_value, step)
                except Exception as e:
                    print(f"Warning: Failed to log metric to TensorBoard: {e}")

    def _log_images_to_tensorboard(self, images, step):
        """
        Log images to TensorBoard
        """
        if not self.tensorboard_enabled or self.writer is None:
            return
            
        try:
            # Select first batch's first few images
            if images.dim() == 5:  # [batch, num_images, channels, height, width]
                images = images[:4, 0]  # Take first 4 samples' first image
            else:  # [batch, channels, height, width]
                images = images[:4]
            
            # Create image grid
            grid = torchvision.utils.make_grid(images, normalize=True)
            self.writer.add_image('input_images', grid, step)
        except Exception as e:
            print(f"Warning: Failed to log images to TensorBoard: {e}")

    def train_epoch(self, train_loader):
        """
        Train one epoch
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, params, targets) in enumerate(pbar):
            images = images.to(self.device)
            params = params.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images, params)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if self.writer is not None:
                try:
                    self.writer.add_scalar('training/batch_loss', loss.item(),
                                        batch_idx + len(train_loader) * self.current_epoch)
                except Exception as e:
                    print(f"Warning: Failed to record batch loss: {e}")
        
        # Calculate average loss and metrics
        avg_loss = total_loss / len(train_loader)
        metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
        
        return avg_loss, metrics
    
    def validate(self, val_loader):
        """
        Validate model
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images, params, target = batch
                images = images.to(self.device)
                params = params.to(self.device)
                target = target.to(self.device)
                
                # Forward propagation
                output = self.model(images, params)
                
                # Calculate loss
                loss = F.mse_loss(output, target)
                
                # Statistics
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                # Collect predictions and targets for metrics calculation
                predictions.append(output.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Calculate metrics
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        metrics = calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """
        Train model
        """
        if num_epochs is None:
            num_epochs = self.config['train_config']['num_epochs']
        
        # Record model structure
        if self.writer is not None:
            try:
                sample_images, sample_params, _ = next(iter(train_loader))
                self.writer.add_graph(self.model, [sample_images.to(self.device), 
                                                sample_params.to(self.device)])
            except Exception as e:
                print(f"Warning: Failed to record model structure: {e}")
        
        start_time = time.time()
        
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train one epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            print(f"Training loss: {train_loss:.4f}, Training RMSE: {train_metrics['rmse']:.4f}")
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            print(f"Validation loss: {val_loss:.4f}, Validation RMSE: {val_metrics['rmse']:.4f}")
            
            # Log to TensorBoard
            if self.writer is not None:
                self._log_metrics_to_tensorboard(train_metrics, epoch, 'train')
                self._log_metrics_to_tensorboard(val_metrics, epoch, 'validation')
                self.writer.add_scalar('loss/train', train_loss, epoch)
                self.writer.add_scalar('loss/validation', val_loss, epoch)
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                # Record images every N epochs
                if epoch % 5 == 0:
                    try:
                        sample_images, _, _ = next(iter(train_loader))
                        self._log_images_to_tensorboard(sample_images, epoch)
                    except Exception as e:
                        print(f"Warning: Failed to record training images: {e}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Update learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(os.path.join(self.save_dir, 'best_model.pth'))
                self.patience_counter = 0
                print("Saved best model")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping: No improvement in validation loss for {self.early_stopping_patience} epochs")
                break
        
        # Save final model
        self.save_model(os.path.join(self.save_dir, 'final_model.pth'))
        
        # Save training history
        self.save_history()
        
        # Close TensorBoard writer
        if self.writer is not None:
            try:
                self.writer.close()
            except:
                pass
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes")
    
    def test(self, test_loader):
        """
        Test model
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, params, targets in tqdm(test_loader, desc='Testing'):
                images = images.to(self.device)
                params = params.to(self.device)
                outputs = self.model(images, params)
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_preds, all_targets)
        
        # Print test metrics
        print("\nTest Metrics:")
        print("Regression Metrics:")
        for metric_name in ['rmse', 'mae', 'r2', 'mse', 'mape']:
            if metric_name in metrics:
                if metric_name == 'mape':
                    print(f"  {metric_name.upper()} = {metrics[metric_name]:.4f}%")
                else:
                    print(f"  {metric_name.upper()} = {metrics[metric_name]:.4f}")
        
        if 'classification_metrics' in metrics:
            print("\nClassification Metrics:")
            for metric_name, value in metrics['classification_metrics'].items():
                print(f"  {metric_name} = {value:.4f}")
        
        return metrics, all_preds, all_targets
    
    def save_model(self, path):
        """
        Save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
    
    def load_model(self, path):
        """
        Load model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def save_history(self):
        """
        Save training history
        """
        # Loss and learning rate history
        history_df = pd.DataFrame({
            'epoch': list(range(1, len(self.history['train_loss']) + 1)),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'learning_rate': self.history['lr']
        })
        
        # Training metrics
        for metric in self.history['train_metrics'][0].keys():
            history_df[f'train_{metric}'] = [m[metric] for m in self.history['train_metrics']]
        
        # Validation metrics
        for metric in self.history['val_metrics'][0].keys():
            history_df[f'val_{metric}'] = [m[metric] for m in self.history['val_metrics']]
        
        # Save to CSV
        history_df.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)
        
        # Plot training history
        self.plot_history()
    
    def plot_history(self):
        """
        Plot training history
        """
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        # Loss plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # RMSE plot
        plt.subplot(1, 2, 2)
        train_rmse = [m['rmse'] for m in self.history['train_metrics']]
        val_rmse = [m['rmse'] for m in self.history['val_metrics']]
        plt.plot(epochs, train_rmse, 'b-', label='Training RMSE')
        plt.plot(epochs, val_rmse, 'r-', label='Validation RMSE')
        plt.title('Training and Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300)
        plt.close()
        
        # Learning rate plot
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, self.history['lr'], 'g-')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.savefig(os.path.join(self.save_dir, 'learning_rate.png'), dpi=300)
        plt.close()

    def calculate_metrics(self, predictions, targets):
        """
        Calculate evaluation metrics
        """
        metrics = {}
        
        # Regression metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['r2'] = r2_score(targets, predictions)
        metrics['mse'] = mean_squared_error(targets, predictions)
        
        # Calculate MAPE, handle zero value cases
        mask = targets != 0
        if np.any(mask):
            mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.nan
        
        # Classification metrics (based on median threshold)
        median = np.median(targets)
        y_true_binary = (targets > median).astype(int)
        y_pred_binary = (predictions > median).astype(int)
        
        try:
            metrics['classification_metrics'] = {
                'accuracy': accuracy_score(y_true_binary, y_pred_binary),
                'precision': precision_score(y_true_binary, y_pred_binary),
                'recall': recall_score(y_true_binary, y_pred_binary),
                'f1': f1_score(y_true_binary, y_pred_binary),
                'auc_roc': roc_auc_score(y_true_binary, predictions)
            }
        except:
            # If classification metrics calculation fails, do not add to results
            pass
        
        return metrics 