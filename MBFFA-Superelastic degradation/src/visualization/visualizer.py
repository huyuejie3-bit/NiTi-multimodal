"""
Visualization module, used to generate heatmaps and various visualization results
"""

import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import shap
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# Configure matplotlib for better font handling and appearance
matplotlib.use('Agg')  # Use non-interactive backend
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a standard font that supports both English and special characters
plt.rcParams['axes.unicode_minus'] = False  # Correctly display negative signs
plt.rcParams['savefig.dpi'] = 300  # Higher resolution for saved figures
plt.rcParams['figure.autolayout'] = True  # Better layout handling

class MaterialVisualizer:
    """
    Material Performance Prediction Visualization Class
    """
    def __init__(self, model, config, device):
        """
        Initialize visualizer
        
        Parameters:
            model: Trained model
            config: Configuration
            device: Device
        """
        self.model = model
        self.config = config
        self.device = device
        self.save_dir = config['visualization_config']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
    
    def visualize_predictions(self, predictions, targets, save_name='predictions.png'):
        """
        Visualize prediction results compared to actual values
        
        Parameters:
            predictions: Predicted values, shape [n_samples, n_outputs]
            targets: Actual values, shape [n_samples, n_outputs]
            save_name: Save file name
        """
        n_outputs = predictions.shape[1]
        
        plt.figure(figsize=(5*n_outputs, 5))
        for i in range(n_outputs):
            plt.subplot(1, n_outputs, i+1)
            plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
            plt.plot([targets[:, i].min(), targets[:, i].max()], 
                    [targets[:, i].min(), targets[:, i].max()], 
                    'r--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Output {i+1} Predictions')
            
            # Calculate R²
            r2 = 1 - np.sum((targets[:, i] - predictions[:, i]) ** 2) / np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
            plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300)
        plt.close()
    
    def visualize_heatmap(self, image, params, target_layer_name='layer4', save_name='heatmap.png'):
        """
        Generate and visualize heatmap
        
        Parameters:
            image: Image tensor with shape [C, H, W]
            model_output: Model output
            target_layer_name: Target layer name
            save_name: Save file name
        """
        # Ensure the image has 3 channels
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        # Ensure image dimension is correct [1, C, H, W]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        if params.dim() == 1:
            params = params.unsqueeze(0)
        params = params.to(self.device)

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
        
        # Generate heatmap
        heatmap = self._generate_gradcam(image, params, target_layer_name)
        
        # Resize heatmap
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))
        
        # Apply heatmap to original image
        img_with_heatmap = self._apply_heatmap(img_np, heatmap)
        
        # Plot original image and heatmap
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title('Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(img_with_heatmap)
        plt.title('Heatmap Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300)
        plt.close()
    
    def _generate_gradcam(self, image_tensor, params_tensor, target_layer_name):
        """
        Generate Grad-CAM heatmap
        """
        # Register hook function
        self.model.eval()
        
        # 查找合适的目标层 - 使用最后一个卷积层或残差块
        target_layer = None
        # 遍历model.features找出最后一个卷积或残差块
        for i, layer in reversed(list(enumerate(self.model.features))):
            # 如果是ConvBlock, ResidualBlock或DilatedConvBlock类型
            if hasattr(layer, 'conv') or hasattr(layer, 'conv1') or hasattr(layer, 'convs'):
                target_layer = layer
                print(f"Using layer {i} for Grad-CAM")
                break
        
        # 如果未找到合适的层，使用最后一个层
        if target_layer is None:
            target_layer = self.model.features[-1]
            print("Using last layer for Grad-CAM")
        
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
            output = self.model(image_tensor, params_tensor)
            output_index = torch.argmax(output)
            
            # Backward propagation
            self.model.zero_grad()
            output[0, output_index].backward()
            
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
            
            return cam
            
        except Exception as e:
            print(f"Failed to generate heatmap: {str(e)}")
            # Remove hooks
            handle_forward.remove()
            handle_backward.remove()
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=np.float32)
    
    def _apply_heatmap(self, img, heatmap):
        """
        Apply heatmap to original image
        """
        # Convert heatmap to JET color mapping
        heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay on original image
        alpha = 0.5
        img_with_heatmap = alpha * heatmap_colored + (1-alpha) * img
        
        # Normalize
        img_with_heatmap = img_with_heatmap / img_with_heatmap.max()
        
        return img_with_heatmap
    
    def visualize_shap_values(self, data_loader, num_samples=100, save_name='shap_values.png'):
        """
        使用SHAP分析多模态特征的重要性
        同时考虑图像特征和数值参数对预测的贡献
        
        参数:
            data_loader: 数据加载器
            num_samples: 用于分析的样本数量
            save_name: 保存文件名
        """
        num_images = self.config['data_config']['num_images']
        num_params = self.config['data_config']['num_parameters']

        try:
            print("开始进行多模态SHAP分析...")
            
            # 收集样本数据
            samples = []
            combined_features_list = []
            targets_list = []
            images_list = []
            params_list = []
            
            # 从数据加载器中获取样本
            for i, (image, param, target) in enumerate(data_loader):
                if i >= num_samples:
                    break
                    
                # 保存数据
                images_list.append(image)
                params_list.append(param)
                targets_list.append(target)
                
                # 处理单个样本，提取特征
                # 将图像和参数转移到设备上
                image = image.to(self.device)
                param = param.to(self.device)
                
                # 提取每个样本的组合特征
                self.model.eval()
                with torch.no_grad():
                    # 提取图像特征
                    batch_size = image.size(0)
                    image_features = []
                    
                    # 处理每张图像
                    for j in range(image.size(1)):  # 遍历每张图像
                        img = image[:, j]
                        # 通过特征提取网络
                        x = self.model.features(img)
                        x = x.view(batch_size, -1)  # 展平
                        # 降维
                        x = self.model.feature_reducer(x)
                        image_features.append(x)
                    
                    # 拼接图像特征
                    image_features = torch.cat(image_features, dim=1)
                    
                    # 将图像特征和数值参数连接
                    combined_features = torch.cat([image_features, param], dim=1)
                    combined_features_list.append(combined_features.cpu())
                    
                    # 打印调试信息
                    print(f"样本 {i+1} 特征维度: 图像={image_features.shape}, 参数={param.shape}, 组合={combined_features.shape}")
            
            # 如果没有足够数据，返回
            if len(combined_features_list) == 0:
                print("警告: 没有足够的数据进行SHAP分析")
                return
            
            # 堆叠特征
            all_combined_features = torch.cat(combined_features_list, dim=0)
            all_images = torch.cat(images_list, dim=0)
            all_params = torch.cat(params_list, dim=0)
            
            print(f"收集到 {len(combined_features_list)} 个样本，特征维度: {all_combined_features.shape}")
            
            # 创建一个包装函数进行SHAP分析
            def model_predict_combined(features_data):
                """
                包装模型预测函数，接受组合特征输入
                """
                # 转换为张量
                if isinstance(features_data, np.ndarray):
                    features_data = torch.tensor(features_data, dtype=torch.float32).to(self.device)
                
                # 确保批次维度
                if len(features_data.shape) == 1:
                    features_data = features_data.unsqueeze(0)
                
                # 设置模型为评估模式
                self.model.eval()
                
                outputs = []
                
                # 对每个特征样本进行预测
                with torch.no_grad():
                    for feature in features_data:
                        # 调整为2D张量
                        feature = feature.view(1, -1)
                        
                        # 通过模型的注意力层和后续层
                        # attended_features = self.model.attention(feature)
                        # 使用transformer_fusion替代fusion_layer
                        # transformed_features = self.model.transformer_fusion(attended_features)
                        # 用FeatureSynergy而不是attention / transformer
                        fused_features = self.model.feature_synergy(feature)
                        # 通过预测头网络
                        fused_features = self.model.prediction_head(fused_features)
                        output = self.model.output_layer(fused_features)
                        
                        outputs.append(output.cpu().numpy())
                
                # 返回结果
                return np.vstack(outputs)
            
            # 使用KernelExplainer，适用于任何模型
            try:
                print("使用KernelExplainer进行SHAP分析...")
                
                # 创建背景数据 - 使用更多样本作为背景
                background_size = min(3, len(combined_features_list))
                background_data = all_combined_features[:background_size].cpu().numpy()
                print(f"使用 {background_size} 个样本作为背景数据，形状: {background_data.shape}")
                
                # 创建解释器
                explainer = shap.KernelExplainer(model_predict_combined, background_data)
                
                # 选择样本进行解释 - 使用所有剩余样本
                test_samples = all_combined_features[background_size:].cpu().numpy()
                if len(test_samples) == 0:  # 如果没有剩余样本，使用第一个样本
                    test_samples = all_combined_features[:1].cpu().numpy()
                
                print(f"用于解释的样本数量: {len(test_samples)}, 形状: {test_samples.shape}")
                
                # 计算SHAP值
                shap_values = explainer.shap_values(test_samples)
                
                # 打印SHAP值信息以便调试
                if isinstance(shap_values, list):
                    print(f"SHAP值是一个列表，长度为 {len(shap_values)}")
                    for i, sv in enumerate(shap_values):
                        print(f"  输出 {i+1} 的SHAP值形状: {sv.shape}")
                else:
                    print(f"SHAP值形状: {shap_values.shape}")
                
                # # 创建特征名称
                # num_images = self.config['data_config']['num_images']
                # num_params = self.config['data_config']['num_parameters']
                
                feature_names = []
                # 为每张图像的每个特征添加名称
                for i in range(num_images):
                    for j in range(3):  # 每张图像提取3个特征
                        feature_names.append(f'Img{i+1}_Feat{j+1}')
                
                # 为每个数值参数添加名称
                for i in range(num_params):
                    feature_names.append(f'Param{i+1}')
                
                print(f"特征名称列表: {feature_names}")
                print(f"特征名称数量: {len(feature_names)}, 特征维度: {test_samples.shape[1]}")
                
                # 确保特征名称数量与特征维度匹配
                if len(feature_names) != test_samples.shape[1]:
                    print(f"警告: 特征名称数量({len(feature_names)})与特征维度({test_samples.shape[1]})不匹配!")
                    # 调整特征名称以匹配特征维度
                    if len(feature_names) < test_samples.shape[1]:
                        for i in range(len(feature_names), test_samples.shape[1]):
                            feature_names.append(f'Feature{i+1}')
                    else:
                        feature_names = feature_names[:test_samples.shape[1]]
                    print(f"调整后的特征名称数量: {len(feature_names)}")
                
                # 生成摘要图
                plt.figure(figsize=(12, 8))
                
                if isinstance(shap_values, list):
                    # 如果返回多个输出的SHAP值，使用第一个
                    shap.summary_plot(shap_values[0], test_samples, feature_names=feature_names, show=False)
                else:
                    # 单一输出的SHAP值
                    shap.summary_plot(shap_values, test_samples, feature_names=feature_names, show=False)
                
                plt.title('多模态特征SHAP重要性分析')
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 额外创建一个图像特征与数值参数对比的SHAP图
                plt.figure(figsize=(12, 8))
                
                # 计算每类特征的总重要性
                if isinstance(shap_values, list):
                    shap_array = np.abs(shap_values[0])
                else:
                    shap_array = np.abs(shap_values)
                
                # 分别计算图像特征和数值参数的重要性
                img_features_count = num_images * 3
                
                # 确保索引不超出范围
                if shap_array.shape[1] < img_features_count + num_params:
                    print(f"警告: SHAP数组维度({shap_array.shape[1]})小于预期的特征总数({img_features_count + num_params})")
                    img_features_count = min(img_features_count, shap_array.shape[1])
                    
                if img_features_count < shap_array.shape[1]:
                    img_importance = np.sum(shap_array[:, :img_features_count])
                    param_importance = np.sum(shap_array[:, img_features_count:])
                    
                    # 打印各特征类型的重要性，便于调试
                    print(f"图像特征重要性: {img_importance:.6f}, 数值参数重要性: {param_importance:.6f}")
                    
                    # 创建饼图
                    plt.pie([img_importance, param_importance], 
                           labels=['图像特征', '数值参数'],
                           autopct='%1.1f%%',
                           startangle=90,
                           colors=['#ff9999', '#66b3ff'])
                    
                    plt.title('图像特征与数值参数重要性比例')
                    plt.axis('equal')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.save_dir, 'modal_importance_pie.png'), dpi=300)
                    plt.close()
                else:
                    print("警告: 无法区分图像特征和数值参数的重要性，因为SHAP数组维度不足")
                
                print("SHAP分析完成")
                
            except Exception as e:
                print(f"KernelExplainer失败: {str(e)}")
                print("尝试使用简化的特征重要性分析...")
                
                # 如果SHAP分析失败，使用简单的特征替换方法
                try:
                    # 设置模型为评估模式
                    self.model.eval()
                    
                    # 基准预测
                    baseline_features = all_combined_features[0].unsqueeze(0).to(self.device)
                    baseline_pred = model_predict_combined(baseline_features)[0, 0]
                    
                    # 逐个扰动每个特征
                    feature_importance = []
                    num_features = all_combined_features.shape[1]
                    
                    for i in range(num_features):
                        # 创建扰动特征（将第i个特征设为零）
                        perturbed_features = baseline_features.clone()
                        perturbed_features[0, i] = 0.0
                        
                        # 预测
                        perturbed_pred = model_predict_combined(perturbed_features)[0, 0]
                        
                        # 计算影响（百分比变化）
                        if abs(baseline_pred) > 1e-10:
                            importance = abs((baseline_pred - perturbed_pred) / baseline_pred)
                        else:
                            importance = abs(baseline_pred - perturbed_pred)
                        
                        feature_importance.append((i, importance))
                    
                    # 按重要性排序
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    
                    # 创建特征名称（与上面相同）
                    feature_names = []
                    for i in range(num_images):
                        for j in range(3):  # 每张图像提取3个特征
                            feature_names.append(f'Img{i+1}_Feat{j+1}')
                    
                    for i in range(num_params):
                        feature_names.append(f'Param{i+1}')
                    
                    # 绘制条形图
                    plt.figure(figsize=(12, 8))
                    top_n = min(15, len(feature_importance))  # 显示最重要的15个特征
                    indices = [f[0] for f in feature_importance[:top_n]]
                    values = [f[1] for f in feature_importance[:top_n]]
                    names = [feature_names[i] if i < len(feature_names) else f'Feature{i+1}' for i in indices]
                    
                    plt.barh(range(top_n), values, align='center')
                    plt.yticks(range(top_n), names)
                    plt.xlabel('特征重要性')
                    plt.title('多模态特征重要性分析（特征替换法）')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.save_dir, save_name), dpi=300)
                    plt.close()
                    
                    # 计算图像特征和数值参数的重要性
                    img_features_count = num_images * 3
                    img_importance = sum(values[i] for i, idx in enumerate(indices) if idx < img_features_count)
                    param_importance = sum(values[i] for i, idx in enumerate(indices) if idx >= img_features_count and idx < img_features_count + num_params)
                    
                    # 创建饼图
                    plt.figure(figsize=(10, 8))
                    plt.pie([img_importance, param_importance], 
                           labels=['图像特征', '数值参数'],
                           autopct='%1.1f%%',
                           startangle=90,
                           colors=['#ff9999', '#66b3ff'])
                    
                    plt.title('图像特征与数值参数重要性比例')
                    plt.axis('equal')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.save_dir, 'modal_importance_pie.png'), dpi=300)
                    plt.close()
                    
                    print("简化的多模态特征重要性分析完成")
                    
                except Exception as e:
                    print(f"简化的特征重要性分析也失败了: {str(e)}")
                
        except Exception as e:
            print(f"警告: SHAP分析失败: {str(e)}")
            print("详细错误:", e.__class__.__name__)
    
    def visualize_feature_importance(self, data_loader, save_name='feature_importance.png'):
        """
        Visualize feature importance, combining the influence of numerical parameters
        
        Parameters:
            data_loader: Data loader
            save_name: Save file name
        """
        # Collect some samples
        params = []
        
        for batch_idx, batch in enumerate(data_loader):
            _, batch_params, _ = batch
            params.append(batch_params.cpu().numpy())
            if batch_idx >= 10:
                break
        
        # Concatenate all samples
        params = np.vstack(params)
        
        # Analyze the distribution of each parameter
        param_names = [f'Param {i+1}' for i in range(params.shape[1])]
        
        plt.figure(figsize=(10, 6))
        
        # Draw boxplot
        plt.boxplot(params, labels=param_names)
        plt.title('Numerical Parameter Distribution')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300)
        plt.close()
        
        # Parameter correlation analysis
        if params.shape[1] > 1:
            plt.figure(figsize=(8, 6))
            sns.heatmap(np.corrcoef(params.T), annot=True, fmt='.2f', cmap='coolwarm',
                        xticklabels=param_names, yticklabels=param_names)
            plt.title('Parameter Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'parameter_correlation.png'), dpi=300)
            plt.close() 