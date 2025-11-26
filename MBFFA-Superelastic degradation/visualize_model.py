"""
模型结构可视化工具
"""


import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import torch.nn as nn
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.models.cnn_model import MaterialCNN
from src.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, VISUALIZATION_CONFIG

def visualize_with_torchviz(model, save_path):
    """
    使用torchviz可视化模型结构
    
    参数:
        model: 要可视化的模型
        save_path: 保存路径
    """
    # 设置模型为评估模式
    model.eval()
    
    try:
        # 创建一个随机输入
        num_images = DATA_CONFIG['num_images']
        image_size = DATA_CONFIG['image_size']
        num_params = DATA_CONFIG['num_parameters']
        
        # 创建随机输入张量
        batch_size = 2  # 使用批量大小>1，避免批归一化问题
        dummy_images = torch.randn(batch_size, num_images, 3, image_size, image_size)
        dummy_params = torch.randn(batch_size, num_params)
        
        # 前向传播
        # with torch.no_grad():
        output = model(dummy_images, dummy_params)
        
        # 创建计算图并保存
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.format = 'png'
        dot.render(save_path)
        
        print(f"模型结构已保存为: {save_path}.png")
    except Exception as e:
        print(f"使用torchviz生成模型结构时出错: {str(e)}")

def visualize_with_tensorboard(model, log_dir):
    """
    使用tensorboard可视化模型结构
    
    参数:
        model: 要可视化的模型
        log_dir: TensorBoard日志目录
    """
    # 设置模型为评估模式
    model.eval()
    
    try:
        # 创建一个SummaryWriter
        writer = SummaryWriter(log_dir)
        
        # 创建随机输入
        num_images = DATA_CONFIG['num_images']
        image_size = DATA_CONFIG['image_size']
        num_params = DATA_CONFIG['num_parameters']
        
        batch_size = 2  # 使用批量大小>1，避免批归一化问题
        dummy_images = torch.randn(batch_size, num_images, 3, image_size, image_size)
        dummy_params = torch.randn(batch_size, num_params)
        
        # 添加模型图
        # 注意：由于输入是两个参数，需要创建一个自定义模型包装器
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model
                
            def forward(self, dummy_input):
                # 分解输入：前半部分是images，后半部分是params
                n_images = DATA_CONFIG['num_images']
                n_channels = 3
                img_size = DATA_CONFIG['image_size']
                n_params = DATA_CONFIG['num_parameters']
                
                # 计算总尺寸
                n_image_elements = n_images * n_channels * img_size * img_size
                batch_size = dummy_input.size(0)
                
                # 手动拆分输入为images和params
                flat_images = dummy_input[:, :n_image_elements]
                params = dummy_input[:, -n_params:]
                
                # 重塑图像
                images = flat_images.view(batch_size, n_images, n_channels, img_size, img_size)
                
                return self.model(images, params)
        
        # 创建包装器
        model_wrapper = ModelWrapper(model)
        
        # 创建一个单一的输入张量，包含images和params
        dummy_input = torch.randn(batch_size, num_images * 3 * image_size * image_size + num_params)
        
        # 添加模型图到TensorBoard
        with torch.no_grad():
            writer.add_graph(model_wrapper, dummy_input)
        writer.close()
        
        print(f"模型结构已添加到TensorBoard。使用以下命令启动TensorBoard：")
        print(f"python -m tensorboard.main --logdir={log_dir} --port=6009")
    except Exception as e:
        print(f"使用TensorBoard生成模型结构时出错: {str(e)}")

def visualize_with_summary(model):
    """
    使用torchsummary生成模型摘要
    
    参数:
        model: 要可视化的模型
    """
    # 设置模型为评估模式
    model.eval()
    
    try:
        # 创建一个函数来包装模型的forward方法
        def prepare_summary(model):
            class ModelSummaryWrapper(nn.Module):
                def __init__(self, model):
                    super(ModelSummaryWrapper, self).__init__()
                    self.model = model
                    self.num_images = DATA_CONFIG['num_images']
                    self.image_size = DATA_CONFIG['image_size']
                    self.num_params = DATA_CONFIG['num_parameters']
                
                def forward(self, x):
                    # 这里假设x是图像输入
                    batch_size = x.size(0)
                    # 创建一个dummy的参数输入
                    params = torch.zeros(batch_size, self.num_params).to(x.device)
                    # 需要重塑图像输入以匹配模型期望的格式
                    x = x.view(batch_size, self.num_images, 3, self.image_size, self.image_size)
                    return self.model(x, params)
            
            return ModelSummaryWrapper(model)
        
        wrapped_model = prepare_summary(model)
        
        # 打印摘要
        print("\n===== 模型参数摘要 =====")
        # 使用一个假的输入形状，对应单个批次中的所有图像展平 (batch_size, num_images*channels*h*w)
        input_size = (DATA_CONFIG['num_images'] * 3, DATA_CONFIG['image_size'], DATA_CONFIG['image_size'])
        summary(wrapped_model, input_size, device="cpu", batch_size=2)
    except Exception as e:
        print(f"生成模型摘要时出错: {str(e)}")
        print("尝试使用替代方法...")
        visualize_model_text(model)

def visualize_model_text(model):
    """
    以文本形式显示模型结构
    
    参数:
        model: 要可视化的模型
    """
    # 设置模型为评估模式
    model.eval()
    
    try:
        # 打印模型结构
        print("\n===== 模型结构 =====")
        print(model)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n===== 模型参数信息 =====")
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        # 获取并显示每个子模块的参数数量
        print("\n===== 各模块参数数量 =====")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {module_params:,} 参数")
            
            # 对于复杂模块，递归显示子模块的参数
            if hasattr(module, 'named_children') and len(list(module.named_children())) > 0:
                for sub_name, sub_module in module.named_children():
                    sub_params = sum(p.numel() for p in sub_module.parameters())
                    print(f"  - {sub_name}: {sub_params:,} 参数")
    except Exception as e:
        print(f"以文本形式显示模型结构时出错: {str(e)}")

def save_detailed_text_summary(model, file_path):
    """
    将详细的模型结构保存到文本文件
    
    参数:
        model: 要可视化的模型
        file_path: 保存路径
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入模型结构
            f.write("===== 模型结构 =====\n")
            f.write(str(model) + "\n\n")
            
            # 写入模型参数信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            f.write("===== 模型参数信息 =====\n")
            f.write(f"总参数数量: {total_params:,}\n")
            f.write(f"可训练参数数量: {trainable_params:,}\n\n")
            
            # 写入各模块参数数量
            f.write("===== 各模块参数数量 =====\n")
            for name, module in model.named_children():
                module_params = sum(p.numel() for p in module.parameters())
                f.write(f"{name}: {module_params:,} 参数\n")
                
                # 对于复杂模块，递归显示子模块的参数
                if hasattr(module, 'named_children') and len(list(module.named_children())) > 0:
                    for sub_name, sub_module in module.named_children():
                        sub_params = sum(p.numel() for p in sub_module.parameters())
                        f.write(f"  - {sub_name}: {sub_params:,} 参数\n")
                        
                        # 继续递归显示子子模块的参数
                        if hasattr(sub_module, 'named_children') and len(list(sub_module.named_children())) > 0:
                            for sub_sub_name, sub_sub_module in sub_module.named_children():
                                sub_sub_params = sum(p.numel() for p in sub_sub_module.parameters())
                                f.write(f"    - {sub_sub_name}: {sub_sub_params:,} 参数\n")
            
            # 写入每个参数的详细信息
            f.write("\n===== 详细参数信息 =====\n")
            for name, param in model.named_parameters():
                f.write(f"{name}: 形状{list(param.shape)}, 参数量{param.numel():,}\n")
        
        print(f"详细模型结构已保存到: {file_path}")
    except Exception as e:
        print(f"保存详细模型结构时出错: {str(e)}")

def visualize_model_hierarchy(model, save_path):
    """
    生成模型结构的层次图
    
    参数:
        model: 要可视化的模型
        save_path: 保存路径
    """
    try:
        # 递归获取模块树
        def get_module_tree(module, prefix=''):
            result = []
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                # 获取该模块的参数数量
                params = sum(p.numel() for p in child.parameters())
                # 保存模块信息：全名，类型，参数量
                result.append((full_name, child.__class__.__name__, params))
                # 递归处理子模块
                result.extend(get_module_tree(child, full_name))
            return result
        
        # 获取模块树
        modules = get_module_tree(model)
        
        # 创建图表
        plt.figure(figsize=(20, 25))
        ax = plt.gca()
        ax.set_xlim(0, 10)
        
        # 计算每个模块的级别
        module_levels = {}
        for name, _, _ in modules:
            level = name.count('.')
            module_levels[name] = level
        
        # 确定每个级别的模块数量，用于计算y坐标
        level_counts = {}
        for name, level in module_levels.items():
            if level not in level_counts:
                level_counts[level] = 0
            level_counts[level] += 1
        
        # 确定每个级别的模块的y坐标
        level_positions = {}
        current_y = 0
        for level in sorted(level_counts.keys()):
            level_positions[level] = current_y
            current_y += level_counts[level] * 1.5
        
        # 每个级别的当前y偏移
        level_offsets = {level: 0 for level in level_counts.keys()}
        
        # 绘制模块框
        module_positions = {}  # 用于存储每个模块的位置，以便绘制连线
        
        # 颜色映射，根据模块类型着色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(set(type_name for _, type_name, _ in modules))))
        type_color = {type_name: colors[i] for i, type_name in enumerate(set(type_name for _, type_name, _ in modules))}
        
        for name, type_name, params in modules:
            level = module_levels[name]
            # 确定x和y坐标
            x = level * (10 / (len(level_counts) + 1))
            y = level_positions[level] + level_offsets[level]
            level_offsets[level] += 1.5
            
            # 存储模块位置
            module_positions[name] = (x, y)
            
            # 创建矩形框
            rect = patches.Rectangle((x - 0.7, y - 0.3), 1.4, 0.6, 
                                    linewidth=1, edgecolor='black', 
                                    facecolor=type_color[type_name], alpha=0.7)
            ax.add_patch(rect)
            
            # 添加标签
            short_name = name.split('.')[-1]  # 只显示最后一部分名称
            params_text = f"{params:,}参数" if params > 0 else ""
            plt.text(x, y, f"{short_name}\n{type_name}\n{params_text}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=8)
        
        # 绘制连线
        for name in module_positions:
            if '.' in name:
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name in module_positions:
                    x1, y1 = module_positions[parent_name]
                    x2, y2 = module_positions[name]
                    plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
        
        # 图表设置
        plt.title("模型组件层次结构", fontsize=16)
        plt.axis('off')  # 隐藏坐标轴
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"模型层次结构已保存为: {save_path}")
    except Exception as e:
        print(f"生成模型层次结构图时出错: {str(e)}")

def create_model_flowchart(model, save_path):
    """
    创建模型处理流程图
    
    参数:
        model: 要可视化的模型
        save_path: 保存路径
    """
    try:
        # 定义各模块阶段
        stages = [
            ("输入", "多模态输入\n(图像和参数)"),
            ("特征提取", "CNN特征提取器\n(卷积、池化、注意力机制)"),
            ("特征融合", "特征降维和融合"),
            ("注意力机制", "多头自注意力"),
            ("全连接层", "融合层\n(全连接网络)"),
            ("输出层", "预测输出")
        ]
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 定义节点位置
        n_stages = len(stages)
        positions = [(i * (12/n_stages), 4) for i in range(n_stages)]
        
        # 绘制节点和连接
        for i, ((stage_name, stage_desc), (x, y)) in enumerate(zip(stages, positions)):
            # 在主流程上创建节点
            rect = patches.Rectangle((x-1.5, y-0.6), 3, 1.2, 
                                    linewidth=1, edgecolor='black', 
                                    facecolor='lightblue', alpha=0.7)
            ax.add_patch(rect)
            plt.text(x, y, stage_desc, horizontalalignment='center', 
                    verticalalignment='center', fontsize=10)
            
            # 如果不是最后一个节点，添加连接线
            if i < n_stages - 1:
                next_x = positions[i+1][0]
                plt.arrow(x+1.5, y, next_x-x-3, 0, head_width=0.1, head_length=0.1, 
                        fc='black', ec='black')
        
        # 添加CNN模块的细节
        if hasattr(model, 'features'):
            y_offset = 2
            # 绘制CNN模块的细节节点
            cnn_stages = [
                "卷积块+池化\n输入处理",
                "残差块\n特征提取",
                "通道注意力\n特征增强",
                "空间注意力\n空间增强",
                "多尺度特征\n空洞卷积",
                "全局池化\n特征聚合"
            ]
            
            cnn_x_min = positions[1][0] - 1.5  # 特征提取阶段的x起点
            cnn_x_max = positions[1][0] + 1.5  # 特征提取阶段的x终点
            cnn_width = cnn_x_max - cnn_x_min
            cnn_step = cnn_width / len(cnn_stages)
            
            for i, stage_desc in enumerate(cnn_stages):
                x = cnn_x_min + i * cnn_step
                rect = patches.Rectangle((x, y_offset-0.4), cnn_step*0.8, 0.8, 
                                        linewidth=1, edgecolor='black', 
                                        facecolor='lightgreen', alpha=0.7)
                ax.add_patch(rect)
                plt.text(x + cnn_step*0.4, y_offset, stage_desc, 
                        horizontalalignment='center', verticalalignment='center', 
                        fontsize=8)
                
                # 连接到主流程
                plt.arrow(x + cnn_step*0.4, y_offset+0.4, 0, 3-y_offset, 
                        head_width=0.05, head_length=0.1, fc='black', ec='black', 
                        linestyle='--', alpha=0.5)
        
        # 添加标题和图例
        plt.title("多模态材料性能预测模型处理流程", fontsize=14)
        plt.axis('off')  # 隐藏坐标轴
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"模型处理流程图已保存为: {save_path}")
    except Exception as e:
        print(f"生成模型处理流程图时出错: {str(e)}")

if __name__ == "__main__":
    # 加载配置
    config = {
        'data_config': DATA_CONFIG,
        'model_config': MODEL_CONFIG,
        'train_config': TRAIN_CONFIG,
        'visualization_config': VISUALIZATION_CONFIG
    }
    
    # 创建模型
    model = MaterialCNN(config)
    # 设置模型为评估模式
    model.eval()
    
    # 确保目录存在
    os.makedirs("model_visualization", exist_ok=True)
    
    print("开始生成模型可视化...")
    
    # 使用torchviz可视化
    visualize_with_torchviz(model, "model_visualization/model_structure")
    
    # 使用tensorboard可视化
    visualize_with_tensorboard(model, "tensorboard_logs/model_graph")
    
    # 使用torchsummary生成摘要
    visualize_with_summary(model)
    
    # 以文本形式显示模型结构
    visualize_model_text(model)
    
    # 保存详细的文本摘要
    save_detailed_text_summary(model, "model_visualization/model_structure.txt")
    
    # 生成模型层次结构图
    visualize_model_hierarchy(model, "model_visualization/model_hierarchy.png")
    
    # 生成模型处理流程图
    create_model_flowchart(model, "model_visualization/model_flowchart.png")
    
    print("\n可视化完成！可以通过以下方式查看模型结构：")
    print("1. 查看 model_visualization/model_structure.png 文件 - 详细的模型计算图")
    print("2. 使用TensorBoard: python -m tensorboard.main --logdir=tensorboard_logs/model_graph --port=6009")
    print("3. 查看 model_visualization/model_structure.txt 文件 - 详细的文本描述")
    print("4. 查看 model_visualization/model_hierarchy.png 文件 - 模型层次结构图")
    print("5. 查看 model_visualization/model_flowchart.png 文件 - 模型处理流程图") 