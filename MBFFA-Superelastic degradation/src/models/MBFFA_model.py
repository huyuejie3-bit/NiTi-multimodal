"""
Multimodal model for predicting superelastic degradation in niti alloys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import DeformConv2d

class ConvBlock(nn.Module):
    """
    A convolutional block contains convolutional layers, batch normalization, activation functions, and optional pooling layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 pool=True, pool_size=2, pool_stride=2, dropout_rate=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = pool
        self.dropout = nn.Dropout2d(dropout_rate)
        self.pool_layer = nn.MaxPool2d(pool_size, pool_stride) if pool else None
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        if self.pool:
            x = self.pool_layer(x)
        return x

class FeatureSynergyModule(nn.Module):
    def __init__(self, feature_dims=[3, 3, 3, 3]):
        super().__init__()
        self.feature_dims = feature_dims  # [Euler, phase, KAM, numerical]
        self.num_dim = feature_dims[-1]          # Current numerical feature dimension

        # Add a linear layer to the numerical features to make its dimensions consistent with the other features (3D).
        self.num_expand = nn.Linear(self.num_dim, 3)

        # Redefine the number of input channels for interaction pairs (all based on the extended 3D numerical features).
        self.interaction_pairs = nn.ModuleDict({
            'ipf_phase': self._build_interaction(6),  # 3+3
            'ipf_kam': self._build_interaction(6),  # 3+3
            'ipf_num': self._build_interaction(6),  # 3+3 (扩展后)
            'phase_kam': self._build_interaction(6),  # 3+3
            'phase_num': self._build_interaction(6),  # 3+3 (扩展后)
            'kam_num': self._build_interaction(6)  # 3+3 (扩展后)
        })

        # 特征合并后的通道注意力（总通道数 = 3 + 3 + 3 + num_dim）
        total_dim = sum(self.feature_dims)

        # 特征合并后的通道注意力
        self.fc = nn.Sequential(
            nn.Linear(total_dim, total_dim // 4),
            nn.ReLU(),
            nn.Linear(total_dim // 4, total_dim),
            nn.Sigmoid()
        )

    def _build_interaction(self, in_channels):
        """为每对特征构建融合模块（统一6输入通道）"""
        return nn.ModuleDict({
            'offset_conv': nn.Sequential(
                nn.Conv2d(in_channels, 18, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(18, 2 * 3 * 3, kernel_size=3, padding=1)  # 2*3*3 offsets
            ),
            'deform_conv': DeformConv2d(3, 3, kernel_size=3, padding=1)
        })

    def _apply_interaction(self, feat1, feat2, pair_name):
        """执行特征对齐：feat1根据feat2调整"""
        combined = torch.cat([feat1, feat2], dim=1).unsqueeze(-1).unsqueeze(-1)  # [B,6,1,1]
        offsets = self.interaction_pairs[pair_name]['offset_conv'](combined)  # [B,18,1,1]
        aligned = self.interaction_pairs[pair_name]['deform_conv'](
            feat1.unsqueeze(-1).unsqueeze(-1),
            offsets
        ).squeeze()
        return aligned

    def forward(self, x):
        batch_size = x.size(0)
        ipf, phase, kam, num = torch.split(x, self.feature_dims, dim=1)

        # 扩展数值特征维度到3
        num_expanded = self.num_expand(num)  # [B, num_dim] -> [B, 3]

        # === 所有两两交互融合 ===
        ipf_phase = self._apply_interaction(ipf, phase, 'ipf_phase')
        ipf_kam = self._apply_interaction(ipf, kam, 'ipf_kam')
        ipf_num = self._apply_interaction(ipf, num_expanded, 'ipf_num')
        phase_kam = self._apply_interaction(phase, kam, 'phase_kam')
        phase_num = self._apply_interaction(phase, num_expanded, 'phase_num')
        kam_num = self._apply_interaction(kam, num_expanded, 'kam_num')

        # === 特征合并 ===
        combined = torch.cat([
            (ipf + ipf_phase + ipf_kam + ipf_num) / 4,
            (phase + ipf_phase + phase_kam + phase_num) / 4,
            (kam + ipf_kam + phase_kam + kam_num) / 4,
            num  # 保留原始数值特征
        ], dim=1)

        # 通道注意力
        weights = self.fc(combined.mean(dim=0, keepdim=True))
        return combined * weights  # [B,total_dim]

class ResidualBlock(nn.Module):
    """
    残差块，包含两个卷积层和跳跃连接
    """
    def __init__(self, channels, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 残差连接
        out = self.relu(out)
        return out

class SEBlock(nn.Module):
    """
    注意力模块 - 通道注意力
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """
    空间注意力模块
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class DilatedConvBlock(nn.Module):
    """
    空洞卷积块
    """
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4, 8]):
        super(DilatedConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(dilation_rates), 
                      kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanisms are used to handle the relationships between features.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # 调整头数，确保 embed_dim 能被 num_heads 整除
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 查询、键、值投影矩阵
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.final_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        参数:
            x: 输入特征 [batch_size, embed_dim]
        返回:
            output: 注意力处理后的特征 [batch_size, embed_dim]
        """
        batch_size = x.size(0)
        residual = x
        
        # 确保输入是二维的 [batch_size, embed_dim]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, embed_dim]
            squeezed = True
        else:
            squeezed = False
        
        # 多头注意力计算
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, heads, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        # 残差连接和层归一化
        x = self.norm(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.final_norm(x + ffn_output)
        
        # 如果输入是被挤压的，则输出也应该是挤压的
        if squeezed:
            x = x.squeeze(1)
        
        return x

class TransformerFusionBlock(nn.Module):
    """
    基于Transformer的特征融合模块
    用于处理多模态特征（图像特征和数值参数）的融合
    """
    def __init__(self, embed_dim, num_heads=4, ff_dim=512, dropout=0.1, num_layers=2):
        super(TransformerFusionBlock, self).__init__()
        self.embed_dim = embed_dim
        
        # 调整头数确保能被embed_dim整除
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        # 堆叠多个Transformer编码器层
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 位置编码（固定的正弦位置编码）
        self.register_buffer("positional_encodings", self._generate_positional_encodings(embed_dim))
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
    
    def _generate_positional_encodings(self, embed_dim, max_len=10):
        """生成固定的正弦位置编码"""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 确保div_term的尺寸正确
        if embed_dim % 2 != 0:
            # 对于奇数维度，特殊处理
            div_term_size = embed_dim // 2 + 1
            div_term = torch.exp(torch.arange(0, div_term_size*2, 2).float() * (-math.log(10000.0) / embed_dim))
            # 确保只使用需要的部分
            pe[:, 0::2] = torch.sin(position * div_term[:embed_dim//2 + embed_dim%2])
            # 对偶数位置
            if embed_dim > 1:  # 确保有足够的维度
                pe[:, 1::2] = torch.cos(position * div_term[:embed_dim//2])
        else:
            # 对于偶数维度，标准处理
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, embed_dim]
    
    def forward(self, x):
        """
        参数:
            x: 输入特征 [batch_size, embed_dim]
        返回:
            output: Transformer处理后的特征 [batch_size, embed_dim]
        """
        batch_size = x.size(0)
        
        # 添加序列维度
        if len(x.shape) == 2:
            # 将特征视为单个标记 [batch_size, 1, embed_dim]
            x = x.unsqueeze(1)
        
        # 仅使用第一个位置的位置编码
        pos_encodings = self.positional_encodings[:, :x.size(1), :].to(x.device)
        
        # 应用位置编码
        x = x + pos_encodings
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 应用输出投影
        x = self.output_proj(x)
        
        # 如果输入是2D的，输出也应该是2D的
        if x.size(1) == 1:
            x = x.squeeze(1)
        
        return x

class MaterialCNN(nn.Module):
    """
    Multimodal Model for Predicting Superelastic Degradation
    Supports multiple images and multiple numerical parameters as inputs
    """
    def __init__(self, config):
        """
        初始化模型

        参数:
            config: 模型配置
        """
        super(MaterialCNN, self).__init__()

        self.config = config
        self.num_images = config['data_config']['num_images']
        self.num_params = config['data_config']['num_parameters']
        self.num_outputs = config['data_config']['num_outputs']
        self.feature_dim = config['model_config']['feature_dim']
        self.dropout_rate = config['model_config']['dropout']

        # 图像输入尺寸
        self.input_size = config['data_config']['image_size']

        # 构建自定义CNN主干网络
        self.features = nn.Sequential(
            # 第一阶段: 输入图像 -> 64通道特征图 (336x336 -> 168x168 -> 84x84)
            ConvBlock(3, 32, kernel_size=7, stride=2, padding=3, pool=True),

            # 第二阶段: 通道扩展 (84x84 -> 84x84 -> 42x42)
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1, pool=True),
            ResidualBlock(64, dropout_rate=self.dropout_rate),

            # 第三阶段: 进一步特征提取 (42x42 -> 42x42 -> 21x21)
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, pool=True),
            # ResidualBlock(128, dropout_rate=self.dropout_rate),
            # SEBlock(128, reduction=8),  # 添加通道注意力

            # 第四阶段: 深层特征提取 (21x21 -> 21x21)
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, pool=False),
            # ResidualBlock(256, dropout_rate=self.dropout_rate),
            # SpatialAttention(),  # 添加空间注意力

            # 第五阶段: 深层特征提取 (21x21 -> 21x21 -> 10x10)
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1, pool=True),
            # ResidualBlock(256, dropout_rate=self.dropout_rate),
            # SpatialAttention(),  # 添加空间注意力

            # 第六阶段: 多尺度特征提取 (10x10 -> 10x10 -> 5x5)
            DilatedConvBlock(256, 512, dilation_rates=[1, 2, 3, 4]),
            nn.MaxPool2d(2, 2),
            # ResidualBlock(512, dropout_rate=self.dropout_rate),
            # SEBlock(512, reduction=16),  # 再次添加通道注意力

            # 最后阶段: 全局特征 (5x5 -> 1x1)
            nn.AdaptiveAvgPool2d(1)
        )

        # 计算卷积后的特征尺寸
        self.feature_size = 512

        # 图像特征降维层 - 仅保留3个神经元节点
        self.feature_reducer = nn.Sequential(
            nn.Linear(self.feature_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 3)  # 改为3个神经元节点
        )

        # 由于图像特征现在只有3个神经元，计算融合后的输入维度
        self.combined_size = 3 * self.num_images + self.num_params

        # 确保注意力机制的头数合适
        num_attention_heads = min(4, self.combined_size // 2) # 确保能被整除
        while self.combined_size % num_attention_heads != 0 and num_attention_heads > 1:
            num_attention_heads -= 1

        # 添加多头自注意力层 - 在特征融合后应用
        self.attention = MultiHeadSelfAttention(
            embed_dim=self.combined_size,
            num_heads=num_attention_heads,
            dropout=self.dropout_rate
        )

        # 特征融合模块Feature Synergy Module
        self.feature_synergy = FeatureSynergyModule(feature_dims=[3, 3, 3, self.num_params])

        # 使用Transformer取代全连接融合层
        self.transformer_fusion = TransformerFusionBlock(
            embed_dim=self.combined_size,
            num_heads=num_attention_heads,
            ff_dim=self.combined_size * 2,
            dropout=self.dropout_rate,
            num_layers=2  # 使用2层Transformer编码器
        )

        # 预测头网络 - 将Transformer编码后的特征转换为输出
        self.prediction_head = nn.Sequential(
            nn.Linear(self.combined_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        # 输出层
        self.output_layer = nn.Linear(32, self.num_outputs)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化模型权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images, params):
        """
        前向传播

        参数:
            images: 形状为 [batch_size, num_images, channels, height, width] 的图像
                   或 [batch_size, channels, height, width] 的单张图像
            params: 形状为 [batch_size, num_params] 的数值参数

        返回:
            predictions: 形状为 [batch_size, num_outputs] 的预测值
        """
        batch_size = images.size(0)

        # 处理输入图像维度
        if len(images.shape) == 4:  # 单张图像
            images = images.unsqueeze(1)  # 添加num_images维度

        # 处理多张图像
        image_features = []
        for i in range(min(self.num_images, images.size(1))):
            # 提取每张图像的特征
            x = images[:, i]
            x = self.features(x)
            x = x.view(batch_size, -1)  # 展平

            # 将特征向量降维到3个神经元节点
            x = self.feature_reducer(x)  # 输出为 [batch_size, 3]
            image_features.append(x)

        # 如果图像数量不足，用零填充
        while len(image_features) < self.num_images:
            x = torch.zeros_like(image_features[0])
            image_features.append(x)

        # 简单拼接所有图像特征
        image_features = torch.cat(image_features, dim=1)  # [batch_size, 3*num_images]

        # 将图像特征和数值参数直接连接起来
        combined_features = torch.cat([image_features, params], dim=1)  # [batch_size, 3*num_images + num_params]

        #应用feature_synergy模块处理特征间关系
        fused_features = self.feature_synergy(combined_features)

        # 应用多头自注意力机制处理特征间关系
        # attended_features = self.attention(combined_features)

        # 通过Transformer融合层进一步处理特征关系
        # transformed_features = self.transformer_fusion(attended_features)

        # 通过预测头网络
        fused_features = self.prediction_head(fused_features)

        # 输出预测
        predictions = self.output_layer(fused_features)

        return predictions

    def get_activation_maps(self, x):
        """
        获取激活映射，用于可视化
        """
        activations = {}

        # 保存特征图的勾子函数
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        # 注册勾子到最后一个卷积层(假设是特征提取器的倒数第二层)
        # 找到最后一个卷积层
        for i, layer in reversed(list(enumerate(self.features))):
            if isinstance(layer, ConvBlock) or isinstance(layer, ResidualBlock) or isinstance(layer, DilatedConvBlock):
                self.features[i].register_forward_hook(get_activation('features'))
                break

        # 前向传播
        with torch.no_grad():
            if len(x.shape) == 4:  # 单张图像
                x = x.unsqueeze(1)  # 添加num_images维度

            # 只处理第一张图像
            img = x[:, 0]
            _ = self.features(img)

        return activations.get('features', None)