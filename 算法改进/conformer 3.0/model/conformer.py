import torch
import torch.nn as nn
from model.attention import RelativeMultiHeadAttention, RelPositionalEncoding
from model.feed_forward import FeedForward
from model.convolution import ConvolutionModule

class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer Block结构: MF 或 CF 结构替代原来的FMCF结构，其余部分和conformer类似
    squeezeformer里使用的是post-LayerNorm结构, 用可学习的缩放层代替了pre-LayerNorm
    NF: x -> MHA (残差连接) -> post-LayerNorm -> scaling -> FeedForward(残差连接) 
    CF: Conv -> FeedForward (MF中的MHA替换为Conv，其余不变)
    交替使用两种结构代替单一的FMCF结构，以达到更好的效率和性能
    """
    def __init__(self, d_model, n_heads, d_ffn, kernel_size, dropout=0.1, block_type='MF'):
        super(SqueezeformerBlock, self).__init__()
        
        self.block_type = block_type
        
        if block_type == 'MF':
            # MHA -> FeedForward 结构
            self.attn = RelativeMultiHeadAttention(d_model, n_heads, dropout)  # 多头注意力机制
            self.ffn = FeedForward(d_model, d_ffn, dropout)  # 前馈网络
            self.attn_norm = nn.LayerNorm(d_model)  # 注意力层归一化
            self.ffn_norm = nn.LayerNorm(d_model)   # 前馈层归一化
        else:  # 'CF'
            # Conv -> FeedForward 结构  
            self.conv = ConvolutionModule(d_model, kernel_size, dropout)
            self.ffn = FeedForward(d_model, d_ffn, dropout)
            self.conv_norm = nn.LayerNorm(d_model)
            self.ffn_norm = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)         # 通用Dropout层
        self.final_norm = nn.LayerNorm(d_model)    # 最终输出归一化层

    def forward(self, x, mask=None, pos_emb=None):
        residual = x    # 记录残差连接
        
        if self.block_type == 'MF':     # 如果是MF结构就执行MHA和FFN
            # Multi-Head Attention
            x = self.attn_norm(x)            # 后归一化
            x = self.attn(x, mask, pos_emb)  # 注意力计算
            x = residual + self.dropout(x)   # 残差连接和Dropout
            
            # Feed Forward
            residual = x
            x = self.ffn_norm(x)             # 后归一化
            x = self.ffn(x)                  # 前馈网络计算
            x = residual + x                 # 残差连接
        else:        # 如果是CF结构就执行Conv和FFN
            # Convolution
            x = self.conv_norm(x)
            x = self.conv(x)
            x = residual + self.dropout(x)
            
            # Feed Forward
            residual = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = residual + x
        
        x = self.final_norm(x)
        return x

class ConformerBlock(nn.Module):   # 论文 Figure 1的Conformer Block，3.0里不启用，可忽略
    """
    Conformer Block 结构 (论文 Figure 1):
    x -> FFN_1 -> MHSA -> Conv -> FFN_2 -> Layernorm -> output
    注意：两个 FFN 都有 0.5 的残差系数
    """
    def __init__(self, d_model, n_heads, d_ffn, kernel_size, dropout=0.1):
        super(ConformerBlock, self).__init__()
        
        self.ffn1 = FeedForward(d_model, d_ffn, dropout)
        self.attn = RelativeMultiHeadAttention(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForward(d_model, d_ffn, dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # 模块间的 Layernorm 和 Dropout 通常集成在模块内或这里处理
        # 根据论文，Attention 和 Conv 后通常有 Post-Norm
        self.attn_norm = nn.LayerNorm(d_model)
        self.conv_norm = nn.LayerNorm(d_model) # Conv模块内部已有BN，这里是残差前的LayerNorm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_emb=None):
        # 1. First Feed Forward Module (Macaron Style: 0.5 * FFN)
        # 残差连接: x = x + 0.5 * FFN(x)
        residual = x
        x = self.ffn1(x) 
        x = residual + x # FFN内部已经乘了0.5，这里直接加
        
        # 2. Multi-Head Self Attention
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, mask, pos_emb)
        x = residual + self.dropout(x)
        
        # 3. Convolution Module
        residual = x
        # Conv模块内部处理了LayerNorm，所以这里直接传
        x = self.conv(x) 
        x = residual + self.dropout(x)
        
        # 4. Second Feed Forward Module
        residual = x
        x = self.ffn2(x)
        x = residual + x
        
        # 5. Final Layernorm
        x = self.norm(x)
        
        return x

class TemporalUpsampling(nn.Module):
    """
    Temporal U-Net 上采样层
    修复长度对齐问题
    """
    def __init__(self, scale_factor=2):
        super(TemporalUpsampling, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x, skip_connection=None):
        # x: [B, T, D]
        batch_size, seq_len, d_model = x.size()
        
        # 上采样到目标长度
        target_length = skip_connection.size(1) if skip_connection is not None else seq_len * self.scale_factor
        
        x_upsampled = torch.nn.functional.interpolate(
            x.transpose(1, 2), 
            size=target_length, 
            mode='linear', 
            align_corners=True
        ).transpose(1, 2)
        
        # 如果有跳跃连接，就相加（现在长度已经对齐）
        if skip_connection is not None:
            x_upsampled = x_upsampled + skip_connection
            
        return x_upsampled

class DepthwiseDownsampling(nn.Module):
    """
    深度可分离下采样层
    代替传统的池化或卷积下采样，更好地保留特征
    """
    def __init__(self, d_model, kernel_size=3, stride=2):
        super(DepthwiseDownsampling, self).__init__()
        self.downsample = nn.Conv1d(            # Depthwise Conv1d 实现下采样
            d_model, d_model, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=(kernel_size-1)//2,    # 计算padding以保持对齐
            groups=d_model  # depthwise
        )
        self.norm = nn.LayerNorm(d_model)       # 归一化层
        self.stride = stride                    # 保存stride信息

    def forward(self, x):                 
        # x: [B, T, D] -> [B, D, T] for conv
        batch_size, seq_len, d_model = x.size()
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ffn, kernel_size, 
                 dropout=0.1, use_squeezeformer=False, downsample_ratio=2, downsample_layer=7):
        super(ConformerEncoder, self).__init__()
        
        self.use_squeezeformer = use_squeezeformer
        self.downsample_ratio = downsample_ratio
        self.downsample_layer = downsample_layer
        self.n_layers = n_layers
        
        # 1. Subsampling (降采样层)
        self.subsampling = ConvSubsampling(input_dim, d_model, dropout)
        
        # 2. Positional Encoding
        self.pos_encoder = RelPositionalEncoding(d_model)
        
        # 3. 根据是否使用Squeezeformer选择不同的Block结构
        if use_squeezeformer:
            # Squeezeformer: 交替使用MF和CF块
            self.layers = nn.ModuleList()
            for i in range(n_layers):
                if i % 2 == 0:
                    # MF块 (MHA -> FFN)
                    block = SqueezeformerBlock(d_model, n_heads, d_ffn, kernel_size, dropout, 'MF')
                else:
                    # CF块 (Conv -> FFN)  
                    block = SqueezeformerBlock(d_model, n_heads, d_ffn, kernel_size, dropout, 'CF')
                self.layers.append(block)
                
            # 4. Temporal U-Net 组件
            if downsample_layer < n_layers:
                self.downsample_op = DepthwiseDownsampling(d_model, stride=downsample_ratio)
                self.upsample_op = TemporalUpsampling(scale_factor=downsample_ratio)
        else:
            # 原始Conformer结构
            self.layers = nn.ModuleList([
                ConformerBlock(d_model, n_heads, d_ffn, kernel_size, dropout)
                for _ in range(n_layers)
            ])

    def forward(self, x, input_lengths):
        """
        x: [batch, time, input_dim]
        input_lengths: [batch] 每一条音频的真实长度
        """
        # 下采样
        x, output_lengths = self.subsampling(x, input_lengths)
        
        # 保存下采样前的特征用于跳跃连接
        if self.use_squeezeformer and self.downsample_layer < self.n_layers:
            skip_connection = x.clone()
            skip_lengths = output_lengths.clone()
        
        # 获取位置编码
        pos_emb = self.pos_encoder(x)
        
        # 创建Padding Mask
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < output_lengths.unsqueeze(1)
        mask = mask.unsqueeze(1)
        
        # 通过多个Squeezeformer块
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask, pos_emb=pos_emb)
            
            # 在指定层后进行时间下采样
            if self.use_squeezeformer and i == self.downsample_layer and self.downsample_layer < self.n_layers:
                x_before_downsample = x.clone()  # 保存下采样前的状态
                x = self.downsample_op(x)
                
                # 更新mask和长度 - 使用更精确的长度计算
                output_lengths = torch.div(output_lengths + self.downsample_ratio - 1, self.downsample_ratio, rounding_mode='floor')
                max_len = x.size(1)
                mask = torch.arange(max_len, device=x.device).unsqueeze(0) < output_lengths.unsqueeze(1)
                mask = mask.unsqueeze(1)
                
                # 重新计算位置编码
                pos_emb = self.pos_encoder(x)
        
        # 上采样恢复时间分辨率
        if self.use_squeezeformer and self.downsample_layer < self.n_layers:
            x = self.upsample_op(x, skip_connection)
            # 恢复原始长度
            output_lengths = skip_lengths
            
        return x, output_lengths

class ConvSubsampling(nn.Module):
    """
    简单的 2层 2D 卷积，stride=2，实现 time 维度 4倍降采样
    """
    def __init__(self, input_dim, d_model, dropout):
        super(ConvSubsampling, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2), # [B, 1, T, F] -> [B, 32, T/2, F/2]
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), # -> [B, 32, T/4, F/4]
            nn.ReLU(),
        )
        
        # 计算卷积后的特征维度
        self.out_dim = 32 * (((input_dim - 1) // 2 - 1) // 2) 
        self.linear = nn.Linear(self.out_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_lengths):
        # x: [B, T, F] -> [B, 1, T, F] (channel first for Conv2d)
        x = x.unsqueeze(1) 
        x = self.conv(x)
        
        # 重塑形状: [B, C, T, F] -> [B, T, C*F]
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        
        x = self.linear(x)
        x = self.dropout(x)
        
        # 更新长度 (除以4)
        output_lengths = (((input_lengths - 1) // 2 - 1) // 2)
        
        return x, output_lengths