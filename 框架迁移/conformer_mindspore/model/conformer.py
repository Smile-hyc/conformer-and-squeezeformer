import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.numpy as mnp
from model.attention import RelativeMultiHeadAttention, RelPositionalEncoding
from model.feed_forward import FeedForward
from model.convolution import ConvolutionModule


class SqueezeformerBlock(nn.Cell):
    """
    Squeezeformer Block结构: MF 或 CF 结构替代原来的FMCF结构，其余部分和conformer类似
    squeezeformer里使用的是post-LayerNorm结构, 用可学习的缩放层代替了pre-LayerNorm
    MF: x -> MHA (残差连接) -> post-LayerNorm -> scaling -> FeedForward(残差连接) 
    CF: Conv -> FeedForward (MF中的MHA替换为Conv，其余不变)
    交替使用两种结构代替单一的FMCF结构，以达到更好的效率和性能
    """
    def __init__(self, d_model, n_heads, d_ffn, kernel_size, dropout=0.1, block_type='MF'):
        super(SqueezeformerBlock, self).__init__()
        
        self.block_type = block_type
        
        if block_type == 'MF':
            # MHA -> FeedForward 结构
            self.attn = RelativeMultiHeadAttention(d_model, n_heads, dropout)
            self.ffn = FeedForward(d_model, d_ffn, dropout)
            self.attn_norm = nn.LayerNorm((d_model,))
            self.ffn_norm = nn.LayerNorm((d_model,))
        else:  # 'CF'
            # Conv -> FeedForward 结构  
            self.conv = ConvolutionModule(d_model, kernel_size, dropout)
            self.ffn = FeedForward(d_model, d_ffn, dropout)
            self.conv_norm = nn.LayerNorm((d_model,))
            self.ffn_norm = nn.LayerNorm((d_model,))
            
        self.dropout = nn.Dropout(p=dropout)
        self.final_norm = nn.LayerNorm((d_model,))

    def construct(self, x, mask=None, pos_emb=None):
        residual = x
        
        if self.block_type == 'MF':
            # Multi-Head Attention
            x = self.attn_norm(x)
            x = self.attn(x, mask, pos_emb)
            x = residual + self.dropout(x)
            
            # Feed Forward
            residual = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = residual + x
        else:
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


class ConformerBlock(nn.Cell):
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
        self.norm = nn.LayerNorm((d_model,))
        
        # 模块间的 Layernorm 和 Dropout 通常集成在模块内或这里处理
        # 根据论文，Attention 和 Conv 后通常有 Post-Norm
        self.attn_norm = nn.LayerNorm((d_model,))
        self.conv_norm = nn.LayerNorm((d_model,))
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, mask=None, pos_emb=None):
        # 1. First Feed Forward Module (Macaron Style: 0.5 * FFN)
        # 残差连接: x = x + 0.5 * FFN(x)
        residual = x
        x = self.ffn1(x) 
        x = residual + x  # FFN内部已经乘了0.5，这里直接加
        
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


class TemporalUpsampling(nn.Cell):
    """
    Temporal U-Net 上采样层
    修复长度对齐问题
    """
    def __init__(self, scale_factor=2):
        super(TemporalUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.transpose = ops.Transpose()

    def construct(self, x, skip_connection=None):
        # x: [B, T, D]
        batch_size, seq_len, d_model = x.shape
        
        # 上采样到目标长度
        if skip_connection is not None:
            target_length = skip_connection.shape[1]
        else:
            target_length = seq_len * self.scale_factor
        
        # 转换维度进行插值 [B, T, D] -> [B, D, T]
        x_transposed = self.transpose(x, (0, 2, 1))
        
        # 使用interpolate进行上采样
        # MindSpore中使用ResizeBilinear或自定义插值
        x_transposed = x_transposed.astype(mindspore.float32)
        
        # 简单的线性插值实现
        x_upsampled = ops.interpolate(
            x_transposed.expand_dims(2),  # [B, D, 1, T]
            sizes=(1, target_length),
            mode='bilinear',
            align_corners=True
        ).squeeze(2)  # [B, D, T]
        
        # 转回 [B, T, D]
        x_upsampled = self.transpose(x_upsampled, (0, 2, 1))
        
        # 如果有跳跃连接，就相加
        if skip_connection is not None:
            x_upsampled = x_upsampled + skip_connection
            
        return x_upsampled


class DepthwiseDownsampling(nn.Cell):
    """
    深度可分离下采样层
    代替传统的池化或卷积下采样，更好地保留特征
    """
    def __init__(self, d_model, kernel_size=3, stride=2):
        super(DepthwiseDownsampling, self).__init__()
        padding = (kernel_size - 1) // 2
        self.downsample = nn.Conv1d(
            d_model, d_model, 
            kernel_size=kernel_size, 
            stride=stride, 
            pad_mode='pad',
            padding=padding,
            group=d_model,  # depthwise
            has_bias=True
        )
        self.norm = nn.LayerNorm((d_model,))
        self.stride = stride
        self.transpose = ops.Transpose()

    def construct(self, x):
        # x: [B, T, D] -> [B, D, T] for conv
        x = self.transpose(x, (0, 2, 1))
        x = self.downsample(x)
        x = self.transpose(x, (0, 2, 1))
        x = self.norm(x)
        return x


class ConvSubsampling(nn.Cell):
    """
    简单的 2层 2D 卷积，stride=2，实现 time 维度 4倍降采样
    """
    def __init__(self, input_dim, d_model, dropout):
        super(ConvSubsampling, self).__init__()
        
        self.conv = nn.SequentialCell(
            nn.Conv2d(1, 32, 3, stride=2, pad_mode='valid', has_bias=True),  # [B, 1, T, F] -> [B, 32, T/2, F/2]
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, pad_mode='valid', has_bias=True),  # -> [B, 32, T/4, F/4]
            nn.ReLU(),
        )
        
        # 计算卷积后的特征维度
        self.out_dim = 32 * (((input_dim - 1) // 2 - 1) // 2) 
        self.linear = nn.Dense(self.out_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x, input_lengths):
        # x: [B, T, F] -> [B, 1, T, F] (channel first for Conv2d)
        x = self.expand_dims(x, 1)
        x = self.conv(x)
        
        # 重塑形状: [B, C, T, F] -> [B, T, C*F]
        b, c, t, f = x.shape
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (b, t, c * f))
        
        x = self.linear(x)
        x = self.dropout(x)
        
        # 更新长度 (除以4)
        output_lengths = ((input_lengths - 1) // 2 - 1) // 2
        
        return x, output_lengths


class ConformerEncoder(nn.Cell):
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
            self.layers = nn.CellList()
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
            self.layers = nn.CellList([
                ConformerBlock(d_model, n_heads, d_ffn, kernel_size, dropout)
                for _ in range(n_layers)
            ])
        
        self.expand_dims = ops.ExpandDims()

    def construct(self, x, input_lengths):
        """
        x: [batch, time, input_dim]
        input_lengths: [batch] 每一条音频的真实长度
        """
        # 下采样
        x, output_lengths = self.subsampling(x, input_lengths)
        
        # 保存下采样前的特征用于跳跃连接
        skip_connection = None
        skip_lengths = None
        if self.use_squeezeformer and self.downsample_layer < self.n_layers:
            skip_connection = x
            skip_lengths = output_lengths
        
        # 获取位置编码
        pos_emb = self.pos_encoder(x)
        
        # 创建Padding Mask
        max_len = x.shape[1]
        arange_tensor = mnp.arange(max_len)
        mask = self.expand_dims(arange_tensor, 0) < self.expand_dims(output_lengths, 1)
        mask = self.expand_dims(mask, 1)
        
        # 通过多个Conformer/Squeezeformer块
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x, mask=mask, pos_emb=pos_emb)
            
            # 在指定层后进行时间下采样
            if self.use_squeezeformer and i == self.downsample_layer and self.downsample_layer < self.n_layers:
                x = self.downsample_op(x)
                
                # 更新mask和长度 - 使用更精确的长度计算
                output_lengths = (output_lengths + self.downsample_ratio - 1) // self.downsample_ratio
                max_len = x.shape[1]
                arange_tensor = mnp.arange(max_len)
                mask = self.expand_dims(arange_tensor, 0) < self.expand_dims(output_lengths, 1)
                mask = self.expand_dims(mask, 1)
                
                # 重新计算位置编码
                pos_emb = self.pos_encoder(x)
        
        # 上采样恢复时间分辨率
        if self.use_squeezeformer and self.downsample_layer < self.n_layers:
            x = self.upsample_op(x, skip_connection)
            # 恢复原始长度
            output_lengths = skip_lengths
            
        return x, output_lengths