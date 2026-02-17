import torch
import torch.nn as nn
from model.attention import RelativeMultiHeadAttention, RelPositionalEncoding
from model.feed_forward import FeedForward
from model.convolution import ConvolutionModule

#马卡龙生产流水线
class ConformerBlock(nn.Module):
    """
    Conformer Block 结构 (论文 Figure 1):
    x -> FFN_1 -> MHSA -> Conv -> FFN_2 -> Layernorm -> output
    该结构采用“马卡龙网络（Macaron-Net）”设计风格，通过两个前馈网络模块将注意力与卷积层包裹在中间，形成对称的夹心结构。
    具体组成如下：
    1.下层前馈子模块：实现输入特征的初次非线性变换。
    2.多头自注意力模块：捕捉序列中元素之间的全局依赖关系。
    3.卷积模块：提取局部上下文特征，增强模型对局部模式的建模能力。
    4.上层前馈子模块：对融合后的特征进行进一步变换与整合。
    5.层归一化：对输出进行规范化，提升训练稳定性。
    """
    def __init__(self, d_model, n_heads, d_ffn, kernel_size, dropout=0.1):
        super(ConformerBlock, self).__init__()
        
        self.ffn1 = FeedForward(d_model, d_ffn, dropout) #第一个前馈神经网络模块
        self.attn = RelativeMultiHeadAttention(d_model, n_heads, dropout) #多头相对位置注意力模块
        self.conv = ConvolutionModule(d_model, kernel_size, dropout) #卷积模块
        self.ffn2 = FeedForward(d_model, d_ffn, dropout) #第二个前馈神经网络模块
        self.norm = nn.LayerNorm(d_model) # 最后的LayerNorm，进行层归一化
        

        # 在进入 Attention 和 Conv 之前，先过一个 LayerNorm。
        # 这种 Pre-Norm 的结构通常更有利于深层模型的训练。
        self.attn_norm = nn.LayerNorm(d_model)
        self.conv_norm = nn.LayerNorm(d_model) # Conv模块内部已有BN，这里是残差前的LayerNorm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_emb=None):
        """
        mask: 用于 Attention 的掩码
        pos_emb: 预计算好的相对位置编码
        """
        # 第一层 Feed Forward
        # 残差连接: x = x + 0.5 * FFN(x)
        residual = x
        # ffn1 内部已经把输出乘了0.5 (在 feed_forward.py解释了)
        x = self.ffn1(x) 
        x = residual + x # 可以直接相加，残差连接
        
        # 多头相对位置注意力机制
        residual = x
        x = self.attn_norm(x) # 先归一化
        x = self.attn(x, mask, pos_emb) # 注意力计算
        x = residual + self.dropout(x) # 带dropout的残差连接
        
        # 卷积模块
        residual = x
        # Conv模块内部处理了LayerNorm，所以这里直接传
        x = self.conv(x) 
        x = residual + self.dropout(x) # 残差连接
        
        # 第二层 Feed Forward
        residual = x
        x = self.ffn2(x)
        x = residual + x
        
        # 整个Block输出前再做一次统一的归一化
        x = self.norm(x)
        
        return x

# Conformer编码器的主体
# Input -> Subsampling -> Positional Encoding -> N x Conformer Blocks -> Output
class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ffn, kernel_size, dropout=0.1):
        super(ConformerEncoder, self).__init__()
        
        # 降采样层
        # 将 80维 Mel 谱图降采样 4倍，并提升维度到 d_model
        self.subsampling = ConvSubsampling(input_dim, d_model, dropout)
        
        # 相对位置编码生成，不直接处理数据，而是生成正弦波矩阵，供Attention使用
        self.pos_encoder = RelPositionalEncoding(d_model)
        
        # 3. N个Conformer Blocks
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, d_ffn, kernel_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, input_lengths):
        """
        x: 记录原始音频特征
        input_lengths: 每一条音频的真实长度
        """
        #压缩
        x, output_lengths = self.subsampling(x, input_lengths)
        
        # 获取相对位置编码
        pos_emb = self.pos_encoder(x) 
        
        # 制作掩码
        # 因为这是 Batch 训练，长短不一，短的音频后面补了 0 (Padding)。
        max_len = x.size(1) #如果索引小于真实长度，就是 True (有效)，否则是 False (无效)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < output_lengths.unsqueeze(1)
        mask = mask.unsqueeze(1) #扩展维数以匹配Attention的输入要求
        
        # 逐层传递数据，每一层都用到了相对位置编码和掩码
        for layer in self.layers:
            x = layer(x, mask=mask, pos_emb=pos_emb)
            
        return x, output_lengths

class ConvSubsampling(nn.Module):
    """
    前端降采样模块 (论文中Frontend Subsampling)
    作用: 将输入音频的长度缩短 4 倍，降低计算量，同时进行初步特征提取。
    结构: 2层 2D 卷积 (Conv2d)，每层 stride=2。
    注意：卷积会改变频率维度，这里改变的是时间和频率
    """
    def __init__(self, input_dim, d_model, dropout):
        super(ConvSubsampling, self).__init__()
        #这是一个连续的卷积序列
        self.conv = nn.Sequential(
            #第一层卷积
            nn.Conv2d(1, 32, 3, stride=2), # [B, 1, T, F] -> [B, 32, T/2, F/2]
            nn.ReLU(),
            #第二层卷积
            nn.Conv2d(32, 32, 3, stride=2), # -> [B, 32, T/4, F/4]
            nn.ReLU(),
        )

        # 动态计算卷积后特征的总维度
        self.out_dim = 32 * (((input_dim - 1) // 2 - 1) // 2) 

        # 最后的线性投影: 把卷积出来的杂乱特征统一映射到模型的 d_model 维度 (比如 256)
        self.linear = nn.Linear(self.out_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_lengths):
        # x: 梅尔频谱，input_lengths: 每条音频的真实长度
        x = x.unsqueeze(1) 
        x = self.conv(x)
        
        # 维度的重塑，要把Channel和Freq_new合并，只保留Batch和Time
        b, c, t, f = x.size()

        # 合并 Channel 和 Freq 维度
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        
        # 线性变换到 d_model 维度
        x = self.linear(x)
        x = self.dropout(x)
        
        # 更新长度，因为时间维度被压缩了4倍，所有的长度记录也要相应除以4
        output_lengths = (((input_lengths - 1) // 2 - 1) // 2)
        
        return x, output_lengths