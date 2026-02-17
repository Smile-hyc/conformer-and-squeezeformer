import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvolutionModule(nn.Module):
    """
    论文 Figure 2: Convolution Module
    支持Squeezeformer的统一激活函数
    LayerNorm -> Pointwise Conv1d -> Swish -> Depthwise Conv1d -> BN -> Swish -> Pointwise Conv1d -> Dropout
    """
    def __init__(self, d_model, kernel_size, dropout=0.1, use_swish=False):
        super(ConvolutionModule, self).__init__()
        
        self.use_swish = use_swish
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise Conv (扩张通道数 x2 用于 GLU)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model * 2, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
        )
        
        # 根据配置选择激活函数
        if use_swish:
            # Squeezeformer风格：使用Swish替代GLU
            self.activation = Swish()
            # 调整通道数，因为Swish不会改变通道数
            self.channel_adjust = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        else:
            # 原始Conformer：使用GLU
            self.glu = nn.GLU(dim=1)
        
        # Depthwise Conv (1D)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=d_model, # 关键点
            bias=True
        )
        
        # Batch Norm
        self.batch_norm = nn.BatchNorm1d(d_model)
        
        # Swish (这里保持使用Swish)
        self.swish = Swish()
        
        # Pointwise Conv 2
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        # 记录残差
        residual = x
        
        # Layernorm
        x = self.layer_norm(x)
        
        # 维度转换: [B, T, D] -> [B, D, T] 以适应 Conv1d
        x = x.transpose(1, 2)
        
        # Pointwise -> 激活函数
        x = self.pointwise_conv1(x)
        
        if self.use_swish:
            # Squeezeformer风格：Swish激活
            x = self.activation(x)
            x = self.channel_adjust(x)  # 调整回d_model通道
        else:
            # 原始Conformer：GLU激活
            x = self.glu(x)
        
        # Depthwise -> BN -> Swish
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        
        # Pointwise 2 -> Dropout
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # 转回维度: [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)
        
        return x + residual # 这里的残差连接是 Module 内部的