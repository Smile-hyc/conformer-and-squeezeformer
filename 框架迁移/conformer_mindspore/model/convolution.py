import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class Swish(nn.Cell):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = ops.Sigmoid()
    
    def construct(self, x):
        return x * self.sigmoid(x)

class GLU(nn.Cell):
    """
    GLU门控线性单元实现
    """
    def __init__(self, dim=1):
        super(GLU, self).__init__()
        self.dim = dim
        self.split = ops.Split(axis=dim, output_num=2)
        self.sigmoid = ops.Sigmoid()
    
    def construct(self, x):
        a, b = self.split(x)
        return a * self.sigmoid(b)

class ConvolutionModule(nn.Cell):
    """
    论文 Figure 2: Convolution Module
    支持Squeezeformer的统一激活函数
    LayerNorm -> Pointwise Conv1d -> Swish -> Depthwise Conv1d -> BN -> Swish -> Pointwise Conv1d -> Dropout
    """
    def __init__(self, d_model, kernel_size, dropout=0.1, use_swish=False):
        super(ConvolutionModule, self).__init__()
        
        self.use_swish = use_swish
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm((d_model,))
        
        # Pointwise Conv (扩张通道数 x2 用于 GLU)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model * 2, 
            kernel_size=1, 
            stride=1, 
            pad_mode='valid',
            has_bias=True
        )
        
        # 根据配置选择激活函数
        if use_swish:
            # Squeezeformer风格：使用Swish替代GLU
            self.activation = Swish()
            # 调整通道数，因为Swish不会改变通道数
            self.channel_adjust = nn.Conv1d(d_model * 2, d_model, kernel_size=1, has_bias=True)
        else:
            # 原始Conformer：使用GLU
            self.glu = GLU(dim=1)
        
        # Depthwise Conv (1D)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            pad_mode='pad',
            padding=padding,
            group=d_model,  # 关键点
            has_bias=True
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
            pad_mode='valid',
            has_bias=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # 转置操作
        self.transpose = ops.Transpose()

    def construct(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        # 记录残差
        residual = x
        
        # Layernorm
        x = self.layer_norm(x)
        
        # 维度转换: [B, T, D] -> [B, D, T] 以适应 Conv1d
        x = self.transpose(x, (0, 2, 1))
        
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
        x = self.transpose(x, (0, 2, 1))
        
        return x + residual  # 这里的残差连接是 Module 内部的