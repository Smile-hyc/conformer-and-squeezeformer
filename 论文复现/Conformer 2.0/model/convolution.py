import torch
import torch.nn as nn

#Conformer 引入了这个卷积模块，专门用来提取局部特征
#这里是一个复用的Swish类，在feedforward模块中已经提到
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 深度可分离卷积，它把卷积过程分成了三步pointwise1->Depthwise->pointwise2
class ConvolutionModule(nn.Module):
    """
    论文 Figure 2: Convolution Module
    结构: 
    Layernorm -> Pointwise Conv (x2通道) -> GLU -> Depthwise Conv -> BatchNorm -> Swish -> Pointwise Conv -> Dropout
    """
    def __init__(self, d_model, kernel_size, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        
        # LayerNorm 在卷积模块的开始，我们先对输入进行归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 第一个Pointwise Conv (逐点卷积，通道数扩展到2倍以供GLU使用)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model * 2, # 这里输出通道翻倍是因为GLU会用一半的通道来做gate
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
        )
        
        # GLU 激活函数 (自带 gating mechanism)，它会把输入通道分成两半，一半做内容，一半做门控
        self.glu = nn.GLU(dim=1) # GLU 会把通道数减半回到 d_model，dim=1 表示在通道维度进行切分
        
        # 深度卷积，每个通道独立卷积
        # groups=d_model 它告诉 PyTorch 让输入通道和输出通道一一对应，互不干扰。
        padding = (kernel_size - 1) // 2 # 保持卷积前后长度不变
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,# 卷积核大小，比如 31 或 15，感受野较大
            stride=1,
            padding=padding,# 保持卷积前后长度不变
            groups=d_model, # 关键点：设为 d_model 即使得它成为 Depthwise Conv
            # 相当于把输入劈成 256 份，每一份只用一个卷积核去卷，互不干扰
            # 参数量瞬间变成 in * 1 * kernel，大大减少了计算量，防止过拟合
            bias=True
        )
        
        # 5. Batch Norm，可以处理[Batch, Channel, Time] 格式的数据
        self.batch_norm = nn.BatchNorm1d(d_model)
        
        # 6. Swish激活
        self.swish = Swish()
        
        # 7. Pointwise Conv 2，再一次进行特征融合，把Depthwise Conv 提取的特征整合起来，且维度不变
        # 这里的输出通道数仍然是 d_model，保持与输入维度相同
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=1, 
            stride=1, 
            padding=0, # 逐点卷积不需要填充，因为它不改变序列长度
            bias=True
        )
        
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
        # LayerNorm 输出的是 [Batch, Time, Channel]
        # 但是 PyTorch 的 Conv1d 需要 [Batch, Channel, Time]
        x = x.transpose(1, 2)
        
        # Pointwise -> GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Depthwise -> BN -> Swish
        #提取时间局部特征，维度形状保持不变
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        
        # Pointwise 2 -> Dropout
        # 这里进行最后的特征整理
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # 转回维度: [B, D, T] -> [B, T, D]，卷积做完后，要把数据格式转回Transformer需要的格式
        x = x.transpose(1, 2)
        
        return x + residual # 这里的残差连接是 Module 内部的，原路信号 + 卷积处理后的信号