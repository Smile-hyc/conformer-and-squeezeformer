import torch
import torch.nn as nn

# Swish 激活函数实现
class Swish(nn.Module):
    """
    Conformer 论文中使用的 Swish 激活函数，是一种非线性变换 
    f(x) = x * sigmoid(x)
    这个函数是光滑的且非单调，在深层网络中的表现优于 ReLU，能帮助梯度更好地流向深层。
    """
    def forward(self, x): #x为输入张量，sigmoid将输入映射到0到1之间
        return x * torch.sigmoid(x)

#前馈神经网络模块类，提炼特征
class FeedForward(nn.Module):
    """
    对应论文 Figure 4: Feed Forward Module
    标准的 Transformer FFN 结构通常是: Linear -> ReLU -> Linear
    Conformer 的 FFN 结构改为: 
    Layernorm -> Linear1 -> Swish -> Dropout -> Linear2 -> Dropout
    并且通常采用 Pre-Norm (先做归一化) 的结构。
    """
    def __init__(self, d_model, d_ffn, dropout, expansion_factor=4):
        super(FeedForward, self).__init__()
        
        self.layer_norm = nn.LayerNorm(d_model) # 进入线性层之前先对输入进行归一化
        self.linear1 = nn.Linear(d_model, d_ffn) # d_ffn通常是d_model的4倍，这样可以提取更丰富的特征
        self.swish = Swish() # 使用Swish激活函数
        self.dropout1 = nn.Dropout(dropout)  # 随机丢弃一部分神经元，防止过拟合
        self.linear2 = nn.Linear(d_ffn, d_model) # 将特征维度还原回d_model，以便后续进行残差连接
        self.dropout2 = nn.Dropout(dropout) # 这里是第二个Dropout层，再次随机丢弃，增加模型鲁棒性
    
    def forward(self, x):
        # 这里x的维度: [batch, seq_len, d_model]
        residual = x
        x = self.layer_norm(x) # 先做 LayerNorm 归一化
        x = self.linear1(x) # 第一个线性变换，升维到 d_ffn
        x = self.swish(x) # Swish 激活函数
        x = self.dropout1(x) # 第一个 Dropout，随机丢弃
        x = self.linear2(x) # 第二个线性变换，降维回 d_model
        x = self.dropout2(x) # 第二个 Dropout，再次随机丢弃
        
        # 论文公式 (1) 中提到 FFN 是 x + 1/2 * FFN(x)，
        # 但这里的 0.5 缩放我们通常放在 Block 级处理，或者在这里处理。
        # 为了通用性，这里只返回 FFN(x) 的结果，残差连接在外层做。
        return x * 0.5  # Macaron Net 风格通常会乘 0.5，这是Conformer和Transformer的最主要区别之一，普通的 Transformer Block 里只有一个 FFN。