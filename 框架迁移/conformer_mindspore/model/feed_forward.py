import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class Swish(nn.Cell):
    """
    Swish 激活函数: x * sigmoid(x)
    Swish 光滑且非单调，在深层网络中的表现优于 ReLU，能帮助梯度更好地流向深层
    """
    def construct(self, x):
        return x * ops.sigmoid(x)

class FeedForward(nn.Cell):
    """
    论文 Figure 4: Feed Forward Module
    结构: Norm -> Linear -> Swish -> Dropout -> Linear -> Dropout
    并采用pre-norm结构
    """
    def __init__(self, d_model, d_ffn, dropout, expansion_factor=4):
        super(FeedForward, self).__init__()
        
        # 这里的 d_ffn 通常是 d_model * expansion_factor
        self.layer_norm = nn.LayerNorm((d_model,))  # 进入线性层之前先对输入进行归一化
        self.linear1 = nn.Dense(d_model, d_ffn)  # d_ffn通常是d_model的4倍，这样可以提取更丰富的特征
        self.swish = Swish()  # 使用Swish激活函数
        self.dropout1 = nn.Dropout(p=dropout)  # 随机丢弃一部分神经元，防止过拟合
        self.linear2 = nn.Dense(d_ffn, d_model)  # 将特征维度还原回d_model，以便后续进行残差连接
        self.dropout2 = nn.Dropout(p=dropout)  # 第二个Dropout层，再次随机丢弃，增加模型鲁棒性

    def construct(self, x):
        # x: [batch, seq_len, d_model]
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        
        # 论文公式 (1) 中提到 FFN 是 x + 1/2 * FFN(x)，
        # 但这里的 0.5 缩放我们通常放在 Block 级处理，或者在这里处理。
        # 为了通用性，这里只返回 FFN(x) 的结果，残差连接在外层做。
        return x * 0.5  # 两个FFN都是1/2残差连接