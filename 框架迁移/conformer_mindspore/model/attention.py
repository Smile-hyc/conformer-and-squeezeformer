import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import XavierUniform
import mindspore.numpy as mnp

class RelativeMultiHeadAttention(nn.Cell):
    """
    论文 Section 2.1: Multi-Headed Self-Attention with Relative Positional Encoding
    采用了 Transformer-XL 的相对位置编码机制。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0    # n_heads 必须是 d_model 的因数
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = nn.Dropout(p=dropout)
        self.scale = Tensor(math.sqrt(self.d_head), mindspore.float32)

        # Q, K, V, Pos 投影层, 其中 Pos 是相对位置编码
        self.linear_q = nn.Dense(d_model, d_model)
        self.linear_k = nn.Dense(d_model, d_model)
        self.linear_v = nn.Dense(d_model, d_model)
        self.linear_pos = nn.Dense(d_model, d_model, has_bias=False)

        # 输出层：把多头的结果拼回去后，再做一次线性变换
        self.linear_out = nn.Dense(d_model, d_model)
        
        # 两个可学习的 bias，用于相对位置计算
        # u_bias 对应 content-position, v_bias 对应 content-content
        self.u_bias = Parameter(Tensor(shape=(self.n_heads, self.d_head), 
                                       dtype=mindspore.float32, init=XavierUniform()))
        self.v_bias = Parameter(Tensor(shape=(self.n_heads, self.d_head), 
                                       dtype=mindspore.float32, init=XavierUniform()))
        
        # 操作符
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.batmatmul = ops.BatchMatMul()
        self.batmatmul_trans = ops.BatchMatMul(transpose_b=True)
        self.expand_dims = ops.ExpandDims()

    def construct(self, x, mask=None, pos_emb=None):
        """
        x: [batch, time, d_model]
        pos_emb: [2*time-1, d_model] 相对位置编码向量
        mask: [batch, 1, time]
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影
        q = self.linear_q(x)
        q = self.reshape(q, (batch_size, seq_len, self.n_heads, self.d_head))
        k = self.linear_k(x)
        k = self.reshape(k, (batch_size, seq_len, self.n_heads, self.d_head))
        v = self.linear_v(x)
        v = self.reshape(v, (batch_size, seq_len, self.n_heads, self.d_head))
        
        # 相对位置编码投影
        p = self.linear_pos(pos_emb)
        p = self.reshape(p, (pos_emb.shape[0], self.n_heads, self.d_head))

        # 维度调整 [Batch, Head, Time, Dim]
        q = self.transpose(q, (0, 2, 1, 3))  # [B, H, T, D_h]
        k = self.transpose(k, (0, 2, 1, 3))
        v = self.transpose(v, (0, 2, 1, 3))
        p = self.transpose(p, (1, 2, 0))     # [H, D_h, 2T-1]，无 batch 维度

        # 计算注意力分数
        # Term AC: Query * Key
        u_bias = self.reshape(self.u_bias, (1, self.n_heads, 1, self.d_head))
        q_with_u = q + u_bias
        ac = self.batmatmul_trans(q_with_u, k)  # [B, H, T, T]

        # Term BD: Query * Pos
        v_bias = self.reshape(self.v_bias, (1, self.n_heads, 1, self.d_head))
        q_with_v = q + v_bias
        # [B, H, T, D_h] x [H, D_h, 2T-1] -> [B, H, T, 2T-1]
        p_expanded = self.expand_dims(p, 0)  # [1, H, D_h, 2T-1]
        bd = self.batmatmul(q_with_v, p_expanded)  # [B, H, T, 2T-1]
        
        # 相对位置移位操作
        bd = self._relative_shift(bd, seq_len)  # [B, H, T, T]

        # 总分数
        scores = (ac + bd) / self.scale

        # Masking & Softmax
        if mask is not None:
            mask_expanded = self.expand_dims(mask, 1)
            scores = ops.masked_fill(scores, ops.equal(mask_expanded, 0), Tensor(-1e9, mindspore.float32))

        attn = self.softmax(scores)
        attn = self.dropout(attn)

        # 加权求和
        output = self.batmatmul(attn, v)  # [B, H, T, D_h]
        
        # 拼接多头与输出投影
        output = self.transpose(output, (0, 2, 1, 3))
        output = self.reshape(output, (batch_size, seq_len, self.d_model))

        # 输出线性变换
        output = self.linear_out(output)

        return output

    def _relative_shift(self, x, seq_len):
        """
        实现 Transformer-XL 的相对位置移位技巧
        输入 x: [Batch, Heads, Seq_Len, 2*Seq_Len - 1]
        输出: [Batch, Heads, Seq_Len, Seq_Len]
        """
        batch_size, n_heads, seq_len_x, total_len = x.shape
        
        # 在最后一维左侧补一列 0
        pad_op = ops.Pad(((0, 0), (0, 0), (0, 0), (1, 0)))
        x = pad_op(x)  # [B, H, T, 2T]
        
        # 展平维度并重塑
        x = self.reshape(x, (batch_size, n_heads, total_len + 1, seq_len_x))
        
        # 切掉第一行
        x = x[:, :, 1:, :]
        
        # 再次重塑回原始维度
        x = self.reshape(x, (batch_size, n_heads, seq_len_x, total_len))
        
        # 截取前 T 列
        return x[:, :, :, :seq_len]


class RelPositionalEncoding(nn.Cell):
    """
    生成相对位置编码向量
    """
    def __init__(self, d_model, max_len=5000):
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 预计算位置编码
        pe = self._compute_pe(max_len)
        self.pe = Parameter(pe, requires_grad=False)
        
    def _compute_pe(self, length):
        """
        计算正弦位置编码
        """
        total_len = length * 2 - 1
        
        # 创建位置索引
        position = mnp.arange(0, total_len).astype(mindspore.float32)
        position = position.reshape(-1, 1)
        
        # 中心对齐
        position = position - (length - 1)
        
        # 计算div_term
        div_term = mnp.arange(0, self.d_model, 2).astype(mindspore.float32)
        div_term = ops.Exp()(-div_term * (math.log(10000.0) / self.d_model))
        
        # 计算正弦/余弦
        pe_sin = ops.Sin()(position * div_term)  # [total_len, d_model//2]
        pe_cos = ops.Cos()(position * div_term)  # [total_len, d_model//2]
        
        # 交替拼接 sin 和 cos
        pe = ops.Zeros()((total_len, self.d_model), mindspore.float32)
        
        # 使用stack和reshape来交替排列
        pe_stack = ops.Stack(axis=2)([pe_sin, pe_cos])  # [total_len, d_model//2, 2]
        pe = pe_stack.reshape(total_len, -1)  # [total_len, d_model]
        
        # 如果d_model是奇数，截断最后一个
        if self.d_model % 2 == 1:
            pe = pe[:, :self.d_model]
        
        return pe

    def construct(self, x):
        batch, seq_len, _ = x.shape
        
        # 截取中心部分
        pe_len = self.pe.shape[0]
        start_idx = pe_len // 2 - seq_len + 1
        end_idx = pe_len // 2 + seq_len
        
        return self.pe[start_idx:end_idx]