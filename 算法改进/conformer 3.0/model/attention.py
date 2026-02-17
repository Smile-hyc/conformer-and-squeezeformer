import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeMultiHeadAttention(nn.Module):
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
        self.dropout = nn.Dropout(dropout)

        # Q, K, V, Pos 投影层, 其中 Pos 是相对位置编码
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)

        #输出层：把多头的结果拼回去后，再做一次线性变换
        self.linear_out = nn.Linear(d_model, d_model)
        
        # 两个可学习的 bias，用于相对位置计算
        # u_bias 对应 content-position, v_bias 对应 content-content
        self.u_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        # 参数初始化
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

    def forward(self, x, mask=None, pos_emb=None):
        """
        x: [batch, time, d_model]
        pos_emb: [2*time-1, d_model] 相对位置编码向量
        mask: [batch, 1, time]
        """
        batch_size, seq_len, _ = x.size()

        # 线性投影
        q = self.linear_q(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.linear_k(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.linear_v(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # 相对位置编码投影
        # pos_emb 的长度通常是 2*seq_len - 1 (从 -L 到 +L)
        # 这里假设输入 pos_emb 已经处理好
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), self.n_heads, self.d_head)

        # 维度调整 [Batch, Head, Time, Dim]
        q = q.permute(0, 2, 1, 3) # [B, H, T, D_h]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        p = p.permute(1, 2, 0)    # [H, D_h, 2T-1]，无 batch 维度

        # 计算注意力分数
        # 根据 Transformer-XL 论文，Attention Score = Term (a) + Term (b) + ...
        # 简化为两部分：Content-Content 和 Content-Position 
        # Term AC: Query * Key
        # q_with_bias = q + u_bias
        q_with_u = (q + self.u_bias.view(1, self.n_heads, 1, self.d_head))
        ac = torch.matmul(q_with_u, k.transpose(-2, -1)) # [B, H, T, T]

        # Term BD: Query * Pos
        # q_with_bias = q + v_bias
        q_with_v = (q + self.v_bias.view(1, self.n_heads, 1, self.d_head))
        bd = torch.matmul(q_with_v, p) # [B, H, T, 2T-1]
        
        # 相对位置移位操作，转换为标准的 Attention Map 格式
        bd = self._relative_shift(bd)  # [B, H, T, T]

        # 总分数
        scores = (ac + bd) / math.sqrt(self.d_head)

        # Masking & Softmax
        if mask is not None:
            # mask 需要扩展维度以匹配 scores
            # mask: 一般为[B, 1, 1, T]
            scores = scores.masked_fill(mask.unsqueeze(1).eq(0), -1e9)

        attn = F.softmax(scores, dim=-1)     # softemax 将分数转化为概率分布（所有列加起来为1）
        attn = self.dropout(attn)            # 对Attention结果做Dropout防止过拟合

        # 加权求和，用算出的概率去加权 Value (V)
        output = torch.matmul(attn, v) # [B, H, T, D_h]
        
        # 拼接多头与输出投影，把多头的数据拼回原来的形状
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)

        # 输出线性变换
        output = self.linear_out(output)

        return output

    def _relative_shift(self, x):
            """
            实现 Transformer-XL 的相对位置移位技巧 (修正稳健版)
            输入 x: [Batch, Heads, Seq_Len, 2*Seq_Len - 1]
            输出: [Batch, Heads, Seq_Len, Seq_Len]
            """
            batch_size, n_heads, seq_len, total_len = x.size()
            
            # 在最后一维左侧补一列 0，让展平后的总长度能被整除，并且实现错位
            # 形状: [B, H, T, 2T-1] -> [B, H, T, 2T]
            x = F.pad(x, (1, 0)) 
            
            # 展平维度并重塑，利用内存连续性实现“错位”
            # 把 (Time, 2*Time) 看成一维数组，然后按照新的宽度 (2*Time) 重新切分
            # [B, H, T, 2T] -> [B, H, 2T, T]
            x = x.view(batch_size, n_heads, total_len + 1, seq_len)
            
            # 切掉第一行 (实际上切掉了刚才补的0和错位产生的无用元素)
            # [B, H, 2T-1, T]
            x = x[:, :, 1:, :]
            
            # 再次重塑回原始维度，此时对角线已经对齐
            # [B, H, T, 2T-1]
            x = x.view(batch_size, n_heads, seq_len, total_len)
            
            # 截取我们需要的前 T 列 (Attention Map 的大小)
            # [B, H, T, T]
            return x[:, :, :, :seq_len]

class RelPositionalEncoding(nn.Module):
    """
    生成相对位置编码向量
    """
    def __init__(self, d_model, max_len=5000):
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.extend_pe(max_len)

    def extend_pe(self, length):
        """
        计算正弦位置编码
        """
        # 使用 hasattr 检查 pe 是否存在，并且长度是否足够
        if hasattr(self, 'pe') and self.pe.size(0) >= length * 2 - 1:
            return

        # 计算位置编码矩阵，这是一个标准的正弦/余弦位置编码公式 (Transformer-XL)    
        pe = torch.zeros(length * 2 - 1, self.d_model)
        position = torch.arange(0, length * 2 - 1).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        
        # 中心对齐，使得 position=0 对应中间位置
        position = position - (length - 1)
        
        # 计算正弦/余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer，不参与反向传播，但随模型保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 根据输入 x 的长度，截取中心部分需要的位置编码
        # 比如输入长度是 10，我们就截取从中心往左9个，往右9个，共19个编码
        batch, seq_len, _ = x.size()
        self.extend_pe(seq_len)
        # 截取中心部分
        start_idx = self.pe.size(0) // 2 - seq_len + 1
        end_idx = self.pe.size(0) // 2 + seq_len
        # 返回形状为 [2*seq_len-1, d_model] 的位置编码
        return self.pe[start_idx:end_idx]