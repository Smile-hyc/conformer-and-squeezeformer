import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#搭配相对位置编码的多头注意力机制
class RelativeMultiHeadAttention(nn.Module):
    """
    对应论文 Section 2.1: Multi-Headed Self-Attention with Relative Positional Encoding
    普通 Transformer 的绝对位置编码直接加在输入上，
    这里把位置信息作为一种独立的“距离特征”参与 Attention 分数的计算。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        
        ## 保证特征维度能被头数整除，否则没法平均分给每个头
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        #d_head: 每个头负责的维度。比如 d_model=256, heads=4, 就是每个头处理 64 维
        self.d_head = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

        # Q, K, V, Pos 投影层。其中的Q, K, V 的投影层，和普通 Transformer 一样
        # Pos 的投影层：用来把位置编码变成和 Q 一样的维度进行计算
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)

        #输出层：把多头的结果拼回去后，再做一次线性变换
        self.linear_out = nn.Linear(d_model, d_model)
        
        # 两个可学习的 bias，用于相对位置计算
        # Query (查询向量) 需要两个通用的“基准点”：
        # u_bias (Content Bias)与v_bias (Position Bias): 
      
        self.u_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head)) #对内容的偏置
        self.v_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head)) #对位置的偏置
        
        #两个参数初始化
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

    def forward(self, x, mask=None, pos_emb=None):
        """
        x: [batch, time, d_model]
        pos_emb: [2*time-1, d_model] 相对位置编码向量
        mask: [batch, 1, time]，用于遮住padding部分
        """
        batch_size, seq_len, _ = x.size()

        # 线性投影
        # 把输入 x 变成 q, k, v
        # .view() 操作是把大维度 d_model 拆分成 n_heads * d_head
        q = self.linear_q(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.linear_k(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.linear_v(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        #把相对位置编码也进行投影
        #pos_emb 的长度是 2*seq_len - 1 ，包含了所有可能的相对距离
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), self.n_heads, self.d_head)

        # 维度调整 [Batch, Head, Time, Dim]
        q = q.permute(0, 2, 1, 3) # [B, H, T, D_h]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        p = p.permute(1, 2, 0)    #位置编码没有Batch维度 (所有样本共享相同的位置逻辑)
        # [H, D_h, 2T-1]，Dim 在前是为了方便后面的矩阵乘法

        # 计算注意力分数
        # 根据 Transformer-XL 论文，Attention Score = Term (a) + Term (b) + ...
        # 简化为两部分：Content-Content 和 Content-Position 
        q_with_u = (q + self.u_bias.view(1, self.n_heads, 1, self.d_head))
        ac = torch.matmul(q_with_u, k.transpose(-2, -1)) # [B, H, T, T]

        # Term BD: Query * Pos
        #Query 结合了“通用位置偏置 v”后，去和 Position (相对距离) 匹配。
        q_with_v = (q + self.v_bias.view(1, self.n_heads, 1, self.d_head))
        bd = torch.matmul(q_with_v, p) # [B, H, T, 2T-1]
        # 这个矩阵目前是“错位”的，它的列并不直接对应第几个 Token，而是对应“相对距离”。

        # 相对位置移位操作，将其转换为标准的 Attention Map 格式
        # 一个是相对索引，一个是绝对索引，为了能相加，必须挪动一下
        bd = self._relative_shift(bd)  # [B, H, T, T]

        # 总分数
        scores = (ac + bd) / math.sqrt(self.d_head)

        # 4. Masking和Softmax
        if mask is not None:
            # mask 需要扩展维度以匹配 scores 形状
            scores = scores.masked_fill(mask.unsqueeze(1).eq(0), -1e9)
        #softemax 将分数转化为概率分布
        attn = F.softmax(scores, dim=-1)
        # 对Attention结果做Dropout
        attn = self.dropout(attn)

        # 加权求和，用算出的概率去加权 Value (V)
        output = torch.matmul(attn, v) 
        
        # 拼接多头与输出投影，把多头的数据拼回原来的形状
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        # 最后再过一层线性层
        output = self.linear_out(output)

        return output

    def _relative_shift(self, x):
            """
            实现 Transformer-XL 的相对位置移位技巧
            输入 x: [Batch, Heads, Seq_Len, 2*Seq_Len - 1]
            输出:   [Batch, Heads, Seq_Len, Seq_Len]
            """
            batch_size, n_heads, seq_len, total_len = x.size()
            
            # 在最后一维左侧补一列 0
            # 原因：为了凑数，让展平后的总长度能被整除，并且实现错位。
            x = F.pad(x, (1, 0)) 
            
            # 展平维度并重塑，利用内存连续性实现“错位”
            # 我们把 (Time, 2*Time) 看成一维数组，然后按照新的宽度 (2*Time) 重新切分
            # 这里的逻辑是：总元素数量不变，只是改变观察方式
            x = x.view(batch_size, n_heads, total_len + 1, seq_len)
            
            # 切掉第一行 (实际上切掉了刚才补的0和错位产生的无用元素)
            x = x[:, :, 1:, :]
            
            # 再次重塑回原始维度，但此时对角线已经对齐
            x = x.view(batch_size, n_heads, seq_len, total_len)
            
            # 5. 截取需要的前 T 列 (Attention Map 的大小)
            # 原本的 2T-1 列里包含了从 -L 到 +L 的所有距离，
            # 移位后，前 T 列正好对应当前时刻所需的相对位置分数。
            return x[:, :, :, :seq_len]

class RelPositionalEncoding(nn.Module):
    """
    生成相对位置编码向量（Attention is all you need中提到的正弦波编码）
    """
    def __init__(self, d_model, max_len=5000):
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        # 预生成一个最大的位置编码，后续根据需要截取
        self.extend_pe(max_len)

    def extend_pe(self, length):
        """
        计算正弦位置编码
        """
        #先检查现有的表够不够长，如果够长则不用重复
        if hasattr(self, 'pe') and self.pe.size(0) >= length * 2 - 1:
            return

        # 计算位置编码矩阵，这是一个标准的正弦/余弦位置编码公式 
        pe = torch.zeros(length * 2 - 1, self.d_model)
        position = torch.arange(0, length * 2 - 1).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        
        # 中心对齐，使得 position=0 对应中间位置
        position = position - (length - 1)
        
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
        return self.pe[start_idx:end_idx]