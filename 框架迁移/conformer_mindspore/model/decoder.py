import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

class TransformerDecoderLayer(nn.Cell):
    """
    单层Transformer解码器
    """
    def __init__(self, d_model, n_heads, d_ffn, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed Forward
        self.ffn = nn.SequentialCell(
            nn.Dense(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(d_ffn, d_model),
            nn.Dropout(p=dropout)
        )
        
        # Layer Norms
        self.norm1 = nn.LayerNorm((d_model,))
        self.norm2 = nn.LayerNorm((d_model,))
        self.norm3 = nn.LayerNorm((d_model,))
        
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # Self-Attention with residual
        residual = tgt
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, 
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = residual + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-Attention with residual
        residual = tgt
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = residual + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed Forward with residual
        residual = tgt
        tgt2 = self.ffn(tgt)
        tgt = residual + tgt2
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerDecoder(nn.Cell):
    """
    标准的 Transformer Decoder
    用于 Autoregressive (自回归) 预测文本
    接收 Encoder 的输出 (语音特征) 和 目标文本 (上一时刻的词)，预测下一个词
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ffn, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # 词嵌入层（Embedding）,将词ID映射到向量空间
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # 解码器层列表
        self.decoder_layers = nn.CellList([
            TransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ffn=d_ffn,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # 输出层：把向量投影回词表大小，并计算出每个词的概率
        self.output_layer = nn.Dense(d_model, vocab_size)

    def construct(self, targets, memory, memory_mask=None, target_mask=None, target_padding_mask=None):
        """
        targets: 目标文本序列，作为 Teacher Forcing 的输入
        memory: Encoder 的输出，即语音特征
        target_mask: 因果掩码，一个下三角矩阵遮住未来的结果，防止抄答案
        target_padding_mask: Padding 掩码，忽略补 0 的部分
        """
        # 文字向量化 Embedding + Dropout
        x = self.embed(targets)
        x = self.dropout(x)
        
        # 逐层通过解码器
        for layer in self.decoder_layers:
            x = layer(x, memory, target_mask, target_padding_mask)
        
        # 输出层，预测下一个词的概率分布
        logits = self.output_layer(x)
        return logits