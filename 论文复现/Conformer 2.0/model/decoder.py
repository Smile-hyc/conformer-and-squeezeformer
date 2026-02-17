import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    """
    延用了标准的 Transformer Decoder
    作用: 接收 Encoder 的输出 (语音特征) 和 目标文本 (上一时刻的词)，预测下一个词。
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ffn, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        #词嵌入层（Embedding）,将词ID映射到向量空间
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 解码层堆叠，直接使用pyTorch官方的TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True 
        )
        
        #将上面定义的一层解码层堆叠n_layers次
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # 输出层：把向量投影回词表大小，并计算出每个词的概率
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, targets, memory, memory_mask=None, target_mask=None, target_padding_mask=None):
        """
        targets: 目标文本序列，作为 Teacher Forcing 的输入
        memory: Encoder 的输出，即语音特征
        注：Q来自 Decoder（我在写什么），K/V来自Encoder（刚才听到了什么）
        掩码由两部分构成
        target_mask: 因果掩码，一个下三角矩阵遮住未来的结果，防止抄答案
        target_padding_mask: Padding 掩码，忽略补 0 的部分
        """
        # 文字向量化
        x = self.embed(targets)
        x = self.dropout(x)
        
        # Transformer Decoder
        # memory_key_padding_mask 用于忽略 Encoder 输出中的 padding 部分
        x = self.decoder(
            tgt=x,#解码器的输入
            memory=memory,#编码器的输出
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask
        )
        
        # 输出层，预测下一个词的概率分布
        logits = self.output_layer(x)
        return logits