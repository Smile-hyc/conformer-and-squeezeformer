import torch
import torch.nn as nn
from model.conformer import ConformerEncoder
from model.decoder import TransformerDecoder

#模型的顶层封装，包含Conformer Encoder和Transformer Decoder
#两个分支：分别是CTC分支和Attention分支
class ConformerASR(nn.Module):
    """
    Conformer ASR 完整模型 (Hybrid CTC/Attention)
    这是一个集大成者，它同时包含 Encoder, Decoder 和 CTC 输出层。
    """
    def __init__(self, config, vocab_size):
        super(ConformerASR, self).__init__()
        
        # 配置参数,直接从config.yaml中读取
        enc_cfg = config['model']['encoder']
        dec_cfg = config['model']['decoder']
        
        # 组装Encoder
        self.encoder = ConformerEncoder(
            input_dim=enc_cfg['d_input'],
            d_model=enc_cfg['d_model'],
            n_layers=enc_cfg['n_layers'],
            n_heads=enc_cfg['n_heads'],
            d_ffn=enc_cfg['d_ffn'],
            kernel_size=enc_cfg['cnn_module_kernel'],
            dropout=enc_cfg['dropout']
        )
        
        # 组装Decoder (Transformer)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=enc_cfg['d_model'], 
            n_layers=dec_cfg['n_layers'],
            n_heads=dec_cfg['n_heads'],
            d_ffn=dec_cfg['d_ffn'],
            dropout=dec_cfg['dropout']
        )
        
        # 组装CTC
        #这是一个简单的线性层，将Encoder的输出映射到词表大小
        self.ctc_layer = nn.Linear(enc_cfg['d_model'], vocab_size)

    def forward(self, audio_feat, audio_len, text_tgt=None):
        """
        前向传播逻辑
        audio_feat: 输入音频
        audio_len: 音频长度
        text_tgt: 目标文本，训练时才有，推理时为None
        """
        # 所有音频先经过一遍Encoder
        encoder_out, encoder_len = self.encoder(audio_feat, audio_len)
        
        # CTC,直接听声转字，暂时不关注上下文语境
        ctc_output = self.ctc_layer(encoder_out) 
        
        # Attention分支计算
        decoder_output = None
        if text_tgt is not None:
            # 只有在训练时，我们才跑 Decoder。
            # 推理时 Decoder 的跑法不一样 (一个字一个字蹦出来)，通常在 inference.py 里单独写。

            # 生成目标文本的掩码
            tgt_mask = self.make_pad_mask(text_tgt)
            # 生成Attention Mask (下三角矩阵，防止偷看未来)
            tgt_attn_mask = self.generate_square_subsequent_mask(text_tgt.size(1)).to(audio_feat.device)
            
            decoder_output = self.decoder(
                targets=text_tgt,
                memory=encoder_out,
                target_mask=tgt_attn_mask,
                target_padding_mask=tgt_mask
            )
            
        return ctc_output, encoder_len, decoder_output # 返回三个东西：CTC的预测，新的长度，Decoder的预测

    def generate_square_subsequent_mask(self, sz):
        """生成类似下三角矩阵的 mask，用于自回归预测"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_pad_mask(self, target):
        # 生成填充掩码
        return target == 0 # 假设 0 是 padding id