#conformer 3.0/model/__init__.py
import torch
import torch.nn as nn
from model.conformer import ConformerEncoder
from model.decoder import TransformerDecoder

#模型的顶层封装，包含Squeezeformer Encoder和Transformer Decoder
#config中可决定是否启用Squeezeformer特性
#同时支持CTC和Attention机制
class ConformerASR(nn.Module):
    """
    Conformer ASR 完整模型 (Hybrid CTC/Attention)
    包含原conformer模型基本架构的同时，支持Squeezeformer优化特性
    """
    def __init__(self, config, vocab_size):
        super(ConformerASR, self).__init__()
        
        # 配置参数
        enc_cfg = config['model']['encoder']
        dec_cfg = config['model']['decoder']
        
        # Squeezeformer 配置选项
        self.use_squeezeformer = enc_cfg.get('use_squeezeformer', False) # 是否使用Squeezeformer特性
        self.downsample_ratio = enc_cfg.get('downsample_ratio', 2)  # 下采样比例
        self.downsample_layer = enc_cfg.get('downsample_layer', 7)  # 在第七层后下采样
        
        # Encoder (Conformer 或 Squeezeformer)
        self.encoder = ConformerEncoder(
            input_dim=enc_cfg['d_input'],
            d_model=enc_cfg['d_model'],
            n_layers=enc_cfg['n_layers'],
            n_heads=enc_cfg['n_heads'],
            d_ffn=enc_cfg['d_ffn'],
            kernel_size=enc_cfg['cnn_module_kernel'],
            dropout=enc_cfg['dropout'],
            use_squeezeformer=self.use_squeezeformer,
            downsample_ratio=self.downsample_ratio,
            downsample_layer=self.downsample_layer
        )
        
        # Decoder (Transformer)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=enc_cfg['d_model'],    # Decoder维度通常需与Encoder一致以便注意力机制
            n_layers=dec_cfg['n_layers'],
            n_heads=dec_cfg['n_heads'],
            d_ffn=dec_cfg['d_ffn'],
            dropout=dec_cfg['dropout']
        )
        
        # CTC 输出层 (Encoder 顶部的线性层)
        # 将 Encoder 输出映射到词汇表空间，用于 CTC 损失计算和解码
        self.ctc_layer = nn.Linear(enc_cfg['d_model'], vocab_size)

    def forward(self, audio_feat, audio_len, text_tgt=None):
        """
        前向传播
        audio_feat: 输入音频
        audio_len: 音频长度
        text_tgt: 目标文本 (训练时提供)
        Training 模式下同时计算 CTC 和 Attention 输出
        """

        # 音频特征先通过 Encoder
        encoder_out, encoder_len = self.encoder(audio_feat, audio_len)
        
        # CTC分支输出
        ctc_output = self.ctc_layer(encoder_out) # [batch, time, vocab]
        
        # Attention分支输出
        decoder_output = None
        if text_tgt is not None:      # 生成 Attention Mask (防止看到未来信息)
            # padding mask(tgt_mask): 标记padding位置，避免计算无效位置的损失
            # 形状[batch_size, target_length]
            tgt_mask = self.make_pad_mask(text_tgt) 
            # casual mask(tgt_attn_mask): 下三角矩阵，保证自回归性质，每个位置只能看到之前的信息
            # 形状[target_length, target_length]
            tgt_attn_mask = self.generate_square_subsequent_mask(text_tgt.size(1)).to(audio_feat.device)
            
            decoder_output = self.decoder(   # 这里传入 encoder_out 作为 memory
                targets=text_tgt,
                memory=encoder_out,
                target_mask=tgt_attn_mask,
                target_padding_mask=tgt_mask
            )

        # 返回 CTC 和 Attention 输出    
        return ctc_output, encoder_len, decoder_output

    def generate_square_subsequent_mask(self, sz):
        '''生成causal mask'''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_pad_mask(self, target):
        '''生成 padding mask'''
        # target: [B, L] -> mask: [B, L] (padded的地方是True)
        return target == 0 