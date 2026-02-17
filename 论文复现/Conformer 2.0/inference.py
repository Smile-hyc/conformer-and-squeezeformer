import os
import yaml
import torch
import torchaudio
import argparse
from model import ConformerASR

import torch
torch.backends.cudnn.enabled = False

class ASRPredictor:
    """
    ASR 预测器类 (The Translator)
    封装了模型加载、音频预处理、以及三种解码方式 (CTC, Attention, Hybrid)
    """
    def __init__(self, config_path, checkpoint_path, vocab_path):
        #检查设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"推理运行设备: {self.device}")

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 加载词表
        self.id2char = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                self.id2char[idx] = line.strip()
        
        self.vocab_size = len(self.id2char)
        # 定义关键的特殊符号ID
        self.sos_id = 1
        self.eos_id = 2
        self.ctc_weight = self.config['model'].get('ctc_weight', 0.8)

        # 初始化模型
        self.model = ConformerASR(self.config, self.vocab_size).to(self.device)

        # 加载权重
        print(f"正在加载模型权重: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")

        # map_location 确保在 CPU 上也能加载 GPU 训练的模型  
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        print("模型加载完毕！")

        # 特征提取器
        # 必须和训练时的配置完全一致 (采样率、FFT点数等)，否则听到的声音就是畸变的
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config['data']['sample_rate'],
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length'],
            n_mels=self.config['data']['n_mels']
        ).to(self.device)

    def preprocess_audio(self, wav_path):
        #前处理工序: 读取 wav -> 重采样 -> 转单声道 -> 提取 Mel 谱 -> 归一化
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # 重采样
        if sample_rate != self.config['data']['sample_rate']:
            resampler = torchaudio.transforms.Resample(sample_rate, self.config['data']['sample_rate'])
            waveform = resampler(waveform)
        
        # 转单声道(模型只听一只耳朵，如果是立体声，取平均值混合)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        waveform = waveform.to(self.device)
        
        # 提取特征 (Mel Spectrogram)
        feature = self.mel_transform(waveform)
        
        # 对数压缩和归一化（与训练一致）
        feature = torch.log(feature + 1e-6)
        feature = (feature - feature.mean()) / (feature.std() + 1e-8)
        
        # 维度调整
        feature = feature.transpose(1, 2)
        
        length = torch.tensor([feature.size(1)], device=self.device)
        
        return feature, length

    def recognize_ctc(self, wav_path):
        #使用CTC解码（更稳定）
        features, length = self.preprocess_audio(wav_path)
        
        with torch.no_grad():
            # 关掉梯度计算，省显存，提速
            decoder_input = torch.tensor([[self.sos_id]], device=self.device)
            ctc_out, enc_lens, dec_out = self.model(features, length, decoder_input)
            
            # CTC解码
            ctc_predictions = ctc_out.argmax(dim=-1)[0]
            
            # 移除重复和blank token
            result_ids = []
            prev_id = -1
            for idx in ctc_predictions.cpu().numpy():
                if idx != prev_id and idx != 0:  # 0是blank token
                    result_ids.append(idx)
                prev_id = idx
        
        # ID 转文字
        text_result = "".join([self.id2char.get(idx, "") for idx in result_ids])
        return text_result

    def recognize_attention(self, wav_path):
        #使用Attention解码
        features, length = self.preprocess_audio(wav_path)
        
        with torch.no_grad():
            # 初始化decoder输入 给 Decoder 第一个字 <sos>
            decoder_input = torch.tensor([[self.sos_id]], device=self.device, dtype=torch.long)
            max_len = 100
            result_ids = []
            
            for i in range(max_len):
                # 使用完整模型（与训练一致）
                ctc_out, enc_lens, dec_out = self.model(features, length, decoder_input)
                
                # 取最后一个时间步
                last_token_logits = dec_out[:, -1, :]
                # 贪婪选择概率最大的字
                predicted_id = torch.argmax(last_token_logits, dim=-1).item()
                # 结束条件: 如果预测到了 <eos> ，就停止
                if predicted_id == self.eos_id:
                    break
                    
                result_ids.append(predicted_id)

                # 把预测出来的字拼接到输入后面，作为下一次的输入
                decoder_input = torch.cat([
                    decoder_input, 
                    torch.tensor([[predicted_id]], device=self.device)
                ], dim=1)
                
                # 防止无限循环
                if len(result_ids) >= max_len:
                    break
        
        text_result = "".join([self.id2char.get(idx, "") for idx in result_ids])
        return text_result

    def recognize_hybrid(self, wav_path):
        #混合解码：优先CTC，失败时用Attention
        # 先尝试CTC
        ctc_result = self.recognize_ctc(wav_path)
        
        # 如果CTC结果太短或异常，用Attention
        if len(ctc_result) <= 1 or ctc_result in ["二", "的", "一"]:
            attn_result = self.recognize_attention(wav_path)
            print(f"CTC结果太短({ctc_result})，使用Attention: {attn_result}")
            return attn_result
        
        return ctc_result

def main():
    #命令行参数解析
    parser = argparse.ArgumentParser(description="Conformer ASR 推理脚本")
    parser.add_argument('--wav', type=str, required=True, help="输入的 wav 音频路径")
    parser.add_argument('--config', type=str, default='conf/config.yaml', help="配置文件路径")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help="模型权重路径")
    parser.add_argument('--output', type=str, default='result.txt', help="输出结果的 txt 文件路径")
    parser.add_argument('--mode', type=str, choices=['ctc', 'attention', 'hybrid'], default='hybrid', help="解码模式")
    
    args = parser.parse_args()
    
    # 实例化预测器
    predictor = ASRPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        vocab_path='data/vocabulary.txt'
    )
    
    # 执行识别
    print(f"正在识别音频: {args.wav} ...")
    try:
        if args.mode == 'ctc':
            text = predictor.recognize_ctc(args.wav)
        elif args.mode == 'attention':
            text = predictor.recognize_attention(args.wav)
        else:
            text = predictor.recognize_hybrid(args.wav)
            
        print(f"识别结果: {text}")
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"结果已保存至: {args.output}")
        
    except Exception as e:
        print(f"识别出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()