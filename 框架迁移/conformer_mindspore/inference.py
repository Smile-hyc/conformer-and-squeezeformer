import os
import yaml
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore import load_checkpoint, load_param_into_net
from model import ConformerASR

# 音频处理库
import scipy.io.wavfile as wav
import librosa

# 测试运行命令：python inference.py --wav "data_aishell/wav/test/S0764/BAC009S0764W0121.wav"

class ASRPredictor:
    """
    ASR 预测器类 (The Translator)
    封装了模型加载、音频预处理、以及三种解码方式 (CTC, Attention, Hybrid)
    """
    def __init__(self, config_path, checkpoint_path, vocab_path):
        # 设置运行环境 (华为云Ascend)
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        print(f"推理运行设备: Ascend NPU")

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 加载词表
        self.char2id = {}
        self.id2char = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                char = line.strip()
                self.char2id[char] = idx
                self.id2char[idx] = char
        
        self.vocab_size = len(self.id2char)
        self.sos_id = self.char2id.get('<sos>', 1)      # 定义特殊符号ID
        self.eos_id = self.char2id.get('<eos>', 2)
        self.blank_id = self.char2id.get('<blank>', 0)  # CTC 空白字符的ID
        self.ctc_weight = self.config['model'].get('ctc_weight', 0.8)  # CTC权重

        # 初始化模型
        self.model = ConformerASR(self.config, self.vocab_size)

        # 加载权重
        print(f"正在加载模型权重: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
        
        # MindSpore 加载checkpoint
        param_dict = load_checkpoint(checkpoint_path)
        
        # 加载权重到模型
        not_loaded, _ = load_param_into_net(self.model, param_dict, strict_load=False)
        if not_loaded:
            print(f"以下参数未加载: {not_loaded[:5]}...")
            
        self.model.set_train(False)
        print("模型加载完毕！")

        # 音频参数
        self.sample_rate = self.config['data']['sample_rate']
        self.n_fft = self.config['data']['n_fft']
        self.hop_length = self.config['data']['hop_length']
        self.n_mels = self.config['data']['n_mels']

    def preprocess_audio(self, wav_path):
        """读取音频并转换为 Mel 谱图"""
        try:
            # 使用librosa读取音频
            waveform, sample_rate = librosa.load(wav_path, sr=None)
            
            # 重采样
            if sample_rate != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # 提取Mel频谱特征
            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # 对数压缩和归一化（与训练一致）
            feature = np.log(mel_spec + 1e-6)
            mean = feature.mean()
            std = feature.std()
            feature = (feature - mean) / (std + 1e-8)
            
            # [n_mels, time] -> [1, time, n_mels]
            feature = feature.T  # [time, n_mels]
            feature = np.expand_dims(feature, axis=0)  # [1, time, n_mels]
            
            # 转换为MindSpore Tensor
            feature = Tensor(feature, mindspore.float32)
            length = Tensor([feature.shape[1]], mindspore.int32)
            
            return feature, length
            
        except Exception as e:
            raise Exception(f"音频预处理失败: {e}")

    def ctc_greedy_decode(self, ctc_logits):
        """CTC贪心解码"""
        # ctc_logits: [1, T, vocab_size]
        argmax_op = ops.Argmax(axis=-1)
        predictions = argmax_op(ctc_logits)[0]  # [T]
        
        # 移除重复和blank token
        result_ids = []
        prev_id = -1
        predictions_np = predictions.asnumpy()
        for idx in predictions_np:
            if idx != prev_id and idx != self.blank_id:
                result_ids.append(int(idx))
            prev_id = idx
        
        return result_ids

    def recognize_ctc(self, wav_path):
        """使用CTC解码"""
        features, length = self.preprocess_audio(wav_path)
        
        # 只使用encoder部分进行CTC解码
        encoder_out, encoder_len = self.model.encoder(features, length)
        ctc_output = self.model.ctc_layer(encoder_out)
        
        # CTC解码
        result_ids = self.ctc_greedy_decode(ctc_output)
        
        # ID 转文字
        text_result = "".join([self.id2char.get(idx, "") for idx in result_ids])
        return text_result

    def recognize_attention(self, wav_path, max_len=50):
        """使用Attention解码"""
        features, length = self.preprocess_audio(wav_path)
        
        # 初始化decoder输入
        decoder_input = Tensor([[self.sos_id]], mindspore.int32)
        result_ids = []
        
        argmax_op = ops.Argmax(axis=-1)
        concat_op = ops.Concat(axis=1)
        
        for i in range(max_len):
            # 前向传播
            ctc_out, enc_lens, dec_out = self.model(features, length, decoder_input)
            
            # 取最后一个时间步
            last_token_logits = dec_out[:, -1, :]
            predicted_id = int(argmax_op(last_token_logits).asnumpy()[0])
            
            if predicted_id == self.eos_id:
                break
                
            result_ids.append(predicted_id)
            new_token = Tensor([[predicted_id]], mindspore.int32)
            decoder_input = concat_op((decoder_input, new_token))
        
        text_result = "".join([self.id2char.get(idx, "") for idx in result_ids])
        return text_result

    def recognize_hybrid(self, wav_path):
        """混合解码：CTC + Attention重打分"""
        # 先尝试CTC
        ctc_result = self.recognize_ctc(wav_path)
        
        # 如果CTC结果合理，直接返回
        if len(ctc_result) >= 2 and len(ctc_result) <= 30:
            return ctc_result
        else:
            # CTC结果不理想，使用Attention
            attn_result = self.recognize_attention(wav_path)
            print(f"CTC结果异常(长度:{len(ctc_result)})，使用Attention解码")
            return attn_result

    def recognize_batch(self, wav_paths, mode='hybrid'):
        """批量识别"""
        results = []
        for wav_path in wav_paths:
            try:
                if mode == 'ctc':
                    text = self.recognize_ctc(wav_path)
                elif mode == 'attention':
                    text = self.recognize_attention(wav_path)
                else:
                    text = self.recognize_hybrid(wav_path)
                results.append((wav_path, text))
                print(f"{os.path.basename(wav_path)}: {text}")
            except Exception as e:
                print(f"{os.path.basename(wav_path)}: 识别失败 - {e}")
                results.append((wav_path, ""))
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Conformer ASR 推理脚本 (MindSpore)")
    parser.add_argument('--wav', type=str, required=True, help="输入的 wav 音频路径")
    parser.add_argument('--config', type=str, default='conf/config.yaml', help="配置文件路径")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.ckpt', help="模型权重路径")
    parser.add_argument('--vocab', type=str, default='data/vocabulary.txt', help="词表路径")
    parser.add_argument('--output', type=str, default='result.txt', help="输出结果的 txt 文件路径")
    parser.add_argument('--mode', type=str, choices=['ctc', 'attention', 'hybrid'], default='hybrid', help="解码模式")
    parser.add_argument('--batch', action='store_true', help="批量处理模式（wav参数改为目录）")
    
    args = parser.parse_args()
    
    # 实例化预测器
    predictor = ASRPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab
    )
    
    try:
        if args.batch:
            # 批量处理模式
            if os.path.isdir(args.wav):
                wav_files = [os.path.join(args.wav, f) for f in os.listdir(args.wav) if f.endswith('.wav')]
                print(f"批量处理 {len(wav_files)} 个音频文件...")
                results = predictor.recognize_batch(wav_files, args.mode)
                
                # 保存结果
                with open(args.output, 'w', encoding='utf-8') as f:
                    for wav_path, text in results:
                        f.write(f"{os.path.basename(wav_path)}\t{text}\n")
                print(f"批量结果已保存至: {args.output}")
            else:
                print("批量模式需要提供目录路径")
        else:
            # 单文件模式
            print(f"正在识别音频: {args.wav} ...")
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