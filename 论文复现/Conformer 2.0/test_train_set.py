# test_train_set.py
import os
import yaml
import torch
import torchaudio
import argparse
import pandas as pd
from tqdm import tqdm
from model import ConformerASR

import torch
torch.backends.cudnn.enabled = False

class TrainSetTesterNoSpace:
    def __init__(self, config_path, checkpoint_path, vocab_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 测试运行设备: {self.device}")

        # 1. 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 2. 加载词表
        self.id2char = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                self.id2char[idx] = line.strip()
        
        self.vocab_size = len(self.id2char)
        self.sos_id = 1
        self.eos_id = 2

        # 3. 初始化模型
        self.model = ConformerASR(self.config, self.vocab_size).to(self.device)

        # 4. 加载权重
        print(f"🔄 正在加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        print("✅ 模型加载完毕！")

        # 5. 特征提取器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config['data']['sample_rate'],
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length'],
            n_mels=self.config['data']['n_mels']
        ).to(self.device)

    def fix_audio_path(self, audio_path):
        """修复音频文件路径"""
        if audio_path.startswith('./data_aishell/./data_aishell/'):
            audio_path = audio_path.replace('./data_aishell/./data_aishell/', 'data_aishell/')
        
        if os.path.exists(audio_path):
            return audio_path
        
        possible_paths = [
            audio_path,
            audio_path.replace('./data_aishell/', 'data_aishell/'),
            audio_path.replace('data_aishell/', './data_aishell/'),
            audio_path.replace('./', ''),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return audio_path

    def preprocess_audio(self, wav_path):
        """读取音频并转换为 Mel 谱图"""
        wav_path = self.fix_audio_path(wav_path)
        
        waveform, sample_rate = torchaudio.load(wav_path)
        
        if sample_rate != self.config['data']['sample_rate']:
            resampler = torchaudio.transforms.Resample(sample_rate, self.config['data']['sample_rate'])
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        waveform = waveform.to(self.device)
        
        feature = self.mel_transform(waveform)
        feature = torch.log(feature + 1e-6)
        feature = (feature - feature.mean()) / (feature.std() + 1e-8)
        feature = feature.transpose(1, 2)
        
        length = torch.tensor([feature.size(1)], device=self.device)
        
        return feature, length

    def recognize_ctc(self, wav_path):
        """使用CTC解码进行识别"""
        features, length = self.preprocess_audio(wav_path)
        
        with torch.no_grad():
            decoder_input = torch.tensor([[self.sos_id]], device=self.device)
            ctc_out, enc_lens, dec_out = self.model(features, length, decoder_input)
            
            ctc_predictions = ctc_out.argmax(dim=-1)[0]
            
            result_ids = []
            prev_id = -1
            for idx in ctc_predictions.cpu().numpy():
                if idx != prev_id and idx != 0:
                    if idx not in [self.sos_id, self.eos_id]:
                        result_ids.append(idx)
                prev_id = idx
        
        text_result = "".join([self.id2char.get(idx, "") for idx in result_ids])
        return text_result

    def calculate_cer_no_space(self, pred_text, true_text):
        """计算字错误率 - 忽略空格"""
        import numpy as np
        
        # 移除所有空格
        pred_chars = list(pred_text.replace(' ', ''))
        true_chars = list(true_text.replace(' ', ''))
        
        n, m = len(pred_chars), len(true_chars)
        
        # 如果预测为空，返回最大错误率
        if n == 0 and m == 0:
            return 0.0, 0, 0
        elif n == 0:
            return 1.0, m, m
        elif m == 0:
            return 1.0, n, n
        
        dp = np.zeros((n+1, m+1))
        
        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j
            
        for i in range(1, n+1):
            for j in range(1, m+1):
                if pred_chars[i-1] == true_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j] + 1,   # 删除
                                 dp[i][j-1] + 1,   # 插入
                                 dp[i-1][j-1] + 1) # 替换
        
        edit_distance = dp[n][m]
        cer = edit_distance / max(m, 1)  # 使用真实文本的无空格长度
        
        return cer, edit_distance, m

    def test_train_set(self, train_csv, output_csv=None, sample_count=1000):
        """测试训练集 - 忽略空格"""
        train_df = pd.read_csv(train_csv)
        
        if sample_count:
            train_df = train_df.sample(n=min(sample_count, len(train_df)), random_state=42)
            print(f"🧪 随机测试 {len(train_df)} 个训练样本 (忽略空格)...")
        
        results = []
        total_cer = 0
        total_edit_distance = 0
        total_chars = 0
        failed_count = 0
        
        print(f"🔍 开始测试 {len(train_df)} 个训练音频...")
        
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
            try:
                audio_path = row['path']
                true_text = row['text']
                
                fixed_path = self.fix_audio_path(audio_path)
                
                if not os.path.exists(fixed_path):
                    failed_count += 1
                    continue
                
                pred_text = self.recognize_ctc(fixed_path)
                
                # 使用忽略空格的CER计算
                cer, edit_distance, num_chars = self.calculate_cer_no_space(pred_text, true_text)
                
                result = {
                    'audio_path': fixed_path,
                    'true_text': true_text,
                    'true_text_no_space': true_text.replace(' ', ''),
                    'pred_text': pred_text,
                    'cer': cer,
                    'edit_distance': edit_distance,
                    'num_chars': num_chars
                }
                results.append(result)
                
                total_cer += cer
                total_edit_distance += edit_distance
                total_chars += num_chars
                
            except Exception as e:
                failed_count += 1
                continue
        
        if not results:
            print("❌ 所有测试都失败了！")
            return [], 0, 0
        
        avg_cer = total_cer / len(results)
        overall_cer = total_edit_distance / total_chars if total_chars > 0 else 0
        
        print(f"\n🎯 训练集测试完成! (忽略空格)")
        print(f"📊 总体统计:")
        print(f"   测试样本数: {len(results)}")
        print(f"   测试失败: {failed_count}")
        print(f"   平均CER: {avg_cer:.4f}")
        print(f"   整体CER: {overall_cer:.4f}")
        print(f"   总编辑距离: {total_edit_distance}")
        print(f"   总字符数: {total_chars} (无空格)")
        
        # 按CER排序
        results_sorted = sorted(results, key=lambda x: x['cer'])
        
        print(f"\n🏆 CER最低的5个样本:")
        for i, result in enumerate(results_sorted[:5]):
            print(f"  CER {result['cer']:.4f}:")
            print(f"    真实: '{result['true_text']}'")
            print(f"    预测: '{result['pred_text']}'")
            print(f"    无空格真实: '{result['true_text_no_space']}'")
        
        print(f"\n❌ CER最高的5个样本:")
        for i, result in enumerate(results_sorted[-5:]):
            print(f"  CER {result['cer']:.4f}:")
            print(f"    真实: '{result['true_text']}'")
            print(f"    预测: '{result['pred_text']}'")
            print(f"    无空格真实: '{result['true_text_no_space']}'")
        
        # CER分布统计
        cer_ranges = {
            'CER < 0.1': 0,
            '0.1 ≤ CER < 0.3': 0,
            '0.3 ≤ CER < 0.5': 0,
            '0.5 ≤ CER < 0.7': 0,
            '0.7 ≤ CER < 0.9': 0,
            'CER ≥ 0.9': 0
        }
        
        for result in results:
            cer = result['cer']
            if cer < 0.1:
                cer_ranges['CER < 0.1'] += 1
            elif cer < 0.3:
                cer_ranges['0.1 ≤ CER < 0.3'] += 1
            elif cer < 0.5:
                cer_ranges['0.3 ≤ CER < 0.5'] += 1
            elif cer < 0.7:
                cer_ranges['0.5 ≤ CER < 0.7'] += 1
            elif cer < 0.9:
                cer_ranges['0.7 ≤ CER < 0.9'] += 1
            else:
                cer_ranges['CER ≥ 0.9'] += 1
        
        print(f"\n📈 CER分布:")
        for range_name, count in cer_ranges.items():
            percentage = count / len(results) * 100
            print(f"   {range_name}: {count} 个样本 ({percentage:.1f}%)")
        
        if output_csv:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"💾 详细结果已保存至: {output_csv}")
        
        return results, avg_cer, overall_cer

def main():
    parser = argparse.ArgumentParser(description="Conformer ASR 训练集测试脚本 (忽略空格)")
    parser.add_argument('--config', type=str, default='conf/config.yaml', help="配置文件路径")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help="模型权重路径")
    parser.add_argument('--train_csv', type=str, default='data/train.csv', help="训练集CSV路径")
    parser.add_argument('--output', type=str, default='train_test_results_no_space.csv', help="输出结果CSV路径")
    parser.add_argument('--sample_count', type=int, default=1000, help="测试样本数量")
    
    args = parser.parse_args()
    
    tester = TrainSetTesterNoSpace(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        vocab_path='data/vocabulary.txt'
    )
    
    results, avg_cer, overall_cer = tester.test_train_set(
        train_csv=args.train_csv,
        output_csv=args.output,
        sample_count=args.sample_count
    )

if __name__ == "__main__":
    main()