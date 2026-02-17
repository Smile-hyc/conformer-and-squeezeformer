import os
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import mindspore
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore import load_checkpoint, load_param_into_net
from model import ConformerASR

# 音频处理库
import librosa

# 设置运行环境 (华为云Ascend)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

# 测试集验证命令：python test_dataset.py --checkpoint checkpoints/best_model.ckpt --sample_count 50


class DatasetTester:
    def __init__(self, config_path, checkpoint_path, vocab_path):
        print(f"测试运行设备: Ascend NPU")

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 加载词表
        self.id2char = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                self.id2char[idx] = line.strip()
        
        self.vocab_size = len(self.id2char)
        self.sos_id = 1
        self.eos_id = 2

        # 初始化模型
        self.model = ConformerASR(self.config, self.vocab_size)

        # 加载权重
        print(f"正在加载模型权重: {checkpoint_path}")
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(self.model, param_dict)
            
        self.model.set_train(False)
        print("模型加载完毕！")

        # 音频参数
        self.sample_rate = self.config['data']['sample_rate']
        self.n_fft = self.config['data']['n_fft']
        self.hop_length = self.config['data']['hop_length']
        self.n_mels = self.config['data']['n_mels']
        
        # 操作符
        self.argmax_op = ops.Argmax(axis=-1)

    def fix_audio_path(self, audio_path):
        """修复音频文件路径"""
        # 移除可能的多余前缀
        if audio_path.startswith('./data_aishell/./data_aishell/'):
            audio_path = audio_path.replace('./data_aishell/./data_aishell/', 'data_aishell/')
        
        # 检查文件是否存在
        if os.path.exists(audio_path):
            return audio_path
        
        # 尝试其他可能的路径
        possible_paths = [
            audio_path,
            audio_path.replace('./data_aishell/', 'data_aishell/'),
            audio_path.replace('data_aishell/', './data_aishell/'),
            audio_path.replace('./', ''),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"路径修复: {audio_path} -> {path}")
                return path
        
        # 如果都不存在，返回原始路径（让后续代码报错）
        return audio_path

    def preprocess_audio(self, wav_path):
        """读取音频并转换为 Mel 谱图"""
        # 先修复路径
        wav_path = self.fix_audio_path(wav_path)
        
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
        
        # 对数压缩和归一化
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

    def recognize_ctc(self, wav_path):
        """使用CTC解码进行识别"""
        features, length = self.preprocess_audio(wav_path)
        
        # 完整模型前向
        decoder_input = Tensor([[self.sos_id]], mindspore.int32)
        ctc_out, enc_lens, dec_out = self.model(features, length, decoder_input)
        
        # CTC解码
        ctc_predictions = self.argmax_op(ctc_out)[0]
        
        # 移除重复和blank token
        result_ids = []
        prev_id = -1
        predictions_np = ctc_predictions.asnumpy()
        for idx in predictions_np:
            if idx != prev_id and idx != 0:  # 0是blank token
                # 过滤特殊标记
                if idx not in [self.sos_id, self.eos_id]:
                    result_ids.append(int(idx))
            prev_id = idx
        
        # ID 转文字
        text_result = "".join([self.id2char.get(idx, "") for idx in result_ids])
        return text_result

    def calculate_cer(self, pred_text, true_text):
        """计算字错误率 (Character Error Rate) - 忽略空格版本"""
        # 移除所有空格
        pred_chars = list(pred_text.replace(' ', ''))
        true_chars = list(true_text.replace(' ', ''))

        n, m = len(pred_chars), len(true_chars)

        # 处理边界情况
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
                    dp[i][j] = min(dp[i-1][j] + 1,     # 删除
                                   dp[i][j-1] + 1,     # 插入
                                   dp[i-1][j-1] + 1)   # 替换
    
        edit_distance = dp[n][m]
        cer = edit_distance / max(m, 1)  # 使用无空格的真实文本长度
    
        return cer, edit_distance, m

    def test_dataset(self, test_csv, output_csv=None, sample_count=None):
        """测试整个数据集"""
        # 读取测试数据
        test_df = pd.read_csv(test_csv)
        
        if sample_count:
            test_df = test_df.head(sample_count)
            print(f"测试前 {sample_count} 个样本...")
        
        results = []
        total_cer = 0
        total_edit_distance = 0
        total_chars = 0
        failed_count = 0
        
        print(f"开始测试 {len(test_df)} 个音频...")
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            try:
                audio_path = row['path']
                true_text = row['text']
                
                # 修复路径
                fixed_path = self.fix_audio_path(audio_path)
                
                if not os.path.exists(fixed_path):
                    print(f"文件不存在: {fixed_path}")
                    failed_count += 1
                    continue
                
                # 识别
                pred_text = self.recognize_ctc(fixed_path)
                
                # 计算CER
                cer, edit_distance, num_chars = self.calculate_cer(pred_text, true_text)
                
                # 保存结果
                result = {
                    'audio_path': fixed_path,
                    'true_text': true_text,
                    'pred_text': pred_text,
                    'cer': cer,
                    'edit_distance': edit_distance,
                    'num_chars': num_chars
                }
                results.append(result)
                
                # 累计统计
                total_cer += cer
                total_edit_distance += edit_distance
                total_chars += num_chars
                
                # 每50个样本打印一次进度
                if (idx + 1) % 50 == 0:
                    if results:  # 防止除零
                        avg_cer = total_cer / len(results)
                        print(f"已测试 {idx + 1} 个样本, 成功: {len(results)}, 失败: {failed_count}, 平均CER: {avg_cer:.4f}")
                    
            except Exception as e:
                print(f"测试失败 {audio_path}: {e}")
                failed_count += 1
                continue
        
        if not results:
            print("所有测试均失败，无法计算统计信息。")
            return [], 0, 0
        
        # 计算总体统计
        avg_cer = total_cer / len(results)
        overall_cer = total_edit_distance / total_chars if total_chars > 0 else 0
        
        print(f"\n测试完成!")
        print(f"总体统计:")
        print(f"   总样本数: {len(test_df)}")
        print(f"   成功测试: {len(results)}")
        print(f"   测试失败: {failed_count}")
        print(f"   平均CER: {avg_cer:.4f}")
        print(f"   整体CER: {overall_cer:.4f}")
        print(f"   总编辑距离: {total_edit_distance}")
        print(f"   总字符数: {total_chars}")
        
        # 保存结果到CSV
        if output_csv:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"详细结果已保存至: {output_csv}")
        
        # 打印一些样本对比
        print(f"\n样本对比 (前5个):")
        for i, result in enumerate(results[:5]):
            print(f"样本 {i+1}:")
            print(f"  真实: {result['true_text']}")
            print(f"  预测: {result['pred_text']}")
            print(f"  CER: {result['cer']:.4f}")
            print()
        
        return results, avg_cer, overall_cer


def main():
    parser = argparse.ArgumentParser(description="Conformer ASR 数据集测试脚本 (MindSpore)")
    parser.add_argument('--config', type=str, default='conf/config.yaml', help="配置文件路径")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.ckpt', help="模型权重路径")
    parser.add_argument('--test_csv', type=str, default='data/test.csv', help="测试集CSV路径")
    parser.add_argument('--output', type=str, default='test_results.csv', help="输出结果CSV路径")
    parser.add_argument('--sample_count', type=int, default=None, help="测试样本数量（None表示全部）")
    
    args = parser.parse_args()
    
    # 实例化测试器
    tester = DatasetTester(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        vocab_path='data/vocabulary.txt'
    )
    
    # 执行测试
    results, avg_cer, overall_cer = tester.test_dataset(
        test_csv=args.test_csv,
        output_csv=args.output,
        sample_count=args.sample_count
    )


if __name__ == "__main__":
    main()