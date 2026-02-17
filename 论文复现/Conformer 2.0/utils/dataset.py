import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import os

class AIShellDataset(torch.utils.data.Dataset):
    """
    AIShell 数据集加载器 (The Chef)
    负责: 读取文件路径 -> 加载音频/文本 -> 预处理 -> 返回一个样本
    """
    def __init__(self, manifest_path, vocab_path, config):
        self.config = config
        
        # 加载数据清单
        self.data = pd.read_csv(manifest_path)
        print(f"加载数据集，样本数: {len(self.data)}")
        
        # 加载词表 - 修正特殊标记处理
        self.char2id = {}
        
        # 首先添加特殊标记
        # 必须保证这些标记的 ID 是固定的 (0, 1, 2, 3)，
        # 因为 config.yaml 和 model 代码里通常默认 pad=0, sos=1
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        for idx, token in enumerate(special_tokens):
            self.char2id[token] = idx
        
        # 加载词汇表中的字符
        with open(vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                char = line.strip()
                if char not in self.char2id:  # 避免重复
                    self.char2id[char] = len(self.char2id)
        
        # 特殊 Token ID
        self.pad_id = self.char2id['<pad>']  # 0
        self.sos_id = self.char2id['<sos>']  # 1
        self.eos_id = self.char2id['<eos>']  # 2
        self.unk_id = self.char2id['<unk>']  # 3
        
        print(f"词汇表大小: {len(self.char2id)}")
        print(f"特殊标记: PAD={self.pad_id}, SOS={self.sos_id}, EOS={self.eos_id}, UNK={self.unk_id}")
        
        # 3. 音频特征提取器
        # 使用 torchaudio 的 MelSpectrogram 算子
        # 这里的参数必须和 inference.py 完全一致
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['data']['sample_rate'],
            n_fft=config['data']['n_fft'],
            hop_length=config['data']['hop_length'],
            n_mels=config['data']['n_mels'],
            power=2.0
        )

    def __len__(self):
        return len(self.data)

    def text_to_ids(self, text):
        """
        文本数字化: "我爱北京" -> [1, 5, 20, 33, 2]
        这里做了非常重要的一步：加头加尾
        """
        ids = []
        text = text.replace(' ', '').strip()
        
        # 添加 <sos> 标记
        ids.append(self.sos_id)
        
        # 转换字符
        for char in text:
            if char in self.char2id:
                ids.append(self.char2id[char])
            else:
                ids.append(self.unk_id)
        
        # 添加 <eos> 标记
        ids.append(self.eos_id)
        
        return ids

    def __getitem__(self, idx):
        """
        获取第 idx 个样本 
        注意：这里加了异常处理，防止因为一个坏文件导致整个训练崩溃
        """
        try:
            item = self.data.iloc[idx]
            wav_path = item['path']
            text = item['text']
            
            # 加载音频
            waveform, sample_rate = torchaudio.load(wav_path)
            
            # 确保单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 提取Mel特征
            mel_spec = self.mel_transform(waveform)
            
            # 动态范围压缩
            mel_spec = torch.log(mel_spec + 1e-6)
            
            # 转置为 [Time, n_mels]
            feature = mel_spec.squeeze(0).transpose(0, 1)
        
            # 特征归一化
            mean = feature.mean(dim=0, keepdim=True)
            std = feature.std(dim=0, keepdim=True)
            feature = (feature - mean) / (std + 1e-8)
            
            # 3. 处理文本 (添加特殊标记)
            target = self.text_to_ids(text)
            
            # 数据验证
            if torch.isnan(feature).any() or torch.isinf(feature).any():
                print(f"警告: 第{idx}个样本包含NaN或Inf")
                feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 过滤异常样本
            if len(target) > 50 or len(target) < 3:  # 太短或太长
                print(f"跳过异常文本样本 {idx}: 长度 {len(target)}")
                # 返回下一个样本
                return self.__getitem__((idx + 1) % len(self))
                
            if feature.shape[0] > 1500 or feature.shape[0] < 100:  # 音频太长或太短
                print(f"跳过异常音频样本 {idx}: 长度 {feature.shape[0]}")
                return self.__getitem__((idx + 1) % len(self))
            
            # 调试信息
            if idx < 3:
                print(f"样本 {idx}: 特征 {feature.shape}, 目标长度 {len(target)}, 文本: {text}")
            
            return feature, torch.tensor(target, dtype=torch.long)
            
        except Exception as e:
            print(f"处理第{idx}个样本时出错: {e}")
            # 返回下一个有效样本
            return self.__getitem__((idx + 1) % len(self))

def collate_fn(batch):
    """
    自定义的 Batch 整理函数 
    Dataset 返回的是单个样本，这里要把它们打包成一个 Batch。
    """
    # 过滤掉无效样本
    batch = [item for item in batch if item[0].shape[0] > 0 and item[1].shape[0] > 0]
    
    if len(batch) == 0:
        # 返回一个最小的有效batch
        dummy_feature = torch.zeros(1, 100, 80)
        dummy_target = torch.tensor([[1, 2]], dtype=torch.long)  # [sos, eos]
        return dummy_feature, torch.tensor([100]), dummy_target, torch.tensor([2])
    
    features = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # 记录原始长度
    feat_lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    
    # 填充序列
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    # 最终检查
    if torch.isnan(padded_features).any() or torch.isinf(padded_features).any():
        print("警告: Batch中包含NaN或Inf值")
        padded_features = torch.nan_to_num(padded_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 返回: (补齐后的音频, 真实音频长度, 补齐后的文本, 真实文本长度)
    return padded_features, feat_lengths, padded_targets, target_lengths