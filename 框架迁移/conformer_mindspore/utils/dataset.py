import numpy as np
import pandas as pd
import os
import librosa

import mindspore
from mindspore import Tensor
import mindspore.dataset as ds


class AIShellDataset:
    """
    AIShell 数据集加载器 (The Chef)
    负责: 读取文件路径 -> 加载音频/文本 -> 预处理 -> 返回一个样本
    适配 MindSpore GeneratorDataset
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
        
        # 音频参数
        self.sample_rate = config['data']['sample_rate']
        self.n_fft = config['data']['n_fft']
        self.hop_length = config['data']['hop_length']
        self.n_mels = config['data']['n_mels']

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
        返回 numpy 数组，供 MindSpore GeneratorDataset 使用
        """
        try:
            item = self.data.iloc[idx]
            wav_path = item['path']
            text = item['text']
            
            # 使用 librosa 加载音频
            waveform, sample_rate = librosa.load(wav_path, sr=None)
            
            # 重采样
            if sample_rate != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # 提取Mel特征
            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0
            )
            
            # 动态范围压缩
            mel_spec = np.log(mel_spec + 1e-6)
            
            # 转置为 [Time, n_mels]
            feature = mel_spec.T  # [Time, n_mels]
        
            # 特征归一化
            mean = feature.mean(axis=0, keepdims=True)
            std = feature.std(axis=0, keepdims=True)
            feature = (feature - mean) / (std + 1e-8)
            
            # 处理文本 (添加特殊标记)
            target = self.text_to_ids(text)
            
            # 数据验证
            if np.isnan(feature).any() or np.isinf(feature).any():
                print(f"警告: 第{idx}个样本包含NaN或Inf")
                feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            
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
            
            # 返回 numpy 数组和长度信息
            feature = feature.astype(np.float32)
            target = np.array(target, dtype=np.int32)
            feat_len = np.array(feature.shape[0], dtype=np.int32)
            target_len = np.array(len(target), dtype=np.int32)
            
            return feature, feat_len, target, target_len
            
        except Exception as e:
            print(f"处理第{idx}个样本时出错: {e}")
            # 返回下一个有效样本
            return self.__getitem__((idx + 1) % len(self))


def pad_sequence(sequences, padding_value=0):
    """
    将不等长的序列填充到相同长度
    sequences: list of numpy arrays
    返回: padded numpy array [batch, max_len, ...]
    """
    max_len = max(seq.shape[0] for seq in sequences)
    
    # 获取序列的形状（除了第一维）
    if len(sequences[0].shape) > 1:
        padded_shape = (len(sequences), max_len) + sequences[0].shape[1:]
    else:
        padded_shape = (len(sequences), max_len)
    
    padded = np.full(padded_shape, padding_value, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0]] = seq
    
    return padded


def collate_fn(batch):
    """
    自定义的 Batch 整理函数 
    Dataset 返回的是单个样本，这里要把它们打包成一个 Batch。
    """
    # 过滤掉无效样本
    batch = [item for item in batch if item[0].shape[0] > 0 and item[2].shape[0] > 0]
    
    if len(batch) == 0:
        # 返回一个最小的有效batch
        dummy_feature = np.zeros((1, 100, 80), dtype=np.float32)
        dummy_target = np.array([[1, 2]], dtype=np.int32)  # [sos, eos]
        return dummy_feature, np.array([100], dtype=np.int32), dummy_target, np.array([2], dtype=np.int32)
    
    features = [item[0] for item in batch]
    feat_lengths = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    target_lengths = [item[3] for item in batch]
    
    # 记录原始长度
    feat_lengths = np.array(feat_lengths, dtype=np.int32)
    target_lengths = np.array(target_lengths, dtype=np.int32)
    
    # 填充序列
    padded_features = pad_sequence(features, padding_value=0)
    padded_targets = pad_sequence(targets, padding_value=0)
    
    # 最终检查
    if np.isnan(padded_features).any() or np.isinf(padded_features).any():
        print("警告: Batch中包含NaN或Inf值")
        padded_features = np.nan_to_num(padded_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 返回: (补齐后的音频, 真实音频长度, 补齐后的文本, 真实文本长度)
    return padded_features, feat_lengths, padded_targets, target_lengths


def create_aishell_dataset(manifest_path, vocab_path, config, batch_size, shuffle=True, num_workers=4):
    """
    创建 MindSpore 数据集
    """
    # 创建数据集实例
    aishell_dataset = AIShellDataset(manifest_path, vocab_path, config)
    
    # 使用 GeneratorDataset 包装
    dataset = ds.GeneratorDataset(
        source=aishell_dataset,
        column_names=["feats", "feat_lens", "targets", "target_lens"],
        shuffle=shuffle,
        num_parallel_workers=num_workers
    )
    
    # 设置 batch（使用自定义的 batch 函数）
    dataset = dataset.batch(
        batch_size, 
        drop_remainder=True,
        per_batch_map=batch_map_fn,
        input_columns=["feats", "feat_lens", "targets", "target_lens"],
        output_columns=["feats", "feat_lens", "targets", "target_lens"]
    )
    
    return dataset


def batch_map_fn(feats, feat_lens, targets, target_lens, batch_info):
    """
    MindSpore batch map 函数，用于填充序列
    """
    # feats: list of arrays with shape [T_i, n_mels]
    # targets: list of arrays with shape [S_i]
    
    batch_size = len(feats)
    
    # 获取最大长度
    max_feat_len = max(f.shape[0] for f in feats)
    max_target_len = max(t.shape[0] for t in targets)
    n_mels = feats[0].shape[1]
    
    # 创建填充后的数组
    padded_feats = np.zeros((batch_size, max_feat_len, n_mels), dtype=np.float32)
    padded_targets = np.zeros((batch_size, max_target_len), dtype=np.int32)
    
    # 填充
    for i in range(batch_size):
        feat_len = feats[i].shape[0]
        target_len = targets[i].shape[0]
        
        padded_feats[i, :feat_len, :] = feats[i]
        padded_targets[i, :target_len] = targets[i]
    
    # 转换长度为数组
    feat_lens_arr = np.array(feat_lens, dtype=np.int32)
    target_lens_arr = np.array(target_lens, dtype=np.int32)
    
    return padded_feats, feat_lens_arr, padded_targets, target_lens_arr


class AIShellDatasetIterable:
    """
    可迭代版本的数据集，用于更简单的 GeneratorDataset 使用
    """
    def __init__(self, manifest_path, vocab_path, config):
        self.dataset = AIShellDataset(manifest_path, vocab_path, config)
        self.char2id = self.dataset.char2id
        
    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self.dataset[i]
    
    def __len__(self):
        return len(self.dataset)


def create_aishell_dataset_simple(manifest_path, vocab_path, config, batch_size, shuffle=True, num_workers=4):
    """
    创建 MindSpore 数据集 (简化版，使用固定长度填充)
    """
    # 创建可迭代数据集实例
    aishell_dataset = AIShellDatasetIterable(manifest_path, vocab_path, config)
    
    # 使用 GeneratorDataset 包装
    dataset = ds.GeneratorDataset(
        source=aishell_dataset,
        column_names=["feats", "feat_lens", "targets", "target_lens"],
        shuffle=shuffle,
        num_parallel_workers=num_workers
    )
    
    # 填充到固定长度
    max_feat_len = 1500  # 最大音频帧数
    max_target_len = 50  # 最大文本长度
    n_mels = config['data']['n_mels']
    
    # 定义 pad 操作
    pad_feat_op = ds.transforms.PadEnd(pad_shape=[max_feat_len, n_mels], pad_value=0.0)
    pad_target_op = ds.transforms.PadEnd(pad_shape=[max_target_len], pad_value=0)
    
    dataset = dataset.map(operations=pad_feat_op, input_columns=["feats"])
    dataset = dataset.map(operations=pad_target_op, input_columns=["targets"])
    
    # 设置 batch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset, aishell_dataset.char2id