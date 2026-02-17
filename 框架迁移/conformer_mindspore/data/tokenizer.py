# 生成词汇表的脚本
import os
import pandas as pd
import yaml
from collections import Counter


def build_vocabulary(config_path):
    """
    从训练集文本中提取所有唯一字符，构建词汇表
    """
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    train_csv = config['data']['train_manifest']
    vocab_path = config['data']['vocab_path']
    
    print("正在构建词表 (Vocabulary)...")
    print(f"训练数据: {train_csv}")
    
    # 检查训练集是否存在
    if not os.path.exists(train_csv):
        print(f"错误: 训练集文件不存在: {train_csv}")
        print("请先运行 aishell_data_process.py 生成数据清单")
        return
    
    # 读取训练集文本
    df = pd.read_csv(train_csv)
    all_text = df['text'].tolist()
    
    print(f"读取了 {len(all_text)} 条训练文本")
    
    # 统计字符及其频率
    char_counter = Counter()
    unique_chars = set()
    
    for text in all_text:
        # 去除空格，AIShell文本中间可能有空格
        text = text.replace(" ", "") 
        for char in text:
            unique_chars.add(char)
            char_counter[char] += 1
            
    print(f"训练集中共有 {len(unique_chars)} 个唯一字符。")
    
    # 显示最常见的字符
    most_common = char_counter.most_common(10)
    print("\n最常见的10个字符:")
    for char, count in most_common:
        print(f"  '{char}': {count} 次")
    
    # 添加特殊符号
    # <pad>: 填充符 (ID 0)
    # <sos>: 句子开始 (ID 1)
    # <eos>: 句子结束 (ID 2)
    # <unk>: 未知字符 (ID 3)
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    
    sorted_chars = sorted(list(unique_chars))
    final_vocab = special_tokens + sorted_chars
    
    # 确保输出目录存在
    vocab_dir = os.path.dirname(vocab_path)
    if vocab_dir and not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
        print(f"创建输出目录: {vocab_dir}")
    
    # 保存到 vocabulary.txt
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for char in final_vocab:
            f.write(char + '\n')
            
    print(f"\n词表已保存至: {vocab_path}")
    print(f"总词表大小: {len(final_vocab)}")
    print(f"  - 特殊标记: {len(special_tokens)}")
    print(f"  - 普通字符: {len(sorted_chars)}")
    
    # 验证词表
    verify_vocabulary(vocab_path)


def verify_vocabulary(vocab_path):
    """
    验证词汇表是否正确生成
    """
    print("\n=== 验证词汇表 ===")
    
    if not os.path.exists(vocab_path):
        print(f"✗ 词表文件不存在: {vocab_path}")
        return
    
    # 读取词表
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip())
    
    print(f"✓ 词表文件存在")
    print(f"✓ 词表大小: {len(vocab)}")
    
    # 检查特殊标记
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    print("\n特殊标记检查:")
    for i, token in enumerate(special_tokens):
        if i < len(vocab) and vocab[i] == token:
            print(f"  ✓ {token} (ID: {i})")
        else:
            print(f"  ✗ {token} 位置错误或缺失")
    
    # 显示前几个普通字符
    print("\n前10个普通字符:")
    for i in range(4, min(14, len(vocab))):
        print(f"  ID {i}: '{vocab[i]}'")
    
    # 检查是否有重复
    if len(vocab) != len(set(vocab)):
        print("\n✗ 警告: 词表中存在重复字符!")
    else:
        print("\n✓ 词表中无重复字符")


def analyze_vocabulary(config_path):
    """
    分析词汇表的统计信息
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    vocab_path = config['data']['vocab_path']
    
    if not os.path.exists(vocab_path):
        print("词表不存在，请先运行 build_vocabulary()")
        return
    
    # 读取词表
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip())
    
    print("=== 词汇表分析 ===")
    print(f"总字符数: {len(vocab)}")
    
    # 统计不同类型的字符
    chinese_chars = 0
    english_chars = 0
    digits = 0
    punctuation = 0
    special = 0
    
    for char in vocab:
        if char.startswith('<') and char.endswith('>'):
            special += 1
        elif '\u4e00' <= char <= '\u9fff':  # 中文字符范围
            chinese_chars += 1
        elif char.isalpha():
            english_chars += 1
        elif char.isdigit():
            digits += 1
        else:
            punctuation += 1
    
    print(f"  特殊标记: {special}")
    print(f"  中文字符: {chinese_chars}")
    print(f"  英文字符: {english_chars}")
    print(f"  数字: {digits}")
    print(f"  标点符号: {punctuation}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'conf', 'config.yaml')
    
    print(f"配置文件路径: {config_path}")
    print("=" * 50)
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
    else:
        build_vocabulary(config_path)
        print("\n" + "=" * 50)
        analyze_vocabulary(config_path)