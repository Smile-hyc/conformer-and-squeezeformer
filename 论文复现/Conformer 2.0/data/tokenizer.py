#生成词汇表的脚本
import os
import pandas as pd
import yaml

def build_vocabulary(config_path):
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    train_csv = config['data']['train_manifest']
    vocab_path = config['data']['vocab_path']
    
    print("正在构建词表 (Vocabulary)...")
    
    # 读取训练集文本
    df = pd.read_csv(train_csv)
    all_text = df['text'].tolist()
    
    # 统计字符
    unique_chars = set()
    for text in all_text:
        # 去除空格，AIShell文本中间可能有空格
        text = text.replace(" ", "") 
        for char in text:
            unique_chars.add(char)
            
    print(f"训练集中共有 {len(unique_chars)} 个唯一字符。")
    
    # 添加特殊符号
    # <pad>: 填充符 (ID 0)
    # <sos>: 句子开始 (ID 1)
    # <eos>: 句子结束 (ID 2)
    # <unk>: 未知字符 (ID 3)
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    
    sorted_chars = sorted(list(unique_chars))
    final_vocab = special_tokens + sorted_chars
    
    # 保存到 vocabulary.txt
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for char in final_vocab:
            f.write(char + '\n')
            
    print(f"词表已保存至: {vocab_path}")
    print(f"总词表大小: {len(final_vocab)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'conf', 'config.yaml')
    
    build_vocabulary(config_path)