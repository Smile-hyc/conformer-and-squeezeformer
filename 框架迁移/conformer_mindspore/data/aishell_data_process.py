import os
import yaml
import pandas as pd


def generate_manifest(config_path):
    """
    生成训练/验证/测试数据清单CSV文件
    从AIShell数据集的标注文件中读取文本，匹配对应的音频文件
    """
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data']['data_path']
    transcript_path = os.path.join(data_dir, 'transcript', 'aishell_transcript_v0.8.txt')
    wav_dir = os.path.join(data_dir, 'wav')

    print(f"正在读取标注文件: {transcript_path}")
    
    # 检查文件是否存在
    if not os.path.exists(transcript_path):
        print(f"错误: 标注文件不存在: {transcript_path}")
        return
    
    # 2. 构建字典 (使用 split() 不带参数，自动处理所有空白符)
    transcripts = {}
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 不指定分隔符，自动处理多空格/制表符
            parts = line.split() 
            
            file_id = parts[0]
            # 把剩下的部分拼回去
            text = ' '.join(parts[1:]) 
            transcripts[file_id] = text

            # 当场验证关键样本
            if file_id == "BAC009S0003W0121":
                print("\n" + "=" * 40)
                print("关键样本核对 (S0003W0121):")
                print(f"原始文本应为: 汉王考勤及其配套...")
                print(f"实际解析结果: {text}")
                print("=" * 40 + "\n")

    print(f"加载了 {len(transcripts)} 条标注。")

    # 3. 匹配音频文件
    subsets = ['train', 'dev', 'test']
    output_dir = os.path.dirname(config['data']['train_manifest'])
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    for subset in subsets:
        subset_dir = os.path.join(wav_dir, subset)
        data_list = []
        
        if not os.path.exists(subset_dir):
            print(f"警告: {subset} 目录不存在: {subset_dir}")
            continue
        
        print(f"正在扫描 {subset} ...")
        for root, dirs, files in os.walk(subset_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_id = file.split('.')[0]
                    
                    if file_id in transcripts:
                        abs_path = os.path.join(root, file).replace('\\', '/')
                        text = transcripts[file_id]
                        data_list.append({
                            'path': abs_path,
                            'text': text
                        })
        
        # 保存 CSV
        if data_list:
            df = pd.DataFrame(data_list)
            save_path = os.path.join(output_dir, f"{subset}.csv")
            df.to_csv(save_path, index=False, encoding='utf-8')
            print(f"已生成 {subset}.csv ({len(df)} 条)")
        else:
            print(f"警告: {subset} 没有匹配到任何数据")


def verify_manifest(config_path):
    """
    验证生成的数据清单是否正确
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n=== 验证数据清单 ===")
    
    for subset in ['train', 'dev', 'test']:
        manifest_path = config['data'].get(f'{subset}_manifest')
        if manifest_path and os.path.exists(manifest_path):
            df = pd.read_csv(manifest_path)
            print(f"{subset}: {len(df)} 条记录")
            
            # 检查前几条数据
            if len(df) > 0:
                # 验证音频文件是否存在
                sample_path = df.iloc[0]['path']
                if os.path.exists(sample_path):
                    print(f"  ✓ 音频文件路径有效")
                else:
                    print(f"  ✗ 警告: 音频文件不存在: {sample_path}")
        else:
            print(f"{subset}: 文件不存在")


if __name__ == "__main__":
    # 定位配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'conf', 'config.yaml')
    
    print(f"配置文件路径: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
    else:
        generate_manifest(config_path)
        verify_manifest(config_path)