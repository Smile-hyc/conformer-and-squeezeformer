import os
import yaml
import pandas as pd

def generate_manifest(config_path):
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data']['data_path']
    transcript_path = os.path.join(data_dir, 'transcript', 'aishell_transcript_v0.8.txt')
    wav_dir = os.path.join(data_dir, 'wav')

    print(f"正在读取标注文件: {transcript_path}")
    
    # 2. 构建字典 (使用 split() 不带参数，自动处理所有空白符)
    transcripts = {}
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 不指定分隔符，自动处理多空格/制表符
            parts = line.split() 
            
            file_id = parts[0]
            # 把剩下的部分拼回去
            text = ' '.join(parts[1:]) 
            transcripts[file_id] = text

            # 当场验证关键样本
            if file_id == "BAC009S0003W0121":
                print("\n" + "="*40)
                print("关键样本核对 (S0003W0121):")
                print(f"原始文本应为: 汉王考勤及其配套...")
                print(f"实际解析结果: {text}")
                print("="*40 + "\n")

    print(f"加载了 {len(transcripts)} 条标注。")

    # 3. 匹配音频文件
    subsets = ['train', 'dev', 'test']
    output_dir = os.path.dirname(config['data']['train_manifest'])
    
    for subset in subsets:
        subset_dir = os.path.join(wav_dir, subset)
        data_list = []
        
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

if __name__ == "__main__":
    # 定位配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'conf', 'config.yaml')
    
    generate_manifest(config_path)