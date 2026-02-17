#用于解压原始数据集的脚本
import os
import tarfile
from tqdm import tqdm  
import yaml

def extract_all_tars(config_path):
    # 1. 读取配置找到路径
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 定位到 wav 文件夹: E:/data_aishell/data_aishell/wav
    wav_dir = os.path.join(config['data']['data_path'], 'wav')
    
    print(f"正在扫描压缩包路径: {wav_dir}")
    
    # 2. 找到所有的 .tar.gz 文件
    files = os.listdir(wav_dir)
    tar_files = [f for f in files if f.endswith('.tar.gz')]
    
    if not tar_files:
        print("错误：在该目录下没有找到 .tar.gz 压缩包！请检查路径或确认是否已经解压过。")
        return

    print(f"发现 {len(tar_files)} 个压缩包，准备开始解压...")
    print("注意：解压过程可能需要几分钟，请耐心等待。")

    # 3. 逐个解压
    for tar_name in tqdm(tar_files, desc="解压进度"):
        tar_path = os.path.join(wav_dir, tar_name)
        
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                # 将文件解压到 wav_dir 当前目录下
                # 由于压缩包内部通常已经包含了 'train/S0002/...' 结构
                # 所以直接解压到当前目录会自动合并到 train/dev/test 文件夹中
                tar.extractall(path=wav_dir)
        except Exception as e:
            print(f"\n解压 {tar_name} 失败: {e}")

    print("\n所有文件解压完成！")
    
    # 4. 简单验证
    if os.path.exists(os.path.join(wav_dir, 'train')):
        print("验证通过：发现了 'train' 文件夹。")
    else:
        print("警告：解压后未发现 'train' 文件夹，请手动检查 E:/data_aishell/data_aishell/wav 目录结构。")

if __name__ == "__main__":
    # 定位配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'conf', 'config.yaml')
    
    extract_all_tars(config_path)