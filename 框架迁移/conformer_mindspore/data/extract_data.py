# 用于解压原始数据集的脚本
import os
import tarfile
from tqdm import tqdm  
import yaml


def extract_all_tars(config_path):
    """
    解压AIShell数据集中所有的tar.gz压缩包
    """
    # 1. 读取配置找到路径
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 定位到 wav 文件夹
    wav_dir = os.path.join(config['data']['data_path'], 'wav')
    
    print(f"正在扫描压缩包路径: {wav_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(wav_dir):
        print(f"错误: wav目录不存在: {wav_dir}")
        print("请确认数据集路径是否正确，或手动创建该目录后放入压缩包。")
        return
    
    # 2. 找到所有的 .tar.gz 文件
    files = os.listdir(wav_dir)
    tar_files = [f for f in files if f.endswith('.tar.gz')]
    
    if not tar_files:
        print("提示：在该目录下没有找到 .tar.gz 压缩包！")
        print("可能的原因：")
        print("  1. 压缩包已经解压过")
        print("  2. 压缩包放置路径不正确")
        print("  3. 数据集尚未下载")
        
        # 检查是否已经解压
        if os.path.exists(os.path.join(wav_dir, 'train')):
            print("\n检测到 'train' 文件夹已存在，数据可能已经解压完成。")
        return

    print(f"发现 {len(tar_files)} 个压缩包，准备开始解压...")
    print("注意：解压过程可能需要几分钟，请耐心等待。")
    print("-" * 50)

    # 3. 逐个解压
    success_count = 0
    failed_files = []
    
    for tar_name in tqdm(tar_files, desc="解压进度"):
        tar_path = os.path.join(wav_dir, tar_name)
        
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                # 将文件解压到 wav_dir 当前目录下
                # 由于压缩包内部通常已经包含了 'train/S0002/...' 结构
                # 所以直接解压到当前目录会自动合并到 train/dev/test 文件夹中
                tar.extractall(path=wav_dir)
            success_count += 1
        except Exception as e:
            print(f"\n解压 {tar_name} 失败: {e}")
            failed_files.append(tar_name)

    print("\n" + "=" * 50)
    print(f"解压完成！成功: {success_count}, 失败: {len(failed_files)}")
    
    if failed_files:
        print(f"失败的文件: {failed_files}")
    
    # 4. 验证解压结果
    verify_extraction(wav_dir)


def verify_extraction(wav_dir):
    """
    验证解压结果
    """
    print("\n=== 验证解压结果 ===")
    
    subsets = ['train', 'dev', 'test']
    
    for subset in subsets:
        subset_dir = os.path.join(wav_dir, subset)
        
        if os.path.exists(subset_dir):
            # 统计wav文件数量
            wav_count = 0
            for root, dirs, files in os.walk(subset_dir):
                wav_count += len([f for f in files if f.endswith('.wav')])
            
            print(f"  ✓ {subset}: 发现 {wav_count} 个音频文件")
        else:
            print(f"  ✗ {subset}: 目录不存在")


def check_disk_space(path):
    """
    检查磁盘空间 (可选功能)
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024 ** 3)
        print(f"磁盘剩余空间: {free_gb:.2f} GB")
        
        if free_gb < 10:
            print("警告: 磁盘空间不足10GB，解压可能失败！")
            return False
        return True
    except Exception as e:
        print(f"无法检查磁盘空间: {e}")
        return True


if __name__ == "__main__":
    # 定位配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, 'conf', 'config.yaml')
    
    print(f"配置文件路径: {config_path}")
    print("=" * 50)
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
    else:
        # 检查磁盘空间
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        data_path = config['data']['data_path']
        if os.path.exists(data_path):
            check_disk_space(data_path)
        
        # 开始解压
        extract_all_tars(config_path)