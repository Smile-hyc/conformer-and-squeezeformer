import torch
import os
import glob

# 删除检查点  rm -rf checkpoints/ logs/
# ls -la checkpoints/

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename="checkpoint.pth"):
    """保存模型和训练状态"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    save_path = os.path.join(checkpoint_dir, filename)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, save_path)
    print(f"模型已保存: {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """加载模型和训练状态"""
    print(f"正在从 {checkpoint_path} 恢复训练...")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    
    print(f"恢复成功！将从第 {start_epoch} 轮继续训练。")
    return start_epoch

def get_latest_checkpoint(checkpoint_dir):
    """自动寻找最新的 checkpoint 文件"""
    if not os.path.exists(checkpoint_dir):
        return None
        
    # 找名字里带 'latest' 的或者按时间排序
    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not files:
        return None
        
    # 优先找 latest_model.pth，如果找不到就找最新的
    latest_path = os.path.join(checkpoint_dir, "latest_model.pth")
    if os.path.exists(latest_path):
        return latest_path
        
    # 如果没有 latest，返回修改时间最新的文件
    files.sort(key=os.path.getmtime)
    return files[-1]