import os
import glob
import mindspore
from mindspore import save_checkpoint as ms_save_checkpoint
from mindspore import load_checkpoint as ms_load_checkpoint
from mindspore import load_param_into_net

# 删除检查点  rm -rf checkpoints/ logs/
# ls -la checkpoints/


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename="checkpoint.ckpt"):
    """保存模型和训练状态"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    save_path = os.path.join(checkpoint_dir, filename)
    
    # MindSpore保存模型参数
    ms_save_checkpoint(model, save_path)
    
    # 保存训练状态到单独的文件
    state_path = save_path.replace('.ckpt', '_state.ckpt')
    
    # 创建包含epoch和loss的参数列表
    append_dict = [
        {"name": "epoch", "data": mindspore.Tensor(epoch, mindspore.int32)},
        {"name": "loss", "data": mindspore.Tensor(loss, mindspore.float32)}
    ]
    ms_save_checkpoint(model, save_path, append_dict=append_dict)
    
    print(f"模型已保存: {save_path}")


def save_checkpoint_simple(model, checkpoint_dir, filename="checkpoint.ckpt"):
    """简化版保存 - 只保存模型参数"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    save_path = os.path.join(checkpoint_dir, filename)
    ms_save_checkpoint(model, save_path)
    print(f"模型已保存: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """加载模型和训练状态"""
    print(f"正在从 {checkpoint_path} 恢复训练...")
    
    # 加载参数字典
    param_dict = ms_load_checkpoint(checkpoint_path)
    
    # 加载模型参数
    not_loaded, _ = load_param_into_net(model, param_dict, strict_load=False)
    if not_loaded:
        print(f"以下参数未加载: {not_loaded[:5]}...")
    
    # 尝试获取epoch
    start_epoch = 0
    if "epoch" in param_dict:
        start_epoch = int(param_dict["epoch"].asnumpy()) + 1
    else:
        # 尝试从文件名解析epoch
        try:
            if 'epoch_' in checkpoint_path:
                start_epoch = int(checkpoint_path.split('epoch_')[1].split('.')[0])
        except:
            pass
    
    print(f"恢复成功！将从第 {start_epoch} 轮继续训练。")
    return start_epoch


def load_checkpoint_simple(model, checkpoint_path):
    """简化版加载 - 只加载模型参数"""
    print(f"正在加载模型: {checkpoint_path}")
    
    param_dict = ms_load_checkpoint(checkpoint_path)
    not_loaded, _ = load_param_into_net(model, param_dict, strict_load=False)
    
    if not_loaded:
        print(f"以下参数未加载: {not_loaded[:5]}...")
    
    print("模型加载完成！")


def get_latest_checkpoint(checkpoint_dir):
    """自动寻找最新的 checkpoint 文件"""
    if not os.path.exists(checkpoint_dir):
        return None
        
    # 找名字里带 'latest' 的或者按时间排序
    files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not files:
        return None
    
    # 过滤掉 _state.ckpt 文件
    files = [f for f in files if '_state.ckpt' not in f]
    if not files:
        return None
        
    # 优先找 latest_model.ckpt，如果找不到就找最新的
    latest_path = os.path.join(checkpoint_dir, "latest_model.ckpt")
    if os.path.exists(latest_path):
        return latest_path
        
    # 如果没有 latest，返回修改时间最新的文件
    files.sort(key=os.path.getmtime)
    return files[-1]