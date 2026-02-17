# 华为云中清理显存的方法
import torch
import gc

# 方法2.1：基本清理
torch.cuda.empty_cache()
gc.collect()

# 方法2.2：强制清理
if torch.cuda.is_available():
    torch.cuda.synchronize()  # 等待所有操作完成
    torch.cuda.empty_cache()
    gc.collect()
    print("显存已清理")
    
# 检查清理效果
print(f"清理后显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

'''
# 强制清理，直接运行
# 查看所有Python进程
ps aux | grep python

# 杀死所有Python进程（彻底清理）
pkill -9 python

# 或者只杀死当前用户的Python进程
pkill -9 -u $USER python

# 等待几秒后检查
sleep 3
nvidia-smi
'''