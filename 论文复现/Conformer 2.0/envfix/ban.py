import torch
import os

# 完全禁用cuDNN
torch.backends.cudnn.enabled = False
print(f"cuDNN已禁用: {not torch.backends.cudnn.enabled}")