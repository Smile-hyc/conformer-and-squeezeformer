import os
import warnings
warnings.filterwarnings("ignore")
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入我们写好的模块
from model import ConformerASR
from utils.dataset import AIShellDataset, collate_fn
from utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from utils.logger import Logger
from utils.metrics import compute_cer

torch.backends.cudnn.enabled = False

# 清理显存与设备检查
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# 设置设备，检查GPU是否可用等
try:
    device = torch.device("cuda")
    # 测试小张量
    test_tensor = torch.randn(100, 100).cuda()
    print("GPU可用")
except:
    device = torch.device("cpu") 
    print("GPU不可用，使用CPU")
    
    

def train():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 读取配置（所有的超参数都写在这里了，可以直接改）
    with open('conf/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 初始化日志
    logger = Logger(config['training']['log_dir'])
    logger.log("=== 开始训练流程 ===")
    
    # 加载数据数据集
    logger.log("正在加载数据集...")
    train_dataset = AIShellDataset(config['data']['train_manifest'], config['data']['vocab_path'], config)
    dev_dataset = AIShellDataset(config['data']['dev_manifest'], config['data']['vocab_path'], config)
    
    #实例化 Dataset 对象，它负责从硬盘读取文件名和文本
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['training']['num_workers']
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['training']['num_workers']
    )
    
    vocab_size = len(train_dataset.char2id)
    id2char = {v: k for k, v in train_dataset.char2id.items()}
    logger.log(f"词表大小: {vocab_size}")

    # 初始化模型
    model = ConformerASR(config, vocab_size).to(device)
    
    # 定义Loss和优化器
    optimizer = optim.AdamW(#Adam的改良版，自适应学习率且权重衰减防止过拟合
        model.parameters(), 
        lr=config['training']['lr'], 
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    #学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        # ReduceLROnPlateau: "高原反应减速策略"
        #如果发现 Loss好几个 Epoch 都不下降了，就把学习率*0.5，继续训练
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        min_lr=1e-6,
        verbose=True
    )
    
    # 可以理解为两个裁判
    # 第一个裁判： CTC Loss，听声辨字，负责发音对齐
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    # 第二个裁判： Attention Loss，关注上下文，负责语义理解
    att_loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean') 
    # CTC 权重的比例 (比如 0.8 CTC + 0.2 Attention)
    ctc_weight = config['model']['ctc_weight']
    
    # 梯度累积
    accumulation_steps = 2

    # 断点续训检查
    start_epoch = 0
    # 检查以前有没有训练过的存档 (Checkpoint)。如果有，就加载进来接着练，而不是从头开始。
    checkpoint_dir = config['training']['checkpoint_dir']
    latest_ckpt = get_latest_checkpoint(checkpoint_dir)
    
    if latest_ckpt:
        logger.log(f"发现检查点: {latest_ckpt}，正在恢复...")
        # 这个函数会把模型权重和优化器状态都加载回来，并返回上次练到了第几轮
        start_epoch = load_checkpoint(model, optimizer, latest_ckpt)
    else:
        logger.log("未发现检查点，从头开始训练。")

    # 训练循环
    total_epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience_counter = 0  # 早停计数器，记录没有进步的次数
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        
        #初始化这一圈的统计数据
        total_loss = 0
        total_ctc_loss = 0
        total_att_loss = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for batch_idx, (feats, feat_lens, targets, target_lens) in enumerate(pbar):
            feats = feats.to(device)#音频特征
            feat_lens = feat_lens.to(device)#音频长度
            targets = targets.to(device)#目标文本
            target_lens = target_lens.to(device)#文本长度
            
            # 准备Decoder的输入和目标
            decoder_input = targets[:, :-1]  # 移除<eos>
            decoder_target = targets[:, 1:]  # 移除<sos>
            
            # 前向传播，得到CTC和Decoder的输出
            ctc_out, enc_lens, dec_out = model(feats, feat_lens, decoder_input)
            
            # Loss 计算
            ctc_out_T = ctc_out.log_softmax(dim=2).transpose(0, 1)# log_softmax: 把输出变成概率的对数形式
            loss_ctc = ctc_loss_fn(ctc_out_T, targets, enc_lens, target_lens)
            loss_att = att_loss_fn(dec_out.reshape(-1, vocab_size), decoder_target.reshape(-1))# 需要把数据拉直 (reshape) 成一维长条，方便和标准答案对比
            loss = ctc_weight * loss_ctc + (1 - ctc_weight) * loss_att# 甲醛总分
            
            # 梯度累积
            loss = loss / accumulation_steps
            loss.backward() # 反向传播：算出每个参数该怎么调 (计算梯度)
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # 更严格的梯度裁剪，可以有效防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step() # 优化器更新参数，更新权重
                optimizer.zero_grad()
            
            # 记录数据
            total_loss += loss.item() * accumulation_steps
            total_ctc_loss += loss_ctc.item()
            total_att_loss += loss_att.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.3f}',
                'ctc': f'{loss_ctc.item():.3f}', 
                'att': f'{loss_att.item():.3f}'
            })
        
        # 处理最后一个不完整的累积批次
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        #计算这一圈的平均分
        avg_train_loss = total_loss / len(train_loader)
        avg_ctc_loss = total_ctc_loss / len(train_loader)
        avg_att_loss = total_att_loss / len(train_loader)
        
        # 验证阶段
        # 这一步非常重要，用于检查模型是否出现过拟合
        model.eval()#评估模式
        val_loss = 0
        val_ctc_loss = 0
        val_att_loss = 0
        
        with torch.no_grad():
            for feats, feat_lens, targets, target_lens in dev_loader:
                feats = feats.to(device)
                feat_lens = feat_lens.to(device)
                targets = targets.to(device)
                target_lens = target_lens.to(device)
                
                decoder_input = targets[:, :-1]
                decoder_target = targets[:, 1:]
                
                #前向传播
                ctc_out, enc_lens, dec_out = model(feats, feat_lens, decoder_input)
                
                #算loss
                ctc_out_T = ctc_out.log_softmax(dim=2).transpose(0, 1)
                loss_ctc = ctc_loss_fn(ctc_out_T, targets, enc_lens, target_lens)
                loss_att = att_loss_fn(dec_out.reshape(-1, vocab_size), decoder_target.reshape(-1))
                loss = ctc_weight * loss_ctc + (1 - ctc_weight) * loss_att
                
                val_loss += loss.item()
                val_ctc_loss += loss_ctc.item()
                val_att_loss += loss_att.item()
        
        avg_val_loss = val_loss / len(dev_loader)
        avg_val_ctc = val_ctc_loss / len(dev_loader)
        avg_val_att = val_att_loss / len(dev_loader)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录日志
        log_msg = (f"Epoch {epoch+1} | "
                  f"Train Loss: {avg_train_loss:.4f} (CTC: {avg_ctc_loss:.4f}, Att: {avg_att_loss:.4f}) | "
                  f"Val Loss: {avg_val_loss:.4f} (CTC: {avg_val_ctc:.4f}, Att: {avg_val_att:.4f}) | "
                  f"LR: {current_lr:.2e}")
        logger.log(log_msg)
        
        # 打印预测样本
        pred_str, target_str = compute_cer(dec_out, decoder_target, train_dataset.char2id, id2char)
        logger.log(f"   Pred: {pred_str[:50]}...")
        logger.log(f"   True: {target_str[:50]}...")
        
        # 保存模型
        save_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_dir, "latest_model.pth")
        
        # 保存最佳并检查早停
        if avg_val_loss < best_val_loss - 0.001:  # 有显著改善
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_dir, "best_model.pth")
            logger.log(f"新的最佳模型已保存，验证损失: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # 定期保存
        if epoch % config['training']['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_dir, f"epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
    
  # 测试：python inference.py --wav "data_aishell/wav/test/S0764/BAC009S0764W0121.wav"

  # 部分验证：python test_dataset.py --checkpoint checkpoints/best_model.pth --sample_count 50

  # 训练集验证：python test_train_set.py --checkpoint checkpoints/best_model.pth --sample_count 1000