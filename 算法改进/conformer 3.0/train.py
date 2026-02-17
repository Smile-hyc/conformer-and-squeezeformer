import os
import warnings
warnings.filterwarnings("ignore")
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入写好的模块
from model import ConformerASR
from utils.dataset import AIShellDataset, collate_fn
from utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from utils.logger import Logger
from utils.metrics import compute_cer

torch.backends.cudnn.enabled = False

# 清理显存
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# 设置设备
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

    # 读取配置
    with open('conf/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 初始化日志
    logger = Logger(config['training']['log_dir'])
    logger.log("=== 开始训练流程 ===")
    
    # 加载数据
    logger.log("正在加载数据集...")
    train_dataset = AIShellDataset(config['data']['train_manifest'], config['data']['vocab_path'], config)
    dev_dataset = AIShellDataset(config['data']['dev_manifest'], config['data']['vocab_path'], config)
    
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
    
    # 定义 Loss 和 优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        min_lr=1e-6,
        verbose=True
    )
    
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    att_loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean') 
    
    ctc_weight = config['model']['ctc_weight']
    
    # 梯度累积
    accumulation_steps = 2

    # 断点续训检查
    start_epoch = 0
    checkpoint_dir = config['training']['checkpoint_dir']
    latest_ckpt = get_latest_checkpoint(checkpoint_dir)
    
    if latest_ckpt:
        logger.log(f"发现检查点: {latest_ckpt}，正在恢复...")
        start_epoch = load_checkpoint(model, optimizer, latest_ckpt)
    else:
        logger.log("未发现检查点，从头开始训练。")

    # 训练循环
    total_epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience_counter = 0  # 早停计数器
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        total_ctc_loss = 0
        total_att_loss = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for batch_idx, (feats, feat_lens, targets, target_lens) in enumerate(pbar):
            feats = feats.to(device)
            feat_lens = feat_lens.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)
            
            # 前向传播
            decoder_input = targets[:, :-1]  # 移除<eos>
            decoder_target = targets[:, 1:]  # 移除<sos>
            
            ctc_out, enc_lens, dec_out = model(feats, feat_lens, decoder_input)
            
            # Loss 计算
            ctc_out_T = ctc_out.log_softmax(dim=2).transpose(0, 1)
            loss_ctc = ctc_loss_fn(ctc_out_T, targets, enc_lens, target_lens)
            loss_att = att_loss_fn(dec_out.reshape(-1, vocab_size), decoder_target.reshape(-1))
            loss = ctc_weight * loss_ctc + (1 - ctc_weight) * loss_att
            
            # 梯度累积
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # 更严格的梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
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
        
        avg_train_loss = total_loss / len(train_loader)
        avg_ctc_loss = total_ctc_loss / len(train_loader)
        avg_att_loss = total_att_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
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
                
                ctc_out, enc_lens, dec_out = model(feats, feat_lens, decoder_input)
                
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
            if patience_counter >= config['training'].get('early_stopping_patience', 8):
                logger.log(f"早停触发，在 Epoch {epoch+1} 停止训练")
                break
        
        # 定期保存
        if epoch % config['training']['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_dir, f"epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
    

