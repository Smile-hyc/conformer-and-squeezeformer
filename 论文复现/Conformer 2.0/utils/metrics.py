import torch

def compute_cer(results, targets, char2id, id2char):
    """
    计算字错率 (Character Error Rate) 的简化版 -> 准确率
    results: [B, T, Vocab]
    targets: [B, T]
    这里为了训练速度，我们只在打印时定性观察，严谨的WER/CER通常在推理阶段算
    """
    # 贪婪解码：取概率最大的 id
    predictions = torch.argmax(results, dim=-1) # [B, T]
    
    # 随便挑一个样本解码打印出来看看效果
    pred_seq = predictions[0].cpu().tolist()
    target_seq = targets[0].cpu().tolist()
    
    pred_text = ""
    target_text = ""
    
    for pid in pred_seq:
        if pid in id2char and pid not in [0, 1, 2]: # 忽略 pad, sos, eos
            pred_text += id2char[pid]
            
    for tid in target_seq:
        if tid in id2char and tid not in [0, 1, 2]:
            target_text += id2char[tid]
            
    return pred_text, target_text