import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor


def compute_cer(results, targets, char2id, id2char):
    """
    计算字错率 (Character Error Rate) 的简化版 -> 准确率
    results: [B, T, Vocab]
    targets: [B, T]
    这里为了训练速度，我们只在打印时定性观察，严谨的WER/CER通常在推理阶段算
    """
    # 贪婪解码：取概率最大的 id
    argmax_op = ops.Argmax(axis=-1)
    predictions = argmax_op(results)  # [B, T]
    
    # 随便挑一个样本解码打印出来看看效果
    pred_seq = predictions[0].asnumpy().tolist()
    target_seq = targets[0].asnumpy().tolist()
    
    pred_text = ""
    target_text = ""
    
    for pid in pred_seq:
        if pid in id2char and pid not in [0, 1, 2]:  # 忽略 pad, sos, eos
            pred_text += id2char[pid]
            
    for tid in target_seq:
        if tid in id2char and tid not in [0, 1, 2]:
            target_text += id2char[tid]
            
    return pred_text, target_text


def compute_cer_score(pred_text, true_text):
    """
    计算真正的字错误率 (Character Error Rate)
    使用编辑距离算法
    """
    # 移除空格
    pred_chars = list(pred_text.replace(' ', ''))
    true_chars = list(true_text.replace(' ', ''))
    
    n, m = len(pred_chars), len(true_chars)
    
    # 边界情况处理
    if n == 0 and m == 0:
        return 0.0
    elif n == 0:
        return 1.0
    elif m == 0:
        return 1.0
    
    # 动态规划计算编辑距离
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_chars[i - 1] == true_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # 删除
                    dp[i][j - 1] + 1,      # 插入
                    dp[i - 1][j - 1] + 1   # 替换
                )
    
    edit_distance = dp[n][m]
    cer = edit_distance / max(m, 1)
    
    return cer


def compute_accuracy(results, targets, ignore_ids=None):
    """
    计算token级别的准确率
    results: [B, T, Vocab] 或 [B, T]
    targets: [B, T]
    ignore_ids: 忽略的token id列表，如 [0, 1, 2] 表示忽略 pad, sos, eos
    """
    if ignore_ids is None:
        ignore_ids = [0, 1, 2]
    
    # 如果results是logits，先取argmax
    if len(results.shape) == 3:
        argmax_op = ops.Argmax(axis=-1)
        predictions = argmax_op(results)
    else:
        predictions = results
    
    # 转换为numpy进行计算
    pred_np = predictions.asnumpy()
    target_np = targets.asnumpy()
    
    # 创建mask，忽略指定的token
    mask = np.ones_like(target_np, dtype=bool)
    for ignore_id in ignore_ids:
        mask = mask & (target_np != ignore_id)
    
    # 计算准确率
    correct = (pred_np == target_np) & mask
    accuracy = correct.sum() / max(mask.sum(), 1)
    
    return accuracy


def compute_batch_cer(results, targets, id2char, ignore_ids=None):
    """
    计算整个batch的平均CER
    results: [B, T, Vocab]
    targets: [B, T]
    """
    if ignore_ids is None:
        ignore_ids = [0, 1, 2]
    
    # 贪婪解码
    argmax_op = ops.Argmax(axis=-1)
    predictions = argmax_op(results)
    
    pred_np = predictions.asnumpy()
    target_np = targets.asnumpy()
    
    batch_size = pred_np.shape[0]
    total_cer = 0.0
    valid_samples = 0
    
    for i in range(batch_size):
        # 解码预测文本
        pred_text = ""
        for pid in pred_np[i]:
            if pid in id2char and pid not in ignore_ids:
                pred_text += id2char[pid]
        
        # 解码真实文本
        true_text = ""
        for tid in target_np[i]:
            if tid in id2char and tid not in ignore_ids:
                true_text += id2char[tid]
        
        # 计算CER
        if len(true_text) > 0:
            cer = compute_cer_score(pred_text, true_text)
            total_cer += cer
            valid_samples += 1
    
    avg_cer = total_cer / max(valid_samples, 1)
    return avg_cer


def decode_predictions(results, id2char, ignore_ids=None):
    """
    将模型输出解码为文本列表
    results: [B, T, Vocab] 或 [B, T]
    """
    if ignore_ids is None:
        ignore_ids = [0, 1, 2]
    
    # 如果results是logits，先取argmax
    if len(results.shape) == 3:
        argmax_op = ops.Argmax(axis=-1)
        predictions = argmax_op(results)
    else:
        predictions = results
    
    pred_np = predictions.asnumpy()
    batch_size = pred_np.shape[0]
    
    decoded_texts = []
    for i in range(batch_size):
        text = ""
        for pid in pred_np[i]:
            if pid in id2char and pid not in ignore_ids:
                text += id2char[pid]
        decoded_texts.append(text)
    
    return decoded_texts


def decode_targets(targets, id2char, ignore_ids=None):
    """
    将目标序列解码为文本列表
    targets: [B, T]
    """
    if ignore_ids is None:
        ignore_ids = [0, 1, 2]
    
    target_np = targets.asnumpy()
    batch_size = target_np.shape[0]
    
    decoded_texts = []
    for i in range(batch_size):
        text = ""
        for tid in target_np[i]:
            if tid in id2char and tid not in ignore_ids:
                text += id2char[tid]
        decoded_texts.append(text)
    
    return decoded_texts