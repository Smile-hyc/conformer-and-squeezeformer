import os
import warnings
warnings.filterwarnings("ignore")
import yaml
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, Parameter
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback
import mindspore.dataset as ds

# 导入写好的模块
from model import ConformerASR
from utils.dataset import AIShellDataset, collate_fn
from utils.checkpoint import save_checkpoint_ms, load_checkpoint_ms, get_latest_checkpoint
from utils.logger import Logger
from utils.metrics import compute_cer

# 设置运行环境 (华为云Ascend)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
print("使用设备: Ascend NPU")


class CTCLoss(nn.Cell):
    """CTC Loss封装"""
    def __init__(self, blank=0, reduction='mean'):
        super(CTCLoss, self).__init__()
        self.ctc_loss = ops.CTCLoss(preprocess_collapse_repeated=False, 
                                     ctc_merge_repeated=True,
                                     ignore_longer_outputs_than_inputs=True)
        self.blank = blank
        self.reduction = reduction
        self.log_softmax = ops.LogSoftmax(axis=2)
        self.transpose = ops.Transpose()
        self.reduce_mean = ops.ReduceMean()
        self.reduce_sum = ops.ReduceSum()
    
    def construct(self, log_probs, targets, input_lengths, target_lengths):
        # log_probs: [T, B, C]
        # targets: [B, S]
        # input_lengths: [B]
        # target_lengths: [B]
        
        loss, _ = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        if self.reduction == 'mean':
            return self.reduce_mean(loss)
        elif self.reduction == 'sum':
            return self.reduce_sum(loss)
        return loss


class CrossEntropyLoss(nn.Cell):
    """CrossEntropy Loss封装，支持ignore_index"""
    def __init__(self, ignore_index=0, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        self.cast = ops.Cast()
        self.not_equal = ops.NotEqual()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
    
    def construct(self, logits, labels):
        # logits: [N, C], labels: [N]
        loss = self.cross_entropy(logits, labels)
        
        # 创建mask，忽略ignore_index
        mask = self.cast(self.not_equal(labels, self.ignore_index), mindspore.float32)
        loss = loss * mask
        
        if self.reduction == 'mean':
            return self.reduce_sum(loss) / (self.reduce_sum(mask) + 1e-8)
        elif self.reduction == 'sum':
            return self.reduce_sum(loss)
        return loss


class ASRWithLoss(nn.Cell):
    """带损失计算的ASR模型封装"""
    def __init__(self, model, vocab_size, ctc_weight=0.3, blank=0):
        super(ASRWithLoss, self).__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        
        self.ctc_loss_fn = CTCLoss(blank=blank, reduction='mean')
        self.att_loss_fn = CrossEntropyLoss(ignore_index=0, reduction='mean')
        
        self.log_softmax = ops.LogSoftmax(axis=2)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
    
    def construct(self, feats, feat_lens, targets, target_lens):
        # 准备decoder输入输出
        decoder_input = targets[:, :-1]  # 移除<eos>
        decoder_target = targets[:, 1:]  # 移除<sos>
        
        # 前向传播
        ctc_out, enc_lens, dec_out = self.model(feats, feat_lens, decoder_input)
        
        # CTC Loss计算
        ctc_log_probs = self.log_softmax(ctc_out)
        ctc_log_probs_T = self.transpose(ctc_log_probs, (1, 0, 2))  # [T, B, C]
        loss_ctc = self.ctc_loss_fn(ctc_log_probs_T, targets, enc_lens, target_lens)
        
        # Attention Loss计算
        batch_size, seq_len, _ = dec_out.shape
        dec_out_flat = self.reshape(dec_out, (-1, self.vocab_size))
        decoder_target_flat = self.reshape(decoder_target, (-1,))
        loss_att = self.att_loss_fn(dec_out_flat, decoder_target_flat)
        
        # 总损失
        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        
        return loss, loss_ctc, loss_att, dec_out, decoder_target


class TrainOneStepCell(nn.Cell):
    """单步训练封装"""
    def __init__(self, network, optimizer, grad_clip=1.0):
        super(TrainOneStepCell, self).__init__()
        self.network = network
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)
        self.grad_clip = grad_clip
        self.hyper_map = ops.HyperMap()
        
    def construct(self, feats, feat_lens, targets, target_lens):
        # 前向传播计算损失
        loss, loss_ctc, loss_att, dec_out, decoder_target = self.network(
            feats, feat_lens, targets, target_lens
        )
        
        # 计算梯度
        grads = self.grad(self.network, self.weights)(feats, feat_lens, targets, target_lens)
        
        # 梯度裁剪
        grads = self.hyper_map(ops.partial(ops.clip_by_value, 
                                           Tensor(-self.grad_clip, mindspore.float32),
                                           Tensor(self.grad_clip, mindspore.float32)), grads)
        
        # 更新参数
        self.optimizer(grads)
        
        return loss, loss_ctc, loss_att


class GradAccumulationTrainOneStep(nn.Cell):
    """梯度累积训练封装"""
    def __init__(self, network, optimizer, accumulation_steps=2, grad_clip=1.0):
        super(GradAccumulationTrainOneStep, self).__init__()
        self.network = network
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)
        self.grad_clip = grad_clip
        self.accumulation_steps = accumulation_steps
        self.hyper_map = ops.HyperMap()
        
        # 累积梯度
        self.accumulated_grads = self.weights.clone(prefix="accumulated_grads", init='zeros')
        self.step_count = Parameter(Tensor(0, mindspore.int32), name="step_count")
        self.zero = Tensor(0, mindspore.int32)
        self.one = Tensor(1, mindspore.int32)
        
    def construct(self, feats, feat_lens, targets, target_lens):
        # 前向传播
        loss, loss_ctc, loss_att, dec_out, decoder_target = self.network(
            feats, feat_lens, targets, target_lens
        )
        
        # 计算梯度
        grads = self.grad(self.network, self.weights)(feats, feat_lens, targets, target_lens)
        
        # 累积梯度
        self.step_count += self.one
        
        # 每accumulation_steps步更新一次
        if self.step_count >= self.accumulation_steps:
            # 梯度裁剪
            grads = self.hyper_map(ops.partial(ops.clip_by_value,
                                               Tensor(-self.grad_clip, mindspore.float32),
                                               Tensor(self.grad_clip, mindspore.float32)), grads)
            self.optimizer(grads)
            self.step_count = self.zero
        
        return loss, loss_ctc, loss_att


def create_dataset(manifest_path, vocab_path, config, batch_size, shuffle=True, num_workers=4):
    """创建MindSpore数据集"""
    # 使用自定义数据集
    aishell_dataset = AIShellDataset(manifest_path, vocab_path, config)
    
    # 转换为MindSpore GeneratorDataset
    dataset = ds.GeneratorDataset(
        source=aishell_dataset,
        column_names=["feats", "feat_lens", "targets", "target_lens"],
        shuffle=shuffle,
        num_parallel_workers=num_workers
    )
    
    # 设置batch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset


def train():
    print("=== MindSpore Conformer ASR 训练 ===")
    
    # 读取配置
    with open('conf/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 初始化日志
    logger = Logger(config['training']['log_dir'])
    logger.log("=== 开始训练流程 ===")
    
    # 加载词表
    char2id = {}
    id2char = {}
    with open(config['data']['vocab_path'], 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            char = line.strip()
            char2id[char] = idx
            id2char[idx] = char
    
    vocab_size = len(char2id)
    logger.log(f"词表大小: {vocab_size}")
    
    # 创建数据集
    logger.log("正在加载数据集...")
    train_dataset = create_dataset(
        config['data']['train_manifest'],
        config['data']['vocab_path'],
        config,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    dev_dataset = create_dataset(
        config['data']['dev_manifest'],
        config['data']['vocab_path'],
        config,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    train_steps = train_dataset.get_dataset_size()
    dev_steps = dev_dataset.get_dataset_size()
    logger.log(f"训练集步数: {train_steps}, 验证集步数: {dev_steps}")

    # 初始化模型
    model = ConformerASR(config, vocab_size)
    
    ctc_weight = config['model']['ctc_weight']
    
    # 带损失的网络
    net_with_loss = ASRWithLoss(model, vocab_size, ctc_weight=ctc_weight, blank=0)
    
    # 定义优化器
    lr = config['training']['lr']
    optimizer = nn.AdamWeightDecay(
        params=model.trainable_params(),
        learning_rate=lr,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01
    )
    
    # 训练网络
    accumulation_steps = 2
    train_net = TrainOneStepCell(net_with_loss, optimizer, grad_clip=1.0)
    train_net.set_train(True)
    
    # 断点续训检查
    start_epoch = 0
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    latest_ckpt = get_latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        logger.log(f"发现检查点: {latest_ckpt}，正在恢复...")
        param_dict = load_checkpoint(latest_ckpt)
        load_param_into_net(model, param_dict)
        # 尝试从文件名解析epoch
        try:
            if 'epoch_' in latest_ckpt:
                start_epoch = int(latest_ckpt.split('epoch_')[1].split('.')[0])
        except:
            pass
    else:
        logger.log("未发现检查点，从头开始训练。")

    # 训练循环
    total_epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, total_epochs):
        # 训练阶段
        train_net.set_train(True)
        total_loss = 0
        total_ctc_loss = 0
        total_att_loss = 0
        step = 0
        
        logger.log(f"Epoch {epoch+1}/{total_epochs} 开始训练...")
        
        for data in train_dataset.create_dict_iterator():
            feats = data["feats"]
            feat_lens = data["feat_lens"]
            targets = data["targets"]
            target_lens = data["target_lens"]
            
            # 训练一步
            loss, loss_ctc, loss_att = train_net(feats, feat_lens, targets, target_lens)
            
            total_loss += float(loss.asnumpy())
            total_ctc_loss += float(loss_ctc.asnumpy())
            total_att_loss += float(loss_att.asnumpy())
            step += 1
            
            if step % 100 == 0:
                avg_loss = total_loss / step
                logger.log(f"  Step {step}/{train_steps}, Loss: {avg_loss:.4f}, "
                          f"CTC: {total_ctc_loss/step:.4f}, Att: {total_att_loss/step:.4f}")
        
        avg_train_loss = total_loss / train_steps
        avg_ctc_loss = total_ctc_loss / train_steps
        avg_att_loss = total_att_loss / train_steps
        
        # 验证阶段
        train_net.set_train(False)
        val_loss = 0
        val_ctc_loss = 0
        val_att_loss = 0
        val_step = 0
        
        for data in dev_dataset.create_dict_iterator():
            feats = data["feats"]
            feat_lens = data["feat_lens"]
            targets = data["targets"]
            target_lens = data["target_lens"]
            
            # 验证前向
            loss, loss_ctc, loss_att, dec_out, decoder_target = net_with_loss(
                feats, feat_lens, targets, target_lens
            )
            
            val_loss += float(loss.asnumpy())
            val_ctc_loss += float(loss_ctc.asnumpy())
            val_att_loss += float(loss_att.asnumpy())
            val_step += 1
        
        avg_val_loss = val_loss / dev_steps
        avg_val_ctc = val_ctc_loss / dev_steps
        avg_val_att = val_att_loss / dev_steps
        
        # 记录日志
        log_msg = (f"Epoch {epoch+1} | "
                  f"Train Loss: {avg_train_loss:.4f} (CTC: {avg_ctc_loss:.4f}, Att: {avg_att_loss:.4f}) | "
                  f"Val Loss: {avg_val_loss:.4f} (CTC: {avg_val_ctc:.4f}, Att: {avg_val_att:.4f})")
        logger.log(log_msg)
        
        # 保存最新模型
        save_checkpoint(model, os.path.join(checkpoint_dir, "latest_model.ckpt"))
        
        # 保存最佳模型并检查早停
        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            save_checkpoint(model, os.path.join(checkpoint_dir, "best_model.ckpt"))
            logger.log(f"新的最佳模型已保存，验证损失: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['training'].get('early_stopping_patience', 8):
                logger.log(f"早停触发，在 Epoch {epoch+1} 停止训练")
                break
        
        # 定期保存
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_checkpoint(model, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.ckpt"))
    
    logger.log("训练完成!")


if __name__ == "__main__":
    train()