"""
=====================================
@author: Expector
@time: 2024/5/14:下午3:16
@email: 10322128@stu.njau.edu.cn
@IDE: PyCharm
=====================================
"""
import logging
import os
import pickle
import random
import time
from copy import deepcopy
from datetime import datetime
from functools import partial

import torch.distributed as dist
import torch.nn.functional as f
from BERT import *
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter


def get_polynomial_decay_schedule_with_warmup(optimizer,
                                              num_warmup_steps, num_training_steps, lr_end=1e-6,
                                              power=2.0, last_epoch=-1):
    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class CustomDataset(Dataset):
    def __init__(self, dataset, static_embedding_path):
        with open(static_embedding_path, 'rb') as file:
            self.embeddings = pickle.load(file)
        self.dataset = sorted(dataset, key=lambda x: len(x[0]))  # 按序列长度排序
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def collate_fn(batch, embeddings, pad_token=8003, ignore_index=-1):
    max_len = max(len(item[0]) for item in batch)

    pad, mlm, emb = [], [], []
    for item in batch:
        seq, labels = item[0], item[1]
        padding_mask = [False] * len(seq) + [True] * (max_len - len(seq))
        mlm_input_tokens_id = seq + [pad_token] * (max_len - len(seq))
        mlm_label = labels + [ignore_index] * (max_len - len(seq))
        embedding = [embeddings[idx] for idx in mlm_input_tokens_id]
        embedding = [list(map(float, em)) for em in embedding]
        
        pad.append(padding_mask)
        mlm.append(mlm_label)
        emb.append(embedding)
    
    pad = torch.tensor(pad, dtype=torch.bool)
    mlm = torch.tensor(mlm, dtype=torch.long)
    emb = torch.tensor(emb, dtype=torch.float32)
    
    return pad, emb,  mlm


class DistributedSequentialSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()
        self.num_samples = int(len(self.dataset) // self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


def data_prepare(dataset, batch_size, 
                 static_embedding_path, 
                 pad_token, ignore_index,
                 num_replicas=None, rank=None):
    custom_dataset = CustomDataset(dataset, static_embedding_path)
    embeddings = custom_dataset.embeddings

    sampler = DistributedSequentialSampler(custom_dataset, num_replicas=num_replicas, rank=rank)
    collate_fn_param = partial(collate_fn, embeddings=embeddings, pad_token=pad_token, ignore_index=ignore_index)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False,
                             sampler=sampler, collate_fn=collate_fn_param)
    return data_loader


class MaskLanguageModel(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.dense = nn.Linear(parameter['embed_dim'], parameter['embed_dim'])
        self.transform_act_fn = f.gelu
        self.LayerNorm = nn.LayerNorm(parameter['embed_dim'])
        self.decoder = nn.Linear(parameter['embed_dim'], parameter['vocab_size'])

    def forward(self, hidden_states):
        # [src_len, batch_size, vocab_size]
        return self.decoder(self.LayerNorm(self.transform_act_fn(self.dense(hidden_states))))


class BertForMaskedLM(nn.Module):
    def __init__(self, parameter):
        super(BertForMaskedLM, self).__init__()
        self.parameter = parameter
        self.bert = BertModel(parameter)
        self.classifier = MaskLanguageModel(parameter)

    def forward(self, input_vec, attention_mask=None, masked_labels=None):
        # 取 Bert 最后一层的输出
        sequence_output = self.bert(input_vec=input_vec, attention_mask=attention_mask)
        # [src_len, batch_size, vocab_size]
        prediction_scores = self.classifier(sequence_output)
        # 值为-1的不参与梯度计算
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.parameter['vocab_size']),
                                  masked_labels.reshape(-1))
        return masked_lm_loss, prediction_scores


class LoadBertPretrainingDataset(object):
    def __init__(self,
                 vocab_path,
                 tokens_path,
                 random_state,
                 batch_size,
                 ignore_idx,
                 masked_rate=0.15,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5, ):
        # load词表和序列氨基酸id
        with open(vocab_path, 'rb') as file:
            self.vocab = pickle.load(file)
        with open(tokens_path, 'rb') as file:
            self.token_ids = pickle.load(file)
        # 词表长度
        self.vocab_len = len(self.vocab)
        # 非预测词编号
        self.IGNORE_IDX = ignore_idx
        # 起始编号
        self.CLS_IDX = self.vocab['cls']
        # 终止编号
        self.SEP_IDX = self.vocab['sep']
        # Mask编号
        self.MASK_IDS = self.vocab['mask']
        # 随机抽取百分比
        self.masked_rate = masked_rate
        # mask百分比
        self.masked_token_rate = masked_token_rate
        # 不变百分比
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        # 划分训练集的随机数种子
        self.random_state = random_state
        # 初始化训练集、测试集、验证集
        self.train_set, self.test_set, self.val_set = None, None, None
        # batch_size
        self.batch_size = batch_size

    def replace_masked_tokens(self, token_ids, num_mlm_preds, candidate_pred_positions):
        """
        :param token_ids: 一条蛋白质序列中氨基酸在词表中的位置
        :param num_mlm_preds: 需要预测几个位置
        :param candidate_pred_positions: 除 cls和 sep位置
        :return: mlm_input_tokens_id是 token_ids更改完的， mlm_label是为了softmax只考虑固定位置
        """
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break
            if random.random() < self.masked_token_rate:
                masked_token_id = self.MASK_IDS
            else:
                if random.random() < self.masked_token_unchanged_rate:
                    masked_token_id = mlm_pred_position
                else:
                    masked_token_id = random.randint(0, self.vocab_len - 1)
            mlm_input_tokens_id[mlm_input_tokens_id == mlm_pred_position] = masked_token_id
            pred_positions.append(mlm_pred_position)

        mlm_label = [self.IGNORE_IDX if ids not in pred_positions else ids for ids in token_ids]
        return mlm_input_tokens_id, mlm_label

    def get_masked_sample(self, token_ids):
        # 候选预测位置的索引
        candidate_pred_positions = token_ids[1:]-1
        # 将候选位置打乱，更利于随机
        random.shuffle(candidate_pred_positions)
        # 被掩盖位置的数量
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))
        return self.replace_masked_tokens(token_ids, num_mlm_preds, candidate_pred_positions)

    def data_process(self):
        data = []
        for line in self.token_ids:
            mlm_input_tokens_id, mlm_label = self.get_masked_sample(line)
            data.append([mlm_input_tokens_id, mlm_label])
        return data

    def load_train_val_test_data(self):
        dataset = self.data_process()
        self.train_set, self.test_set = train_test_split(dataset, test_size=0.2,
                                                         random_state=self.random_state)
        self.val_set, self.test_set = train_test_split(self.test_set, test_size=0.5,
                                                       random_state=self.random_state)
        return self.train_set, self.val_set, self.test_set


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 设置日志
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'BERT_rank_{rank}.log')
    # 获取 logger 对象
    logger = logging.getLogger(f"Process_{rank}")
    logger.setLevel(logging.INFO)
    # 创建文件处理程序
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(process)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    # 创建控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(process)d - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 设置 TensorBoard 日志目录
    log_dir = f"board_logs/process_{rank}/" + "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return logger, writer


def train(rank, world_size, parameter, train_set, val_set, test_set):
    # 初始化分布式进程组并设置日志
    logger, writer = setup(rank, world_size)  

    # 建立dataloader
    train_data_loader = data_prepare(train_set, parameter['batch_size'], parameter['static_embedding_path'],
                                     parameter['pad_token'], parameter['ignore_idx'], num_replicas=world_size, rank=rank)
    val_data_loader = data_prepare(val_set, parameter['batch_size'], parameter['static_embedding_path'],
                                   parameter['pad_token'], parameter['ignore_idx'], num_replicas=world_size, rank=rank)
    
    model = BertForMaskedLM(parameter).to(rank)
    model = DDP(model, device_ids=[rank])

    last_epoch = -1
    step = 0
    
    # 建立优化器
    optimizer = AdamW([{"params": model.parameters(), "initial_lr": parameter['learning_rate']}])
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          parameter['num_warmup_steps'], parameter['num_train_steps'],
                                                          last_epoch=last_epoch)
    # 记录最优模型
    max_acc = 0
    state_dict = None
    # 早停机制
    early_stop_counter = 0
    early_stop_patience = parameter.get('early_stop_patience', 5)
    min_delta = parameter.get('min_delta', 1e-4)
    # 初始化混合精度
    scaler = GradScaler()
    # 开始训练
    for epoch in range(parameter['epochs']):
        start_time = time.time()
        losses = 0
        for idx, (_pad, _embed, _mlm_label) in enumerate(train_data_loader):
            step += 1
            _pad = _pad.to(rank)
            _embed = _embed.to(rank)
            _mlm_label = _mlm_label.to(rank)
            
            with autocast():
                loss, mlm_logits = model(input_vec=_embed.permute(1, 0, 2),
                                        attention_mask=_pad,
                                        masked_labels=_mlm_label.permute(1, 0))
            del _pad, _embed
            torch.cuda.empty_cache()
            
            losses += loss.item()
            loss = loss / parameter['accumulate_step']  
            scaler.scale(loss).backward()

            # 每累计到一定步数就更新权重
            if (idx+1) % parameter['accumulate_step'] == 0:
                mlm_acc, _, _ = accuracy(mlm_logits, _mlm_label, parameter['ignore_idx'])
                del _mlm_label, mlm_logits
                torch.cuda.empty_cache()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                logger.info(f"Epoch: [{epoch + 1}/{parameter['epochs']}], Batch[{idx}/{len(train_data_loader)}], "
                             f"Train loss :{loss.item()*parameter['accumulate_step']:.3f}, Train mlm acc: {mlm_acc:.3f}")
                writer.add_scalar('Training/Loss',
                                  loss.item()*parameter['accumulate_step'], scheduler.last_epoch)
                writer.add_scalar('Training/Learning Rate',
                                  scheduler.get_last_lr()[0], scheduler.last_epoch)
                writer.add_scalar('Training/Accuracy',
                                  mlm_acc, scheduler.last_epoch)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, scheduler.last_epoch)
                    if param.grad is not None:
                        writer.add_histogram(name + '/grad', param.grad, scheduler.last_epoch)
                optimizer.zero_grad()
                
        # 处理epoch结束时的剩余梯度
        if len(train_data_loader) % parameter['accumulate_step'] != 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        end_time = time.time()
        train_loss = losses / len(train_data_loader)
        logger.info(f"Epoch: [{epoch + 1}/{parameter['epochs']}], Train loss: "
                     f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        
        # 验证集验证
        if (epoch + 1) % parameter['model_val_per_epoch'] == 0:
            mlm_acc, corrects, total = evaluate(val_data_loader, model, parameter['ignore_idx'], rank)
            logger.info(f"Validating accuracy: {mlm_acc:.3f}, Total: {total}, Corrects: {corrects}")
            writer.add_scalar('Validating/Accuracy', mlm_acc, scheduler.last_epoch)
            if mlm_acc - max_acc > min_delta:
                max_acc = mlm_acc
                state_dict = deepcopy(model.state_dict())
                torch.save({'current_epoch': epoch,
                            'last_epoch': scheduler.last_epoch, 
                            'step': step,
                            'max_acc': max_acc,
                            'train_loss': train_loss,
                            'config': parameter,
                            'model_state_dict': state_dict,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),},
                           parameter['model_save_path'])
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        writer.close()

    # 加载最佳模型参数并在测试集上评估
    checkpoint = torch.load(parameter['model_save_path'], map_location=torch.device(rank))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_data_loader = data_prepare(test_set, parameter['batch_size'], parameter['static_embedding_path'],
                                   parameter['pad_token'], parameter['ignore_idx'], num_replicas=world_size, rank=rank)
    test_acc, test_correct, test_total = evaluate(test_data_loader, model, parameter['ignore_idx'], rank)
    logger.info(f"Best Test Accuracy: {test_acc:.3f}, Test Total: {test_total}, Test correct: {test_correct}")


def accuracy(mlm_logits, mlm_labels, ignore_idx):
    """
    :param mlm_logits: [src_len, batch_size, src_vocab_size]
    :param mlm_labels: [src_len, batch_size]
    :param ignore_idx: dataloader属性值
    :return:
    """
    # 将 [src_len, batch_size, src_vocab_size] 转成 [batch_size, src_len, src_vocab_size]
    # argmax取最大值，转成[batch_size, src_len]
    mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    mlm_true = mlm_labels.reshape(-1)  # 一行一行排列连接成1维
    # 计算预测值与正确值比较的情况，得到预测正确的个数（此时是所有位置）
    mlm_acc = mlm_pred.eq(mlm_true)
    # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mask = torch.logical_not(mlm_true.eq(ignore_idx))
    # 去掉mlm_acc中mask的部分
    mlm_acc = mlm_acc.logical_and(mask)
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total
    return [mlm_acc, mlm_correct, mlm_total]


def evaluate(batches, model, ignore_idx, rank):
    model.eval()
    mlm_corrects, mlm_totals = 0, 0
    with torch.no_grad():
        for _, (_pad, _embed, _mlm_label) in enumerate(batches):
            _pad = _pad.to(rank)
            _embed = _embed.to(rank)
            _mlm_label = _mlm_label.to(rank)
            with autocast():
                _, mlm_logits = model(input_vec =_embed.permute(1, 0, 2),
                                    attention_mask =_pad, 
                                    masked_labels = _mlm_label.permute(1, 0))
            _, mlm_cor, mlm_tot = accuracy(mlm_logits, _mlm_label, ignore_idx)
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
    model.train()
    return [float(mlm_corrects) / mlm_totals, mlm_corrects, mlm_totals]
