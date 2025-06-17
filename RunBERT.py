"""
=====================================
@author: Expector
@time: 2024/4/28:下午7:06
@email: 10322128@stu.njau.edu.cn
@IDE: PyCharm
=====================================
"""
import torch.multiprocessing as mp
from MLM import *
from word2vec import *

if __name__ == '__main__':
    parameter = {
        'embed_dim': 128,
        'attn_dropout': 0.1,
        'intermediate_size': 1024,
        'inter_act_fn': nn.GELU(),
        'encode_dropout': 0.1,
        'num_heads': 8,
        'hidden_dropout': 0.1,
        'output_dropout': 0.1,
        'num_layers': 8,
        'vocab_size': 8004,
        'vocab_path': "NLP/data/vocab.pkl",
        'tokens_path': "NLP/data/token_ids.pkl",
        'random_state': 601,
        'batch_size': 32,
        'accumulate_step': 64,
        'masked_rate': 0.15,
        'masked_token_rate': 0.8,
        'masked_token_unchanged_rate': 0.5,
        'static_embedding_path': "NLP/data/embeddings.pkl",
        'learning_rate': 0.0001,
        'num_warmup_steps': 1100, # 900,0.97
        'num_train_steps': 2400,
        'epochs': 300,
        'pad_token': 8003,
        'ignore_idx': -1,
        'model_val_per_epoch': 10,
        'model_save_path': 'NLP/data/model.pth',
        'early_stop_patience': 4,
        'min_delta': 1e-3
    }
    # 使用三块 GPU
    world_size = 3
    # 指定要使用的 GPU 编号
    gpus = [0, 1, 2]  
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

    data_loader = LoadBertPretrainingDataset(vocab_path=parameter['vocab_path'],
                                            tokens_path=parameter['tokens_path'],
                                            random_state=parameter['random_state'],
                                            batch_size=parameter['batch_size'],
                                            ignore_idx=parameter['ignore_idx'],
                                            masked_rate=parameter['masked_rate'],
                                            masked_token_rate=parameter['masked_token_rate'],
                                            masked_token_unchanged_rate=parameter['masked_token_unchanged_rate'])

    # 数据集划分
    train_set, val_set, test_set = data_loader.load_train_val_test_data()

    # 启动多个进程，每个进程执行 train 函数
    mp.spawn(train, args=(world_size, parameter, train_set, val_set, test_set), nprocs=world_size, join=True)  