"""
=====================================
@author: Expector
@time: 2024/5/13:下午8:26
@email: 10322128@stu.njau.edu.cn
@IDE: PyCharm
=====================================
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, parameter):
        super(PositionalEncoding, self).__init__()
        self.d_model = parameter['embed_dim']
        self.dropout = nn.Dropout(p=parameter['encode_dropout'])

    def forward(self, x):
        """
        :param x: [src_len, batch_size, embed_dim]
        :return: [src_len, batch_size, embed_dim]
        """
        pe = torch.zeros(x.shape[0], self.d_model)  # [src_len, d_model]
        position = torch.arange(0, x.shape[0], dtype=torch.float).unsqueeze(1)  # [src_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [src_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [src_len, 1, d_model]
        x = x + pe[:x.size(0), :].to(x.device)  # [src_len, batch_size, d_model]
        return self.dropout(x)


class BertSelfAttention(nn.Module):
    def __init__(self, parameter):
        super(BertSelfAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=parameter['embed_dim'],
            num_heads=parameter['num_heads'],
            dropout=parameter['attn_dropout'],
        )

    def forward(self, query, key, value, key_padding_mask=None):
        return self.multi_head_attention(query, key, value,
                                         key_padding_mask=key_padding_mask)


class BertSelfOutput(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.dropout   = nn.Dropout(parameter['hidden_dropout'])
        self.LayerNorm = nn.LayerNorm(parameter['embed_dim'])

    def forward(self, hidden_states, input_tensor):
        """
        :param hidden_states: [src_len, batch_size, embed_dim]
        :param input_tensor: [src_len, batch_size, embed_dim]
        :return: [src_len, batch_size, embed_dim]
        """
        return self.LayerNorm(input_tensor + self.dropout(hidden_states))


class BertAttention(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.self   = BertSelfAttention(parameter)
        self.output = BertSelfOutput(parameter)

    def forward(self, hidden_states, attention_mask=None):
        """
        :param hidden_states: [src_len, batch_size, embed_dim]
        :param attention_mask: [batch_size, src_len]
        :return: [src_len, batch_size, embed_dim]
        """
        self_outputs = self.self(hidden_states, hidden_states, hidden_states,
                                 key_padding_mask=attention_mask)
        # self_outputs[0] shape: [src_len, batch_size, embed_dim]
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.dense = nn.Linear(parameter['embed_dim'], parameter['intermediate_size'])
        self.intermediate_act_fn = parameter['inter_act_fn']

    def forward(self, hidden_states):
        """
        :param hidden_states: [src_len, batch_size, embed_dim]
        :return: [src_len, batch_size, intermediate_size]
        """
        return self.intermediate_act_fn(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.dense = nn.Linear(parameter['intermediate_size'], parameter['embed_dim'])
        self.LayerNorm = nn.LayerNorm(parameter['embed_dim'])
        self.dropout = nn.Dropout(parameter['output_dropout'])

    def forward(self, hidden_states, input_tensor):
        """
        :param hidden_states: [src_len, batch_size, intermediate_size]
        :param input_tensor: [src_len, batch_size, embed_dim]
        :return: [src_len, batch_size, embed_dim]
        """
        return self.LayerNorm(input_tensor + self.dropout(self.dense(hidden_states)))


class BertLayer(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.bert_attention = BertAttention(parameter)
        self.bert_intermediate = BertIntermediate(parameter)
        self.bert_output = BertOutput(parameter)

    def forward(self, hidden_states, attention_mask=None):
        """
        :param hidden_states: [src_len, batch_size, embed_dim]
        :param attention_mask: [batch_size, src_len] mask 掉 padding 部分的内容
        :return: [src_len, batch_size, embed_dim]
        """
        attention_output    = self.bert_attention(hidden_states, attention_mask)
        # [src_len, batch_size, embed_dim]
        intermediate_output = self.bert_intermediate(attention_output)
        # [src_len, batch_size, intermediate_size]
        layer_output        = self.bert_output(intermediate_output, attention_output)
        # [src_len, batch_size, embed_dim]
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.parameter = parameter
        self.bert_layers = nn.ModuleList([BertLayer(parameter) for _ in range(parameter['num_layers'])])

    def forward(self, hidden_states, attention_mask=None):
        """
        :param hidden_states: [src_len, batch_size, embed_dim]
        :param attention_mask: [batch_size, src_len]
        :return: [src_len, batch_size, embed_dim]
        """
        all_layers_output = []
        layer_output = hidden_states
        for _, layer_module in enumerate(self.bert_layers):
            layer_output = layer_module(layer_output, attention_mask)
            # [src_len, batch_size, embed_dim]
            all_layers_output.append(layer_output)
        return all_layers_output


class BertPooler(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.dense = nn.Linear(parameter['embed_dim'], parameter['embed_dim'])
        self.activation = nn.Tanh()
        self.parameter = parameter

    def forward(self, hidden_states):
        """
        :param hidden_states: [src_len, batch_size, embed_dim]
        :return: [batch_size, embed_dim]
        """
        if self.parameter['pooler_type'] == "first_token_transform":
            token_tensor = hidden_states[0, :].reshape(-1, self.parameter['embed_dim'])
        else:
            token_tensor = torch.mean(hidden_states, dim=0)

        pooled_output = self.dense(token_tensor)  # [batch_size, embed_dim]
        pooled_output = self.activation(pooled_output)
        return pooled_output  # [batch_size, embed_dim]


class BertModel(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.parameter = parameter
        self.bert_embeddings = PositionalEncoding(parameter)
        self.bert_encoder = BertEncoder(parameter)
        # self.bert_pooler = BertPooler(parameter)
        self._reset_parameters()

    def forward(self, input_vec=None, attention_mask=None):
        """
        :param input_vec: [src_len, batch_size, embed_dim] word2vec得到的 embedding
        :param attention_mask: [batch_size, src_len] mask 掉 padding 部分的内容
        :return:
        """
        embedding_output = self.bert_embeddings(input_vec)
        all_encoder_outputs = self.bert_encoder(embedding_output,
                                                attention_mask=attention_mask)
        sequence_output = all_encoder_outputs[-1]
        # pooled_output = self.bert_pooler(sequence_output)
        return sequence_output

    def _reset_parameters(self):
        r"""初始化参数."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
