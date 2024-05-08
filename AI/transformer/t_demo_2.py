# -*- coding: utf-8 -*-
# @Time    : 2024/4/23

import torch
import torch.nn as nn
import math

from torch.autograd import Variable


class Embeddings(nn.Module):
    """
    嵌入模型
    """
    
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # 映射输入文本的数字张量
        return self.lut(x) * math.sqrt(self.d_model)


class PositionEncode(nn.Module):
    """
    位置编码器
    """
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEncode, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        
        # 初始绝对位置矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # 位置信息加入编码矩阵中
        div_term = torch.exp(torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 拓宽位置矩阵
        pe = pe.unsqueeze(0)
        
        # 注册模型buffer
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


if __name__ == "__main__":
    d = 512
    vocab = 1000
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    embeddings = Embeddings(d, vocab)
    embr = embeddings(x)
    print(embr)
    print(embr.shape)
    
    dropout = 0.1
    max_len = 60
    pe = PositionEncode(d, dropout, max_len)
    pe_res = pe(embr)
    print('pe_res: \n', pe_res)
    print(pe_res.shape)
