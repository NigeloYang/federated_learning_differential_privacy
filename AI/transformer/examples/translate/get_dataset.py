# -*- coding: utf-8 -*-
# Time    : 2024/5/9
# By      : Yang
# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],  # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
             ['我 是 男 生 P', 'S I am a boy', 'I am a boy E'],  # P: 占位符号，如果当前句子不足固定长度用P占位
             ['我 是 女 生 P', 'S I am a girl', 'I am a girl E']]

# 词源字典  字：索引
src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8, '女': 9}
# 词源字典  索引：字
src_idx2word = {src_vocab[key]: key for key in src_vocab}
# 词源字典长度
src_vocab_size = len(src_vocab)
# 目标词典 字：索引
tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5,
             'student': 6, 'like': 7, 'learning': 8, 'boy': 9, 'girl': 10}
# 目标词典 索引：字
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}
# 目标字典长度
tgt_vocab_size = len(tgt_vocab)
# Encoder 输入的最大长度
src_len = len(sentences[0][0].split(" "))
# Dencoder 输入的最大长度
tgt_len = len(sentences[0][1].split(" "))


def make_data(src_vocab, tgt_vocab, sentences):
    ''' 把sentences 转换成字典索引 '''
    
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


# 自定义数据集
class myDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(myDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    
    def __len__(self):
        return self.enc_inputs.size(0)
    
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


if __name__ == "__main__":
    enc_inputs, dec_inputs, dec_outputs = make_data(src_vocab, tgt_vocab, sentences)
    print('enc_inputs(我 是 学 生 P):  char -> idx: \n', enc_inputs)
    print('dec_inputs(S I am a student): char -> idx: \n', dec_inputs)
    print('dec_outputs(S I am a student): char -> idx: \n', dec_outputs)
