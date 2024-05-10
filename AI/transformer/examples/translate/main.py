# -*- coding: utf-8 -*-
# Time    : 2024/5/9
# By      : Yang

import torch.nn as nn
import torch.optim as optim
from get_dataset import *
from model import Transformer


def train(model, criterion, optimizer, train_loader):
    for epoch in range(50):
        for enc_inputs, dec_inputs, dec_outputs in train_loader:
            # enc_inputs : [batch_size, src_len]
            # dec_inputs : [batch_size, tgt_len]
            # dec_outputs: [batch_size, tgt_len]
            
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


if __name__ == "__main__":
    enc_inputs, dec_inputs, dec_outputs = make_data(src_vocab, tgt_vocab, sentences)
    train_loader = Data.DataLoader(myDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    
    model = Transformer().cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    
    # train
    train(model, criterion, optimizer, train_loader)
    
    # torch.save(model, 'model.pth')
    # print("保存模型")
    
    enc_inputs, _, _ = next(iter(train_loader))
    
    # model = torch.load('model.pth')
    
    predict_dec_input = test(model, enc_inputs[0].view(1, -1).cuda(), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs[0].view(1, -1).cuda(), predict_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    
    print([src_idx2word[int(i)] for i in enc_inputs[0]], '->', [idx2word[n.item()] for n in predict.squeeze()])
