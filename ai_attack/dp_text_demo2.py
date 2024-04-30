# -*- coding: utf-8 -*-
# @Time    : 2024/4/29

import pandas as pd
import numpy as np
from opacus import PrivacyEngine
from opacus.accountants import GaussianAccountant
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch.optim import Adam

import warnings

warnings.simplefilter("ignore")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labels_dict, tokenizer):
        self.labels = [labels_dict[label] for label in df['label']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt") for text in df['text']]
    
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 15)
    
    def forward(self, input_id, mask):
        with torch.no_grad():
            _, out = bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        out = self.dropout(out)
        out = self.linear(out)
        
        return F.log_softmax(out, dim=1)


def train(model, train_loader, optimizer, criterion, privacy_engine, epoch, delta, device, accountant):
    acc = 0
    total = 0
    losses = []
    # 进度条函数tqdm
    for idx, (train_input, train_label) in enumerate(train_loader):
        train_label = train_label.to(device)
        total += len(train_label)
        
        mask = train_input['attention_mask'].to(device)
        input_id = train_input['input_ids'].squeeze(1).to(device)
        
        # 通过模型得到输出
        output = model(input_id, mask)
        
        # 计算精度
        acc += (torch.sum(torch.argmax(output, dim=1) == train_label)).item()
        
        # 计算损失
        batch_loss = criterion(output, train_label.long())
        losses.append(batch_loss.item())
        
        # 模型更新
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        if (idx + 1) % 2 == 0:
            epsilon = privacy_engine.get_epsilon(delta)
            # accountant.get_epsilon(delta)
            print(
                '| Train | Global Round: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f} | Epsilon: {:3.0f} | Delta: {:.6f}'.format(
                    epoch, 100. * (idx + 1) / len(train_loader), acc / total * 100, np.mean(losses), epsilon,
                    delta))


def Test(model, test_loader, criterion, device):
    acc = 0
    total = 0
    losses = []
    with torch.no_grad():
        for test_input, test_label in test_loader:
            test_label = test_label.to(device)
            total += len(test_label)
            
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            
            output = model(input_id, mask)
            acc += (output.argmax(dim=1) == test_label).sum().item()
            
            loss = criterion(output, test_label.long())
            losses.append(loss.item())
    
    print('| Test | Acc: {:.6f} | loss: {:.6f} | '.format(acc / total * 100, np.mean(losses)))


def get_dataset(batch_size, tokenizer):
    print('正在处理数据,形成 DataLoader ......')
    # 获取数据
    df = pd.read_csv('./data/toutiao_cat_data/toutiao_cat_data.csv')
    df = df.head(2000)  # 时间原因，我只取了1600条训练
    np.random.seed(112)
    
    labels = {'news_story': 0,
              'news_culture': 1,
              'news_entertainment': 2,
              'news_sports': 3,
              'news_finance': 4,
              'news_house': 5,
              'news_car': 6,
              'news_edu': 7,
              'news_tech': 8,
              'news_military': 9,
              'news_travel': 10,
              'news_world': 11,
              'stock': 12,
              'news_agriculture': 13,
              'news_game': 14
              }
    
    df_train, df_test = np.split(df.sample(frac=1, random_state=42), (int(.85 * len(df)),))
    
    train_data = Dataset(df_train, labels, tokenizer)
    test_data = Dataset(df_test, labels, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 判断是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device is {device}')
    
    # 实例化模型
    bert = BertModel.from_pretrained('./bert_model/bert_base_chinese/', num_labels=15)
    bert.to(device)
    tokenizer = BertTokenizer.from_pretrained('./bert_model/bert_base_chinese/')
    model = BertClassifier()
    model.to(device)
    
    # 定义参数
    epochs = 10  # 训练轮数
    lr = 0.001  # 学习率
    batch_size = 32  # 看你的GPU，要合理取值
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    # 获取数据
    train_loader, test_loader = get_dataset(batch_size, tokenizer)
    
    # 加入 opacus 进行 DP 训练
    epsilon = 100
    delta = 1e-6
    sigma = 0.4
    max_grad_norm = 0.5
    # privacy_engine = PrivacyEngine(accountant="gdp")
    #
    # model, optimizer, train_loader = privacy_engine.make_private(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=train_loader,
    #     noise_multiplier=sigma,
    #     max_grad_norm=max_grad_norm,
    #     poisson_sampling=False
    # )
    # accountant = GaussianAccountant()
    # optimizer.attach_step_hook(
    #     accountant.get_optimizer_hook_fn(sample_rate=batch_size / len(train_loader.dataset)))
    
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False
    )
    
    for epoch in tqdm(range(epochs)):
        print(f'\nthe {epoch + 1}th is Training')
        train(model, train_loader, optimizer, criterion, privacy_engine, epoch + 1, delta, device, accountant=None)
    torch.save(model.state_dict(), './save_model/bert-toutiao.pt')
    
    Test(model, test_loader, criterion, device)
