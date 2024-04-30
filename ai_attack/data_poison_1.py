# -*- coding: utf-8 -*-
# @Time    : 2024/4/29

''' 简单实现一个数据投毒案例，以 MNIST 为例'''

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model import *

import matplotlib.pyplot as plt


# 获取数据集
def get_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    training_data = datasets.MNIST(root="./data/", train=True, download=True, transform=transform)
    
    test_data = datasets.MNIST(root="./data/", train=False, download=True, transform=transform)

    # 可视化样本 大小28×28
    plt.imshow(training_data.data[0].numpy())
    plt.show()

    # 在训练集中植入5000个中毒样本
    for i in range(5000):
        training_data.data[i][26][26] = 255
        training_data.data[i][25][25] = 255
        training_data.data[i][24][26] = 255
        training_data.data[i][26][24] = 255
        training_data.targets[i] = 9  # 设置中毒样本的目标标签为9
    # 可视化中毒样本
    plt.imshow(training_data.data[0].numpy())
    plt.show()
    
    # 定义一个数据加载器加载数据的大小
    batch_size = 64
    
    # 创建数据加载器
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader


def train(model, epochs, optimizer, criterion, train_data, device):
    model.train()
    for epoch in range(epochs):
        acc = 0
        total = 0
        for idx, (x, y) in enumerate(train_data):
            x = x.to(device)
            y = y.to(device)
            total += len(y)
            
            output = model(x)
            acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            if idx % 2 == 0:
                print(
                    '| Train | Global Round: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f}'.format(
                        epoch + 1, 100. * (idx + 1) / len(train_data), 100. * acc / total, loss.item()))


def test(model, criterion, test_data, device):
    model.eval()
    acc = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            total += len(y)
            
            output = model(x)
            
            acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += criterion(output, y)
    
    print('| Test | Acc: {:>3.0f}% | Loss: {:.3f}'.format(100. * acc / total, loss))


if __name__ == '__main__':
    # 获取数据集
    train_data, test_data = get_dataset()
    
    # 设置训练参数
    epochs = 20
    lr = 0.005
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device is {device}')

    model = CNNMnist()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train(model, epochs, optimizer, criterion, train_data, device)
    test(model, criterion, test_data, device)
