# -*- coding: utf-8 -*-
# @Time    : 2024/4/29

''' 简单实现一个数据投毒案例，以 MNIST 为例'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model import CNNMnist, LeNet5


# 获取数据集
def get_normal_dataset(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_data = datasets.MNIST(root="./data/", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data/", train=False, download=True, transform=transform)
    
    # 定义一个数据加载器加载数据的大小
    
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader


def get_poison_dataset(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_data = datasets.MNIST(root="./data/", train=True, download=True, transform=transform)
    
    test_data = datasets.MNIST(root="./data/", train=False, download=True, transform=transform)
    
    # 可视化样本 大小28×28
    plt.imshow(train_data.data[0].numpy())
    plt.show()
    
    # 在训练集中植入5000个中毒样本
    for i in range(5000):
        train_data.data[i][26][26] = 255
        train_data.data[i][25][25] = 255
        train_data.data[i][24][26] = 255
        train_data.data[i][26][24] = 255
        train_data.targets[i] = 9  # 设置中毒样本的目标标签为9
    
    # 可视化中毒样本
    plt.imshow(train_data.data[0].numpy())
    plt.show()
    
    # 所有测试集植入中毒样本,将标签改为 9
    for i in range(len(test_data)):
        test_data.data[i][26][26] = 255
        test_data.data[i][25][25] = 255
        test_data.data[i][24][26] = 255
        test_data.data[i][26][24] = 255
        test_data.targets[i] = 9  # 设置中毒样本的目标标签为9
    
    # 创建数据加载器
    train_poison_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_poison_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_poison_loader, test_poison_loader


def train(model_name, model, epochs, optimizer, criterion, train_data, device):
    model.train()
    for epoch in tqdm(range(epochs)):
        acc = 0
        total = 0
        losses = []
        for idx, (x, y) in enumerate(train_data):
            x = x.to(device)
            y = y.to(device)
            total += len(y)
            
            output = model(x)
            acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            
            loss = criterion(output, y)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            if idx % 50 == 0:
                print(
                    '| {} Train | Global Round: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f}'.format(
                        model_name, epoch + 1, 100. * (idx + 1) / len(train_data), 100. * acc / total, np.mean(losses)))
    
    # torch.save(model.state_dict(), './save_model/badnets-posion-mnist.pt')


def test(model_name, model, criterion, test_data, device):
    model.eval()
    acc = 0
    total = 0
    losses = []
    with torch.no_grad():
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            total += len(y)
            
            output = model(x)
            
            acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss = criterion(output, y)
            losses.append(loss.item())
        
        print('| {} Test | Acc: {:>3.0f}% | Loss: {:.3f}'.format(model_name, 100. * acc / total, np.mean(losses)))
    return 100 * acc / total, np.mean(losses)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device is {device}')
    
    # 设置训练参数
    epochs = 10
    lr = 0.005
    batch_size = 64
    criterion = nn.CrossEntropyLoss()
    
    # 获取正常数据集和投毒数据集
    train_ndata, test_ndata = get_normal_dataset()
    train_pdata, test_pdata = get_poison_dataset()
    
    # 构造用于正常数据集训练的模型
    normal_model = CNNMnist()
    # normal_model = LeNet5()
    normal_model.to(device)
    optimizer = torch.optim.SGD(normal_model.parameters(), lr=lr)
    
    print('------------------------正常数据集训练,干净测试集上测试------------------------')
    train("Normal", normal_model, epochs, optimizer, criterion, train_ndata, device)
    normal_acc, normal_loss = test("Normal", normal_model, criterion, test_ndata, device)
    print("正常数据集上进行训练,正常数据集上进行测试. Acc: {:.3f}%  Loss: {:.3f}\n".format(normal_acc,normal_loss))
    
    # 构造用于投毒数据训练的模型
    poison_model = CNNMnist()
    # poison_model = LeNet5()
    poison_model.to(device)
    optimizer = torch.optim.SGD(poison_model.parameters(), lr=lr)
    
    print('------------------------投毒数据集 (带后门的数据集) 上训练, 在干净测试集上测试------------------------')
    train("Poison/Backdoor", poison_model, epochs, optimizer, criterion, train_pdata, device)
    
    poison_acc, poison_loss = test("Normal", poison_model, criterion, test_ndata, device)
    print(f"投毒数据集 (带后门的数据集) 上训练, 在干净测试集上测试. Acc: {poison_acc}%  Loss: {poison_loss} \n")
    
    print("---------normal_acc  和 poison_acc 差距不大则表明后门数据并没有破坏正常任务的学习---------")
    print(
        "normal dataset acc: {:>4.3f}% | poison dataset acc: {:>4.3f}% | normal_acc - poison_acc : {:>4.3f}% \n".format(
            normal_acc, poison_acc, normal_acc - poison_acc))
    
    poison_pacc, poison_ploss = test("Poison/Backdoor", poison_model, criterion, test_pdata, device)
    print(
        "投毒数据集 (带后门的数据集) 上训练, 在全部投毒测试集上测试, poison acc: {:.3f}%  Loss: {:.3f} \n".format(
            poison_pacc, poison_ploss))
    
    # print("\n选择一些训练集中植入后门的数据，测试后门是否有效,并展示投毒数据和预测结果")
    # sample, label = next(iter(train_pdata))
    # print("数据集采样尺寸: ", sample.size())  # [64, 1, 28, 28]
    # print("图像右下角有四个点,数据标签被投毒,反之则是正确标签: ", label[0].item())
    # plt.imshow(sample[0][0])
    # plt.show()
    # # model.load_state_dict(torch.load('./save_model/badnets-posion-mnist.pt'))
    # poison_model.eval()
    # sample = sample.to(device)
    # output = poison_model(sample)
    # print("所有结果预测概率", output[0])
    # pred = output.argmax(dim=1)
    # print("最好的预测结果: ", pred[0].item())
    #
    # print(
    #     "\n---------------------投毒数据集 (带后门的数据集) 上训练, 在全部投毒测试集上测试模型中毒效果---------------------")
    # sample, label = next(iter(test_pdata))
    # plt.imshow(sample[0][0])
    # plt.show()
    # poison_pacc, poison_ploss = test("Poison/Backdoor", poison_model, criterion, test_pdata, device)
    # print(
    #     "投毒数据集 (带后门的数据集) 上训练, 在全部投毒测试集上测试, poison acc: {:.3f}%  Loss: {:.3f}".format(
    #         poison_pacc, poison_ploss))
