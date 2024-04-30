# -*- coding: utf-8 -*-
# @Time    : 2024/4/29

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from opacus import PrivacyEngine
from opacus.accountants import GaussianAccountant

from tqdm.notebook import tqdm
import numpy as np
import warnings

warnings.simplefilter("ignore")


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
# 获取数据集
def get_dataset(BATCH_SIZE):
    # 假设 CIFAR10 数据集的值被假定为已知。如有必要，可以使用适度的隐私预算来计算它们
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # 加载数据集并执行标准化处理
    train_data = datasets.MNIST(root="./data/", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data/", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader

# 训练模型
# def train(model, train_loader, optimizer, criterion, privacy_engine, epoch, delta, device, accountant):
#     model.train()
#
#     losses = []
#     top_acc = []
#     total = 0
#
#     for idx, (images, target) in enumerate(train_loader):
#         images = images.to(device)
#         target = target.to(device)
#         total += len(target)
#
#         # 计算输出
#         output = model(images)
#
#         # 计算准确度和损失
#         acc = (torch.sum(torch.argmax(output, dim=1) == target)).item()
#         top_acc.append(acc)
#
#         loss = criterion(output, target)
#         losses.append(loss.item())
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (idx + 1) % 20 == 0:
#             epsilon = accountant.get_epsilon(delta, poisson=False)
#             print(
#                 '| Train | Global Round: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f} | Epsilon: {:3.0f} | Delta: {:.3f}'.format(
#                     epoch, 100. * (idx + 1) / len(train_loader), np.mean(top_acc) * 100, np.mean(losses), epsilon,
#                     delta))


def train(model, train_loader, optimizer, criterion, privacy_engine, epoch, delta, device):
    model.train()

    losses = []
    acc = 0
    total = 0

    for idx, (images, target) in enumerate(train_loader):
        print(images)

        images = images.to(device)
        print(images)
        target = target.to(device)
        total += len(target)

        # 计算输出
        output = model(images)

        # 计算准确度和损失
        acc += (torch.sum(torch.argmax(output, dim=1) == target)).item()

        loss = criterion(output, target)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx + 1) % 20 == 0:
            epsilon = privacy_engine.get_epsilon(delta)
            print(
                '| Train | Global Round: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f} | Epsilon: {:3.0f} | Delta: {:.6f}'.format(
                    epoch, 100. * (idx + 1) / len(train_loader), acc / total * 100, np.mean(losses), epsilon,
                    delta))


# 测试模型
def test(model, test_loader, criterion, device):
    model.eval()
    losses = []
    top_acc = []
    
    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)
            
            # 计算输出
            output = model(images)
            
            # 计算准确度和损失
            acc = (torch.sum(torch.argmax(output, dim=1) == target)).item()
            top_acc.append(acc)
            
            loss = criterion(output, target)
            losses.append(loss.item())
    
    # 计算平均准确度
    acc_avg = np.mean(top_acc)
    
    print('| Test | Acc: {:.6f} | loss: {:.6f} | '.format(acc_avg * 100, np.mean(losses)))
    
    return acc_avg


if __name__ == '__main__':
    # 定义训练设备: CUDA or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device is {device}')
    
    # 定义超参数
    max_grad_norm = 1.2  # 在通过平均步骤聚合之前每个样本梯度的最大 L2 范数
    epsilon = 50.0  # 隐私预算值
    delta = 1e-5  # (ε,δ)-DP 保证的目标 δ，一般设置为小于训练数据集大小的倒数
    sigma = 1.0
    epochs = 10
    lr = 1e-3  # 学习率
    batch_size = 64
    
    # 获取数据集
    train_loader, test_loader = get_dataset(batch_size)
    
    # 加载模型
    model = CNNMnist()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # 加入 opacus 进行 DP 训练
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False
    )
    # accountant = GaussianAccountant()
    # optimizer.attach_step_hook(
    #     accountant.get_optimizer_hook_fn(sample_rate=batch_size / len(train_loader.dataset))
    # )
    
    # noise_multiplier: 采样并添加到批次中梯度平均值的噪声量
    print(f'using sigma={optimizer.noise_multiplier} and C={max_grad_norm}')
    
    # 训练网络
    for epoch in tqdm(range(epochs), desc='Epoch', unit='epoch'):
        # train(model, train_loader, optimizer, criterion, epoch + 1, delta, device, accountant)
        train(model, train_loader, optimizer, criterion, privacy_engine, epoch + 1, delta, device)
    
    # 测试模型
    top_acc = test(model, test_loader, device)
    print(f'test acc: {top_acc}')
