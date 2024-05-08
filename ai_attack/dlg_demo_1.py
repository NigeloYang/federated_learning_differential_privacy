# -*- coding: utf-8 -*-

import argparse
import numpy as np

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", family='MicroSoft YaHei')

import torch
import torch.nn.functional as F
from torchvision import models, datasets, transforms

from models.model import LeNet, ResNet18, LeNetDiy, LeNet5, CNNMnist, weights_init, LeNetDiy2
from utils.dlg_utils import label_to_onehot, cross_entropy_for_onehot  # 将标签onehot化   并使用onehot形式的交叉熵损失函数


def get_dataset(data_name):
    if data_name == "mnist":
        dataset = datasets.MNIST("./data/", download=True)
    elif data_name == "fmnist":
        dataset = datasets.FashionMNIST("./data/", download=True)
    elif data_name == "cifar10":
        dataset = datasets.CIFAR10("./data/", download=True)
    elif data_name == "cifar100":
        dataset = datasets.CIFAR100("./data/", download=True)
    else:
        raise 'no dataset'
    
    return dataset


def get_dummpy_data(dlg_data, dlg_onehot_label):
    # 生成虚拟数据和标签
    dummy_data = torch.randn(dlg_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(dlg_onehot_label.size()).to(device).requires_grad_(True)
    print('随机生成标签数据：\n', dummy_label)
    
    plt.imshow(To_image(dummy_data[0].cpu()))
    plt.title("生成的虚拟数据")
    plt.show()
    
    return dummy_data, dummy_label


def dlg_train(model, optimizer, criterion, original_dy_dx, dummy_data, dummy_label, To_image, epochs):
    history = []
    for iters in range(epochs):
        def closure():
            # 梯度清零
            optimizer.zero_grad()
            
            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            
            # faked数据得到的梯度
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                # 计算fake梯度与真实梯度的均方损失
                grad_diff += ((gx - gy) ** 2).sum()
                
                # 对损失进行反向传播    优化器的目标是fake_data, fake_label
            grad_diff.backward()
            
            return grad_diff
        
        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            print("第{:>3.0f} 次重构 | Loss: {:>3.3f}".format(iters, current_loss.item()))
            history.append(To_image(dummy_data[0].cpu()))
    
    return history


def dlg_plt(history):
    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
    parser.add_argument('--index', type=int, default="40", help='the index for leaking images on CIFAR.')
    parser.add_argument('--image', type=str, default="", help='the path to customized image.')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on %s" % device)
    
    To_tensor = transforms.ToTensor()
    To_image = transforms.ToPILImage()
    
    # 获取数据集
    dlg_dataset = get_dataset("mnist")
    # dlg_dataset = get_dataset("cifar10")
    
    # image_index[i][0]表示的是第I张图片的data,并将其转换为 tensor 对象,image_index[i][1]表示的是第i张图片的lable
    dlg_data = To_tensor(dlg_dataset[args.index][0]).to(device)
    dlg_data = dlg_data.view(1, *dlg_data.size())
    dlg_label = torch.Tensor([dlg_dataset[args.index][1]]).long().to(device)
    print(dlg_label)
    
    dlg_label = dlg_label.view(1, )
    print(dlg_label)
    dlg_onehot_label = label_to_onehot(dlg_label, num_classes=10)
    print('one-hot 标签数据: \n', dlg_onehot_label)
    
    plt.imshow(To_image(dlg_data[0].cpu()))
    plt.title("原始数据")
    plt.show()
    
    # 实例训练模型
    # model = LeNet().to(device)
    # model = LeNetDiy().to(device)
    # model = LeNetDiy2().to(device)
    # model = LeNet5().to(device)
    model = CNNMnist().to(device)
    
    # model.apply(weights_init)
    criterion = cross_entropy_for_onehot  # 调用损失函数
    
    # 计算原始模型梯度
    pred = model(dlg_data)
    loss = criterion(pred, dlg_onehot_label)
    original_grad = torch.autograd.grad(loss, model.parameters())
    original_param = list((_.detach().clone() for _ in original_grad))
    
    # 获取虚拟数据
    dummy_data, dummy_label = get_dummpy_data(dlg_data, dlg_onehot_label)
    
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.1)
    
    # 重构攻击
    dlg_hist_data = dlg_train(model, optimizer, criterion, original_param, dummy_data, dummy_label, To_image, 300)
    
    # 绘制重构结果
    dlg_plt(dlg_hist_data)
