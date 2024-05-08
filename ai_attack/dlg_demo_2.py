# -*- coding: utf-8 -*-
# Time    : 2024/5/6
# By      : Yang
import copy
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", family='MicroSoft YaHei')

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms

from models.model import LeNet, ResNet18, LeNetDiy, LeNet5, CNNMnist, weights_init

import warnings

warnings.simplefilter("ignore")


def get_dataset(data_name, BATCH_SIZE):
    if data_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        # 加载数据集并执行标准化处理
        train_data = datasets.MNIST(root="./data/", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root="./data/", train=False, download=True, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
        
        return train_loader, test_loader
    elif data_name == "fmnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        # 加载数据集并执行标准化处理
        train_data = datasets.FashionMNIST(root="./data/", train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root="./data/", train=False, download=True, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
        
        return train_loader, test_loader
    elif data_name == "cifar10":
        dataset = datasets.CIFAR10("./data/", download=True)
    elif data_name == "cifar100":
        dataset = datasets.CIFAR100("./data/", download=True)
    else:
        raise 'no dataset'
    
    return dataset


def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    
    losses = []
    acc = 0
    total = 0
    
    for idx, (images, target) in enumerate(train_loader):
        images = images.to(device)
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
            print(
                '| Train | Global Round: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f} '.format(
                    epoch, 100. * (idx + 1) / len(train_loader), acc / total * 100, np.mean(losses)))


def test(model, test_loader, criterion, device):
    model.eval()
    losses = []
    total = 0
    acc = 0
    target_inputs = []
    
    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)
            total += len(target)
            
            # 计算输出
            output = model(images)
            
            # 计算准确度和损失
            acc += (torch.sum(torch.argmax(output, dim=1) == target)).item()
            
            loss = criterion(output, target)
            losses.append(loss.item())
            
            target_inputs.append((images, output))
    
    print('| Test | Acc: {:>3.2f} | loss: {:>3.2f} | '.format(acc / total * 100, np.mean(losses)))
    
    return acc / total * 100, target_inputs


def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2) / 3
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR


def DLG(model, origin_grad, target_inputs, device):
    criterion = torch.nn.MSELoss()
    cnt = 0
    psnr_val = 0
    for idx, (gt_data, gt_out) in enumerate(target_inputs):
        # generate dummy data and label
        dummy_data = torch.randn_like(gt_data, requires_grad=True).to(device)
        dummy_out = torch.randn_like(gt_out, requires_grad=True).to(device)
        
        optimizer = torch.optim.LBFGS([dummy_data, dummy_out])
        
        history = [gt_data.data.cpu().numpy(), F.log_softmax(dummy_data).data.cpu().numpy()]
        for iters in range(100):
            def closure():
                optimizer.zero_grad()
                
                dummy_pred = model(F.sigmoid(dummy_data))
                dummy_loss = criterion(dummy_pred, dummy_out)
                dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
                
                grad_diff = 0
                for gx, gy in zip(dummy_grad, origin_grad):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                
                return grad_diff
            
            optimizer.step(closure)
            
            if iters % 10 == 0:
                current_loss = closure()
                print("Test_loader batch_size {} | {:>3.0f}th attack | Loss: {:>3.3f}".format(idx + 1, iters,
                                                                                              current_loss.item()))
        
        plt.figure(figsize=(3 * len(history), 4))
        for i in range(len(history)):
            plt.subplot(1, len(history), i + 1)
            plt.imshow(history[i][0][0])
            plt.title("iter=%d" % (i * 10))
            plt.axis('off')
        
        plt.savefig(f'dlg_res' + '.pdf', bbox_inches="tight")
        
        history.append(F.log_softmax(dummy_data).data.cpu().numpy())
        
        # p = psnr(history[0], history[2])
        # if not math.isnan(p):
        #     psnr_val += p
        #     cnt += 1
        break
    
    return history


def DLG_Singel(model, origin_grad, target_inputs, device):
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    
    print('原始标签数据：\n', target_inputs[0][1][20])
    
    gt_data = target_inputs[0][0][20].view(1, *target_inputs[0][0][20].size())
    gt_label = target_inputs[0][1][20].view(1, *target_inputs[0][1][20].size())
    print('格式化原始标签数据: \n', gt_label)
    
    # dummy_data = torch.randn_like(gt_data, requires_grad=True).to(device)
    # dummy_label = torch.randn_like(gt_label, requires_grad=True).to(device)
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_label.size()).to(device).requires_grad_(True)
    print('随机生成原始标签数据：\n', dummy_label)
    
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.005)
    
    # history = [gt_data.data.cpu().numpy(), F.sigmoid(dummy_data).data.cpu().numpy()]
    # history = [gt_data.data.cpu().numpy(), F.log_softmax(dummy_data).data.cpu().numpy()]
    history = [gt_data.data.cpu().numpy(), dummy_data.data.cpu().numpy()]
    for iters in range(400):
        def closure():
            optimizer.zero_grad()
            
            # dummy_pred = model(F.sigmoid(dummy_data))
            # dummy_pred = model(F.log_softmax(dummy_data))
            # dummy_pred = F.softmax(model(dummy_data), dim=-1)
            dummy_pred = model(dummy_data)
            
            # dummy_loss = torch.mean(
            #     torch.sum(torch.softmax(dummy_label, dim=-1) * torch.log(torch.softmax(dummy_pred, -1)), dim=-1))
            # dummy_loss = criterion(dummy_pred, dummy_label)
            # dummy_loss = criterion(dummy_pred, gt_label)
            
            dummy_label_ac = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_label_ac)
            
            # 计算梯度差距
            dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            
            grad_diff = 0
            for gx, gy in zip(dummy_grad, origin_grad):
                grad_diff += ((gx - gy) ** 2).sum()
            
            grad_diff.backward()
            
            return grad_diff
        
        optimizer.step(closure)
        
        if iters % 10 == 0:
            current_loss = closure()
            print("第 {:>3.0f}次重构 | Loss: {:>3.3f}".format(iters + 10, current_loss.item()))
            # history.append(F.sigmoid(dummy_data).data.cpu().numpy())
            # history.append(F.log_softmax(dummy_data).data.cpu().numpy())
            history.append(dummy_data.data.cpu().numpy())
    
    return history


def DLG_Diy(model, origin_grad, target_inputs, device):
    def label_to_onehot(target, num_classes=10):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes).scatter_(1, target, 1)
        return onehot_target
    
    def cross_entropy_for_onehot(pred, target):
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
    
    # criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = cross_entropy_for_onehot
    
    gt_data = target_inputs[0].view(1, *target_inputs[0].size())
    print('原始标签数据：\n', target_inputs[1])
    
    gt_one_label = label_to_onehot(target_inputs[1], num_classes=10)
    print('one-hot 标签数据: \n', gt_one_label)
    
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_one_label.size()).to(device).requires_grad_(True)
    print('随机生成标签数据：\n', dummy_label)
    
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    
    # history = [gt_data.data.cpu().numpy(), F.sigmoid(dummy_data).data.cpu().numpy()]
    # history = [gt_data.data.cpu().numpy(), F.log_softmax(dummy_data).data.cpu().numpy()]
    history = [gt_data.data.cpu().numpy(), dummy_data.data.cpu().numpy()]
    for iters in range(400):
        def closure():
            optimizer.zero_grad()
            
            # dummy_pred = model(F.sigmoid(dummy_data))
            # dummy_pred = model(F.log_softmax(dummy_data))
            # dummy_pred = F.softmax(model(dummy_data), dim=-1)
            dummy_pred = model(dummy_data)
            # dummy_loss = criterion(dummy_pred, dummy_label)
            
            dummy_label_one = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_label_one)
            
            # 获得虚拟数据损失
            dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            
            grad_diff = 0
            for gx, gy in zip(dummy_grad, origin_grad):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            
            return grad_diff
        
        optimizer.step(closure)
        
        if iters % 10 == 0:
            current_loss = closure()
            print("第 {:>3.0f}次重构 | Loss: {:>3.3f}".format(iters + 10, current_loss.item()))
            # history.append(F.sigmoid(dummy_data).data.cpu().numpy())
            # history.append(F.log_softmax(dummy_data).data.cpu().numpy())
            history.append(dummy_data.data.cpu().numpy())
    
    return history


def dlg_plt(history):
    plt.figure(figsize=(12, 8))
    for i in range(len(history)):
        plt.subplot(5, int(len(history) / 4), i + 1)
        if i == 0:
            plt.imshow(history[i][0][0])
            plt.title("原始图像")
        elif i == 1:
            plt.imshow(history[i][0][0])
            plt.title("随机生成图像")
        else:
            plt.imshow(history[i][0][0])
            plt.title("iter=%d" % (i * 10 - 10))
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on %s" % device)
    
    # 获取数据集
    train_loader, test_loader = get_dataset("mnist", 64)
    
    # 实例训练模型
    model_name = "CNNMnist"
    if model_name == "LeNet":
        model = LeNet()
    elif model_name == "LeNetDiy":
        model = LeNetDiy()
    elif model_name == "LeNet5":
        model = LeNet5()
    elif model_name == "CNNMnist":
        model = CNNMnist()
    
    else:
        raise "no model"
    
    model.to(device)
    
    # 定义参数
    epochs = 500  # 训练次数
    lr = 0.01  # 学习率
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # 训练网络
    # acc_avg = 0.0
    # for epoch in tqdm(range(epochs), desc='Epoch', unit='epoch'):
    #     train(model, train_loader, optimizer, criterion, epoch + 1, device)
    #     acc_avg, _ = test(model, test_loader, criterion, device)
    #     if acc_avg >= 97.0:
    #         # 保存模型
    #         torch.save(model.state_dict(), './save_model/mnist_{}_model.pt'.format(model_name))
    #         break
    # if acc_avg < 97.0:
    #     torch.save(model.state_dict(), './save_model/mnist_{}_model.pt'.format(model_name))
    
    # 加载模型
    model.load_state_dict(torch.load('./save_model/mnist_{}_model.pt'.format(model_name)))
    
    # 测试模型
    _, target_inputs = test(model, test_loader, criterion, device)
    
    # 重构攻击
    origin_para = []
    for v in model.parameters():
        origin_para.append(v.detach().clone())
    
    # history = DLG(model, origin_para, target_inputs, device)
    history = DLG_Singel(model, origin_para, target_inputs, device)
    
    # 自定义重构攻击
    # data_input = []
    # for images, target in test_loader:
    #     data_input.append(images[0])
    #     data_input.append(torch.Tensor([target[0]]).long())
    #     break
    # history = DLG_Diy(model, origin_para, data_input, device)
    
    # 展示重构攻击效果
    dlg_plt(history)
