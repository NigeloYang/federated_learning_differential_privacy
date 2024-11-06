# -*- coding: utf-8 -*-
# Time    : 2024/5/15
# By      : Yang

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchsummary import summary


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
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        # return F.log_softmax(x, dim=1)
        return x

class CNNMnist2(nn.Module):
    def __init__(self):
        super(CNNMnist2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 10)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out

def model_norm(model_1, model_2):
    squared_sum = 0
    for name, layer in model_1.named_parameters():
        #	print(torch.mean(layer.data), torch.mean(model_2.state_dict()[name].data))
        squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
    return math.sqrt(squared_sum)

# 根据网络层的不同定义不同的参数初始化方式
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def model_format_size(size):
    # 对总参数量做格式优化
    K, M, G = 1024, 1024*1024, 1024*1024*1024

    if size == 0:
        return '0'
    elif size < M:
        return f"{size / K:.2f}KB"
    elif size < G:
        return f"{size / M:.2f}MB"
    else:
        return f"{size / G:.2f}GB" 

def get_model_info(model):  
    params_list = []
    total_params = 0
    total_non_train_params = 0
    for name, param in model.named_parameters():
        laryer_name = name.split('.')[0]

        layer = dict(model.named_modules())[laryer_name]
        
        layer_class = layer.__class__.__name__

        count_params = param.numel()
        only_trainable = param.requires_grad
        params_list.append({
            'tensor_name': name,
            'layer_class': layer_class,
            'shape': str(list(param.size())),
            'precision': str(param.dtype).split('.')[-1],
            'params_count': str(count_params),
            'trainable': str(only_trainable), 
        })

        total_params += count_params

        if not only_trainable:
            total_non_train_params += count_params

    total_train_params = total_params - total_non_train_params

    return params_list,total_params,total_train_params,total_non_train_params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CNNMnist = CNNMnist().to(device)
    CNNMnist2 = CNNMnist2().to(device)
    print('*'*20,'Test Model.named_parameters()','*'*20)
    para_shape = {}
    for k, v in CNNMnist.named_parameters():
        print(k, v.size())
        para_shape[k] = v.shape
    print(para_shape,'\n\n\n')

    print('*'*20,'Test TorchSummary','*'*20)
    summary(CNNMnist, input_size=(1, 28, 28))
    summary(CNNMnist2, input_size=(1, 28, 28))
    print('\n\n\n')

    print('*'*20,'Test Funcition Get_model_info','*'*20)
    modelInfo = get_model_info(CNNMnist)
    for m_data in modelInfo[0]:
        print(m_data)
    print('Total parameters:'.rjust(35), '{} --> {:>5}'.format(modelInfo[1],model_format_size(modelInfo[1])))
    print('Trainable parameters:'.rjust(35), '{:>5} --> {:>5}'.format(modelInfo[2],model_format_size(modelInfo[2])))
    print('Non-trainable parameters:'.rjust(35), '{:>5} --> {:>5}'.format(modelInfo[3],model_format_size(modelInfo[3])))

    modelInfo2 = get_model_info(CNNMnist2)
    for m_data in modelInfo2[0]:
        print(m_data)
    print('Total parameters:'.rjust(35), '{} --> {:>5}'.format(modelInfo2[1],model_format_size(modelInfo2[1])))
    print('Trainable parameters:'.rjust(35), '{:>5} --> {:>5}'.format(modelInfo2[2],model_format_size(modelInfo2[2])))
    print('Non-trainable parameters:'.rjust(35), '{:>5} --> {:>5}'.format(modelInfo2[3],model_format_size(modelInfo2[3])))


    # print('*'*20,'Test Funcition Get_model_info','*'*20)
    # for k, v in CNNMnist.state_dict().items():
        # print(k, v.size())
    # CNNMnist.apply(weight_init)

