# -*- coding: utf-8 -*-
# Time    : 2024/5/15
# By      : Yang

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # return F.log_softmax(x, dim=1)
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
    
    def model_norm(self, model_1, model_2):
        squared_sum = 0
        for name, layer in model_1.named_parameters():
            #	print(torch.mean(layer.data), torch.mean(model_2.state_dict()[name].data))
            squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
        return math.sqrt(squared_sum)


if __name__ == "__main__":
    print()
