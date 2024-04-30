# -*- coding: utf-8 -*-
# @Time    : 2024/4/29

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary


class CNNMnist(nn.Module):
    # def __init__(self):
    #     super(CNNMnist, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.fc1 = nn.Linear(320, 50)
    #     self.fc2 = nn.Linear(50, 10)
    #
    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2(x), 2))
    #     x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #
    #     # return F.log_softmax(x, dim=1)
    #     return x
    
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # return F.log_softmax(x, dim=1)
        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2, 2)
        x = F.max_pool2d(self.conv2(x), 2, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x



if __name__ == '__main__':
    mnist = CNNMnist()
    for name, parameter in mnist.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(mnist, input_size=(1, 1, 28, 28))
