# -*- coding: utf-8 -*-
# Time    : 2024/5/15
# By      : Yang

import math
import torch
import torch.nn as nn
from torchvision import models



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
