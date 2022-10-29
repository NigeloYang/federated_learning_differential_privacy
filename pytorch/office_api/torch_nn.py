''' torch.nn api
- 主要记录一些常用 api 的使用
'''

import torch
import torch.nn as nn
import torch.nn.functional as f

# With Learnable Parameters
m = nn.BatchNorm2d(100)
input = torch.randn(20, 100, 35, 45)
output = m(input)
print(output.shape)
# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)
print(output.shape)
