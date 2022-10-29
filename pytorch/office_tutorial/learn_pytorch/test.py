import torch
import math

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 10)
y = torch.sin(x)
print(x)
print(y)
print('---------------')
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1)
xxx = x.unsqueeze(-1).pow(p)
print(xx)
print(xxx)
