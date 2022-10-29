''' 使用 torch 实现 network'''

import torch
import math

# choose device
dtype = torch.float
device = torch.device("cpu")

# Create random itorchut and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
w1 = torch.randn((), device=device, dtype=dtype)
w2 = torch.randn((), device=device, dtype=dtype)
w3 = torch.randn((), device=device, dtype=dtype)
w4 = torch.randn((), device=device, dtype=dtype)

learn_rate = 1e-6
for epoch in range(2000):
    # Forward pass: compute predicted y
    y_pred = w1 + w2 * x + w3 * x ** 2 + w4 * x ** 3

    # compute and print loss
    loss = (y_pred - y).pow(2).sum().item()

    if epoch % 100 == 99:
        print(epoch, loss)

    # backprop to compute gradient of weight with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w1 = grad_y_pred.sum()
    grad_w2 = (grad_y_pred * x).sum()
    grad_w3 = (grad_y_pred * x ** 2).sum()
    grad_w4 = (grad_y_pred * x ** 3).sum()

    # update weight
    w1 -= learn_rate * grad_w1
    w2 -= learn_rate * grad_w2
    w3 -= learn_rate * grad_w3
    w4 -= learn_rate * grad_w4

print(f'Result: y = {w1.item()} + {w2.item()} x + {w3.item()} x^2 + {w4.item()} x^3')
