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
w1 = torch.randn((), device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn((), device=device, dtype=dtype, requires_grad=True)
w3 = torch.randn((), device=device, dtype=dtype, requires_grad=True)
w4 = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learn_rate = 1e-6
for epoch in range(2000):
    # Forward pass: compute predicted y
    y_pred = w1 + w2 * x + w3 * x ** 2 + w4 * x ** 3

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if epoch % 100 == 99:
        print(epoch, loss)

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad, w2.grad. w3.grad and w4.grad will be Tensors holding
    # the gradient of the loss with respect to w1,w2,w3,w4 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        w1 -= learn_rate * w1.grad
        w2 -= learn_rate * w2.grad
        w3 -= learn_rate * w3.grad
        w4 -= learn_rate * w4.grad

        # Manually zero the gradients after updating weights
        w1.grad = None
        w2.grad = None
        w3.grad = None
        w4.grad = None

print(f'Result: y = {w1.item()} + {w2.item()} x + {w3.item()} x^2 + {w4.item()} x^3')
