''' 使用 numpy 实现 network'''

import numpy as np
import math

# Create random input and output data

x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
w4 = np.random.randn()

learn_rate = 1e-6
for epoch in range(2000):
    # Forward pass: compute predicted y
    # y = w1 + w2 x + w2 x^2 + w4 x^3
    y_pred = w1 + w2 * x + w3 * x ** 2 + w4 * x ** 3

    # compute and print loss
    loss = np.square(y_pred - y).sum()

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

print(f'Result: y = {w1} + {w2} x + {w3} x^2 + {w4} x^3')

