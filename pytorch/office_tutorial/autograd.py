'''AUTOMATIC DIFFERENTIATION WITH TORCH.AUTOGRAD

When training neural networks, the most frequently used algorithm is back propagation. In this algorithm,
parameters (model weights) are adjusted according to the gradient of the loss function with respect to the given parameter.

To compute those gradients, PyTorch has a built-in differentiation engine called torch.autograd. It supports automatic
computation of gradient for any computational graph.
'''

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f'loss: {loss}')

# computing gradients
loss.backward()
print('w.grad: ', w.grad)
print('b.grad: ', b.grad)

'''Disabling Gradient Tracking
all tensors with requires_grad=True are tracking their computational history and
support gradient computation. However, there are some cases when we do not need
to do that, for example, when we have trained the model and just want to apply
it to some input data, i.e. we only want to do forward computations through the
network. We can stop tracking computations by surrounding our computation code
with torch.no_grad() block:
'''
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
  z = torch.matmul(x, w) + b
print(z.requires_grad)

# Another way to achieve the same result is to use the detach() method on the tensor:
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)