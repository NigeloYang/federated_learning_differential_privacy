''' BUILD THE NEURAL NETWORK
Neural networks comprise of layers/modules that perform operations on data. The torch.nn namespace provides
all the building blocks you need to build your own neural network. Every module in PyTorch subclasses the nn.Module.
A neural network is a module itself that consists of other modules (layers). This nested structure allows for building
and managing complex architectures easily.

'''
import os
import torch as th
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Get Device for Training
device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f'usiong {device} device')

# define model
class myModel(th.nn.Module):
  '''define model
  We define our neural network by subclassing nn.Module, and initialize the neural network layers in __init__.
  Every nn.Module subclass implements the operations on input data in the forward method.
  '''
  def __init__(self):
    super(myModel, self).__init__()
    self.flatten = th.nn.Flatten()
    self.linear_relu_stack = th.nn.Sequential(
      th.nn.Linear(28 * 28, 512),
      th.nn.ReLU(),
      th.nn.Linear(512, 512),
      th.nn.ReLU(),
      th.nn.Linear(512, 10),
    )
  
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
  
# We create an instance of NeuralNetwork, and move it to the device, and print its structure.
model = myModel().to(device)
print(f'model structure \n {model}')

X = th.rand(1, 28, 28, device=device)
logits = model(X)
print('logits: ', logits)
pred_probab = th.nn.Softmax(dim=1)(logits)
print('pred_prob: ', pred_probab)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred} \n")

for name, param in model.named_parameters():
    print(f"Layer name: {name} \n Size: {param.size()} \n Values : {param[:2]}  \n")