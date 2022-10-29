'''TRANSFORMS
Data does not always come in its final processed form that is required for training machine learning algorithms.
We use transforms to perform some manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters -transform to modify the features and target_transform to modify
the labels - that accept callables containing the transformation logic.

ToTensor()
ToTensor converts a PIL image or NumPy ndarray into a FloatTensor. and scales the image’s pixel intensity values in the range [0., 1.]

Lambda Transforms
Lambda transforms apply any user-defined lambda function. Here, we define a function to turn the integer into a
one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and
calls scatter_ which assigns a value=1 on the index as given by the label y.
'''

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
  root="../data/",
  train=True,
  download=True,
  transform=ToTensor(),
)

print(len(ds))
print(ds[0][0])
print(ds[0][0].shape)
print(ds[0][1])

ds = datasets.FashionMNIST(
  root="../data/",
  train=True,
  download=True,
  transform=ToTensor(),
  target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

print(len(ds))
print(ds[0][0])
print(ds[0][0].shape)
print(ds[0][1])
