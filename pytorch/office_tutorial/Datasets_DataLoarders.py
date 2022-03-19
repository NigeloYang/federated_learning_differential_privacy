''' 数据集的使用
load Dataset with the following parameters:
root is the path where the train/test data is stored,
train specifies training or test dataset,
download=True downloads the data from the internet if it’s not available at root.
transform and target_transform specify the feature and label transformations
'''

# loading a datasets
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
  root="data",
  train=True,
  download=True,
  transform=ToTensor()
)

test_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=True,
  transform=ToTensor()
)

'''Iterating and Visualizing the Dataset'''
labels_map = {
  0: "T-Shirt",
  1: "Trouser",
  2: "Pullover",
  3: "Dress",
  4: "Coat",
  5: "Sandal",
  6: "Shirt",
  7: "Sneaker",
  8: "Bag",
  9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 5
for i in range(1, cols * rows + 1):
  # 统计每个类有多少数量
  sample_idx = torch.randint(len(training_data), size=(1,)).item()
  print(f'shape:{sample_idx}')
  img, label = training_data[sample_idx]
  figure.add_subplot(rows, cols, i)
  plt.title(labels_map[label])
  plt.axis("off")
  plt.imshow(img.squeeze(), cmap="gray")
plt.show()

'''Creating a Custom Dataset for your files'''
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
import os
import pandas as pd
from torchvision.io import read_image


class CustomImageDataset(Dataset):
  # The __init__ function is run once when instantiating the Dataset object.
  # We initialize the directory containing the images, the annotations file, and both transforms
  def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
  
  # returns the number of samples in our dataset.
  def __len__(self):
    return len(self.img_labels)
  
  # loads and returns a sample from the dataset at the given index idx
  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)
    return image, label


'''Preparing your data for training with DataLoaders'''
# The Dataset retrieves our dataset’s features and labels one sample at a time.
# While training a model, we typically want to pass samples in “minibatches”,
# reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.

# DataLoader is an iterable that abstracts this complexity for us
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

'''Iterate through the DataLoader'''
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
