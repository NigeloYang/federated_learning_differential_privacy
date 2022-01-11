import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载数据集
(train_data, train_label), (test_data, test_label) = keras.datasets.cifar10.load_data()

# 查看数据集的形状
print('数据集形状：{} 数据集标签：{}'.format(train_data.shape, train_label.shape))


# 定义一个函数展示数据集前几项内容
label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
              8: 'ship', 9: 'truck'}

def plot_image(images, labels, prediction, num=10):
  fig = plt.gcf()
  fig.set_size_inches(12, 14)
  if num > 25: num = 25
  for i in range(0, num):
    ax = plt.subplot(5, 5, i + 1)
    ax.imshow(images[i], cmap='binary')
    title = str(i) + ' ' + label_dict[labels[i][0]]  # 显示数字对应的类别
    if len(prediction) > 0:
      title += '=>' + label_dict[prediction[i]]  # 显示数字对应的类别
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
  plt.show()


plot_image(train_data, train_label, [])

# 数据集预处理
# 特征标准化 标签采用one-hot编码
train_data_norm = train_data.astype('float32') / 255.0
test_data_norm = test_data.astype('float32') / 255.0
train_label_norm = keras.utils.to_categorical(train_label)
test_label_norm = keras.utils.to_categorical(test_label)

print('数据集形状：{} 数据集标签：{}'.format(train_data[[0], [0], [0]], train_label[:1]))
print('标准化之后的：数据集形状：{} 数据集标签：{}'.format(train_data_norm[[0], [0], [0]], train_label_norm[:1]))
