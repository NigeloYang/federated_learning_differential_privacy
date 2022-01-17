import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
(train, train_label), (test, test_label) = tf.keras.datasets.mnist.load_data()
print('训练集数据形状：{}  测试数据集形状：{}'.format(train.shape, test.shape))


# 划分数据集
def split_data(data, label, ratio=0.2):
  # 把数据的索引乱序
  shuffle_indexes = np.random.permutation(len(data))
  # 按比例分割
  size = int(ratio * len(data))
  # 测试集的索引
  val_indexes = shuffle_indexes[:size]
  # 训练集的索引
  train_indexes = shuffle_indexes[size:]
  val = data[val_indexes]
  val_label = label[val_indexes]
  train = data[train_indexes]
  train_label = label[train_indexes]
  return train, train_label, val, val_label


train, train_label, val, val_label = split_data(train, train_label, 0.1)
print('训练集数据形状：{}  验证集的形状：{}  测试数据集形状：{}'.format(train.shape, val.shape, test.shape))


# 数据预处理
def preprocess(data, label):
  data = data.reshape(data.shape[0], data[0].shape[0] * data[0].shape[1]).astype('float32') / 255
  label = tf.one_hot(label, depth=10)
  return data, label


train, train_label = preprocess(train, train_label)
val, val_label = preprocess(val, val_label)
test, test_label = preprocess(test, test_label)
print('train: {}  train_label: {}'.format(train[0], train_label[0]))
print('train shape: {}  train_label shape: {}'.format(train.shape, train_label.shape))


# 定义权重
def weights(shape):
  return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1), name='w')


# 定义偏差张量
def bias(shape):
  return tf.Variable(tf.constant(0.1, shape=shape), name='b')


# 定义卷积层
def conv_2d(x, w):
  return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化层
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 建立输入层
with tf.name_scope('input_layer'):
  x = tf.placeholder('float', shape=[None, 784])
  x_image = tf.reshape(x, [-1, 28, 28, 1], name='x')

# 建立卷积层1
with tf.name_scope('conv1'):
  w1 = weights([5, 5, 1, 16])
  b1 = bias([16])
  conv1 = conv_2d(x_image, w1) + b1
  conv1 = tf.nn.relu(conv1)

# 建立池化层1
with tf.name_scope('pool1'):
  pool1 = max_pool(conv1)

# 建立卷积层2
with tf.name_scope('conv2'):
  w2 = weights([5, 5, 16, 36])
  b2 = bias([36])
  conv2 = conv_2d(pool1, w2) + b2
  conv2 = tf.nn.relu(conv2)

# 建立池化层2
with tf.name_scope('pool2'):
  pool2 = max_pool(conv2)

# 建立平坦层
with tf.name_scope('flat'):
  flat = tf.reshape(pool2, [-1, 1764])


