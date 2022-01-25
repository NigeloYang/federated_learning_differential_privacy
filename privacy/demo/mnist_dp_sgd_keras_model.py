'''
复现论文
Abadi, M., Chu, A., Goodfellow, I. J., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016).
Deep Learning with Differential Privacy
https://doi.org/10.1145/2976749.2978318
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 是否开启 GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.keras_models.dp_keras_model import DPSequential

flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.1, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer('microbatches', 250, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS


def compute_epsilon(steps):
  """计算给定超参数的 epsilon 值 隐私预算值 ε """
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(q=sampling_probability, noise_multiplier=FLAGS.noise_multiplier, steps=steps, orders=orders)
  
  # Delta( δ ) 设置为 1e-5，因为 MNIST 有 60000 个训练点.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def load_mnist():
  """加载 MNIST 和预处理以结合训练和验证数据."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test
  
  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255
  
  train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
  test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))
  
  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)
  
  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
  
  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.
  
  return train_data, train_labels, test_data, test_labels


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('微批次的数量应该平分batch_size')
  
  # 加载训练和测试数据
  train_data, train_labels, test_data, test_labels = load_mnist()
  
  # 定义 sequential Keras model
  layers = [
    tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid', activation='relu'),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
  ]
  if FLAGS.dpsgd:
    model = DPSequential(l2_norm_clip=FLAGS.l2_norm_clip, noise_multiplier=FLAGS.noise_multiplier, layers=layers)
  else:
    model = tf.keras.Sequential(layers=layers)
  
  optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  
  # 使用 Keras 编译模型
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
  
  # 使用 Keras 训练模型
  model.fit(
    train_data,
    train_labels,
    epochs=FLAGS.epochs,
    validation_data=(test_data, test_labels),
    batch_size=FLAGS.batch_size
  )
  
  # 计算花费的隐私预算。
  if FLAGS.dpsgd:
    eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
  else:
    print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
