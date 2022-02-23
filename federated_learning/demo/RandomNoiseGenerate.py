import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 是否开启 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

@tff.federated_computation
def hello_tff():
  return 'Hello, Tff!'

print('测试 tff 是否可以使用', hello_tff())

