import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 是否开启 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow_federated as tff


@tff.tf_computation(tf.int32)
def data_filter(x):
  return x > 10

print(data_filter(45))
print(data_filter.type_signature)

data_filter_2 = tff.tf_computation(lambda x: x > 10, tf.int32)
print(data_filter_2(5))
print(data_filter_2.type_signature)

@tff.tf_computation(tf.int32)
def add_half(x):
  # tf 代码封装在tff.tf_computation装饰器中
  return tf.add(x, 2)


print(add_half.type_signature)


@tff.federated_computation(tff.type_at_clients(tf.int32))
def foo(x):
  return tff.federated_map(add_half, x)

print(foo.type_signature)
print(foo([1, 4, 7]))
