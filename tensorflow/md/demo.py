# 此文件主要用来运行案例使用
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 因为安装的是 GPU 版本的 rensorflow, 所以在不加任何配置情况下，是默认使用gpu的，加上下面这句代码就使用cpu了 进程量不大的情况下性能： cpu > gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf

# tf.data.Dataset.from_generator
# 运行这个代码需要在 tensorflow > v2.5.0
def gen():
  ragged_tensor = tf.ragged.constant([[1, 2], [3]])
  yield 42, ragged_tensor


dataset = tf.data.Dataset.from_generator(
  gen,
  output_signature=(tf.TensorSpec(shape=(), dtype=tf.int32), tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32))
)

print(list(dataset.take(1)))


# tf.data.Dataset.from_tensor_slices
# Slicing a 2D tensor produces 1D tensor elements.
# dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
# print(list(dataset.as_numpy_iterator()))
#
# # Two tensors can be combined into one Dataset object.
#
# features = tf.constant([[1, 3], [2, 1], [3, 3]])  # ==> 3x2 tensor
# labels = tf.constant(['A', 'B', 'A'])  # ==> 3x1 tensor
# dataset = tf.data.Dataset.from_tensor_slices((features, labels))
# print(list(dataset.as_numpy_iterator()))
#
# # Both the features and the labels tensors can be converted to a Dataset object separately and combined after.
# features_dataset = tf.data.Dataset.from_tensor_slices(features)
# print(list(features_dataset.as_numpy_iterator()))
# labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
# print(list(labels_dataset.as_numpy_iterator()))
# dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
# print(list(dataset.as_numpy_iterator()))
#
# # A batched feature and label set can be converted to a Dataset in similar fashion.
# batched_features = tf.constant([[[1, 3], [2, 3]], [[2, 1], [1, 2]], [[3, 3], [3, 2]]], shape=(3, 2, 2))
# batched_labels = tf.constant([['A', 'A'], ['B', 'B'], ['A', 'B']], shape=(3, 2, 1))
# dataset = tf.data.Dataset.from_tensor_slices((batched_features, batched_labels))
# for element in dataset.as_numpy_iterator():
#   print(element)

#
# x = tf.constant([[1., 2., 3., 4.],
#                  [5., 6., 7., 8.],
#                  [9., 10., 11., 12.]])
# x = tf.reshape(x, [1, 3, 4, 1])
# print(x)
