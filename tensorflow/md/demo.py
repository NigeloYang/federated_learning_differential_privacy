# 此文件主要用来运行案例使用
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 因为安装的是 GPU 版本的 rensorflow, 所以在不加任何配置情况下，是默认使用gpu的，加上下面这句代码就使用cpu了 进程量不大的情况下性能： cpu > gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf

# features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
# labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor
# dataset = tf.data.Dataset.from_tensor_slices((features, labels))
# print(list(dataset.as_numpy_iterator()))

x = tf.constant([[1., 2., 3., 4.],
                 [5., 6., 7., 8.],
                 [9., 10., 11., 12.]])
x = tf.reshape(x, [1, 3, 4, 1])
print(x)