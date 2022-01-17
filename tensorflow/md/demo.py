# 此文件主要用来运行案例使用
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 因为安装的是 GPU 版本的 rensorflow, 所以在不加任何配置情况下，是默认使用gpu的，加上下面这句代码就使用cpu了 进程量不大的情况下性能： cpu > gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf

dataset = tf.data.Dataset.range(8)
# dataset = dataset.batch(3)
# print(list(dataset.as_numpy_iterator()))

dataset = dataset.batch(3, drop_remainder=True)
print(list(dataset.as_numpy_iterator()))