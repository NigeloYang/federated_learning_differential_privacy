# 此文件主要用来运行案例使用
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 因为安装的是 GPU 版本的 rensorflow, 所以在不加任何配置情况下，是默认使用gpu的，加上下面这句代码就使用cpu了 进程量不大的情况下性能： cpu > gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

x = tf.constant([1.8, 2.2], dtype=tf.float32)
print(tf.cast(x, tf.int32))
