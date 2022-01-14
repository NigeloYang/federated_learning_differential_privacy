import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# 由于tensorflow安装的不对，需要从源代码安装tensorflow 通过引入接下来的两个语句解决报错问题
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(0)

print(tff.federated_computation(lambda: 'Hello, World!')())