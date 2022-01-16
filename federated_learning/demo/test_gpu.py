import tensorflow as tf
from tensorflow.python.client import device_lib

# 由于tensorflow安装的不对，需要从源代码安装tensorflow, 通过引入接下来的两个语句解决报错问题
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)
print('------------------------------------')

print('打印 gpu 是否可用：', tf.test.is_gpu_available())
print('------------------------------------')

print('打印gpu 设备：', device_lib.list_local_devices())

