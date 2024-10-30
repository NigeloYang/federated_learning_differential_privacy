# 由于tensorflow安装的不对，需要从源代码安装tensorflow, 通过引入接下来的两个语句解决报错问题
import os

'''
TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 是否开启 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.python.client import device_lib

print(tf.__version__)
print('------------------------------------')

print('打印 gpu 是否可用：', tf.test.is_gpu_available())
print('------------------------------------')

print('打印gpu 设备：', device_lib.list_local_devices())
print('------------------------------------')

print('打印GPU 设备：', tf.config.list_physical_devices('GPU'))
print('打印CPU 设备：', tf.config.list_physical_devices('CPU'))
