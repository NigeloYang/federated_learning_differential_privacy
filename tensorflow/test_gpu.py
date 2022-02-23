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
