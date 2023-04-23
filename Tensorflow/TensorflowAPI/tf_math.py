'''主要记录 tf.Math FL_API 方法'''

import tensorflow as tf

'''四则运算方法 主要是张量对应元素的加减乘除
tf.add(张量1，张量2) 实现两个张量的对应元素相加
tf.subtract(张量1，张量2) 实现两个张量的对应元素相减
tf.multiply(张量1，张量2) 实现两个张量的对应元素相乘
tf.divide(张量1，张量2) 实现两个张量的对应元素相除

只有维度相同的张量才可以做四则运算
'''
a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)
print(a)
print(b)
print(tf.add(a, b))
print(tf.subtract(a, b))
print(tf.multiply(a, b))
print(tf.divide(b, a))
# tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float32)
# tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
# tf.Tensor([[4. 4. 4.]], shape=(1, 3), dtype=float32)
# tf.Tensor([[-2. -2. -2.]], shape=(1, 3), dtype=float32)
# tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
# tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)

'''平方、次方、开方
tf.square(张量名) 计算某个张量的平方
tf.pow(张量名, n次方数) 计算某个张量的n次方
tf.sqrt(张量名) 计算某个张量的开方
'''
a = tf.fill([1, 2], 3.)
print(a)
print(tf.square(a))
print(tf.pow(a, 3))
print(tf.sqrt(a))
# tf.Tensor([[3. 3.]], shape=(1, 2), dtype=float32)
# tf.Tensor([[9. 9.]], shape=(1, 2), dtype=float32)
# tf.Tensor([[27. 27.]], shape=(1, 2), dtype=float32)
# tf.Tensor([[1.7320508 1.7320508]], shape=(1, 2), dtype=float32)

'''矩阵相乘
tf.matmul(矩阵1，矩阵2) 实现两个矩阵的相乘
'''
a = tf.ones([3, 2])
b = tf.fill([2, 3], 3.)
print(tf.matmul(a, b))
# tf.Tensor([[6. 6. 6.],[6. 6. 6.],[6. 6. 6.]], shape=(3, 3), dtype=float32)

'''计算张量的值 一般用于维度计算
tf.reduce_mean
tf.reduce_sum
'''
x = tf.constant([[1, 2, 3],
                 [2, 2, 3]])
print(x)
# tf.Tensor([[1 2 3], [2 2 3]], shape=(2, 3), dtype=int32)

# tf.reduce_mean(张量名，axis=操作轴) 计算张量沿着指定维度的平均值
print(tf.reduce_mean(x))
print(tf.reduce_mean(x, axis=0))
print(tf.reduce_mean(x, axis=1))
# tf.Tensor(2, shape=(), dtype=int32)
# tf.Tensor([1 2 3], shape=(3,), dtype=int32)
# tf.Tensor([2 2], shape=(2,), dtype=int32)


# tf.reduce_sum (张量名，axis=操作轴) 计算张量沿着指定维度的和
print(tf.reduce_sum(x))
print(tf.reduce_sum(x, axis=0))
print(tf.reduce_sum(x, axis=1))
# tf.Tensor(13, shape=(), dtype=int32)
# tf.Tensor([3 4 6], shape=(3,), dtype=int32)
# tf.Tensor([6 7], shape=(2,), dtype=int32)
