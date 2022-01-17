# tf.data
Dataset: 表示一组可能很大的元素.

DatasetSpec: tf.data.Dataset 的类型规范.

FixedLengthRecordDataset: 来自一个或多个二进制文件的固定长度记录的数据集.

Iterator: 表示 tf.data.Dataset 的迭代器.

IteratorSpec: tf.data.Iterator 的类型规范.

Options: 表示 tf.data.Dataset 的选项.

TFRecordDataset: 包含来自一个或多个 TFRecord 文件的记录的数据集.

TextLineDataset: 包含来自一个或多个文本文件的行的数据集.

## tf.data.Dataset(variant_tensor) 
### tf.data.Dataset(variant_tensor) 函数用法
variant_tensor: 代表数据集的 DT_VARIANT 张量。.

tf.data.Dataset API 支持编写描述性和高效的输入管道。数据集的使用遵循一个常见的模式：
1、从输入数据创建源数据集。2、应用数据集转换来预处理数据。3、迭代数据集并处理元素。
迭代以流式方式发生，因此整个数据集不需要放入内存中。

```python
# 创建一个dataset数据集，如何对数据集进行操作呢？看下面的几个例子

import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
for element in dataset:
  print(element)
# tf.Tensor(1, shape=(), dtype=int32)
# tf.Tensor(2, shape=(), dtype=int32)
# tf.Tensor(3, shape=(), dtype=int32)

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.map(lambda x: x*2)
print(list(dataset.as_numpy_iterator()))  # [2, 4, 6]

# dataset 的元素可以是元组、命名元组和字典的嵌套结构。注意，Python 列表不被视为组件的嵌套结构。相反，列表被转换为张量并被视为组件。
# 例如，元素 (1, [1, 2, 3]) 只有两个组件；张量 1 和张量 [1, 2, 3]。元素组件可以是 tf.TypeSpec 可以表示的任何类型，
# 包括 tf.Tensor、tf.data.Dataset、tf.sparse.SparseTensor、tf.RaggedTensor 和 tf.TensorArray。

```
### tf.data.Dataset 方法
#### apply(transformation_func)
apply 启用自定义数据集转换的链接，这些转换表示为接受一个数据集参数并返回转换后的数据集的函数。

transformation_func: 一个函数，它接受一个 Dataset 参数并返回一个 Dataset。

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(100)
def dataset_fn(ds):
  return ds.filter(lambda x: x < 5)
dataset = dataset.apply(dataset_fn)
print(list(dataset.as_numpy_iterator()))  # [0, 1, 2, 3, 4]
```

#### as_numpy_iterator 该更迭代结果返回的形式 tensor 形式 --> 数字形式
```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
for element in dataset:
  print(element)
# tf.Tensor(1, shape=(), dtype=int32)
# tf.Tensor(2, shape=(), dtype=int32)
# tf.Tensor(3, shape=(), dtype=int32)

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
for element in dataset.as_numpy_iterator():
  print(element) #1,2,3

```

#### batch(batch_size, drop_remainder=False)
batch_size:	一个 tf.int64 标量 tf.Tensor，表示要在单个批次中组合的此数据集的连续元素的数量。

drop_remainder:	（可选。）一个 tf.bool 标量 tf.Tensor，表示在最后一批少于 batch_size 元素的情况下是否应该丢弃它；默认行为是不丢弃较小的批次。

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(8)
dataset = dataset.batch(3)
print(list(dataset.as_numpy_iterator()))
# [array([0, 1, 2], dtype=int64), array([3, 4, 5], dtype=int64), array([6, 7], dtype=int64)]

dataset = dataset.batch(3, drop_remainder=True)
print(list(dataset.as_numpy_iterator()))
# [array([0, 1, 2], dtype=int64), array([3, 4, 5], dtype=int64)]

```

