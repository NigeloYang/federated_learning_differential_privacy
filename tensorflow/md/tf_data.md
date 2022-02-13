# tf.data
- Dataset: 表示一组可能很大的元素.
- DatasetSpec: tf.data.Dataset 的类型规范. FixedLengthRecordDataset: 来自一个或多个二进制文件的固定长度记录的数据集.
- Iterator: 表示 tf.data.Dataset 的迭代器.
- IteratorSpec: tf.data.Iterator 的类型规范.
- Options: 表示 tf.data.Dataset 的选项.
- TFRecordDataset: 包含来自一个或多个 TFRecord 文件的记录的数据集.
- TextLineDataset: 包含来自一个或多个文本文件的行的数据集.

## Dataset
函数  
tf.data.Dataset(variant_tensor)

参数：  
variant_tensor: 代表数据集的 DT_VARIANT 张量

作用：  
tf.data.Dataset API 支持编写描述性和高效的输入管道。数据集的使用遵循一个常见的模式： 1、从输入数据创建源数据集。 2、应用数据集转换来预处理数据。 3、迭代数据集并处理元素。
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
dataset = dataset.map(lambda x: x * 2)
print(list(dataset.as_numpy_iterator()))  # [2, 4, 6]

# dataset 的元素可以是元组、命名元组和字典的嵌套结构。注意，Python 列表不被视为组件的嵌套结构。相反，列表被转换为张量并被视为组件。
# 例如，元素 (1, [1, 2, 3]) 只有两个组件；张量 1 和张量 [1, 2, 3]。元素组件可以是 tf.TypeSpec 可以表示的任何类型，
# 包括 tf.Tensor、tf.data.Dataset、tf.sparse.SparseTensor、tf.RaggedTensor 和 tf.TensorArray。

```

### Dataset 方法

#### apply(transformation_func)

参数  
transformation_func: 一个函数，它接受一个 Dataset 参数并返回一个 Dataset。

作用  
apply 启用自定义数据集转换的链接，这些转换表示为接受一个数据集参数并返回转换后的数据集的函数。

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(100)


def dataset_fn(ds):
  return ds.filter(lambda x: x < 5)


dataset = dataset.apply(dataset_fn)
print(list(dataset.as_numpy_iterator()))  # [0, 1, 2, 3, 4]
```

#### as_numpy_iterator
作用  
检查数据集的内容。 要查看元素形状和类型，请直接打印数据集元素，而不是使用 as_numpy_iterator

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
print(list(dataset.as_numpy_iterator()))  # [1, 2, 3]

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
for element in dataset.as_numpy_iterator():
  print(element)
# tf.Tensor(1, shape=(), dtype=int32)
# tf.Tensor(2, shape=(), dtype=int32)
# tf.Tensor(3, shape=(), dtype=int32)
```

#### batch
函数  
batch( batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None,name=None)

作用    
结果元素的组件将有一个额外的外部维度，它将是 batch_size （如果 batch_size 没有将输入元素的数量 N 均分并且 drop_remainder 为 False，则为最后一个元素的 N %
batch_size）。如果您的程序依赖于具有相同外部尺寸的批次，则应将 drop_remainder 参数设置为 True 以防止生成较小的批次。

参数
- batch_size 一个 tf.int64 标量 tf.Tensor，表示要在单个批次中组合的此数据集的连续元素的数量。
- drop_remainder （可选。）一个 tf.bool 标量 tf.Tensor，表示在最后一批少于 batch_size 元素的情况下是否应该丢弃它；默认行为是不丢弃较小的批次。
- num_parallel_calls （可选。）一个 tf.int64 标量 tf.Tensor，表示要并行异步计算的批次数。如果未指定，批次将按顺序计算。如果使用值
  tf.data.AUTOTUNE，则并行调用的数量是根据可用资源动态设置的。
- deterministic （可选。）指定 num_parallel_calls 时，如果指定此布尔值（True 或 False），它控制转换生成元素的顺序。如果设置为
  False，则允许转换产生无序的元素，以用确定性换取性能。如果未指定，则 tf.data.Options.deterministic 选项（默认为 True）控制行为。
- name （可选。） tf.data 操作的名称。

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

#### filter
函数  
filter( predicate, name=None )  

作用  
根据制定词过滤此数据集

参数  
- predicate 将数据集元素映射到布尔值的函数。 
- name  (可选) tf.data 操作的名称。

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.filter(lambda x: x < 3)
print(list(dataset.as_numpy_iterator()))  # [1,2]


def filter_fn(x):
  return tf.math.equal(x, 1)


dataset = dataset.filter(filter_fn)
print(list(dataset.as_numpy_iterator())) 
```

#### from_generator
函数  
from_generator(generator, output_types=None, output_shapes=None, args=None, output_signature=None, name=None  )

作用  
生成器参数必须是一个可调用对象，该对象返回一个支持 iter() 协议的对象（例如生成器函数）。 生成器生成的元素必须与给定的 output_signature 参数或给定的 output_types 和（可选） output_shapes
参数兼容，以指定者为准。 调用 from_generator 的推荐方法是使用 output_signature 参数。在这种情况下，将假定输出由具有类、形状和类型的对象组成，这些对象由 output_signature 参数中的
tf.TypeSpec 对象定义：

参数
- generator 返回支持 iter() 协议的对象的可调用对象。如果未指定 args，则生成器不得带任何参数；否则，它必须采用与 args 中的值一样多的参数。
- output_types （可选。） tf.DType 对象的（嵌套）结构，对应于生成器产生的元素的每个组件。
- output_shapes （可选。） tf.TensorShape 对象的（嵌套）结构，对应于生成器产生的元素的每个组件。
- args （可选。） tf.Tensor 对象的元组，将被评估并作为 NumPy 数组参数传递给生成器。
- output_signature （可选。） tf.TypeSpec 对象的（嵌套）结构，对应于生成器产生的元素的每个组件。
- name （可选。）from_generator 使用的 tf.data 操作的名称。

```python
import tensorflow as tf


def gen():
  ragged_tensor = tf.ragged.constant([[1, 2], [3]])
  yield 42, ragged_tensor


dataset = tf.data.Dataset.from_generator(
  gen,
  output_signature=(tf.TensorSpec(shape=(), dtype=tf.int32), tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32))
)

print(list(dataset.take(1)))
# [(<tf.Tensor: shape=(), dtype=int32, numpy=42>, <tf.RaggedTensor [[1, 2], [3]]>)]
```

#### from_tensor_slices
函数  
from_tensor_slices(tensors, name=None)

作用  
给定的张量沿它们的第一维进行切片。此操作保留输入张量的结构，删除每个张量的第一个维度并将其用作数据集维度。所有输入张量的第一个维度必须具有相同的大小。

参数
- tensors 一个数据集元素，其组件具有相同的第一维。此处记录了支持的值。
- name （可选。） tf.data 操作的名称。

```python
import tensorflow as tf

# Slicing a 1D tensor produces scalar tensor elements.
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
print(list(dataset.as_numpy_iterator()))  # [1,2,3]

# Slicing a 2D tensor produces 1D tensor elements.
dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
print(list(dataset.as_numpy_iterator()))  # 
# [array([1, 2]), array([3, 4])]

features = tf.constant([[1, 3], [2, 1], [3, 3]])  # ==> 3x2 tensor
labels = tf.constant(['A', 'B', 'A'])  # ==> 3x1 tensor
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(list(dataset.as_numpy_iterator()))
# [(array([1, 3]), b'A'), (array([2, 1]), b'B'), (array([3, 3]), b'A')]

batched_features = tf.constant([[[1, 3], [2, 3]], [[2, 1], [1, 2]], [[3, 3], [3, 2]]], shape=(3, 2, 2))
batched_labels = tf.constant([['A', 'A'], ['B', 'B'], ['A', 'B']], shape=(3, 2, 1))
dataset = tf.data.Dataset.from_tensor_slices((batched_features, batched_labels))
for element in dataset.as_numpy_iterator():
  print(element)
# (array([[1, 3], [2, 3]]), array([[b'A'], [b'A']], dtype=object))
# (array([[2, 1], [1, 2]]), array([[b'B'], [b'B']], dtype=object))
# (array([[3, 3], [3, 2]]), array([[b'A'], [b'B']], dtype=object))
```

#### map
函数  
map(map_func, num_parallel_calls=None, deterministic=None, name=None)

作用  
此转换将 map_func 应用于此数据集的每个元素，并返回一个包含转换后元素的新数据集，其顺序与它们在输入中出现的顺序相同。 map_func 可用于更改数据集元素的值和结构。 此处记录了支持的结构构造。

参数  
- map_func 将数据集元素映射到另一个数据集元素的函数。	  
- num_parallel_calls （可选。）一个 tf.int64 标量 tf.Tensor，表示要并行异步处理的数字元素。如果未指定，元素将按顺序处理。如果使用值 tf.data.AUTOTUNE，则并行调用的数量是根据可用 CPU
动态设置的。  
- deterministic （可选。）指定 num_parallel_calls 时，如果指定此布尔值（True 或 False），它控制转换生成元素的顺序。如果设置为
- False，则允许转换产生无序的元素，以用确定性换取性能。如果未指定，则 tf.data.Options.deterministic 选项（默认为 True）控制行为。  
- name  (可选) tf.data 操作的名称。

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(5)
# `map_func` 接受一个 `tf.Tensor` 类型的单个参数，具有相同的 shape 和 dtype。
result = dataset.map(lambda x: x + 1)
for el in result:
  print(el)
print(list(dataset.as_numpy_iterator()))
# tf.Tensor(1, shape=(), dtype=int64)
# tf.Tensor(2, shape=(), dtype=int64)
# tf.Tensor(3, shape=(), dtype=int64)
# tf.Tensor(4, shape=(), dtype=int64)
# tf.Tensor(5, shape=(), dtype=int64)
# [0, 1, 2, 3, 4]

# 每个元素都是一个包含两个 `tf.Tensor` 对象的元组。
elements = [(1, "foo"), (2, "bar"), (3, "baz")]
dataset = tf.data.Dataset.from_generator(lambda: elements, (tf.int32, tf.string))
for data in dataset:
  print(data)
# (<tf.Tensor: shape=(), dtype=int32, numpy=1>, <tf.Tensor: shape=(), dtype=string, numpy=b'foo'>)
# (<tf.Tensor: shape=(), dtype=int32, numpy=2>, <tf.Tensor: shape=(), dtype=string, numpy=b'bar'>)
# (<tf.Tensor: shape=(), dtype=int32, numpy=3>, <tf.Tensor: shape=(), dtype=string, numpy=b'baz'>)

# `map_func` 接受两个 `tf.Tensor` 类型的参数。此功能仅投影出第一个组件。
result = dataset.map(lambda x_int, y_str: x_int)
for el in result:
  print(el)
print(list(result.as_numpy_iterator()))
# tf.Tensor(1, shape=(), dtype=int32)
# tf.Tensor(2, shape=(), dtype=int32)
# tf.Tensor(3, shape=(), dtype=int32)
# [1, 2, 3]
```

```python
import tensorflow as tf

```

