### tf.Variable() 
作用：创建变量

函数使用     
tf.Variable(
        initial_value=None, trainable=None, validate_shape=True, caching_device=None,
        name=None, variable_def=None, dtype=None, import_scope=None, constraint=None,
        synchronization=tf.VariableSynchronization.AUTO,
        aggregation=tf.compat.v1.VariableAggregation.NONE, shape=None
    )

参数             
- initial_value    变量的初始值,可以是所有转换为Tensor的类型
- trainable        如果为True，会把它加入到GraphKeys.TRAINABLE_VARIABLES, 才能对它使用Optimizer
- validate_shape	 用于进行类型和维度检查，false 不检查/true 检查
- caching_device   描述变量应缓存位置以供读取的可选设备字符串。默认为变量的设备。如果不是 ，则在另一台设备上缓存。此参数仅在使用 v1 样式 时有效。
- name	           变量的名称，如果没有指定则系统会自动分配一个唯一的值,'Variable'
- variable_def	   VariableDef协议缓冲区。如果不是 ，则重新创建 Variable 对象及其内容，引用图形中变量的节点，这些节点必须已存在。
- import_scope	   "仅从协议缓冲区初始化时使用"的名称范围。
- dtype	           如果设置，initial_value将转换为给定类型。true 设置， false 不设置
- aggregation	     指示分布式变量的聚合方式。
- shape	           变量的形状

构造函数需要变量的初始值，该值可以是任何类型和形状的。此初始值定义变量的类型和形状。构造后，变量的类型和形状是固定的。可以使用其中一种赋值方法更改该值。

```python
import tensorflow as tf

v1 = tf.Variable(3)
v2 = tf.Variable(3, name='v2')
print(v1)  # <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=3>
print(v2)  # <tf.Variable 'v2:0' shape=() dtype=int32, numpy=3>

v3 = tf.Variable(1., shape=tf.TensorShape(None))
v4 = tf.Variable([[1.], [2.]])
print(v3)  # <tf.Variable 'Variable:0' shape=<unknown> dtype=float32, numpy=1.0>
print(v4)  # <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=array([[1.],[2.]], dtype=float32)>

# assign() 函数为变量赋值
print(v1.assign(5))  # <tf.Variable 'UnreadVariable' shape=() dtype=int32, numpy=5>
print(
  v3.assign([[2.]]))  # <tf.Variable 'UnreadVariable' shape=<unknown> dtype=float32, numpy=array([[2.]], dtype=float32)>

# assign.add() 为变量增加一个值
print(v1.assign_add(5))  # <tf.Variable 'UnreadVariable' shape=() dtype=int32, numpy=10>
print(v1 + 5)  # tf.Tensor(15, shape=(), dtype=int32) 变量转换为张量

# assign_sub() 与 assign.add() 使用方法一样，assign_sub() 是减去一个值

# eval() 函数 在会话中，计算并返回此变量的值。
tf.compat.v1.disable_eager_execution()
v5 = tf.Variable([1, 2])
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
  sess.run(init)
  print(v5.eval(sess))
  print(v5.eval())
```

### tf.constant()
作用: 从类似张量的对象创建常量张量  

> 使用:  
tf.constant(value, dtype=None, shape=None, name='Const')

参数  
- value	  输出类型的常量值（或列表）。dtype
- dtype	  生成的张量的元素的类型。1d,2d,3d,4d......
- shape	  所得张量的可选尺寸。
- name	  张量的可选名称。

constant函数提供在tensorflow中定义常量(不可更改的张量)的方法
```python
import tensorflow as tf

# dtype: 如果未指定参数，则从 类型推断类型。dtypevalue
c1 = tf.constant([1, 2, 3, 4, 5, 6])
print(c1)  # tf.Tensor([1 2 3 4 5 6], shape=(6,), dtype=int32)

c2 = np.array([[1, 2, 3], [4, 5, 6]])
print(tf.constant(c2))  # tf.Tensor([[1 2 3], [4 5 6]], shape=(2, 3), dtype=int32)

# dtype: 如果指定，则生成的张量值将转换为请求的
c3 = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float64)
print(c3)  # tf.Tensor([1. 2. 3. 4. 5. 6.], shape=(6,), dtype=float64)

# shape 如果设置了，则 重新调整以匹配。标量被扩展以填充
c4 = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
print(c4)  # tf.Tensor([[1 2 3], [4 5 6]], shape=(2, 3), dtype=int32)
```

### tf.constant_initializer()
作用：生成具有常量值的张量的初始值设定项
> 使用：tf.constant_initializer(value=0)

参数
- value	  Python 标量、值的列表或元组，或 N 维 numpy 数组。初始化变量的所有元素都将设置为参数中的相应值

tf.constant_initializer允许您预先指定初始化策略（在初始值设定项对象中编码），而无需知道要初始化的变量的形状和 dtype。
tf.constant_initializer返回一个对象，该对象在调用时返回一个填充了构造函数中指定的张量。如果是一个列表，则列表的长度必须等于所需
张量形状所隐含的元素数。如果元素总数不等于张量形状所需的元素数，则初始值设定项将引发 TypeError

```python
import tensorflow as tf

def make_variables(k, initializer):
  return (
    tf.Variable(initializer(shape=[k], dtype=tf.float32)), tf.Variable(initializer(shape=[k, k], dtype=tf.float32))
  )


v1, v2 = make_variables(3, tf.constant_initializer(2.))
print(v1)  # <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([2., 2., 2.], dtype=float32)>
print(v2)  # array([[2., 2., 2.],[2., 2., 2.], [2., 2., 2.]], dtype=float32)>

value = [0, 1, 2, 3, 4, 5, 6, 7]
init = tf.constant_initializer(value)

print(tf.Variable(init(shape=[2, 4], dtype=tf.float32)))
# <tf.Variable 'Variable:0' shape=(2, 4) dtype=float32, numpy=array([[0., 1., 2., 3.],[4., 5., 6., 7.]], dtype=float32)>

print(tf.Variable(init(shape=[3, 4], dtype=tf.float32)))
# TypeError: Eager execution of tf.constant with unsupported shape (value has 8 elements, shape is (3, 4) with 12 elements).

print(tf.Variable(init(shape=[2, 3], dtype=tf.float32)))
# TypeError: Eager execution of tf.constant with unsupported shape (value has 8 elements, shape is (2, 3) with 12 elements).

```

### tf.Tensor()
作用： 创建张量
> 使用 tf.Tensor(op, value_index, dtype)

参数
- op：An Operation. Operation that computes this tensor.
- value_index：	    An int. Index of the operation's endpoint that produces this tensor.
- dtype：	          A DType. Type of elements stored in this tensor.

属性
- device：	  The name of the device on which this tensor will be produced, or None.
- dtype：	    The DType of elements in this tensor.
- graph：	    The Graph that contains this tensor.
- name：	    The string name of this tensor.
- op：	      The Operation that produces this tensor as an output.
```python
import tensorflow as tf
import numpy as np

# Compute some values using a Tensor
c1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
c2 = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c3 = tf.matmul(c1, c2)  # tf.matmul()   张量相乘的算法
print(c3)  # tf.Tensor([[1. 3.], [3. 7.]], shape=(2, 2), dtype=float32)
print(c3.shape)  # 打印张量的形状 (2, 2)
print(c3.get_shape)
# <bound method _EagerTensorBase.get_shape of <tf.Tensor: shape=(2, 2), dtype=float32, numpy=array([[1., 3.], [3., 7.]], dtype=float32)>>

c4 = np.array([1, 2, 3])
c5 = tf.constant(c4)
c4[0] = 4
print(c5)  # tf.Tensor([4 2 3], shape=(3,), dtype=int64)
```
### tf.TensorArray

> tf.TensorArray(
    dtype, size=None, dynamic_size=None, clear_after_read=None,
    tensor_array_name=None, handle=None, flow=None, infer_shape=True,
    element_shape=None, colocate_with_first_write_call=True, name=None
)

参数
- dtype	                          (required) data type of the TensorArray.
- size	                          (optional) int32 scalar Tensor: the size of the TensorArray. Required if handle is not provided.
- dynamic_size	                  (optional) Python bool: If true, writes to the TensorArray can grow the TensorArray past its initial size. Default: False.
- clear_after_read	              Boolean (optional, default: True). If True, clear TensorArray values after reading them. This disables read-many semantics, but allows early release of memory.
- tensor_array_name	              (optional) Python string: the name of the TensorArray. This is used when creating the TensorArray handle. If this value is set, handle should be None.
- handle	                        (optional) A Tensor handle to an existing TensorArray. If this is set, tensor_array_name should be None. Only supported in graph mode.
- flow	                          (optional) A float Tensor scalar coming from an existing TensorArray.flow. Only supported in graph mode.
- infer_shape	                    (optional, default: True) If True, shape inference is enabled. In this case, all elements must have the same shape.
- element_shape	                  (optional, default: None) A TensorShape object specifying the shape constraints of each of the elements of the TensorArray. Need not be fully defined.
- colocate_with_first_write_call	If True, the TensorArray will be colocated on the same device as the Tensor used on its first write (write operations include write, unstack, and split). If False, the TensorArray will be placed on the device determined by the device context available during its initialization.
- name	                          A name for the operation (optional).

属性
- dtype	        The data type of this TensorArray.
- dynamic_size	Python bool; if True the TensorArray can grow dynamically.
- element_shape	The tf.TensorShape of elements in this TensorArray.
- flow	        The flow Tensor forcing ops leading to this TensorArray state.
- handle	      The reference to the TensorArray.

```python
import tensorflow as tf

ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
ta = ta.write(0, 10)
ta = ta.write(1, 20)
ta = ta.write(2, 30)
print(ta.read(0))  # tf.Tensor(10.0, shape=(), dtype=float32)
print(ta.read(1))  # tf.Tensor(20.0, shape=(), dtype=float32)
print(ta.read(2))  # tf.Tensor(30.0, shape=(), dtype=float32)
print(ta.stack())  # tf.Tensor([10. 20. 30.], shape=(3,), dtype=float32)
```

### tf.GradientTape()
作用：一般用来进行梯度下降使用

函数  
tf.GradientTape(
    persistent=False, watch_accessed_variables=True
)

参数  
- persistent	布尔值控制是否创建持久性渐变磁带。默认情况下为 False，这意味着最多可以对此对象的 gradient（） 方法进行一次调用。  
- watch_accessed_variables	布尔值控制磁带在磁带处于活动状态时是否自动访问任何（可训练的）变量。默认值为 True 含义梯度，可以从磁带中通过读取可训练的 .如果 False 用户必须显式表示他们想要从中请求渐变的任何 s。watchVariablewatchVariable

```python
# 案例 计算 y = x * x 的梯度  x = 3.0
import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x
dy_dx = g.gradient(y, x)
print(dy_dx)  # tf.Tensor(6.0, shape=(), dtype=float32)

# GradientTapes 可以嵌套以计算高阶导数。例如
x = tf.constant(5.0)
with tf.GradientTape() as g:
  g.watch(x)
  with tf.GradientTape() as gg:
    gg.watch(x)
    y = x * x
  dy_dx = gg.gradient(y, x)  # dy_dx = 2 * x
d2y_dx2 = g.gradient(dy_dx, x)  # d2y_dx2 = 2
print(dy_dx)  # tf.Tensor(10.0, shape=(), dtype=float32)
print(d2y_dx2)  # tf.Tensor(2.0, shape=(), dtype=float32)

# 默认情况下，GradientTape 持有的资源会在 GradientTape.gradient（）方法被调用后立即释放。
# 要在同一计算中计算多个梯度，请创建持久性梯度磁带。这允许多次调用 gradient（） 方法，
# 因为在对磁带对象进行垃圾回收时会释放资源。例如：
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as g:
  g.watch(x)
  y = x * x
  z = y * y
dz_dx = g.gradient(z, x)  # (4*x^3 at x = 3)
print(dz_dx)  # tf.Tensor(108.0, shape=(), dtype=float32)
dy_dx = g.gradient(y, x)
print(dy_dx) # tf.Tensor(6.0, shape=(), dtype=float32)

# 默认情况下，GradientTape 将自动监视在上下文中访问的任何可训练变量。
# 如果要对监视的变量进行细粒度控制，可以通过传递到磁带构造函数来禁用自动跟踪：watch_accessed_variables=False
x = tf.Variable(2.0)
w = tf.Variable(5.0)
with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
  tape.watch(x)
  y = x ** 2  # Gradients will be available for `x`.
  z = w ** 3  # No gradients will be available as `w` isn't being watched.
dy_dx = tape.gradient(y, x)
print(dy_dx) # tf.Tensor(4.0, shape=(), dtype=float32)

# No gradients will be available as `w` isn't being watched.
dz_dy = tape.gradient(z, w)
print(dz_dy) # None

with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
  tape.watch(x)
  tape.watch(w)
  y = x ** 2  # Gradients will be available for `x`.
  z = w ** 3  # Gradients will be available for `w`.
dy_dx_2 = tape.gradient(y, x)
print(dy_dx_2) # tf.Tensor(4.0, shape=(), dtype=float32)

# No gradients will be available as `w` isn't being watched.
dz_dy_2 = tape.gradient(z, w)
print(dz_dy_2) # tf.Tensor(75.0, shape=(), dtype=float32)
```

### tf.argsort

```python
import tensorflow as tf
```



### tf.one_hot
作用：函数实现用独热码表示标签  

函数  
tf.one_hot(
    indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None
)

注意事项  
由索引表示的位置取值，而所有其他位置取值。i`ndiceson_valueoff_value`  
on_value并且必须具有匹配的数据类型。如果还提供了，则它们必须与 指定的数据类型相同。`off_value dtype dtype`  
如果未提供，它将默认为具有类型的值：`on_value 1 dtype`
如果未提供，它将默认为具有类型的值：`off_value 0 dtype` 
如果输入是秩，则输出将具有秩。新轴是在尺寸处创建的（缺省值：新轴附加在末尾）。`indices N N+1 axis`  
如果是标量，则输出形状将是长度的向量`indices depth`  
如果 是长度的向量，则输出形状将为：`indices features`  

参数  
- indices	A 指数。Tensor
- depth	定义一个热维度深度的标量。
- on_value	一个标量，用于定义在 以下情况下要在输出中填充的值。（默认值：1）indices[j] = i
- off_value	一个标量，用于定义在 以下情况下要在输出中填充的值。（默认值：0）indices[j] != i
- axis	要填充的轴（默认：-1，最内侧的新轴）。
- dtype	输出张量的数据类型。
- name	操作的名称（可选）。

```python
import tensorflow as tf

indices = [0, 1, 2]
depth = 3
tf.one_hot(indices, depth)  # output: [3 x 3]
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]

classes = 3
labels = tf.constant([1, 0, 2])  # 输入的元素值最小为0，最大为2
output = tf.one_hot(labels, depth=classes)
print("result of labels1:", output)
print("\n")
# result of labels1: tf.Tensor(
# [[0. 1. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]], shape=(3, 3), dtype=float32)

indices = [0, 2, -1, 1]
depth = 3
tf.one_hot(indices, depth,
           on_value=5.0, off_value=0.0,
           axis=-1)  # output: [4 x 3]
# [[5.0, 0.0, 0.0],  # one_hot(0)
#  [0.0, 0.0, 5.0],  # one_hot(2)
#  [0.0, 0.0, 0.0],  # one_hot(-1)
#  [0.0, 5.0, 0.0]]  # one_hot(1)

indices = [[0, 2], [1, -1]]
depth = 3
tf.one_hot(indices, depth,
           on_value=1.0, off_value=0.0,
           axis=-1)  # output: [2 x 2 x 3]
# [[[1.0, 0.0, 0.0],   # one_hot(0)
#   [0.0, 0.0, 1.0]],  # one_hot(2)
#  [[0.0, 1.0, 0.0],   # one_hot(1)
#   [0.0, 0.0, 0.0]]]  # one_hot(-1)

indices = tf.ragged.constant([[0, 1], [2]])
depth = 3
tf.one_hot(indices, depth)  # output: [2 x None x 3]
# [[[1., 0., 0.],
#   [0., 1., 0.]],
#  [[0., 0., 1.]]]
```

### tf.cast
作用：将张量转换为新类型

函数  
tf.cast(
    x, dtype, name=None
)

参数  
- x 输入的数据
- dtype 要转换的数据类型
- name 给函数的定义名称

```python
import tensorflow as tf

x = tf.constant([1.8, 2.2], dtype=tf.float32)
tf.cast(x, tf.int32) # tf.Tensor([1 2], shape=(2,), dtype=int32)
```


### tf.fill

```python
import tensorflow as tf
```