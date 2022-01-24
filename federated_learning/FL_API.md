# 常用 FL API

## tff
#### tff.Computation
说明  
代表计算的所有类的抽象接口，继承自 TypedObject

属性  
type_signature 返回此对象的 TFF 类型（tff.Type 的实例）。

#### tff.TypedObject

说明  
具有 TFF 类型签名的事物的抽象接口。

属性  
type_signature 返回此对象的 TFF 类型（tff.Type 的实例）。

#### Value
说明  
出现在 TFF 计算中的值的通用基类， 继承自 TypedObject

使用  
tff.Value(comp: tff.framework.ComputationBuildingBlock)

使用说明  
如果此类中的值是包含 StructType 的 StructType 或 FederatedType，则可以通过名称访问内部字段（例如 y = my_value_impl.y）。

参数  
comp building_blocks.ComputationBuildingBlock 的实例，其中包含计算此值的逻辑。

属性   
comp type_signature Returns the TFF type of this object (an instance of tff.Type).

#### tff.data
说明  
使用给定的 URI 和 TFF 类型构造 TFF data 计算。

使用  
tff.data(
    uri: str,
    type_spec: tff.types.Type
)

参数  
uri	数据的字符串 (str) URI。  
type_spec	 表示此数据类型的 tff.Type 实例。

返回  
联合计算主体中具有给定 URI 和 TFF 类型的数据的表示。 

引发  
TypeError 如果参数不是上面指定的类型。

#### tff.federated_aggregate 
说明  
将值从 tff.CLIENTS 聚合到 tff.SERVER。

使用  
tff.federated_aggregate(
    value, zero, accumulate, merge, report
)

参数  
value	 放置在 tff.CLIENTS 要聚合的 TFF 联合类型的值。  
zero	归约算子代数中 U 类型的零，如上所述。  
accumulate	在流程的第一阶段使用的归约运算符。如果 value 是 {T}@CLIENTS 类型，并且零是 U 类型，则此运算符应该是 (<U,T> -> U) 类型。  
merge	 在过程的第二阶段使用的归约算子。必须是 (<U,U> -> U) 类型，其中 U 定义如上。  
report	在过程的最后阶段使用的投影运算符来计算聚合的最终结果。如果 tff.federated_aggregate 返回的预期结果是 R@SERVER 类型，则此运算符必须是 (U -> R) 类型。  

Returns  
tff.SERVER 上使用上述多阶段过程聚合值的结果的表示。

Raises  
TypeError 如果参数不是上面指定的类型。

多阶段聚合过程定义如下:
- 客户被组织成组。在每个组中，首先使用归约算子对组中客户贡献的所有成员价值成分的集合进行归约，以零作为代数中的零。如果 value 的成员是 T 类型，而零（归约空集的结果）是 U 类型，则在此阶段使用的归约运算符累积应该是 (<U,T> -> U) 类型。此阶段的结果是一组 U 类型的项目，每组客户一个项目。  

- 接下来，使用类型为 (<U,U> -> U) 的二元交换关联运算符合并，将前一阶段生成的 U 类型项合并。这个阶段的结果是一个单一的顶级 U，它出现在 tff.SERVER 的层次结构的根部。实际实现可以将此步骤构建为多个层级联。  

- 最后，使用report 作为映射函数，将在前一阶段执行的归约的U 型结果投影到结果值中（例如，如果要合并的结构由计数器组成，最后一步可能包括计算它们的比率）。  

#### tff.federated_broadcast 
作用  
将联合值从 tff.SERVER 广播到 tff.CLIENTS。

使用  
tff.federated_broadcast(
    value
)

参数  
value	放置在 tff.SERVER 的 TFF 联合类型的值，其所有成员均相等（value 的 tff.FederatedType.all_equal 属性为 True）。  

返回  
广播结果的表示：放置在 tff.CLIENTS 的 TFF 联合类型的值，其所有成员都是相等的。

#### tff.federated_computation
作用  
Decorates/wraps 将 Python 函数包装为 TFF 联合复合计算。

使用  
tff.federated_computation(
    *args, tff_internal_types=None
)

说明  
此处使用的术语联合计算是指使用 TFF 编程抽象的任何计算。这种计算的示例可以包括联合训练或联合评估，其涉及客户端和服务器端逻辑并涉及网络通信。
但是，此装饰器包装器也可用于构建仅涉及客户端或服务器上的本地处理的复合计算。 Python 中的联合计算函数体与 TensorFlow defuns 的主体的
主要区别在于，后者是使用各种 TensorFlow 操作对 tf.Tensor 实例进行切片和切块，而前者对 tff.Value 实例进行切片和切块使用 TFF 运算符。
支持的使用模式与 tff.tf_computation 相同。

```python
import tensorflow as tf
import tensorflow_federated as tff

@tff.tf_computation(tf.int32)
def add_half(x):
  # tf 代码封装在tff.tf_computation装饰器中
  return tf.add(x, 2)


print(add_half.type_signature) # (int32 -> int32)


@tff.federated_computation(tff.type_at_clients(tf.int32))
def foo(x):
  return tff.federated_map(add_half, x)

print(foo.type_signature)
print(foo([1, 4, 7]))
# ({int32}@CLIENTS -> {int32}@CLIENTS)
# [<tf.Tensor: shape=(), dtype=int32, numpy=3>, <tf.Tensor: shape=(), dtype=int32, numpy=6>, <tf.Tensor: shape=(), dtype=int32, numpy=9>]

```


#### tff.federated_zip




#### tff.tf_computation  
作用  
将 Python 函数和 defuns 装饰为 TFF TensorFlow 计算。


```python
import tensorflow as tf
import tensorflow_federated as tff


@tff.tf_computation(tf.int32)
def data_filter(x):
  return x > 10

print(data_filter(45))
print(data_filter.type_signature)
# True
# (int32 -> bool)

data_filter_2 = tff.tf_computation(lambda x: x > 10, tf.int32)
print(data_filter_2(5))
print(data_filter_2.type_signature)
# False
# (int32 -> bool)
```
####  



## tff.aggregators


## tff.learning


## tff.simulation


## tff.types

#### FederatedType: 表示 TFF 中的联合类型。
暂无

#### FunctionType: 表示 TFF 中的功能类型。
函数类型，是一个函数式编程框架，其中函数被视为这些函数的紧凑表示法为 (T -> U)，其中 T 为参数类型，U 为结果类型；或者，如果没有参数
（虽然无参数函数是一个大部分情况下仅在 Python 级别存在的过时概念），则可以表示为 ( -> U)。例如，(int32* -> int32) 表示一种将整数
序列缩减为单个整数值的函数类型。第一类值。函数最多有一个参数，并且只有一个结果。  

使用  
tff.types.FunctionType(
    parameter, result
)


#### SequenceType: 表示 TFF 中序列类型的 。 

序列类型是 TFF 中等效于 TensorFlow 中 tf.data.Dataset 的具体概念的抽象。用户可以按顺序使用序列的元素，并且可以包含复杂的类型。 序列类型的紧凑表示法为 T*，其中 T 是元素的类型。例如，int32* 表示整数序列。


#### StructType: 表示 TFF 中的结构类型。 

命名元组类型，这些是 TFF 使用指定类型构造具有预定义数量元素的元组或字典式结构（无论命名与否）的方式。重要的一点是，TFF 的命名元组 概念包含等效于
Python 参数元组的抽象，即元组的元素集合中有一部分（并非全部）是命名元素，还有一部分是位置元素。 命名元组的紧凑表示法为 <n_1=T_1, ..., n_k=T_k>，其中 n_k 是可选元素名称，T_k 是元素类型。 例如，<
int32,int32> 是一对未命名整数的紧凑表示法，<X=float32,Y=float32> 是命名为 X 和 Y（可能代表平面上的一个点）的一对浮点数 的紧凑表示法。元组可以嵌套，也可以与其他类型混用，例如，<
X=float32,Y=float32>* 可能是一系列点的紧凑表示法


#### StructWithPythonType: 与 Python 容器类型配对的结构的表示。
暂无

#### TensorType: 表示 TFF 中的张量类型。
张量类型，对象不仅限于在 TensorFlow 计算图中表示 TensorFlow 运算输出的 Python 的 tf.Tensor 实例，而是也可能包括可产生的数据单位， 例如，作为分布聚合协议的输出。张量类型的紧凑表示法为 dtype 或 dtype[shape]。例如，int32 和 int32[10] 分别是整数和整数向量的类型。



