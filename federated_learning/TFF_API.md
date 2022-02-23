# tff
#### Computation 代表计算的所有类的抽象接口，继承自 TypedObject
属性  
type_signature 返回此对象的 TFF 类型（tff.Type 的实例）。


#### TypedObject 具有 TFF 类型签名的事物的抽象接口。
属性  
type_signature 返回此对象的 TFF 类型（tff.Type 的实例）。


#### Value 出现在 TFF 计算中的值的通用基类， 继承自 TypedObject
    api 
    tff.Value(comp: tff.framework.ComputationBuildingBlock)

使用说明  
如果此类中的值是包含 StructType 的 StructType 或 FederatedType，则可以通过名称访问内部字段（例如 y = my_value_impl.y）。

参数  
comp building_blocks.ComputationBuildingBlock 的实例，其中包含计算此值的逻辑。

属性   
comp type_signature Returns the TFF type of this object (an instance of tff.Type).

#### data 使用给定的 URI 和 TFF 类型构造 TFF data 计算。

使用  


参数  
uri	数据的字符串 (str) URI。  
type_spec	 表示此数据类型的 tff.Type 实例。

返回  
联合计算主体中具有给定 URI 和 TFF 类型的数据的表示。 

引发  
TypeError 如果参数不是上面指定的类型。

#### federated_aggregate(...): 将值从 tff.CLIENTS 聚合到 tff.SERVER。
    api   
    tff.federated_aggregate(
        value, zero, accumulate, merge, report
    )

参数  
- value	 放置在 tff.CLIENTS 要聚合的 TFF 联合类型的值。  
- zero	归约算子代数中 U 类型的零，如上所述。  
- accumulate	在流程的第一阶段使用的归约运算符。如果 value 是 {T}@CLIENTS 类型，并且零是 U 类型，则此运算符应该是 (<U,T> -> U) 类型。  
- merge	 在过程的第二阶段使用的归约算子。必须是 (<U,U> -> U) 类型，其中 U 定义如上。  
- report	在过程的最后阶段使用的投影运算符来计算聚合的最终结果。如果 tff.federated_aggregate 返回的预期结果是 R@SERVER 类型，则此运算符必须是 (U -> R) 类型。  

Returns  
tff.SERVER 上使用上述多阶段过程聚合值的结果的表示。

Raises  
TypeError 如果参数不是上面指定的类型。

多阶段聚合过程定义如下:
- 客户被组织成组。在每个组中，首先使用归约算子对组中客户贡献的所有成员价值成分的集合进行归约，以零作为代数中的零。如果 value 的成员是 T 类型，而零（归约空集的结果）是 U 类型，则在此阶段使用的归约运算符累积应该是 (<U,T> -> U) 类型。此阶段的结果是一组 U 类型的项目，每组客户一个项目。  

- 接下来，使用类型为 (<U,U> -> U) 的二元交换关联运算符合并，将前一阶段生成的 U 类型项合并。这个阶段的结果是一个单一的顶级 U，它出现在 tff.SERVER 的层次结构的根部。实际实现可以将此步骤构建为多个层级联。  

- 最后，使用report 作为映射函数，将在前一阶段执行的归约的U 型结果投影到结果值中（例如，如果要合并的结构由计数器组成，最后一步可能包括计算它们的比率）。  

#### federated_broadcast(...): 将联合值从 tff.SERVER 广播到 tff.CLIENTS。
    api   
    tff.federated_broadcast(
        value
    )

参数  
value:	放置在 tff.SERVER 的 TFF 联合类型的值，其所有成员均相等（value 的 tff.FederatedType.all_equal 属性为 True）。  

返回  
广播结果的表示：放置在 tff.CLIENTS 的 TFF 联合类型的值，其所有成员都是相等的。

#### federated_collect(...): Returns a federated value from tff.CLIENTS as a tff.SERVER sequence.

#### federated_computation(...): 将 Python 函数装饰/包装为 TFF 联合/复合计算。
    api   
    tff.federated_computation(
        *args, tff_internal_types=None
    )

参数  
*args: Python 函数或 TFF 类型规范，或两者（函数优先），或两者都不是。另请参阅 tff.tf_computation 以获取扩展文档。

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


#### federated_eval(...): Evaluates a federated computation at placement, returning the result.

#### federated_map(...): 使用映射函数逐点映射联合值。
    api    
    tff.federated_map(
        fn, arg
    )

参数  
- fn：将逐点应用于 arg 的成员成分的映射函数。 此函数的参数必须与 arg 的成员成分类型相同。    
- arg 放置在 tff.CLIENTS 或 tff.SERVER 的 TFF 联合类型的值（或可以隐式转换为 TFF 联合类型的值，例如，通过压缩）。

返回  
将 fn 逐点应用于 arg 的每个元素的序列，或者如果 arg 是联合的，则为联合序列，其结果是在每个位置本地且独立地在成员序列上调用 sequence_map。

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
# [<tf.Tensor: shape=(), dtype=int32, numpy=3>, 
# <tf.Tensor: shape=(), dtype=int32, numpy=6>, 
# <tf.Tensor: shape=(), dtype=int32, numpy=9>]
```

#### federated_mean(...): Computes a tff.SERVER mean of value placed on tff.CLIENTS.

#### federated_secure_select(...): Sends privately-selected values from a server database to clients.

#### federated_secure_sum(...): Computes a sum at tff.SERVER of a value placed on the tff.CLIENTS.

#### federated_select(...): Sends selected values from a server database to clients.

#### federated_sum(...): Computes a sum at tff.SERVER of a value placed on the tff.CLIENTS.

#### federated_value(...): Returns a federated value at placement, with value as the constituent.

#### federated_zip(...): Converts an N-tuple of federated values into a federated N-tuple value.

#### sequence_map(...): Maps a TFF sequence value pointwise using a given function fn.

#### sequence_reduce(...): Reduces a TFF sequence value given a zero and reduction operator op.

#### sequence_sum(...): Computes a sum of elements in a sequence.

#### structure_from_tensor_type_tree(...): Constructs a structure from a type_spec tree of tff.TensorTypes.

#### tf_computation(...): 将 Python 函数和 defuns 装饰为 TFF TensorFlow 计算。
    api  
    tff.tf_computation(
        *args, tff_internal_types=None
    )

参数  
*args: 如上面的 3 种模式和使用示例中所述，要么是 functiondefun，要么是 TFF 类型规范，或者两者都（函数优先），或者两者都不是。

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

#### to_type(...): Converts the argument into an instance of tff.Type.

#### to_value(...): Converts the argument into an instance of the abstract class tff.Value.




# tff.aggregators 用于构建联合聚合的库。
## Classes
#### DifferentiallyPrivateFactory: UnweightedAggregationFactory for tensorflow_privacy DPQueries.

#### EncodedSumFactory: UnweightedAggregationFactory for encoded sum.

#### MeanFactory: Aggregation factory for weighted mean.

#### PrivateQuantileEstimationProcess: A tff.templates.EstimationProcess for estimating private quantiles.

#### SecureSumFactory: AggregationProcess factory for securely summing values.

#### SumFactory: UnweightedAggregationFactory for sum.

#### UnweightedAggregationFactory: Factory for creating tff.templates.AggregationProcess without weights.

#### UnweightedMeanFactory: Aggregation factory for unweighted mean.

#### UnweightedReservoirSamplingFactory: An UnweightedAggregationFactory for reservoir sampling values.

#### WeightedAggregationFactory: Factory for creating tff.templates.AggregationProcess with weights.

## Functions
#### clipping_factory(...): Creates an aggregation factory to perform L2 clipping.

#### federated_max(...): Computes the maximum at tff.SERVER of a value placed at tff.CLIENTS.

#### federated_min(...): Computes the minimum at tff.SERVER of a value placed at tff.CLIENTS.

#### federated_sample(...): Aggregation to produce uniform sample of at most max_num_samples values.

#### secure_quantized_sum(...): Quantizes and sums values securely.

#### zeroing_factory(...): Creates an aggregation factory to perform zeroing.



# tff.analytics




# tff.backends




# tff.framework




# tff.learning
## Classes
#### BatchOutput 保存 tff.learning.Model 输出的结构
    api
    tff.learning.BatchOutput(
        loss, predictions, num_examples
    )

#### ClientFedAvg 用于联合平均的客户端 TensorFlow 逻辑。

#### ClientWeighting 用于称重客户端的内置方法的枚举。

#### Model 表示用于 TensorFlow Federated 的模型。

#### ModelWeights 模型的可训练和不可训练变量的容器。

## Functions
#### build_federated_averaging_process 构建执行联合平均的迭代过程
    api 
    tff.learning.build_federated_averaging_process(
        model_fn: Callable[[], tff.learning.Model],
        client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
        server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer] = DEFAULT_SERVER_OPTIMIZER_FN,
        *,
        client_weighting: Optional[tff.learning.ClientWeighting] = None,
        broadcast_process: Optional[tff.templates.MeasuredProcess] = None,
        aggregation_process: Optional[tff.templates.MeasuredProcess] = None,
        model_update_aggregation_factory: Optional[tff.aggregators.WeightedAggregationFactory] = None,
        use_experimental_simulation_loop: bool = False
    ) -> tff.templates.IterativeProcess

api参数  
- model_fn:	返回 tff.learning.Model 的无参数函数。此方法不得捕获 TensorFlow 张量或变量并使用它们。
模型必须在每次调用时完全从头构建，每次调用返回相同的预构建模型将导致错误。
- client_optimizer_fn:	返回 tf.keras.Optimizer 的无参数可调用对象。
- server_optimizer_fn:	返回 tf.keras.Optimizer 的无参数可调用对象。默认情况下，使用 tf.keras.optimizers.SGD，学习率为 1.0
- client_weighting:	一个 tff.learning.ClientWeighting 的值，它指定一个内置的加权方法，或者一个可调用的，
它接受 model.report_local_outputs 的输出并返回一个提供模型增量联合平均权重的张量。 如果没有，默认按示例数加权。
- broadcast_process: 一个 tff.templates.MeasuredProcess 将服务器上的模型权重广播到客户端。 
它必须支持签名 (input_values@SERVER -> output_values@CLIENT)。 
如果设置为默认无，则使用默认的 tff.federated_broadcast 向客户端广播服务器模型。
- aggregation_process: 一个 tff.templates.MeasuredProcess 将客户端上的模型更新聚合回服务器。 
它必须支持签名 ({input_values}@CLIENTS-> output_values@SERVER)。 
如果 model_update_aggregation_factory 不是 None，则必须为 None。
- model_update_aggregation_factory:	一个可选的 tff.aggregators.WeightedAggregationFactory 
或 tff.aggregators.UnweightedAggregationFactory 构造 tff.templates.AggregationProcess 用于在服务器上聚合客户端模型更新。
如果没有，使用 tff.aggregators.MeanFactory。如果aggregation_process 不是None，则必须为None。
- use_experimental_simulation_loop:	控制输入数据集的 reduce 循环函数。一个实验性的 reduce 循环用于模拟。当前需要将此标志设置为 True 以进行高性能 GPU 模拟。


api 说明  
此函数创建一个 tff.templates.IterativeProcess 对客户端模型执行联合平均。迭代过程有以下方法继承自 tff.templates.IterativeProcess：

- initialize: 具有功能类型签名的 tff.Computation ，其中 a 表示服务器的初始状态。(S@SERVER) S tff.learning.framework.ServerState

- next: 具有函数类型签名的 tff.Computation 其中是 tff.learning.framework.ServerState，其类型与 的输出匹配，并表示客户端数据集，
其中是单个批次的类型。此计算返回一个 tff.learning.framework.ServerState 表示更新的服务器状态和指标，
这些指标是客户端训练期间 tff.learning.Model.federated_output_computation 的结果以及来自广播和聚合过程的任何其他指标。
(<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>) S 初始化 {B*}@CLIENTS B

迭代过程还有以下方法没有继承自 tff.templates.IterativeProcess：  
- get_model_weights: 将 tff.learning.framework.ServerState 作为输入的 tff.Computation，并返回包含状态模型权重的 tff.learning.ModelWeights

每次调用该方法时，使用广播函数将服务器模型广播到每个客户端。对于每个客户端，通过客户端优化器的 tf.keras.optimizers.Optimizer.apply_gradients 方法执行一个 epoch 的本地训练。每个客户端计算训练后的客户端模型与初始广播模型之间的差异。然后使用一些聚合函数在服务器上聚合这些模型增量。通过使用服务器 optimizer.next 的 tf.keras.optimizers.Optimizer.apply_gradients 方法在服务器上应用聚合模型增量



#### build_federated_evaluation 为给定模型的联合评估构建 TFF 计算。

#### build_federated_sgd_process 使用联合 SGD 构建 TFF 计算以进行优化。

#### build_personalization_eval 构建用于评估个性化策略的 TFF 计算。

#### compression_aggregator 创建具有压缩和自适应归零和裁剪的聚合器。

#### dp_aggregator 创建具有自适应归零和差分隐私的聚合器

#### federated_aggregate_keras_metric 将放置在 CLIENTS 的 keras 度量变量聚合到 SERVER。

#### from_keras_model 从 tf.keras.Model 构建 tff.learning.Model
    api  
    tff.learning.from_keras_model(
        keras_model: tf.keras.Model,
        loss: Loss,
        input_spec,
        loss_weights: Optional[List[float]] = None,
        metrics: Optional[List[tf.keras.metrics.Metric]] = None
    )

#### robust_aggregator 使用自适应归零和裁剪创建均值聚合器。
    api
    tff.learning.robust_aggregator(
        *,
        zeroing: bool = True,
        clipping: bool = True,
        weighted: bool = True
    ) -> tff.aggregators.AggregationFactory
#### secure_aggregator 使用自适应归零和裁剪创建均值聚合器。

#### state_with_new_model_weights 返回具有更新模型权重的 ServerState。




# tff.simulation




# tff.templates 常用计算的模板。
#### AggregationProcess 聚合值的有状态过程。
    api  
        tff.templates.AggregationProcess(  
            initialize_fn: tff.Computation,  
            next_fn: tff.Computation  
        )
api 参数  
- initialize_fn：返回聚合过程初始状态的无参数 tff.Computation。返回的状态必须是服务器放置的联合值。 让这种状态的类型称为 S@SERVER。
- next_fn：表示迭代函数的 tff.Computation。next_fn 必须至少接受两个参数，第一个是状态类型 S@SERVER，第二个是 V@CLIENTS 类型的客户端放置数据。 next_fn 必须返回一个 MeasuredProcessOutput，其中状态属性与类型 S@SERVER 匹配，结果属性与类型 V@SERVER 匹配。

#### EstimationProcess 可以计算某个值的估计值的有状态过程。
    api
    tff.templates.EstimationProcess(
        initialize_fn: tff.Computation,
        next_fn: tff.Computation,
        report_fn: tff.Computation,
        next_is_multi_arg: Optional[bool] = None
    )


#### IterativeProcess 包括初始化和迭代计算的过程。
    api  
    tff.templates.IterativeProcess(
        initialize_fn: tff.Computation,
        next_fn: tff.Computation,
        next_is_multi_arg: Optional[bool] = None
    )

api 参数  
- initialize_fn: 返回迭代过程初始状态的无参数 tff.Computation。设这种状态的类型称为 S。
- next_fn: 表示迭代函数的 tff.Computation。第一个或唯一的参数必须与状态类型 S 匹配。第一个或唯一的返回值也必须与状态类型 S 匹配。
- next_is_multi_arg: 一个可选的布尔值，指示 next_fn 将接收的不仅仅是状态参数（如果为真）或仅状态参数（如果为假）。此参数主要用于提供更好的错误消息。

```python
# 迭代过程通常由控制循环驱动，例如：

import tensorflow as tf
import tensorflow_federated as tff

num_rounds = 2

def initialize_fn():
  ...

def next_fn(state):
  ...

iterative_process = tff.IterativeProcess(initialize_fn, next_fn)
state = iterative_process.initialize()
for round in range(num_rounds):
  state = iterative_process.next(state)

# initialize_fn 函数必须返回一个对象，该对象预期作为 next_fn 函数的输入并由 next_fn 函数返回。按照惯例，我们将此对象称为状态。

# 迭代步骤（next_fn 函数）可以接受除状态（必须是第一个参数）之外的参数，并返回其他参数，状态是第一个输出参数：
def initialize_fn():
  ...

def next_fn(state):
  ...

iterative_process = tff.IterativeProcess(initialize_fn, next_fn)
state = iterative_process.initialize()
for round in range(num_rounds):
  state, output = iterative_process.next(state, round)
```
#### MeasuredProcess 产生指标的有状态过程。 继承 IterativeProcess
    api
    tff.templates.MeasuredProcess(
        initialize_fn: tff.Computation,
        next_fn: tff.Computation,
        next_is_multi_arg: Optional[bool] = None
    )

api 参数  
- initialize_fn: 返回迭代过程初始状态的无参数 tff.Computation。设这种状态的类型称为 S。
- next_fn: 表示迭代函数的 tff.Computation。 第一个或唯一的参数必须与状态类型 S 匹配。返回值必须是 MeasuredProcessOutput，其状态成员与状态类型 S 匹配。
- next_is_multi_arg: 一个可选的布尔值，指示 next_fn 将接收的不仅仅是状态参数（如果为真）或仅状态参数（如果为假）。 此参数主要用于提供更好的错误消息。

api 说明  
1. 一个 tff.templates.MeasuredProcess 是一个 tff.templates.IterativeProcess ，它的下一个计算返回一个 tff.templates.MeasuredProcessOutput
2. 组合指南 两个 MeasuredProcesses F(x) 和 G(y) 可以组合成一个名为 C 的新 MeasuredProcess，具有以下属性：
- `C.state` is the concatenation `<F.state, G.state>`. 
- `C.next(C.state, x).result == G.next(G.state, F.next(F.state, x).result).result` 
- `C.measurements` is the concatenation `<F.measurements, G.measurements>`.
3. 生成的组合 C 将具有以下类型签名  
- initialize: (<F.initialize, G.initialize>) 
- next: (<<F.state, G.state>, F.input> -> <state=<F.state, G.State>, result=G.result, measurements=<F.measurements, G.measurements>>)


#### MeasuredProcessOutput 包含 MeasuredProcess.next 计算输出的结构。
    api
    tff.templates.MeasuredProcessOutput(
        state, result, measurements
    )

api 参数  
- state：将传递给 MeasuredProcess.next 调用的结构。 不用于外部检查，包含流程的实施细节。
- result：给定当前输入和状态的过程结果。使用组合规则，或者传递给链接的 MeasuredProcess 的输入参数，或者与并行 MeasuredProcesses 的输出连接。
- measurements：从结果计算得出的指标。 用于显示值以跟踪未发送到链接的 MeasuredProcesses 的过程的进度。




# tff.types
#### FederatedType: 表示 TFF 中的联合类型。
api  
tff.types.FederatedType(
    member, placement, all_equal=None
)

参数  
- member：tff.Type 的实例,表示此联合类型的每个值的成员组件的类型。
- placement: 此联合类型的成员组件所在的放置规范。 必须是诸如 tff.SERVER 或 tff.CLIENTS 之类的放置文字以引用全局定义的放置，
或者是放置标签以引用在类型签名的其他部分中定义的放置。 尚未实施指定展示位置标签。
- all_equal=None: 一个 bool 值，指示联合类型的所有成员是相等 (True) 还是允许不同 (False)。 
如果 all_equal 为 None，则选择该值作为展示位置的默认值，例如，对于 tff.SERVER 为 True，对于 tff.CLIENTS 为 False。

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
api  
tff.types.StructType(
    elements, enable_wf_check=True
)

api参数
- elements：元素规范的可迭代。每个元素规范要么是元素的类型规范（tff.Type 的实例或可通过 tff.to_type 转换为它的东西），
要么是已定义名称的元素的 (name, spec)。或者，可以在此处提供一个 collections.OrderedDict 实例，将元素名称映射到它们的类型（或可转换为类型的事物）。
- enable_wf_check=True：此标志仅存在以便 StructWithPythonType 可以禁用格式良好检查，因为在子类完成自己的初始化之前类型不会是格式良好的。

命名元组类型，这些是 TFF 使用指定类型构造具有预定义数量元素的元组或字典式结构（无论命名与否）的方式。重要的一点是，TFF 的命名元组 概念包含等效于
Python 参数元组的抽象，即元组的元素集合中有一部分（并非全部）是命名元素，还有一部分是位置元素。 
命名元组的紧凑表示法为 <n_1=T_1, ..., n_k=T_k>，其中 n_k 是可选元素名称，T_k 是元素类型。 
例如，<int32,int32> 是一对未命名整数的紧凑表示法，<X=float32,Y=float32> 是命名为 X 和 Y（可能代表平面上的一个点）的一对浮点数的紧凑表示法。
元组可以嵌套，也可以与其他类型混用，例如，<X=float32,Y=float32>* 可能是一系列点的紧凑表示法


#### StructWithPythonType: 与 Python 容器类型配对的结构的表示。
暂无

#### TensorType: 表示 TFF 中的张量类型。
张量类型，对象不仅限于在 TensorFlow 计算图中表示 TensorFlow 运算输出的 Python 的 tf.Tensor 实例，而是也可能包括可产生的数据单位， 例如，作为分布聚合协议的输出。张量类型的紧凑表示法为 dtype 或 dtype[shape]。例如，int32 和 int32[10] 分别是整数和整数向量的类型。




#### type_at_clients(...): Constructs a federated type of the form {T}@CLIENTS.


#### type_at_server(...): Constructs a federated type of the form T@SERVER.