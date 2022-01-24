import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 是否开启 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import numpy as np
import collections
from typing import Callable, List, OrderedDict

'''
os.environ[" xxxxxx "]='x'
TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息
'''

# 测试是否可以使用tff
print(tff.federated_computation(lambda: 'Hello, TFF!')())

# 加载数据集,并打印数据集的形状
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
print(len(emnist_train.client_ids))
print(emnist_train.element_type_structure)

# 抽取一个客户端的数据集，查看一下数据集的内容
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
fig = plt.figure(figsize=(20, 4))
idx = 0
for example in example_dataset.take(20):
  plt.subplot(5, 5, idx + 1)
  plt.imshow(example['pixels'].numpy(), cmap='gray', aspect='equal')
  plt.axis('off')
  idx += 1

# 抽取几个客户端，并查看每个客户端数据集的基本分布
f = plt.figure(figsize=(12, 7))
f.suptitle('Label Counts for a Sample of Clients')
for i in range(6):
  client_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[i])
  plot_data = collections.defaultdict(list)
  for example in client_dataset:
    label = example['label'].numpy()
    plot_data[label].append(label)
  plt.subplot(2, 3, i + 1)
  plt.title('Client {}'.format(i))
  for j in range(10):
    plt.hist(plot_data[j], density=False, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 随机抽取几个客户端进行mnist数据集平均图像像素的可视化表达，
for i in range(5):
  client_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[i])
  plot_data = collections.defaultdict(list)
  for example in client_dataset:
    plot_data[example['label'].numpy()].append(example['pixels'].numpy())
  f = plt.figure(i, figsize=(12, 5))
  f.suptitle('Client {}`s mean image per label:'.format(i))
  for idx in range(10):
    mean_img = np.mean(plot_data[idx], 0)
    plt.subplot(2, 5, idx + 1)
    plt.imshow(mean_img.reshape((28, 28)))
    plt.axis('off')
plt.show()

# 预处理数据集
NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess(dataset):
  def batch_format_fn(element):
    return collections.OrderedDict(
      x=tf.reshape(element['pixels'], [-1, 784]),
      y=tf.reshape(element['label'], [-1, 1])
    )
  
  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(BATCH_SIZE).map(batch_format_fn).prefetch(
    PREFETCH_BUFFER)


# 验证一下这个函数是否对数据集进行了修改
preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))
print('打印修改后的数据集样式', sample_batch)


# 创建联邦数据集
def make_federated_data(client_data, client_ids):
  return [
    preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids
  ]


sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)
print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))

print('--------------------------------------基于 Keras 的模型------------------------------------------')
# 基于 Keras 的模型创建
def create_keras_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(10, kernel_initializer='zeros'),
    tf.keras.layers.Softmax(),
  ])


'''
为了将任何模型与TFF一起使用，它需要包装在 tff.learning.Model 接口的实例中，该接口公开了标记模型的正向传递，元数据属性等的方法，
类似于Keras，但也引入了其他元素，例如控制计算联合指标过程的方法。现在让我们不要担心这一点;如果有一个像上面刚刚定义的Keras模型，
可以通过调用tff.learning.from_keras_model，将模型和示例数据批处理作为参数传递给TFF来为你包装它，如下所示。
'''


# 我们在这里创建一个新模型，并且不从外部范围使用，而是通过 TFF 将在不同的图形上下文中调用它。
def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
    keras_model,
    input_spec=preprocessed_example_dataset.element_spec,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
  )


'''
现在我们有一个模型包装为tff.learning.Model 以与 TFF 一起使用，然后可以让 TFF 通过调用帮助器函数
tff.learning.build_federated_averaging_process来构造联合平均算法，如下所示。
'''
iterative_process = tff.learning.build_federated_averaging_process(
  model_fn,
  client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
  server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)
print(str(iterative_process.initialize.type_signature))

# 构造服务器状态
state = iterative_process.initialize()

# 训练几次并可视化结果，将上面已经生成的联合数据用于用户示例。
state, metrics = iterative_process.next(state, federated_train_data)
NUM_ROUNDS = 11
for round_num in range(1, NUM_ROUNDS):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('keras_model: round {:2d}, metrics={}'.format(round_num, metrics))

# 评估模型
evaluation = tff.learning.build_federated_evaluation(model_fn)
print(str(evaluation.type_signature))

train_metrics = evaluation(state.model, federated_train_data)
print(str(train_metrics))

federated_test_data = make_federated_data(emnist_test, sample_clients)
print(len(federated_test_data), federated_test_data[0])

test_metrics = evaluation(state.model, federated_test_data)
print(str(test_metrics))

print('--------------------------------------自定义模型------------------------------------------')
'''
虽然我们可以在 TFF 中通过 tff.learning.from_keras_model 使用 Keras 模型。但是，tff.learning 提供的是一个较低级别的
模型接口 tff.learning.Model，它公开了使用模型进行联合学习所需的最小功能。直接实现这个接口（可能仍然使用像tf.keras.layers
这样的构建块）允许在不修改联合学习算法内部的情况下进行最大的定制。因此我们需要进行一个自定义的模型
'''
# 定义模型变量、正向传递和指标
# 定义一个新的数据结构用来表示真个数据集
MnistVariables = collections.namedtuple('MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')


# 创建变量，并将初始的变量保存下来
def create_mnist_variables():
  return MnistVariables(
    weights=tf.Variable(
      lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
      name='weights',
      trainable=True),
    bias=tf.Variable(
      lambda: tf.zeros(dtype=tf.float32, shape=(10)),
      name='bias',
      trainable=True),
    num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
    loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
    accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False)
  )


# 定义前向传递方法，用于计算损失、发出预测并更新单批输入数据的累积统计信息
def predict_on_batch(variables, x):
  return tf.nn.softmax(tf.matmul(x, variables.weights) + variables.bias)


def mnist_forward_pass(variables, batch):
  y = predict_on_batch(variables, batch['x'])
  predictions = tf.cast(tf.argmax(y, 1), tf.int32)
  
  flat_labels = tf.reshape(batch['y'], [-1])
  loss = -tf.reduce_mean(
    tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1])
  )
  
  accuracy = tf.reduce_mean(
    tf.cast(tf.equal(predictions, flat_labels), tf.float32)
  )
  
  num_examples = tf.cast(tf.size(batch['y']), tf.float32)
  
  variables.num_examples.assign_add(num_examples)
  variables.loss_sum.assign_add(loss * num_examples)
  variables.accuracy_sum.assign_add(accuracy * num_examples)
  
  return loss, predictions


# 我们定义一个函数，该函数再次使用 TensorFlow 返回一组本地指标。在这个函数中，可以确定一些是有资格在联合学习或评估过程中聚合到服务器的
# 值（除了自动处理的模型更新之外）。在这里，我们只需返回平均值和 ，以及在计算联合聚合时，我们需要正确加权来自不同用户的贡献
def get_local_mnist_metrics(variables):
  return collections.OrderedDict(
    num_examples=variables.num_examples,
    loss=variables.loss_sum / variables.num_examples,
    accuracy=variables.accuracy_sum / variables.num_examples
  )


# 最后，我们需要确定如何聚合每个设备发出的本地指标，在这里是使用 TFF 表示的联合计算。
@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
  return collections.OrderedDict(
    num_examples=tff.federated_sum(metrics.num_examples),
    loss=tff.federated_mean(metrics.loss, metrics.num_examples),
    accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))


# 构造 tff.learning.Model 的实例
class MnistModel(tff.learning.Model):
  
  def __init__(self):
    self._variables = create_mnist_variables()
  
  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]
  
  @property
  def non_trainable_variables(self):
    return []
  
  @property
  def local_variables(self):
    return [self._variables.num_examples, self._variables.loss_sum, self._variables.accuracy_sum]
  
  @property
  def input_spec(self):
    return collections.OrderedDict(
      x=tf.TensorSpec([None, 784], tf.float32),
      y=tf.TensorSpec([None, 1], tf.int32)
    )
  
  @tf.function
  def predict_on_batch(self, x, training=True):
    del training
    return predict_on_batch(self._variables, x)
  
  @tf.function
  def forward_pass(self, batch, training=True):
    del training
    loss, predictions = mnist_forward_pass(self._variables, batch)
    num_exmaples = tf.shape(batch['x'])[0]
    return tff.learning.BatchOutput(loss=loss, predictions=predictions, num_examples=num_exmaples)
  
  @tf.function
  def report_local_outputs(self):
    return get_local_mnist_metrics(self._variables)
  
  @property
  def federated_output_computation(self):
    return aggregate_mnist_metrics_across_clients
  
  """为未最终确定的值创建度量名称的： OrderedDict """
  
  @tf.function
  def report_local_unfinalized_metrics(self) -> OrderedDict[str, List[tf.Tensor]]:
    return collections.OrderedDict(
      num_examples=[self._variables.num_examples],
      loss=[self._variables.loss_sum, self._variables.num_examples],
      accuracy=[self._variables.accuracy_sum, self._variables.num_examples]
    )
  
  """ 为最终值创建度量名称： OrderedDict """
  
  def metric_finalizers(self) -> OrderedDict[str, Callable[[List[tf.Tensor]], tf.Tensor]]:
    return collections.OrderedDict(
      num_examples=tf.function(func=lambda x: x[0]),
      loss=tf.function(func=lambda x: x[0] / x[1]),
      accuracy=tf.function(func=lambda x: x[0] / x[1])
    )


# 使用新模型进行训练
iterative_process = tff.learning.build_federated_averaging_process(
  MnistModel,
  client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02)
)

# 创建服务器状态
state = iterative_process.initialize()
# 打印自己创建模型的的结果
for round_num in range(1, 10):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('create_model_self:  round {:2d}, metrics={}'.format(round_num, metrics))

# 评估模型
evaluation = tff.learning.build_federated_evaluation(MnistModel)
print(str(evaluation.type_signature))

train_metrics = evaluation(state.model, federated_train_data)
print(str(train_metrics))

federated_test_data = make_federated_data(emnist_test, sample_clients)
print(len(federated_test_data), federated_test_data[0])

test_metrics = evaluation(state.model, federated_test_data)
print(str(test_metrics))
