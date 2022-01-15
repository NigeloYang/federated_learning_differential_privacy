import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import numpy as np
import collections
import os

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
print(sample_batch)


# 创建联邦数据集
def make_federated_data(client_data, client_ids):
  return [
    preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids
  ]


sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)
print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))


# 创建模型
def create_keras_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(10, kernel_initializer='zeros'),
    tf.keras.layers.Softmax(),
  ])


'''
为了将任何模型与TFF一起使用，它需要包装在tff.learning.Model接口的实例中，该接口公开了标记模型的正向传递，元数据属性等的方法，
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