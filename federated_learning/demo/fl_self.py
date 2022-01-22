import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 是否开启 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow_federated as tff

# 加载数据集
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
print('总共有多少个客户端：', len(emnist_train.client_ids))
print('第一个客户端有多少数据：', len(emnist_train.client_ids[0]))

NUM_CLIENTS = 10
BATCH_SIZE = 20


# 数据预处理
def preprocess(dataset):
  # 展平一批 EMNIST 数据并返回一个（feature，label）元组
  def batch_format_fn(element):
    return (tf.reshape(element['pixels'], [-1, 784]), tf.reshape(element['label'], [-1, 1]))
  
  return dataset.batch(BATCH_SIZE).map(batch_format_fn)


# 选择少量客户端，并将上述预处理应用于其数据集
client_ids = sorted(emnist_train.client_ids)[:NUM_CLIENTS]
federated_train_data = [preprocess(emnist_train.create_tf_dataset_for_client(x)) for x in client_ids]


# 创建模型
def create_keras_model():
  main_input = tf.keras.Input(shape=(784,), dtype='int32', name='main_input')
  dense1 = tf.keras.layers.Dense(256, activation='relu')(main_input)
  dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
  dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
  dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
  dense3 = tf.keras.layers.Dense(64, activation='relu')(dropout2)
  output = tf.keras.layers.Dense(10, activation='relu')(dense3)
  
  return tf.keras.models.Model(inputs=main_input, outputs=output)
  
  # initializer = tf.keras.initializers.GlorotNormal(seed=0)
  # return tf.keras.models.Sequential([
  #   tf.keras.layers.Input(shape=(784,)),
  #   tf.keras.layers.Dense(256, kernel_initializer=initializer),
  #   tf.keras.layers.Dense(128, kernel_initializer=initializer),
  #   tf.keras.layers.Dense(10, kernel_initializer=initializer),
  #   tf.keras.layers.Softmax(),
  # ])


def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
    keras_model,
    input_spec=federated_train_data[0].element_spec,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()]
  )


# 构建自己的联合学习算法
'''
tff.learning API允许创建联合平均的许多变体，但还有其他联合算法不能完全适应这个框架。例如，您可能希望添加正则化、裁剪或更复杂的算法
对于这些更高级的算法，我们必须使用TFF编写自己的自定义算法。在许多情况下，联合算法有 4 个主要组件：
  1 服务器到客户端广播步骤。
  2 本地客户端更新步骤。
  3 客户端到服务器上载步骤。
  4 服务器更新步骤。
  
同时，我们应该重写 initialize 和 next 函数。
'''


# 2 本地客户端更新步骤 返回客户端的权重
@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
  # 使用当前服务器权重初始化客户端模型。
  client_weights = model.trainable_variables
  
  # 将服务器权重分配给客户端模型。
  tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)
  
  # 使用 client_optimizer 更新本地模型。
  for batch in dataset:
    with tf.GradientTape() as tape:
      # Compute a forward pass on the batch of data
      outputs = model.forward_pass(batch)
    
    # Compute the corresponding gradient
    grads = tape.gradient(outputs.loss, client_weights)
    grads_and_vars = zip(grads, client_weights)
    
    # Apply the gradient using a client optimizer.
    client_optimizer.apply_gradients(grads_and_vars)
  
  return client_weights


# 4 服务器更新 返回模型权重
@tf.function
def server_update(model, mean_client_weights):
  model_weights = model.trainable_variables
  tf.nest.map_structure(lambda x, y: x.assign(y), model_weights, mean_client_weights)
  return model_weights


# 创建初始化计算 tff.tf_computation来分离出TensorFlow代码
@tff.tf_computation
def server_init():
  model = model_fn()
  return model.trainable_variables


# tff.federated_value 将其直接传递到联合计算中
@tff.federated_computation
def initialize_fn():
  # 具有给定放置位置的联合值，并且成员组成值在所有位置均相等。
  return tff.federated_value(server_init(), tff.SERVER)


# 我们现在使用我们的客户端和服务器更新代码来编写实际的算法。
# 首先将我们的转换成一个 tff.tf_computation，它接受客户端数据集和服务器权重，并输出更新的客户端权重 tensor.client_update
whimsy_model = model_fn()
tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)  # 输入规范
model_weights_type = server_init.type_signature.result  # 输出规范

print('客户端输入规范 tf_dataset_type ：', str(tf_dataset_type))
print('服务端输出规范 model_weights_type ：', str(model_weights_type))


@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
  model = model_fn()
  client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  return client_update(model, tf_dataset, server_weights, client_optimizer)


@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
  model = model_fn()
  return server_update(model, mean_client_weights)


federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)


# 重写 next function 状态是 server_weights.
@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
  # step1. Broadcast the server weights to the clients.
  server_weights_at_client = tff.federated_broadcast(server_weights)
  
  # step2. Each client computes their updated weights.
  client_weights = tff.federated_map(client_update_fn, (federated_dataset, server_weights_at_client))
  
  # step3. The server averages these updates.
  mean_client_weights = tff.federated_mean(client_weights)
  
  # step4. The server averages these updates.
  server_weights = tff.federated_map(server_update_fn, mean_client_weights)
  
  return server_weights


federated_algorithm = tff.templates.IterativeProcess(
  initialize_fn=initialize_fn,
  next_fn=next_fn
)

print('迭代过程和函数的类型签名 federated_algorithm.initialize: ', str(federated_algorithm.initialize.type_signature))
print('迭代过程和函数的类型签名 federated_algorithm.next: ', str(federated_algorithm.next.type_signature))

# 评估算法
# 创建一个集中式评估数据集，然后应用用于训练数据的相同预处理
central_emnist_test = emnist_test.create_tf_dataset_from_all_clients()
central_emnist_test = preprocess(central_emnist_test)

# 接下来，我们编写一个接受服务器状态的函数，并使用 Keras 对测试数据集进行评估
def evaluate(server_state):
  keras_model = create_keras_model()
  keras_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
  )
  keras_model.set_weights(server_state)
  keras_model.evaluate(central_emnist_test)

# 初始化算法并在测试集上进行评估
server_state = federated_algorithm.initialize()
evaluate(server_state)

for round in range(30):
  server_state = federated_algorithm.next(server_state, federated_train_data)
evaluate(server_state)