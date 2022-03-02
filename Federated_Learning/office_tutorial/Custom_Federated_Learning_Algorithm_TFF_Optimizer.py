import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 是否开启 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import functools
import attr
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

only_digits = True

# 加载数据集
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(True)


# 定义预处理函数
def preprocess_fn(dataset, batch_size=16):
  def batch_format_fn(element):
    return (tf.expand_dims(element['pixels'], -1), element['label'])
  
  return dataset.batch(batch_size).map(batch_format_fn)


# 用于原型设计的预处理和示例客户端
train_client_ids = sorted(emnist_train.client_ids)
print(train_client_ids)
train_data = emnist_train.preprocess(preprocess_fn)
central_test_data = preprocess_fn(emnist_train.create_tf_dataset_for_client(train_client_ids[0]))


# 定义模型
def create_keras_model():
  data_format = 'channels_last'
  input_shape = [28, 28, 1]
  
  max_pool = functools.partial(
    tf.keras.layers.MaxPooling2D,
    pool_size=(2, 2),
    padding='same',
    data_format=data_format)
  conv2d = functools.partial(
    tf.keras.layers.Conv2D,
    kernel_size=5,
    padding='same',
    data_format=data_format,
    activation=tf.nn.relu)
  
  model = tf.keras.models.Sequential([
    conv2d(filters=32, input_shape=input_shape),
    max_pool(),
    conv2d(filters=64),
    max_pool(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10 if only_digits else 62),
  ])
  
  return model


# 包装为 `tff.learning.Model`.
def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
    keras_model,
    input_spec=central_test_data.element_spec,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  )


'''
自定义迭代过程 （构建联合平均 （FedAvg） 算法）
联合算法有 4 个主要组件：服务器到客户端广播步骤。本地客户端更新步骤。客户端到服务器上载步骤。服务器更新步骤。
'''

# TF 块：客户端和服务器更新
"""Performs local training on the client's dataset."""


@tf.function
def client_update(model, dataset, server_weights, clients_optimizer):
  # 使用当前服务器权重初始化客户端模型
  client_weights = model.trainable_variables
  
  # 将服务器权重分配给客户端模型。
  tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)
  
  # 初始化客户端优化器
  trainable_tensor_specs = tf.nest.map_structure(lambda v: tf.TensorSpec(v.shape, v.dtype), client_weights)
  optimizer_state = clients_optimizer.initialize(trainable_tensor_specs)
  
  # 使用 client_optimizer 更新本地模型。
  for batch in iter(dataset):
    with tf.GradientTape() as tape:
      # 计算这批数据的前向传递。
      outputs = model.forward_pass(batch)
    
    # 计算相应的梯度。
    grads = tape.gradient(outputs.loss, client_weights)
    
    # 使用客户端优化器应用梯度
    optimizer_state, updated_weights = clients_optimizer.next(optimizer_state, client_weights, grads)
    tf.nest.map_structure(lambda a, b: a.assign(b), client_weights, updated_weights)
  
  # Return model deltas.
  return tf.nest.map_structure(tf.subtract, client_weights, server_weights)


@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
  trainable_weights = attr.ib()
  optimizer_state = attr.ib()


''' Updates the server model weights. '''
@tf.function
def server_update(server_state, mean_model_delta, server_optimizer):
  # Use aggregated negative model delta as pseudo gradient.
  negative_weights_delta = tf.nest.map_structure(lambda w: -1.0 * 2, mean_model_delta)
  new_optimizer_state, update_weights = server_optimizer.next(server_state.optimizer_state,
                                                              server_state.trainable_weights,
                                                              negative_weights_delta)
  
  return tff.structure.updata_struct(server_state, trainable_weights=update_weights,
                                     optimizer_state=new_optimizer_state)


'''
TFF 块：tff.tf_computation 和 tff.federated_computation. 我们现在使用 TFF 进行编排，并为 FedAvg 构建迭代过程。
此时，我们必须用 tff.tf_computation 包装上面定义的 TF 块(client_update, server_update)，
并在 tff.federated_computation 函数中使用 TFF 方法 tff.federated_broadcast、tff.federated_map、tff.federated_mean。
而且在定义自定义迭代过程时，很容易使用 tff.learning.optimizers.Optimizer API 和 functions 。initialize next
'''

# 1. Server and client optimizer to be used.
server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.05, momentum=0.9)
client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)


# 2. Functions return initial state on server.
@tff.tf_computation
def server_init():
  model = model_fn()
  trainable_tensor_specs = tf.nest.map_structure(
    lambda v: tf.TensorSpec(v.shape, v.dtype), model.trainable_variables
  )
  optimizer_state = server_optimizer.initialize(trainable_tensor_specs)
  return ServerState(
    trainable_weights=model.trainable_variables,
    optimizer_state=optimizer_state
  )


@tff.federated_computation
def server_init_tff():
  return tff.federated_value(server_init(), tff.SERVER)


# 3 一轮计算和通信
server_state_type = server_init.type_signature.result
print('server_state_type:\n', server_state_type.formatted_representation())

trainable_weights_type = server_state_type.trainable_weights
print('trainable_weights_type:\n', trainable_weights_type.formatted_representation())


# 3-1. 使用 `tff.tf_computation` 包装服务器和客户端 TF 块。
@tff.tf_computation(server_state_type, trainable_weights_type)
def server_update_fn(server_state, model_delta):
  return server_update(server_state, model_delta, server_optimizer)


whimsy_model = model_fn()
tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)
print('tf_dataset_type:\n', tf_dataset_type.formatted_representation())


@tff.tf_computation(tf_dataset_type, trainable_weights_type)
def client_update_fn(dataset, server_weights):
  model = model_fn()
  return client_update(model, dataset, server_weights, client_optimizer)


# 3-2. 使用“tff.federated_computation”进行编排。
federated_server_type = tff.FederatedType(server_state_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)


@tff.federated_computation(federated_server_type, federated_dataset_type)
def run_one_round(server_state, federated_dataset):
  # 1 Server-to-client broadcast.
  server_weights_at_client = tff.federated_broadcast(server_state.trainable_weights)
  
  # 2 Local client update.
  model_deltas = tff.federated_map(client_update_fn, (federated_dataset, server_weights_at_client))
  
  # 3 Client-to-server upload and aggregation.
  mean_model_delta = tff.federated_mean(model_deltas)
  
  # 4 Server update.
  server_state = tff.federated_map(server_update_fn, (server_state, mean_model_delta))
  
  return server_state


# 4. Build the iterative process for FedAvg.
fedavg_process = tff.templates.IterativeProcess(initialize_fn=server_init_tff, next_fn=run_one_round)

print('type signature of `initialize`:\n', fedavg_process.initialize.type_signature.formatted_representation())
print('type signature of `next`:\n', fedavg_process.next.type_signature.formatted_representation())


# 评估算法
def evaluate(server_state):
  keras_model = create_keras_model()
  tf.nest.map_structure(lambda var, t: var.assign(t), keras_model.trainable_weights, server_state.trainable_weights)
  metric = tf.keras.metrics.SparseCategoricalAccuracy()
  for batch in iter(central_test_data):
    preds = keras_model(batch[0], training=False)
    metric.update_state(y_true=batch[1], y_pred=preds)
  return metric.result().numpy()


server_state = fedavg_process.initialize()
acc = evaluate(server_state)
print('Initial test accuracy', acc)

# Evaluate after a few rounds
CLIENTS_PER_ROUND = 20
sampled_clients = train_client_ids[:CLIENTS_PER_ROUND]
sampled_train_data = [train_data.create_tf_dataset_for_client(client) for client in sampled_clients]
for round in range(20):
  server_state = fedavg_process.next(server_state, sampled_train_data)
acc = evaluate(server_state)
print('Test accuracy', acc)
