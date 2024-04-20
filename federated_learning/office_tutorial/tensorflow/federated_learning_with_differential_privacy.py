import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 是否开启 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp
import matplotlib.pyplot as plt
import seaborn as sns


# 获取数据集，以及重新格式化
def get_mnist_dataset():
  train_data, test_data = tff.simulation.datasets.emnist.load_data(only_digits=True)
  
  def element_fn(element):
    return collections.OrderedDict(
      x=tf.expand_dims(element['pixels'], -1), y=element['label']
    )
  
  def preprocess_train_dataset(dataset):
    return dataset.map(element_fn).shuffle(buffer_size=418).repeat(1).batch(32, drop_remainder=False)
  
  def preprocess_test_dataset(dataset):
    return dataset.map(element_fn).batch(128, drop_remainder=False)
  
  train_data = train_data.preprocess(preprocess_train_dataset())
  test_data = preprocess_test_dataset(test_data.create_tf_dataset_from_all_clients())
  return train_data, test_data


train_data, test_data = get_mnist_dataset()


# 定义训练模型
def my_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10)
  ])
  
  return tff.learning.from_keras_model(
    keras_model=model,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    input_spec=test_data.element_spec,
    metrics=[tf.keras.metrics.SparseCategoricalCrossentropy]
  )


# 确定模型的噪声敏感度
# Run five clients per thread. Increase this if your runtime is running out of memory.
# Decrease it if you have the resources and want to speed up execution.
tff.backends.native.set_local_python_execution_context(clients_per_thread=5)

total_clients = len(train_data.client_ids)


def train(rounds, noise_multiplier, clients_per_round, data_frame):
  # Using the `dp_aggregator` here turns on differential privacy with adaptive clipping.
  aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
    noise_multiplier, clients_per_round)
  
  # We use Poisson subsampling which gives slightly tighter privacy guarantees
  # compared to having a fixed number of clients per round. The actual number of
  # clients per round is stochastic with mean clients_per_round.
  sampling_prob = clients_per_round / total_clients
  
  # Build a federated averaging process.
  # Typically a non-adaptive server optimizer is used because the noise in the
  # updates can cause the second moment accumulators to become very large prematurely.
  learning_process = tff.learning.build_federated_averaging_process(
    my_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0, momentum=0.9),
    model_update_aggregation_factory=aggregation_factory
  )
  
  eval_process = tff.learning.build_federated_evaluation(my_model)
  
  # Training loop.
  state = learning_process.initialize()
  for round in range(rounds):
    if round % 5 == 0:
      metrics = eval_process(state.model, [test_data])['eval']
      if round < 25 or round % 25 == 0:
        print(f'Round {round:3d}: {metrics}')
      data_frame = data_frame.append({'Round': round, 'NoiseMultiplier': noise_multiplier, **metrics},
                                     ignore_index=True)
    
    # Sample clients for a round. Note that if your dataset is large and
    # sampling_prob is small, it would be faster to use gap sampling.
    x = np.random.uniform(size=total_clients)
    sampled_clients = [
      train_data.client_ids[i] for i in range(total_clients) if x[i] < sampling_prob
    ]
    sampled_train_data = [
      train_data.create_tf_dataset_for_client(client) for client in sampled_clients
    ]
    
    # Use selected clients for update.
    state, metrics = learning_process.next(state, sampled_train_data)
  
  metrics = eval_process(state.model, [test_data])['eval']
  print(f'Round {rounds:3d}: {metrics}')
  data_frame = data_frame.append({'Round': rounds, 'NoiseMultiplier': noise_multiplier, **metrics}
                                 , ignore_index=True)
  
  return data_frame


data_frame = pd.DataFrame()
rounds = 100
clients_per_round = 50

for noise_multiplier in [0.0, 0.5, 0.75, 1.0]:
  print(f'Starting training with noise multiplier: {noise_multiplier}')
  data_frame = train(rounds, noise_multiplier, clients_per_round, data_frame)
  print()


def make_plot(data_frame):
  plt.figure(figsize=(15, 5))

  dff = data_frame.rename(columns={'sparse_categorical_accuracy': 'Accuracy', 'loss': 'Loss'})

  plt.subplot(121)
  sns.lineplot(data=dff, x='Round', y='Accuracy', hue='NoiseMultiplier', palette='dark')
  plt.subplot(122)
  sns.lineplot(data=dff, x='Round', y='Loss', hue='NoiseMultiplier', palette='dark')
  
print('开始绘图：')
make_plot(data_frame)

# 基于 RDP 的隐私保护
rdp_orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] + list(range(5, 64)) + [128, 256, 512])
total_clients = 3383
base_noise_multiplier = 0.5
base_clients_per_round = 50
target_delta = 1e-5
target_eps = 2

def get_epsilon(clients_per_round):
  # If we use this number of clients per round and proportionally
  # scale up the noise multiplier, what epsilon do we achieve?
  q = clients_per_round / total_clients
  noise_multiplier = base_noise_multiplier
  noise_multiplier *= clients_per_round / base_clients_per_round
  rdp = tfp.compute_rdp(q, noise_multiplier=noise_multiplier, steps=rounds, orders=rdp_orders)
  eps, _, _ = tfp.get_privacy_spent(rdp_orders, rdp, target_delta=target_delta)
  return clients_per_round, eps, noise_multiplier

def find_needed_clients_per_round():
  hi = get_epsilon(base_clients_per_round)
  if hi[1] < target_eps:
    return hi

  # Grow interval exponentially until target_eps is exceeded.
  while True:
    lo = hi
    hi = get_epsilon(2 * lo[0])
    if hi[1] < target_eps:
      break

  # Binary search.
  while hi[0] - lo[0] > 1:
    mid = get_epsilon((lo[0] + hi[0]) // 2)
    if mid[1] > target_eps:
      lo = mid
    else:
      hi = mid

  return hi

clients_per_round, _, noise_multiplier = find_needed_clients_per_round()
print(f'To get ({target_eps}, {target_delta})-DP, use {clients_per_round} '
      f'clients with noise multiplier {noise_multiplier}.')


rounds = 100
noise_multiplier = 1.2
clients_per_round = 120
data_frame = pd.DataFrame()
data_frame = train(rounds, noise_multiplier, clients_per_round, data_frame)

make_plot(data_frame)
