'''
  待完善的文件
'''

import collections
import logging
import time
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# 测试有没有使用 GPU
gpu_devices = tf.config.list_physical_devices('GPU')
if not gpu_devices:
  raise ValueError('Cannot detect physical GPU device in TF')
tf.config.set_logical_device_configuration(
  gpu_devices[0],
  [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
   tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
print(tf.config.list_logical_devices())


# 确保 tff 框架可以使用
@tff.federated_computation
def hello_world():
  return 'Hello, TFF!'


print(hello_world())

# 加载数据集
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=True)

# 我们按照simple_fedavg示例定义一个预处理 EMNIST 示例的函数。请注意，该参数控制联合学习中客户端上的本地纪元数。client_epochs_per_round
def preprocess_emnist_dataset(client_epochs_per_round, batch_size, test_batch_size):
  def element_fn(element):
    return collections.OrderedDict(
      x=tf.expand_dims(element['pixels'], -1), y=element['label'])
  
  def preprocess_train_dataset(dataset):
    # Use buffer_size same as the maximum client dataset size,
    # 418 for Federated EMNIST
    return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
      count=client_epochs_per_round).batch(batch_size, drop_remainder=False)
  
  def preprocess_test_dataset(dataset):
    return dataset.map(element_fn).batch(test_batch_size, drop_remainder=False)
  
  train_set = emnist_train.preprocess(preprocess_train_dataset)
  test_set = preprocess_test_dataset(
    emnist_test.create_tf_dataset_from_all_clients())
  return train_set, test_set


# 我们按照simple_fedavg示例定义一个预处理 EMNIST 示例的函数。请注意，该参数控制联合学习中客户端上的本地纪元数。client_epochs_per_round
def _conv_3x3(input_tensor, filters, strides):
  """2D Convolutional layer with kernel size 3x3."""
  
  x = tf.keras.layers.Conv2D(
    filters=filters,
    strides=strides,
    kernel_size=3,
    padding='same',
    kernel_initializer='he_normal',
    use_bias=False,
  )(input_tensor)
  return x


def _basic_block(input_tensor, filters, strides):
  """A block of two 3x3 conv layers."""
  
  x = input_tensor
  x = _conv_3x3(x, filters, strides)
  x = tf.keras.layers.Activation('relu')(x)
  
  x = _conv_3x3(x, filters, 1)
  x = tf.keras.layers.Activation('relu')(x)
  return x


def _vgg_block(input_tensor, size, filters, strides):
  """A stack of basic blocks."""
  x = _basic_block(input_tensor, filters, strides=strides)
  for _ in range(size - 1):
    x = _basic_block(x, filters, strides=1)
  return x


def create_cnn(num_blocks, conv_width_multiplier=1, num_classes=10):
  """Create a VGG-like CNN model.

  The CNN has (6*num_blocks + 2) layers.
  """
  input_shape = (28, 28, 1)  # channels_last
  img_input = tf.keras.layers.Input(shape=input_shape)
  x = img_input
  x = tf.image.per_image_standardization(x)
  
  x = _conv_3x3(x, 16 * conv_width_multiplier, 1)
  x = _vgg_block(x, size=num_blocks, filters=16 * conv_width_multiplier, strides=1)
  x = _vgg_block(x, size=num_blocks, filters=32 * conv_width_multiplier, strides=2)
  x = _vgg_block(x, size=num_blocks, filters=64 * conv_width_multiplier, strides=2)
  
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(num_classes)(x)
  
  model = tf.keras.models.Model(
    img_input,
    x,
    name='cnn-{}-{}'.format(6 * num_blocks + 2, conv_width_multiplier))
  return model


def keras_evaluate(model, test_data, metric):
  metric.reset_states()
  for batch in test_data:
    preds = model(batch['x'], training=False)
    metric.update_state(y_true=batch['y'], y_pred=preds)
  return metric.result()


def run_federated_training(client_epochs_per_round,
                           train_batch_size,
                           test_batch_size,
                           cnn_num_blocks,
                           conv_width_multiplier,
                           server_learning_rate,
                           client_learning_rate,
                           total_rounds,
                           clients_per_round,
                           rounds_per_eval,
                           logdir='logdir'):
  train_data, test_data = preprocess_emnist_dataset(client_epochs_per_round, train_batch_size, test_batch_size)
  data_spec = test_data.element_spec
  
  def _model_fn():
    keras_model = create_cnn(cnn_num_blocks, conv_width_multiplier)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return tff.learning.from_keras_model(
      keras_model, input_spec=data_spec, loss=loss)
  
  def _server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=server_learning_rate)
  
  def _client_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=client_learning_rate)
  
  iterative_process = tff.learning.build_federated_averaging_process(
    model_fn=_model_fn,
    server_optimizer_fn=_server_optimizer_fn,
    client_optimizer_fn=_client_optimizer_fn,
    use_experimental_simulation_loop=True)
  
  metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  eval_model = create_cnn(cnn_num_blocks, conv_width_multiplier)
  logging.info(eval_model.summary())
  
  server_state = iterative_process.initialize()
  start_time = time.time()
  for round_num in range(total_rounds):
    sampled_clients = np.random.choice(
      train_data.client_ids,
      size=clients_per_round,
      replace=False)
    sampled_train_data = [
      train_data.create_tf_dataset_for_client(client)
      for client in sampled_clients
    ]
    if round_num == total_rounds - 1:
      with tf.profiler.experimental.Profile(logdir):
        server_state, train_metrics = iterative_process.next(
          server_state, sampled_train_data)
    else:
      server_state, train_metrics = iterative_process.next(
        server_state, sampled_train_data)
    print(f'Round {round_num} training loss: {train_metrics["train"]["loss"]}, '
          f'time: {(time.time() - start_time) / (round_num + 1.)} secs')
    if round_num % rounds_per_eval == 0 or round_num == total_rounds - 1:
      server_state.model.assign_weights_to(eval_model)
      accuracy = keras_evaluate(eval_model, test_data, metric)
      print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')


run_federated_training(
  client_epochs_per_round=1,
  train_batch_size=16,
  test_batch_size=128,
  cnn_num_blocks=2,
  conv_width_multiplier=4,
  server_learning_rate=1.0,
  client_learning_rate=0.01,
  total_rounds=10,
  clients_per_round=16,
  rounds_per_eval=2,
)

# 比较 cpu 和 gpu
cpu_device = tf.config.list_logical_devices('CPU')[0]
tff.backends.native.set_local_python_execution_context(server_tf_device=cpu_device, client_tf_devices=[cpu_device])

run_federated_training(
  client_epochs_per_round=1,
  train_batch_size=16,
  test_batch_size=128,
  cnn_num_blocks=2,
  conv_width_multiplier=4,
  server_learning_rate=1.0,
  client_learning_rate=0.01,
  total_rounds=10,
  clients_per_round=16,
  rounds_per_eval=2,
)
