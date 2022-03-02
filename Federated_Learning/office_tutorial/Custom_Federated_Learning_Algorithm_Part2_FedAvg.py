import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 是否开启 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt

'''准备联合数据集'''
mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
print([(x.dtype, x.shape) for x in mnist_train])
print(mnist_train[0].shape)
print(mnist_train[1].shape)
# [(dtype('uint8'), (60000, 28, 28)), (dtype('uint8'), (60000,))]
# (60000, 28, 28)
# (60000,)


NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100


def get_data_for_digit(source, digit):
  output_sequence = []
  all_samples = [i for i, d in enumerate(source[1]) if d == digit]
  for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
    batch_samples = all_samples[i:i + BATCH_SIZE]
    output_sequence.append({
      'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples], dtype=np.float32),
      'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)
    })
  return output_sequence


federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]
federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]

# 查看数据标准化后的数据格式
print(federated_train_data[0][0]['x'].shape)
print(federated_train_data[0][0])

# 查看数据是否正确
plt.imshow(federated_train_data[5][-1]['x'][-1].reshape(28, 28), cmap='gray')
plt.grid(False)
plt.show()

'''定义与模型相关的内容'''
# 定义 输入类型 的 TFF 元组
BATCH_SPEC = collections.OrderedDict(
  x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
  y=tf.TensorSpec(shape=[None], dtype=tf.int32))
BATCH_TYPE = tff.to_type(BATCH_SPEC)
print('BATCH_TYPE: ', str(BATCH_TYPE))

# 定义 权重和偏差 的 TFF 元组
MODEL_SPEC = collections.OrderedDict(
  weights=tf.TensorSpec(shape=[784, 10], dtype=tf.float32),
  bias=tf.TensorSpec(shape=[10], dtype=tf.float32))
MODEL_TYPE = tff.to_type(MODEL_SPEC)
print('MODEL_TYPE: ', MODEL_TYPE)

# 建立初始模型
initial_model = collections.OrderedDict(
  weights=np.zeros([784, 10], dtype=np.float32),
  bias=np.zeros([10], dtype=np.float32)
)


# 因为 @tf.function 修饰的方法不能调用 @tff.tf_computation
# 所以，把 forward_pass 与 batch_loss 分开定义的，以便以后可以从另一个 tf.function 中调用它。
@tf.function
def forward_pass(model, batch):
  predicted_y = tf.nn.softmax(tf.matmul(batch['x'], model['weights']) + model['bias'])
  return -tf.reduce_mean(
    tf.reduce_sum(tf.one_hot(batch['y'], 10) * tf.math.log(predicted_y), axis=[1])
  )


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
  return forward_pass(model, batch)


print('查看batch_loss 类型', str(batch_loss.type_signature))

# 进行测试
sample_batch = federated_train_data[5][-1]
print('打印测试的 batch_loss: ', batch_loss(initial_model, sample_batch))

# 单个批次上的梯度下降
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
  # 定义一组模型变量并将它们设置为 initial_model， 必须在 @tf.function 之外定义
  model_vars = collections.OrderedDict([
    (name, tf.Variable(name=name, initial_value=value)) for name, value in initial_model.items()
  ])
  optimizer = tf.keras.optimizers.SGD(learning_rate)
  
  @tf.function
  def _train_on_batch(model_vars, batch):
    # 使用来自 `batch_loss` 的损失执行梯度下降的一步
    with tf.GradientTape() as tape:
      loss = forward_pass(model_vars, batch)
    grads = tape.gradient(loss, model_vars)
    optimizer.apply_gradients(zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars)))
    return model_vars
  
  return _train_on_batch(model_vars, batch)


print('batch_train.type_signature: ', str(batch_train.type_signature))

model = initial_model
losses = []
for _ in range(5):
  model = batch_train(model, sample_batch, 0.1)
  losses.append(batch_loss(model, sample_batch))
print('loss: ', losses)


'''本地数据序列上的梯度下降
因为 batch_train 可以正常工作，现在编写一个类似的训练函数 local_train，它会使用一个用户所有批次的整个序列，
而不仅仅是一个批次。现在，新的计算将需要使用 tff.SequenceType(BATCH_TYPE) 而不是 BATCH_TYPE
'''
LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)


@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):
  # 映射函数应用于每个批次
  @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
  def batch_fn(model, batch):
    return batch_train(model, batch, learning_rate)
  
  return tff.sequence_reduce(all_batches, initial_model, batch_fn)


print('local_train.type_signature： ', str(local_train.type_signature))

locally_trained_model = local_train(initial_model, 0.1, federated_train_data[5])


# 本地评估
@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
  return tff.sequence_sum(
    tff.sequence_map(
      tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE), all_batches)
  )


print('local_eval.type_signature: ', str(local_eval.type_signature))
print('initial_model loss =', local_eval(initial_model, federated_train_data[5]))
print('locally_trained_model loss =', local_eval(locally_trained_model, federated_train_data[5]))

print('initial_model loss =', local_eval(initial_model, federated_train_data[0]))
print('locally_trained_model loss =', local_eval(locally_trained_model, federated_train_data[0]))

'''联合的内容'''
# 联合评估
SERVER_MODEL_TYPE = tff.type_at_server(MODEL_TYPE)
CLIENT_DATA_TYPE = tff.type_at_clients(LOCAL_DATA_TYPE)


@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
  return tff.federated_mean(tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))


print('initial_model loss =', federated_eval(initial_model, federated_train_data))
print('locally_trained_model loss =', federated_eval(locally_trained_model, federated_train_data))

'''联合训练'''
SERVER_FLOAT_TYPE = tff.type_at_server(tf.float32)


@tff.federated_computation(SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
  return tff.federated_mean(
    tff.federated_map(local_train, [
      tff.federated_broadcast(model),
      tff.federated_broadcast(learning_rate),
      data
    ])
  )


model = initial_model
learning_rate = 0.1
for round_num in range(5):
  model = federated_train(model, learning_rate, federated_train_data)
  learning_rate = learning_rate * 0.9
  loss = federated_eval(model, federated_train_data)
  print('round {}, loss={}'.format(round_num, loss))

print('initial_model test loss =', federated_eval(initial_model, federated_test_data))
print('trained_model test loss =', federated_eval(model, federated_test_data))