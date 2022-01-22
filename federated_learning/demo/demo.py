import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

tf.compat.v1.enable_v2_behavior()

# TODO(b/148678573,b/148685415): must use the ReferenceExecutor because it
# supports unbounded references and tff.sequence_* intrinsics.
tff.framework.set_default_executor(tff.framework.ReferenceExecutor())

mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100


def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x':
                np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                         dtype=np.float32),
            'y':
                np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
    return output_sequence


federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]

federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]

BATCH_SPEC = collections.OrderedDict(
    x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
    y=tf.TensorSpec(shape=[None], dtype=tf.int32))
BATCH_TYPE = tff.to_type(BATCH_SPEC)

print(str(BATCH_TYPE))

MODEL_SPEC = collections.OrderedDict(
    weights=tf.TensorSpec(shape=[784, 10], dtype=tf.float32),
    bias=tf.TensorSpec(shape=[10], dtype=tf.float32))
MODEL_TYPE = tff.to_type(MODEL_SPEC)

print(MODEL_TYPE)

@tf.function
def forward_pass(model, batch):
    predicted_y = tf.nn.softmax(
        tf.matmul(batch['x'], model['weights']) + model['bias'])
    return -tf.reduce_mean(
        tf.reduce_sum(
            tf.one_hot(batch['y'], 10) * tf.math.log(predicted_y), axis=[1]))

@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    return forward_pass(model, batch)

initial_model = collections.OrderedDict(
    weights=np.zeros([784, 10], dtype=np.float32),
    bias=np.zeros([10], dtype=np.float32))

sample_batch = federated_train_data[5][-1]

print(batch_loss(initial_model, sample_batch))

@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    # Define a group of model variables and set them to `initial_model`. Must
    # be defined outside the @tf.function.
    model_vars = collections.OrderedDict([
      (name, tf.Variable(name=name, initial_value=value))
      for name, value in initial_model.items()
    ])
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    @tf.function
    def _train_on_batch(model_vars, batch):
        # Perform one step of gradient descent using loss from `batch_loss`.
        with tf.GradientTape() as tape:
            loss = forward_pass(model_vars, batch)
        grads = tape.gradient(loss, model_vars)
        optimizer.apply_gradients(
            zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars)))
        return model_vars

    return _train_on_batch(model_vars, batch)

print(str(batch_train.type_signature))

model = initial_model
losses = []
for _ in range(5):
    model = batch_train(model, sample_batch, 0.1)
    losses.append(batch_loss(model, sample_batch))

print("5 loops loss:", losses)

LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)

@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):

    # Mapping function to apply to each batch.
    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)

    return tff.sequence_reduce(all_batches, initial_model, batch_fn)

print(str(local_train.type_signature))

locally_trained_model = local_train(initial_model, 0.1, federated_train_data[5])

@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    # TODO(b/120157713): Replace with `tff.sequence_average()` once implemented.
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
            all_batches))

print(str(local_eval.type_signature))

print('initial_model loss [num 5] =', local_eval(initial_model, federated_train_data[5]))
print('locally_trained_model loss [num 5] =', local_eval(locally_trained_model, federated_train_data[5]))
print('initial_model loss [num 0] =', local_eval(initial_model, federated_train_data[0]))
print('locally_trained_model loss [num 0] =', local_eval(locally_trained_model, federated_train_data[0]))


SERVER_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.SERVER)
CLIENT_DATA_TYPE = tff.FederatedType(LOCAL_DATA_TYPE, tff.CLIENTS)

@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(
        tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))

print(str(federated_eval.type_signature))

print('initial_model loss =', federated_eval(initial_model,
                                             federated_train_data))
print('locally_trained_model loss =',
      federated_eval(locally_trained_model, federated_train_data))


SERVER_FLOAT_TYPE = tff.FederatedType(tf.float32, tff.SERVER)

@tff.federated_computation(SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE,
                           CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
    return tff.federated_mean(
        tff.federated_map(local_train, [
            tff.federated_broadcast(model),
            tff.federated_broadcast(learning_rate), data
        ]))


model = initial_model
learning_rate = 0.1
for round_num in range(5):
    # 每一轮，把大家的模型分别更新一下，取平均之后拿回来（做赋值替换）
    model = federated_train(model, learning_rate, federated_train_data)
    # 把学习率减小一点
    learning_rate = learning_rate * 0.9
    # 算个loss输出一下
    loss = federated_eval(model, federated_train_data)
    print('round {}, loss={}'.format(round_num, loss))
    # 下一轮S端的模型又发给各位clients去更新