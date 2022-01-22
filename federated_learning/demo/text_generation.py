import os

'''
os.environ[" xxxxxx "]='x'
TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 是否可以使用 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import collections
import functools
import time
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)

# 查看 TFF 是否可以使用
print(tff.federated_computation(lambda: 'Hello, TFF!')())

# 此案例中，我们将使用 TFF 提供的联合版本数据，通过联合学习针对莎士比亚作品微调此模型
# 生成词汇表
vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&am*.26:\neiquyAEIMQUY]!%)-159\r')

# 建立词汇表的 map 并将之转换为 数组
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


# 加载预训练模型并生成一些文本
def load_model(batch_size):
  urls = {
    1: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch1.kerasmodel',
    8: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch8.kerasmodel'}
  assert batch_size in urls, 'batch_size must be in ' + str(urls.keys())
  url = urls[batch_size]
  local_file = tf.keras.utils.get_file(os.path.basename(url), origin=url)
  return tf.keras.models.load_model(local_file, compile=False)


def generate_text(model, start_string):
  # From https://tensorflow.google.cn/tutorials/sequences/text_generation
  num_generate = 200
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = 1.0
  
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(idx2char[predicted_id])
  
  return (start_string + ''.join(text_generated))


# 文本生成需要一个 batch_size=1 模型
keras_model_batch1 = load_model(batch_size=1)
print(generate_text(keras_model_batch1, 'What of TensorFlow Federated, you ask? '))

# 加载并预处理联合莎士比亚数据
'''
shakespeare.load_data() 提供的数据集由一系列字符串构成，一个字符串代表莎士比亚戏剧中特定角色的一句台词。
客户端键由 戏剧名和参演角色名 构成，例如 即对应于 Othello（奥赛罗）角色在戏剧 Much Ado About Nothing（《无事生非》）中的台词。
'''
train_data, test_data = tff.simulation.datasets.shakespeare.load_data()

# 创建李尔王的数据集
raw_example_dataset = train_data.create_tf_dataset_for_client('THE_TRAGEDY_OF_KING_LEAR_KING')

# 为了允许未来的扩展，每个条目 x 都是一个 OrderedDict，带有一个包含文本的键“snippets”。
for x in raw_example_dataset.take(2):
  print(x['snippets'])

# 设定预处理参数
SEQ_LENGTH = 100
BATCH_SIZE = 8
BUFFER_SIZE = 100

# 使用上面加载的词汇构造一个查找表以将字符串字符映射到索引：
table = tf.lookup.StaticHashTable(
  tf.lookup.KeyValueTensorInitializer(
    keys=vocab,
    values=tf.constant(list(range(len(vocab))), dtype=tf.int64)
  ),
  default_value=0
)


def to_ids(x):
  s = tf.reshape(x['snippets'], shape=[1])
  chars = tf.strings.bytes_split(s).values
  ids = table.lookup(chars)
  return ids


def split_input_target(chunk):
  input_text = tf.map_fn(lambda x: x[:-1], chunk)
  target_text = tf.map_fn(lambda x: x[1:], chunk)
  return (input_text, target_text)


def preprocess(dataset):
  return (dataset.map(to_ids).unbatch().batch(SEQ_LENGTH + 1, drop_remainder=True)
          .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).map(split_input_target)
          )


example_dataset = preprocess(raw_example_dataset)
print(example_dataset.element_spec)


# 编译模型并基于预处理的数据进行测试
# 自定义一个评估标准
class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
  def __init__(self, name='accuracy', dtype=tf.float32):
    super().__init__(name, dtype=dtype)
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
    return super().update_state(y_true, y_pred, sample_weight)


# 案例其余部分的训练和评估批量大小。
BATCH_SIZE = 8
keras_model = load_model(batch_size=BATCH_SIZE)
keras_model.compile(
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[FlattenedCategoricalAccuracy()]
)

# 确认莎士比亚的损失远低于随机数据
loss, accuracy = keras_model.evaluate(example_dataset.take(5), verbose=0)
print('Evaluating on an example Shakespeare character: {a:3f}'.format(a=accuracy))

# 作为健全性检查，我们可以构建一些完全随机的数据，我们期望准确度基本上是随机的：
random_guessed_accuracy = 1.0 / len(vocab)
print('Expected accuracy for random guessing: {a:.3f}'.format(a=random_guessed_accuracy))
random_indexes = np.random.randint(low=0, high=len(vocab), size=1 * BATCH_SIZE * (SEQ_LENGTH + 1))
data = collections.OrderedDict(
  snippets=tf.constant(''.join(np.array(vocab)[random_indexes]), shape=[1, 1])
)
random_dataset = preprocess(tf.data.Dataset.from_tensor_slices(data))
loss, accuracy = keras_model.evaluate(random_dataset, steps=10, verbose=0)
print('Evaluating on completely random data: {a:.3f}'.format(a=accuracy))


# 通过联合学习微调模型
# 在 `create_tff_model()` 中克隆 keras_model，TFF 将调用它以在将序列化的图中生成模型的新副本。
# 注意：我们想要构造我们需要的所有必要的对象_inside_这个方法。
def create_tff_model():
  # TFF 使用“input_spec”，因此它知道您的模型期望的类型和形状。
  input_spec = example_dataset.element_spec
  keras_model_clone = tf.keras.models.clone_model(keras_model)
  return tff.learning.from_keras_model(
    keras_model_clone,
    input_spec=input_spec,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[FlattenedCategoricalAccuracy()]
  )


# 此命令构建所有 TensorFlow 图并对其进行序列化：
fed_avg = tff.learning.build_federated_averaging_process(
  model_fn=create_tff_model,
  client_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=0.5)
)

state = fed_avg.initialize()
state, metrics = fed_avg.next(state, [example_dataset.take(5)])
print('loss={:.3f}, accuracy={:.3f}'.format(loss, accuracy))


def data(client, source=train_data):
  return preprocess(source.create_tf_dataset_for_client(client)).take(5)


clients = ['ALL_S_WELL_THAT_ENDS_WELL_CELIA', 'MUCH_ADO_ABOUT_NOTHING_OTHELLO', ]
train_datasets = [data(client) for client in clients]
test_dataset = tf.data.Dataset.from_tensor_slices([data(client, test_data) for client in clients]).flat_map(lambda x: x)

NUM_ROUNDS = 5
state = fed_avg.initialize()
state = tff.learning.state_with_new_model_weights(
  state,
  trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
  non_trainable_weights=[v.numpy() for v in keras_model.non_trainable_weights]
)


def keras_evaluate(state, round_num):
  # 获取我们的全局模型权重并将它们推回到 Keras 模型中，以使用其标准的 .evaluate() 方法。
  keras_model = load_model(batch_size=BATCH_SIZE)
  keras_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[FlattenedCategoricalAccuracy()]
  )
  tff.learning.assign_weights_to_keras_model(keras_model, state.model)
  loss, accuracy = keras_model.evaluate(example_dataset, steps=2, verbose=0)
  print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))


for round_num in range(NUM_ROUNDS):
  print('Round {r}'.format(r=round_num))
  keras_evaluate(state, round_num)
  state, metrics = fed_avg.next(state, train_datasets)
  print('\tTrain: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))

keras_evaluate(state, NUM_ROUNDS + 1)

keras_model_batch1.set_weights([v.numpy() for v in keras_model.weights])
print(generate_text(keras_model_batch1, 'What of TensorFlow Federated, you ask? '))
