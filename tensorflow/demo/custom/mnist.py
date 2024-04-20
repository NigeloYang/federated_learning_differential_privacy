import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

# 载入数据
(train_data, train_label), (test_data, test_label) = keras.datasets.mnist.load_data()
print('train_data.shape {}  test_data.shape {}'.format(train_data.shape, test_data.shape))

# 格式化
train_data = train_data.reshape(60000, 784).astype('float32') / 255
test_data = test_data.reshape(10000, 784).astype('float32') / 255
print('Format train_data.shape {}  test_data.shape {}'.format(train_data.shape, test_data.shape))


# 划分数据集
def split_data(data, label, ratio=0.2):
  # 把数据的索引乱序
  shuffle_indexes = np.random.permutation(len(data))
  # 按比例分割
  size = int(ratio * len(data))
  # 测试集的索引
  val_indexes = shuffle_indexes[:size]
  # 训练集的索引
  train_indexes = shuffle_indexes[size:]
  val = data[val_indexes]
  val_label = label[val_indexes]
  train = data[train_indexes]
  train_label = label[train_indexes]
  return train, train_label, val, val_label


train_data, train_label, val_data, val_label = split_data(train_data, train_label, 0.1)
print('训练集数据形状：{}  验证集的形状：{}  测试数据集形状：{}'.format(train_data.shape, val_data.shape, test_data.shape))


# 创建模型
def build_model():
  inputs = keras.Input(shape=(784,))
  dense1 = layers.Dense(256, activation='relu')(inputs)
  dense2 = layers.Dense(128, activation='relu')(dense1)
  dense3 = layers.Dense(64, activation='relu')(dense2)
  outputs = layers.Dense(10, activation='softmax')(dense3)
  model = keras.Model(inputs, outputs)
  return model


# 正常编译
model = build_model()
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# 训练模型
history = model.fit(train_data, train_label, batch_size=128, epochs=3, validation_data=(val_data, val_label))
print('history:')
print(history.history)

result = model.evaluate(test_data, test_label, batch_size=128)
print('evaluate:')
print(result)

pred = model.predict(test_data[:2])
print('predict:')
print(np.argmax(pred, 1))
