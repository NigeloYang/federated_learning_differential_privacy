import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
(train, train_label), (test, test_label) = tf.keras.datasets.mnist.load_data()
print('训练集数据形状：{}  测试数据集形状：{}'.format(train.shape, test.shape))


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


train, train_label, val, val_label = split_data(train, train_label, 0.1)
print('训练集数据形状：{}  验证集的形状：{}  测试数据集形状：{}'.format(train.shape, val.shape, test.shape))


# 数据预处理
def preprocess(data, label):
  # data = data.reshape(data.shape[0], data[0].shape[0] * data[0].shape[1]).astype('float32') / 255
  data = tf.expand_dims(data, -1)
  data = data / 255
  label = tf.one_hot(label, depth=10)
  return data, label


train, train_label = preprocess(train, train_label)
val, val_label = preprocess(val, val_label)
test, test_label = preprocess(test, test_label)
print('train: {}  train_label: {}'.format(train[0], train_label[0]))
print('train shape: {}  train_label shape: {}'.format(train.shape, train_label.shape))

train_ds = tf.data.Dataset.from_tensor_slices((train, train_label)).shuffle(1000).batch(128)
val_ds = tf.data.Dataset.from_tensor_slices((val, val_label)).shuffle(1000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((test, test_label)).shuffle(1000).batch(128)

input_data = tf.keras.Input([28, 28, 1])
conv1 = tf.keras.layers.BatchNormalization()(input_data)
conv2 = tf.keras.layers.Conv2D(32, 3, padding='SAME', activation=tf.nn.relu)(conv1)
conv3 = tf.keras.layers.Conv2D(64, 3, padding='SAME', activation=tf.nn.relu)(conv2)
maxpool = tf.keras.layers.MaxPool2D(strides=[1, 1])(conv3)
conv4 = tf.keras.layers.Conv2D(128, 3, padding='SAME', activation=tf.nn.relu)(maxpool)
flat1 = tf.keras.layers.Flatten()(conv4)
dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat1)
dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(dense1)
logits = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dense2)

model = tf.keras.Model(inputs=input_data, outputs=logits)
print(model.summary())

model.compile(optimizer=tf.optimizers.Adam(1e-3),
              loss=tf.losses.categorical_crossentropy,
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
  monitor="val_loss",
  patience=4,
  restore_best_weights=True
)

history = model.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=[early_stopping])
print(history.history())
# plt = plt.subplot(1, 2, 1)
# plt.plot(history.history[loss])
# plt.plot(history.history[validation])
# plt.title('Train History')
# plt.ylabel(train)
# plt.xlabel('Epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

score = model.evaluate(test_ds)
print('last score:', score)

prediction = model.predict(test_ds)
print('打印预测种类：{}'.format(np.argmax(prediction[0])))
