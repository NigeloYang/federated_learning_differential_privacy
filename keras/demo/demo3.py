import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载数据集
(train_data, train_label), (test_data, test_label) = keras.datasets.cifar10.load_data()

# 查看数据集的形状
print('数据集形状：{} 数据集标签：{}'.format(train_data.shape, train_label.shape))

# 定义数据字典
label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
              8: 'ship', 9: 'truck'}


# 定义一个函数展示数据集前几项内容
def plot_image(images, labels, prediction, num=10):
  fig = plt.gcf()
  fig.set_size_inches(12, 14)
  if num > 10: num = 10
  for i in range(0, num):
    ax = plt.subplot(2, 5, i + 1)
    ax.imshow(images[i], cmap='binary')
    title = str(i) + ' ' + label_dict[labels[i][0]]  # 显示数字对应的类别
    if len(prediction) > 0:
      title += '=>' + label_dict[prediction[i]]  # 显示数字对应的类别
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
  plt.show()


plot_image(train_data, train_label, [])


# 定义函数用于展示模型训练历史记录
def show_history(train_history, train, validation):
  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train History')
  plt.ylabel(train)
  plt.xlabel('Epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()


# 定义函数用于展示预测的准确率
def show_predicted(test, test_label, prediction, prediction_probability, i):
  print("test label: {}  predict label: {}".format(label_dict[test_label[i][0]], label_dict[prediction[i]]))
  plt.figure(figsize=(2, 2))
  plt.imshow(test[i])
  plt.show()
  for j in range(10):  # 输出10个类别概率
    print(label_dict[j] + ' probability: %1.9f' % (prediction_probability[i][j]))


# 数据集预处理
# 特征标准化 标签采用one-hot编码
train_data_norm = train_data.astype('float32') / 255.0
train_label_norm = keras.utils.to_categorical(train_label)
print('数据集：{} 数据集标签：{}'.format(train_data[[0], [0], [0]], train_label[:1]))
print('标准化之后的：数据集：{} 数据集标签：{}'.format(train_data_norm[[0], [0], [0]], train_label_norm[:1]))

val_num = int(train_label_norm.shape[0] * 0.2)
print('val_num: ', val_num)

val_data_norm = train_data_norm[0:val_num]
val_label_norm = train_label_norm[0:val_num]
train_data_norm = train_data_norm[val_num:]
train_label_norm = train_label_norm[val_num:]
test_data_norm = test_data.astype('float32') / 255.0
test_label_norm = keras.utils.to_categorical(test_label)

# 构建模型
model = keras.models.Sequential()

# 卷积层1，池化层1
model.add(
  keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', input_shape=(32, 32, 3),
                      activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


# 卷积层2，池化层2
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=1, padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=1, padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


# 卷积层3，池化层3
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=1, padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=1, padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 卷积层4，池化层4
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', strides=1, padding='same'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', strides=1, padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# 平坦层
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))

# 2层隐藏层
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.2))

# 输出层
model.add(keras.layers.Dense(10, activation='softmax'))

print(model.summary())

# 模型编译
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#
early_stopping = tf.keras.callbacks.EarlyStopping(
  monitor="val_loss",
  patience=3,
  restore_best_weights=True
)

# 训练模型
history = model.fit(
  train_data_norm, train_label_norm,
  validation_split=0.2, epochs=30, batch_size=128,
  validation_data=(val_data_norm, val_label_norm),
  validation_freq=1, verbose=1, shuffle=True,
  callbacks=[early_stopping]
)

# 绘制训练结果
show_history(history, 'acc', 'val_acc')
show_history(history, 'loss', 'val_loss')

# 模型评估
score = model.evaluate(test_data_norm, test_label_norm, verbose=0)
print('loss value: {} metrics values(acc): {}'.format(score[0], score[1]))

if score[1] > 0.9:
  model.save('model_weight/demo4_cifar.h5')

# 预测分类，预测分类的概率, model.predict_classes()在tensorflow 2.6 中已经删除
prediction = np.argmax(model.predict(test_data_norm), axis=1)
prediction_probably = model.predict(test_data_norm)
print('打印预测种类：{}  打印预测种类的概率：{}'.format(prediction[0], prediction_probably[0]))

# 打印预测的结果 只显示前十个
plot_image(test_data, test_label, prediction)

# 展示预测结果属于哪一个的概率
show_predicted(test_data, test_label, prediction, prediction_probably, 0)

# 显示混淆矩阵
# pandas.crosstab() 查看混淆矩阵,且输入必须是一维数组，所以传入的 prediction，label必须是一维的，使用 .reshape转为一维数组
print(pd.crosstab(test_label.reshape(-1), prediction, rownames=['label'], colnames=['prediction']))
