import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 使用keras.utils.get_file函数获取远程数据集
origin = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
dataset_path = keras.utils.get_file("D:/WorkSpace/tensorflow-practice/tensorflow/dp_office_tutorial/data/auto-mpg.data", origin)
print('打印数据的存储路径', dataset_path)


# 设置数据集格式
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.head(5))
# 打印数据集中一些未知的值
print(dataset.isna().sum())
dataset = dataset.dropna()

# "Origin" 列实际上代表分类，而不仅仅是一个数字。所以把它转换为独热码 （one-hot）
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['EU'] = (origin == 2) * 1.0
dataset['JP'] = (origin == 3) * 1.0
# 查看更新后的数据结构
print(dataset.head(5))

# 划分数据集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 查看训练数据的划分
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# 生成标签
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# 数据标准化
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


# 绘制训练进度
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
  plt.ylim([0, 5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
  plt.ylim([0, 20])
  plt.legend()
  plt.show()
  

# 构建模型
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
  optimizer = keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse']
                )
  return model


model = build_model()

# 检查模型
print('打印模型描述')
model.summary()

# 测试数据
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print('打印测试的结果')
print(example_result)


# 训练模型
print('模型训练1')
history1 = model.fit(
  normed_train_data,
  train_labels,
  epochs=1000,
  validation_split=0.2,
  verbose=0,
  callbacks=[PrintDot()]
)

# 使用history 的信息查看训练进度
print('查看训练进度')
hist = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch
print(hist.tail())
plot_history(history1)

# 评估模型
print('评估模型')
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set loss Error: {:5.2f} MPG".format(loss))
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
print("Testing set Mean squared error: {:5.2f} MPG".format(mse))

# 开始预测
print('开始预测')
test_predictions = model.predict(normed_test_data).flatten()

print('绘制预测结果')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

# 训练模型
print('模型训练2')
model2 = build_model()

# patience 值用来检查改进 epochs 的数量
early_stop = keras.callbacks.EarlyStopping(
  monitor='val_loss',
  patience=10
)

history2 = model2.fit(
  normed_train_data,
  train_labels,
  epochs=1000,
  validation_split=0.2,
  verbose=0,
  callbacks=[early_stop, PrintDot()]
)

plot_history(history2)

# 评估模型
print('评估模型2')
loss, mae, mse = model2.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set loss Error: {:5.2f} MPG".format(loss))
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
print("Testing set Mean squared error: {:5.2f} MPG".format(mse))

# 开始预测
print('开始预测')
test_predictions = model2.predict(normed_test_data).flatten()

print('绘制预测结果')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

