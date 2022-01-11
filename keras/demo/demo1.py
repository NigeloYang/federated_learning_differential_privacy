import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 标准化处理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 选择一部分数据进行检测判断是否正确
plt.figure(figsize=(10, 10))
for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])
plt.show()

# 创建模型
# keras.layers.Flatten 把二维数据转换成为一维数据
# keras.layers.Dense： 第一个创建隐藏层处理特征数据，第二个输出层输出是个结果，选择最好的一个结果
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10)
])

# 模型进行编译
# 损失函数 loss：用于测量模型在训练期间的准确率。您会希望最小化此函数，以便将模型“引导”到正确的方向上。
# 优化器 optimizer: 决定模型如何根据其看到的数据和自身的损失函数进行更新。
# 指标 metrics: 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

# 训练模型
model.fit(train_images, train_labels, epochs=10)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 模型预测
# 模型具有线性输出，即 logits。您可以附加一个 softmax 层，将 logits 转换成更容易理解的概率。
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print("打印预测的标签的输出值: {0} 并选择最大的一个结果{1} 真实的结果: {2}".format(predictions[0], np.argmax(predictions[0]), test_labels[0]))
print('打印预测的标签的输出值: %s 并选择最大的一个结果: %s 真实的结果: %s' % (predictions[0], np.argmax(predictions[0]), test_labels[0]))


# 定义函数如何制成图表，查看模型对于全部 10 个类的预测
# 正确的预测标签为蓝色，错误的预测标签为红色。数字表示预测标签的百分比（总计为 100）。
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel(
    "{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]),
    color=color)


def plot_value_array(i, predictions_array, true_label, class_names):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10), class_names, rotation=45)
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# 验证预测结果
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 如何对单个图片进行预测，随机选取一个图像
img = test_images[1]
print(img.shape)

# tf.keras 模型经过了优化，可同时对一个批或一组样本进行预测。因此，即便只使用一个图像，也需要将其添加到列表中：
img = (np.expand_dims(img,0))
print(img.shape)

# 现在预测这个图像的正确标签：
predictions_single = probability_model.predict(img)
print(predictions_single)

plt.figure(figsize=(5, 5))
plot_value_array(1, predictions_single[0], test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

# keras.Model.predict 会返回一组列表，每个列表对应一批数据中的每个图像。在批次中获取对我们（唯一）图像的预测：
print(np.argmax(predictions_single[0]))