import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# 整数 <===> 单词
# 因为评论是以数字形式存储的，所以需要对评论进行转换：整数 <===> 单词

# 下载一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# 把(key, val) <===> (val, key) 进行一个转换
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
  return ' '.join([reverse_word_index.get(i, '?') for i in text])


print('编译输出train_data[0]评论：', decode_review(train_data[0]))
print('没有编译输出train_data[0]评论：', train_data[0])

# 文字不好做文本分类，以数字的形式进行文本分类，但是评论数量不一致，且不能构建统一的神经网络进行张量运算，这种转换可以通过以下两种方式来完成

# 1、将数组转换为表示单词出现与否的由 0 和 1 组成的向量，类似于 one-hot 编码。例如，序列[3, 5]将转换为一个 10,000 维的向量，该向量
# 除了索引为 3 和 5 的位置是 1 以外，其他都为 0。然后，将其作为网络的首层一个可以处理浮点型向量数据的稠密层。不过，这种方法需要大量
# 的内存，需要一个大小为 num_words * num_reviews 的矩阵。

# 2、通过填充数组来保证输入数据具有相同的长度，然后创建一个大小为 max_length * num_reviews 的整型张量。我们可以使用能够处理此形状数
# 据的嵌入层作为网络中的第一层。

# 在这里我们使用了pad_sequences 来使数据保持统一的数据维度

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post',
                                                       maxlen=256)

print('打印填充的数据长度train_data[0]: ', train_data[0])
print('打印填充的数据长度train_data[0]: ', train_data[0].shape)

# 创建一个验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 构建模型
# 神经网络由堆叠的层来构建，这需要从两个主要方面来进行体系结构决策：模型里有多少层？ 每个层里有多少隐层单元（hidden units）？

# 第一层，是嵌入（Embedding）层。该层采用整数编码的词汇表，并查找每个词索引的嵌入向量（embedding vector）。这些向量是通过模型训练学习
# 到的。向量向输出数组增加了一个维度。得到的维度为：(batch, sequence, embedding)。
# 第二层，GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入。
# 第三层，定长输出向量通过一个有 16 个隐层单元的全连接（Dense）层传输。
# 第四层，与单个输出结点密集连接。使用 Sigmoid 激活函数，其函数值为介于 0 与 1 之间的浮点数，表示概率或置信度。

# 输入形状是用于电影评论的词汇数目（10,000 词）
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

# 损失函数与优化器
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(partial_x_train, partial_y_train, epochs=100, batch_size=512, validation_data=(x_val, y_val),
                    verbose=1)

# 评估模型
results = model.evaluate(test_data, test_labels, verbose=2)
print('输出评估结果', results)

# 创建一个准确率（accuracy）和损失值（loss）随时间变化的图表
history_dict = history.history
print('打印模型在训练期间的参数值', history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()  # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
