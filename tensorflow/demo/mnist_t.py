import tensorflow as tf

# 定义权重
def weights(shape):
  return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1), name='w')


# 定义偏差张量
def bias(shape):
  return tf.Variable(tf.constant(0.1, shape=shape), name='b')


# 定义卷积层
def conv_2d(x, w):
  return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化层
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 建立输入层
input_data = tf.constant(shape=[None, 784])
x_image = tf.reshape(input_data, [-1, 28, 28, 1], name='x')

# 建立卷积层1
w1 = weights([5, 5, 1, 16])
b1 = bias([16])
conv1 = conv_2d(x_image, w1) + b1
conv1 = tf.nn.relu(conv1)

# 建立池化层1
pool1 = max_pool(conv1)

# 建立卷积层2

w2 = weights([5, 5, 16, 36])
b2 = bias([36])
conv2 = conv_2d(pool1, w2) + b2
conv2 = tf.nn.relu(conv2)

# 建立池化层2
pool2 = max_pool(conv2)

# 建立平坦层
flat = tf.reshape(pool2, [-1, 1764])

# 建立隐藏层
w3 = weights([1764, 256])
b3 = bias([256])
hidden = tf.nn.relu(tf.matmul(flat, w3) + b3)
hidden_dropout = tf.nn.dropout(hidden, keep_prop=0.8)

w4 = weights([256, 128])
b4 = bias([128])
hidden = tf.nn.relu(tf.matmul(flat, w3) + b3)
hidden_dropout = tf.nn.dropout(hidden, keep_prop=0.8)

# 建立输出层
w5 = weights([128, 10])
b5 = bias([10])
result = tf.nn.softmax(tf.matmul(hidden_dropout, w4) + b4)

# 定义训练方式
y_label = tf.placeholder('float', shape=[None, 10], name='y_label')
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=y_label))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

# 定义评估模型
correct = tf.equal(tf.argmax(result, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))