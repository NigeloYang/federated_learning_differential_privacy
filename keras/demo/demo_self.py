import tensorflow as tf


# 实现自定义层
# __init__ , 您可以在其中进行所有与输入无关的初始化
# build, 知道输入张量的形状并可以进行其余的初始化
# call, 在哪里进行前向计算
# 实际上，你不必等到调用build()来创建网络结构，您也可以在__init()中创建它们。 但是，在build()中创建它们的优点是它可以根据图层
# 将要操作的输入的形状启用后期的网络构建。 另一方面，在__init__中创建变量意味着需要明确指定创建变量所需的形状。
class MyDense(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(MyDense, self).__init__()
    self.output_dim = output_dim
  
  # 定义权重，这个方法必须设 self.built = True，可以通过调用 super([Layer], self).build() 完成
  def build(self, input_shape):
    self.w = self.add_weight(shape=(int(input_shape[-1]), self.output_dim), initializer='random_normal', trainable=True)
    self.b = self.add_weight(shape=(self.output_dim,), initializer='random_normal', trainable=True)
    super(MyDense, self).build(input_shape)
  
  # 编写层的功能逻辑的地方。你只需要关注传入 call 的第一个参数：输入张量
  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b


layer = MyDense(10)
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)


print('-----------------------END--------------------------')
# 模型：组合层
class resnet(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(resnet, self).__init__(name='')
    filters1, filters2, filters3 = filters
    
    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()
    
    self.conv2b = tf.keras.layers.Conv2D(filters1, kernel_size)
    self.bn2b = tf.keras.layers.BatchNormalization()
    
    self.conv2c = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()
  
  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)
    
    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)
    
    x = self.conv2c(x)
    x = self.bn2c(x, training=training)
    
    x += input_tensor
    return tf.nn.relu(x)


block = resnet(1, [1, 2, 3])
_ = block(tf.zeros([1, 2, 3, 3]))
print(block.layers)
print(len(block.variables))
block.summary()
