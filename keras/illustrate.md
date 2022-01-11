# keras 学习记录（tensorflow2.0）

## Keras简介

    Keras是基于TensorFlow和Theano（由加拿大蒙特利尔大学开发的机器学习框架）的深度学习库，是由纯python编写而成的高层神经网络API，也仅支持
    python开发。它是为了支持快速实践而对tensorflow或者Theano的再次封装，让我们可以不用关注过多的底层细节，能够把想法快速转换为结果。它也很
    灵活，且比较容易学。Keras默认的后端为tensorflow，如果想要使用theano可以自行更改。tensorflow和theano都可以使用GPU进行硬件加速，往往
    可以比CPU运算快很多倍。因此如果你的显卡支持cuda的话，建议尽可能利用cuda加速模型训练。（当机器上有可用的GPU时，代码会自动调用GPU 进行并行
    计算。）目前Keras已经被TensorFlow收录，添加到TensorFlow 中，成为其默认的框架，成为TensorFlow官方的高级API。

## Keras 安装

    一般在安装 tensorflow2.0 的时候会自动安装 keras 的适配版本
    安装方式：因为笔者使用的 anaconda 对 python 环境进行控制的。所以，先创建了一个 python == 3.8 环境(Windows 10)
    第一步：conda create -n your_env_name python=x.x 例如 conda create -n tf python=3.8
    第二步：激活环境 conda activate your_env_name  例如 conda activate tf
    第三步：进入环境以后输入：pip install tensorflow==2.x.0

## Keras 基础

### Keras Moddel

    在 Keras 中有两类主要的模型：Sequential 顺序模型 和 使用函数式 API 的 Model 类模型。
    这些模型有许多共同的方法和属性： 
        model.layers 是包含模型网络层的展平列表。
        model.inputs 是模型输入张量的列表。
        model.outputs 是模型输出张量的列表。
        model.summary() 打印出模型概述信息。 它是 utils.print_summary 的简捷调用。
        model.get_config() 返回包含模型配置信息的字典。
        model.get_weights() 返回模型中所有权重张量的列表，类型为 Numpy 数组。
        model.set_weights(weights) 从 Numpy 数组中为模型设置权重。列表中的数组必须与 get_weights() 返回的权重具有相同的尺寸。
        model.to_json() 以 JSON 字符串的形式返回模型的表示。请注意，该表示不包括权重，仅包含结构。你可以通过以下方式从 JSON 字符串
            重新实例化同一模型（使用重新初始化的权重）：
        model.to_yaml() 以 YAML 字符串的形式返回模型的表示。请注意，该表示不包括权重，只包含结构。
        model.save_weights(filepath) 将模型权重存储为 HDF5 文件。
        model.load_weights(filepath, by_name=False): 从 HDF5 文件（由 save_weights 创建）中加载权重。默认情况下，模型的结
            构应该是不变的。 如果想将权重载入不同的模型（部分层相同）， 设置 by_name=True 来载入那些名字相同的层的权重。

#### Keras Sequential 顺序模型
```python
# 在开始讲述之前我们先看一个完整的Keras Sequential 顺序模型
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 生成虚拟数据
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

# model 构建
model = Sequential()

# 添加神经网络层
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# model.complile() 函数用于模型编译
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# model.fit() 函数用于模型训练
model.fit(x_train, y_train,epochs=20,batch_size=128)

# model.evaluate() 函数用于模型评估
score = model.evaluate(x_test, y_test, batch_size=128)

```
    看完这个案例是不是觉得 Keras 很简单，当然这只是最简单的模型，接下来将会详细了解 Keras Sequential 顺序模型主要函数参数的讲解
##### model.complie() 函数
    compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, 
        weighted_metrics=None, target_tensors=None)
    参数：
        optimizer: 字符串（优化器名）或者优化器对象。详见 optimizers。
        loss: 字符串（目标函数名）或目标函数或 Loss 实例。详见 losses。 如果模型具有多个输出，则可以通过传递损失函数的字典或列表，在每
            个输出上使用不同的损失。模型将最小化的损失值将是所有单个损失的总和。
        metrics: 在训练和测试期间的模型评估标准。 通常你会使用 metrics = ['accuracy']。要为多输出模型的不同输出指定不同的评估标准， 
            还可以传递一个字典，如 metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}。 你也可以传递一个评
            估指标序列的序列 (len = len(outputs)) 例如 metrics=[['accuracy'], ['accuracy', 'mse']] 或 
            metrics=['accuracy', ['accuracy', 'mse']]。
        loss_weights: 指定标量系数（Python浮点数）的可选列表或字典，用于加权不同模型输出的损失贡献。 模型将要最小化的损失值将是所有单
            个损失的加权和，由 loss_weights 系数加权。 如果是列表，则期望与模型的输出具有 1:1 映射。 如果是字典，则期望将输出名称（字
            符串）映射到标量系数。
        sample_weight_mode: 如果你需要执行按时间步采样权重（2D 权重），请将其设置为 temporal。 默认为 None，为采样权重（1D）。如果
            模型有多个输出，则可以通过传递 mode 的字典或列表，以在每个输出上使用不同的 sample_weight_mode。
        weighted_metrics: 在训练和测试期间，由 sample_weight 或 class_weight 评估和加权的度量标准列表。
        target_tensors: 默认情况下，Keras 将为模型的目标创建一个占位符，在训练过程中将使用目标数据。 相反，如果你想使用自己的目标张量
            （反过来说，Keras 在训练期间不会载入这些目标张量的外部 Numpy 数据）， 您可以通过 target_tensors 参数指定它们。它应该是
            单个张量（对于单输出 Sequential 模型）。
        **kwargs: 当使用 Theano/CNTK 后端时，这些参数被传入 K.function。当使用 TensorFlow 后端时，
            这些参数被传递到 tf.Session.run。
    异常
        ValueError: 如果 optimizer, loss, metrics 或 sample_weight_mode 这些参数不合法。

#### 函数式 API 的 Model 类模型

    暂无

### Keras Network Structure

    暂无

### Keras Model Parameter

    暂无