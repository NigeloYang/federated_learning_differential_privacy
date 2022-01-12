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
        **kwargs: 当使用 Theano/CNTK 后端时，这些参数被传入 K.function。
                  当使用 TensorFlow 后端时，这些参数被传递到 tf.Session.run。
    异常
        ValueError: 如果 optimizer, loss, metrics 或 sample_weight_mode 这些参数不合法。
##### model.fit() 函数
    fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, 
        validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, 
        steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, 
        use_multiprocessing=False)
    
    参数
        x: 输入数据。可以是：
            一个 Numpy 数组（或类数组），或者数组的序列（如果模型有多个输入）。
            一个将名称匹配到对应数组/张量的字典，如果模型具有命名输入。
            一个返回 (inputs, targets) 或 (inputs, targets, sample weights) 的生成器或 keras.utils.Sequence。
            None（默认），如果从本地框架张量馈送（例如 TensorFlow 数据张量）。
        y: 目标数据。它可以是:
            Numpy 数组（序列）、 本地框架张量（序列）、Numpy数组序列（如果模型有多个输出） 
            None（默认）如果从本地框架张量馈送（例如 TensorFlow 数据张量）。 
            如果模型输出层已命名，你也可以传递一个名称匹配Numpy 数组的字典。 
            如果 x 是一个生成器，或 keras.utils.Sequence 实例，则不应该 指定 y（因为目标可以从 x 获得）。
        batch_size: 整数或 None。每次梯度更新的样本数。如果未指定，默认为 32。 
            如果你的数据是符号张量、生成器或 Sequence 实例形式，不要指定 batch_size， 因为它们会生成批次。
        epochs: 整数。训练模型迭代轮次。一个轮次是在整个 x 或 y 上的一轮迭代。 
            请注意，与 initial_epoch 一起，epochs 被理解为「最终轮次」。 
            模型并不是训练了 epochs 轮，而是到第 epochs 轮停止训练。
        verbose: 整数，0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
        callbacks: 一系列的 keras.callbacks.Callback 实例。一系列可以在训练和验证（如果有）时使用的回调函数。 详见 callbacks。
        validation_split: 0 和 1 之间的浮点数。用作验证集的训练数据的比例。 模型将分出一部分不会被训练的验证数据，
            并将在每一轮结束时评估这些验证数据的误差和任何其他模型指标。 验证数据是混洗之前 x 和y 数据的最后一部分样本中。 
            这个参数在 x 是生成器或 Sequence 实例时不支持。
        validation_data: 用于在每个轮次结束后评估损失和任意指标的数据。 模型不会在这个数据上训练。
            validation_data 会覆盖 validation_split。 validation_data 可以是：
                元组 (x_val, y_val) 或 Numpy 数组或张量
                元组 (x_val, y_val, val_sample_weights) 或 Numpy 数组。
                数据集或数据集迭代器。
            对于前两种情况，必须提供 batch_size。 对于最后一种情况，必须提供 validation_steps。
        shuffle: 布尔值（是否在每轮迭代之前混洗数据）或者字符串 (batch)。 batch 是处理 HDF5 数据限制的特殊选项，
            它对一个 batch 内部的数据进行混洗。 当 steps_per_epoch 非 None 时，这个参数无效。
        class_weight: 可选的字典，用来映射类索引（整数）到权重（浮点）值，用于加权损失函数（仅在训练期间）。 
            这可能有助于告诉模型来自代表性不足的类的样本。
        sample_weight: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。 
            你可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1:1 映射）， 
            或者在时序数据的情况下，可以传递尺寸为 (samples, sequence_length) 的 2D 数组，以对每个样本的每个时间步施加不同的权重。
            在这种情况下，你应该确保在 compile() 中指定 sample_weight_mode="temporal"。 
            这个参数在 x 是生成器或 Sequence 实例时不支持，应该提供 sample_weights 作为 x 的第 3 元素。
        initial_epoch: 整数。开始训练的轮次（有助于恢复之前的训练）。
        steps_per_epoch: 整数或 None。 在声明一个轮次完成并开始下一个轮次之前的总步数（样品批次）。 
            使用 TensorFlow 数据张量等输入张量进行训练时，默认值 None 等于数据集中样本的数量除以 batch 的大小，如果无法确定，则为 1。
        validation_steps: 只有在提供了 validation_data 并且是一个生成器时才有用。 表示在每个轮次结束时执行验证时，
            在停止之前要执行的步骤总数（样本批次）。
        validation_freq: 只有在提供了验证数据时才有用。整数或列表/元组/集合。 
            如果是整数，指定在新的验证执行之前要执行多少次训练，例如，validation_freq=2 在每 2 轮训练后执行验证。 
            如果是列表、元组或集合，指定执行验证的轮次，例如，validation_freq=[1, 2, 10] 表示在第 1、2、10 轮训练后执行验证。
        max_queue_size: 整数。仅用于生成器或 keras.utils.Sequence 输入。 生成器队列的最大尺寸。
            若未指定，max_queue_size 将默认为 10。
        workers: 整数。仅用于生成器或 keras.utils.Sequence 输入。 当使用基于进程的多线程时的最大进程数。
            若未指定，workers 将默认为 1。若为 0，将在主线程执行生成器。
        use_multiprocessing: 布尔值。仅用于生成器或 keras.utils.Sequence 输入。 如果是 True，使用基于进程的多线程。
            若未指定，use_multiprocessing 将默认为 False。 注意由于这个实现依赖于 multiprocessing，
            你不应该像生成器传递不可选的参数，因为它们不能轻松地传递给子进程。
        **kwargs: 用于向后兼容。

    返回
        一个 History 对象。其 History.history 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）。

    异常 
        RuntimeError: 如果模型从未编译。
        ValueError: 在提供的输入数据与模型期望的不匹配的情况下。

##### 一个案例
```python
# 完整的Keras Sequential 顺序模型
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

#### 函数式 API 的 Model 类模型

    暂无

### Keras Network Structure

    暂无

### Keras Model Parameter

    暂无