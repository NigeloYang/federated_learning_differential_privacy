# 联合平均的最小独立实现
这旨在成为联合平均的灵活且最小化的实现，并且代码被设计为模块化和可重用的。这种联合平均算法的实现仅使用关键的 TFF 函数，不依赖于 tff.learning 中的高级特性。

## 介绍
此处提供了联合平均算法的最小实现，以及联合 EMNIST 实验示例。该实现演示了典型联邦学习模拟的三种主要逻辑类型。

- 通过从数据集中选择模拟客户端然后执行联邦学习算法来驱动模拟的外部 Python 脚本。


- 在单个位置（例如，在客户端或服务器上）运行的单个 TensorFlow 代码。代码片段通常是可以在 TFF 之外使用和测试的 tf.functions。


- 编排逻辑通过将本地计算包装为 tff.tf_computations 并在 tff.federated_computation 中使用关键 TFF 函数（如 tff.federated_broadcast 和 tff.federated_map）将它们绑定在一起。  

这个 EMNIST 示例可以很容易地适应实验性更改：

- 在驱动文件 emnist_fedavg_main 中，我们可以更改数据集、神经网络架构、server_optimizer 和 client_optimizer 以用于定制应用程序。请注意，我们需要一个模型包装器，并使用 TFF 构建一个迭代过程。我们在 simple_fedavg_tf 中为 keras 模型定义了一个独立的模型包装器，可以通过调用 tff.learning.from_keras_model 将其替换为 tff.learning.Model。请注意，tff.learning.Model 的内部 keras_model 可能无法直接访问以进行评估。我们还鼓励感兴趣的用户研究 metrics_manager 、 checkpoint_manager 和其他帮助函数来自定义 python 驱动程序文件。  


- 在 TF 函数文件 simple_fedavg_tf 中，我们可以更好地控制优化过程中执行的局部计算。在每一轮中，在服务器端，我们会在 server_update 函数中更新 ServerState；然后我们使用 build_server_broadcast_message 函数构建一个 BroadcastMessage 来准备从服务器到客户端的广播；在客户端，我们使用 client_update 函数执行本地更新并返回 ClientOutput 以发送回服务器。注意 emnist_fedavg_main 中定义的 server_optimizer 用于 server_update 函数；在 emnist_fedavg_main 中定义的 client_optimizer 在 client_update 中使用。这些函数用作整个 TFF 计算中的本地计算构建块，它处理服务器和客户端之间的广播和聚合。


- 在 TFF 文件 simple_fedavg_tff 中，我们可以控制编排策略。我们采用客户端更新的加权平均值来更新保持在服务器状态的模型。更多关于 TFF 函数 federated_broadcast、federated_map 和 federated_mean 的使用说明可以在教程中找到。
