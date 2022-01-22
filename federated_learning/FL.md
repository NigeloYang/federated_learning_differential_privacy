# 联邦学习
## 如何安装
因为笔者使用的是 Windows 系统学习 Tensorflow Federated Learning 所以适配的时候会出现部分问题（Tensorflow-Federated  > v.0.17.0 以后）。
所以，在这里我们使用最适合 Windows 系统的安装包，在自己的 python 环境中安装如下：

1、Tensorflow:  pip install tensorflow==2.3.0  
2、Federated:  pip install tensorflow_federated==0.17.0

## 什么是联邦学习
可以参考文献深度理解联邦学习，为什么要用联邦学习

1、Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated Machine Learning: Concept and Applications. ACM Trans. Intell. Syst. Technol., 10(2), 12:11-12:19. https://doi.org/10.1145/3298981 

## 案例
可以参考 demo 文件夹，此文件是案例学习记录

## FL 相关知识
### 数据类型
Federated Core 提供了以下几种类型：
- 张量类型(tff.TensorType)。对象不仅限于在 TensorFlow 计算图中表示 TensorFlow 运算输出的 Python 的 tf.Tensor 实例，而是也可能包括可产生的数据单位，例如，作为分布聚合协议的输出。张量类型的紧凑表示法为 dtype 或 dtype[shape]。例如，int32 和 int32[10] 分别是整数和整数向量的类型。
- 序列类型 (tff.SequenceType)。这些是 TFF 中等效于 TensorFlow 中 tf.data.Dataset 的具体概念的抽象。用户可以按顺序使用序列的元素，并且可以包含复杂的类型。序列类型的紧凑表示法为 T*，其中 T 是元素的类型。例如，int32* 表示整数序列。
- 命名元组类型 (tff.StructType)。这些是 TFF 使用指定类型构造具有预定义数量元素的元组或字典式结构（无论命名与否）的方式。重要的一点是，TFF 的命名元组概念包含等效于 Python 参数元组的抽象，即元组的元素集合中有一部分（并非全部）是命名元素，还有一部分是位置元素。命名元组的紧凑表示法为 <n_1=T_1, ..., n_k=T_k>，其中 n_k 是可选元素名称，T_k 是元素类型。例如，<int32,int32> 是一对未命名整数的紧凑表示法，<X=float32,Y=float32> 是命名为 X 和 Y（可能代表平面上的一个点）的一对浮点数的紧凑表示法。元组可以嵌套，也可以与其他类型混用，例如，<X=float32,Y=float32>* 可能是一系列点的紧凑表示法。
- 函数类型 (tff.FunctionType)。TFF 是一个函数式编程框架，其中函数被视为这些函数的紧凑表示法为 (T -> U)，其中 T 为参数类型，U 为结果类型；或者，如果没有参数（虽然无参数函数是一个大部分情况下仅在 Python 级别存在的过时概念），则可以表示为 ( -> U)。例如，(int32* -> int32) 表示一种将整数序列缩减为单个整数值的函数类型。第一类值。函数最多有一个参数，并且只有一个结果。