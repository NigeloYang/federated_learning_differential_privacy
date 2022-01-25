# TensorFlow Privacy
## 介绍
Tensorflow Privacy (TF Privacy) 是 Google 研究团队开发的一个开源库。该库包含一些常用 TensorFlow 优化器的实现，
可用于通过 DP 来训练机器学习模型。该库的目标是让使用标准 TensorFlow API 的机器学习从业者只需更改几行代码即可训练能够
保护隐私的模型。

差分隐私优化器可与使用 Optimizer 类的高阶 API（特别是 Keras）结合使用。此外，还可以找到一些 Keras 模型的差分隐私实现。

## 文件说明
- [db_basic.md 记录差分隐私基本原理](dp_basic.md)
- demo 记录案例
1. [demo1 链接攻击](./demo/demo1.py)
2. [demo2 k-匿名](./demo/demo2.py)
3. [demo3 Laplace 机制](./demo/demo3.py)
4. [demo4 DP 属性组合](./demo/demo4.py)