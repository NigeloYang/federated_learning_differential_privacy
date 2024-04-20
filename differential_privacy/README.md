# TensorFlow Privacy
## 介绍
Tensorflow Privacy (TF Privacy) 是 Google 研究团队开发的一个开源库。该库包含一些常用 TensorFlow 优化器的实现，
可用于通过 DP 来训练机器学习模型。该库的目标是让使用标准 TensorFlow API 的机器学习从业者只需更改几行代码即可训练能够
保护隐私的模型。

差分隐私优化器可与使用 Optimizer 类的高阶 API（特别是 Keras）结合使用。此外，还可以找到一些 Keras 模型的差分隐私实现。

## 文件说明
- [db_basic.md 差分隐私基础知识](dp_basic.md)
- customization 记录差分隐私基础实现
- office_tutorial 记录差分隐私官方案例
- paper_model 文献中的模型
- data 用于存放数据集