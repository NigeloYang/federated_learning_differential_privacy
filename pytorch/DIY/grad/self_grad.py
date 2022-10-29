''' 手动更新模型梯度参数

'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# 随机生成数据,
x_data = torch.from_numpy(np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])).float()
y_data = torch.from_numpy(np.array([0, 1, 2, 0, 0, 2])).long()

# 定义一些基本的参数
num_features = 2
num_classes = 3
n_hidden_1 = 5
learning_rate = 0.01
epochs = 100

# 定义权重,这种直接创建的tensor 称为叶子节点，叶子节点对应的grad_fn是None
w1 = torch.randn(num_features, n_hidden_1, requires_grad=True)
b1 = torch.randn(n_hidden_1, requires_grad=True)
wout = torch.randn(n_hidden_1, num_classes, requires_grad=True)
bout = torch.randn(num_classes, requires_grad=True)

# 训练时没有 zero() 属性是因为 梯度里面此时还没有有数据，解决方案1 先运行一遍梯度计算
# z1 = torch.add(torch.matmul(x_data, w1), b1)
# zout = torch.add(torch.matmul(F.relu(z1), wout), bout)
#
# log_softmax = F.log_softmax(zout, dim=1)
# loss = F.nll_loss(log_softmax, y_data)
#
# w1.data -= learning_rate * w1.grad.data
# b1.data -= learning_rate * b1.grad.data
# wout.data -= learning_rate * wout.grad.data
# bout.data -= learning_rate * bout.grad.data

# 循环迭代
for epoch in range(epochs):
    # 神经网络层权重计算
    z1 = torch.add(torch.matmul(x_data, w1), b1)
    zout = torch.add(torch.matmul(F.relu(z1), wout), bout)

    log_softmax = F.log_softmax(zout, dim=1)
    loss = F.nll_loss(log_softmax, y_data)

    # 后向传递，权重更新
    loss.backward()
    with torch.no_grad():
        w1.data -= learning_rate * w1.grad.data
        b1.data -= learning_rate * b1.grad.data
        wout.data -= learning_rate * wout.grad.data
        bout.data -= learning_rate * bout.grad.data

    # w1.grad.data.zero()
    # b1.grad.data.zero()
    # wout.grad.data.zero()
    # bout.grad.data.zero()


    if epoch % 10 == 0:
        with torch.no_grad():
            z1 = torch.add(torch.matmul(x_data, w1), b1)
            zout = torch.add(torch.matmul(F.relu(z1), wout), bout)
            predicted = torch.argmax(zout, 1)
            train_acc = np.sum(predicted.numpy() == y_data.numpy()) / y_data.size()
            print('Epoch: %d, loss: %.4f, train_acc: %.3f' % (epoch + 1, loss.item(), train_acc))
    # 解决方案2 循环之后在清零，运行前把上面的清零方法注释掉
    # w1.grad.data.zero()
    # b1.grad.data.zero()
    # wout.grad.data.zero()
    # bout.grad.data.zero()
print("Finished")

# Result
print('Predicted :', predicted.numpy())
print('Truth :', y_data)
print('Accuracy : %.2f' %train_acc)