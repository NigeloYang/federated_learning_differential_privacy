''' 手动更新模型梯度参数

'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 随机生成数据
x_data = torch.from_numpy(np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])).float()
y_data = torch.from_numpy(np.array([0, 1, 2, 0, 0, 2])).long()

# define parameter
num_features = 2
num_classes = 3
n_hidden_1 = 5
learning_rate = 0.01
global_epochs = 1000


class NetModel(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(NetModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        z1 = self.linear1(x)
        zout = self.linear2(z1)
        return zout


model = NetModel(num_features, n_hidden_1, num_classes)
print('model structure: ', model)

# define model relevance parameter
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# for epoch exe
for epoch in range(global_epochs):
    input = x_data
    label = y_data

    model.train()

    # forward backward optimizer
    output = model(input)
    loss = criterion(output, label)

    # print statistics
    if epoch % 100 == 0:
        model.eval()
        pred_output = model(input)
        predicted = torch.argmax(pred_output, dim=1)
        train_acc = np.sum(y_data.numpy() == predicted.numpy()) / y_data.size()
        print('%d,  loss: %.4f, train_acc: %.4f' % (epoch + 1, loss.item(), train_acc))

    # zero the parameter gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Finished Training')
