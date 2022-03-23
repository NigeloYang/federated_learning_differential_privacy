'''A typical training procedure for a neural network is as follows:

Define the neural network that has some learnable parameters (or weights)
Iterate over a dataset of inputs
Process input through the network
Compute the loss (how far is the output from being correct)
Propagate gradients back into the network’s parameters
Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient
'''
import torch
import torch.nn as nn
import torch.nn.functional as f


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = myModel()
print('model construct \n', model)
params = list(model.parameters())
print('model params len \n', len(params))
# print('model params \n', params)
# print('model params shape\n', params.size())
print('model params conv1.shape \n', params[0].size())
print('model params conv1.weight \n', params[0])

input = torch.randn(1, 1, 32, 32)
out = model(input)
print(out)
model.zero_grad()
out.backward(torch.randn(1, 10))

output = model(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
