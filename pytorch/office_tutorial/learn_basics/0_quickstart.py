'''快速上手一个pytorch模型步骤
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 从 DataSets 下载数据集
training_data = datasets.FashionMNIST(
    root="../data/", train=True, download=True, transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="../data/", train=False, download=True, transform=ToTensor(),
)

# 定义一个数据加载器加载数据的大小
batch_size = 64

# 创建数据加载器
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W] {}: ".format(X.shape))
    print("Shape of y.shape: {} ---- y.dtype: {}: ".format(y.shape, y.dtype))
    break

# 确定使用的训练设备：GPU or CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# 定义一个模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)
print('before update model parameter:')

# 模型参数获取
# model.named_parameters 可以显示模型每一层的名称和对应的参数
print(f'\n ------------model.named_parameters()--------------')
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

print(f'\n ------------model.parameters()-------------')
for param in model.parameters():
    print(f"Size: {param.size()} | Values : {param[:2]} \n")

print(f'\n ------------model.state_dict()--------------')
for item in model.state_dict():
    print(f'name: {item} -- params size: {model.state_dict()[item].size()} \n')
    # print(f'name: {item}')

# 定义一个优化器和损失函数.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# 定义训练过程
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print(size)
    model.train()
    acc = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} --- Acc: [{int(acc):>5d}/{size:5d}] --- [{current:>5d}/{size:>5d}]")


# test
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 3
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

print('after update model parameter:')
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# 保存模型数据
torch.save(model.state_dict(), '../model/mnist.pth')
print('saving pytorch model state to mnist.pth')

# loading models
model = NeuralNetwork()
torch.load('../model/mnist.pth')
print('finishing loading model')

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

