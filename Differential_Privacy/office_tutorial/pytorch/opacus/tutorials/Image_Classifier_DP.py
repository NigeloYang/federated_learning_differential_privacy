'''In this tutorial we will learn to do the following:

Learn about privacy specific hyper-parameters related to DP-SGD
Learn about ModelInspector, incompatible layers, and use model rewriting utility.
Train a differentially private ResNet18 for image classification.
'''

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision import models

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from tqdm.notebook import tqdm
import numpy as np
import warnings

warnings.simplefilter("ignore")

# 定义超参数
MAX_GRAD_NORM = 1.2  # 在通过平均步骤聚合之前每个样本梯度的最大 L2 范数
EPSILON = 50.0  # 隐私预算值
DELTA = 1e-5  # (ε,δ)-DP 保证的目标 δ，一般设置为小于训练数据集大小的倒数
EPOCHS = 10
LR = 1e-3  # 学习率
BATCH_SIZE = 128
MAX_PHYSICAL_BATCH_SIZE = 64

# 假设 CIFAR10 数据集的值被假定为已知。如有必要，可以使用适度的隐私预算来计算它们
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)
])

# 加载数据集并执行标准化处理
data_dir = "../../data"

train_data = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
test_data = CIFAR10(root=data_dir, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

# 加载模型
model = models.resnet18(num_classes=10)

# 因为 opacus 不一定兼容所有的 pytorch layer 所以运行下面的内容检测不兼容的部分
errors = ModuleValidator.validate(model, strict=False)
print(f'打印不兼容内容：{errors[-5:]}')

# 处理不兼容，ModuleValidator.fix(model) 试图找到不兼容模块的最佳替代品
model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)

# 更改计算设备： CUDA and CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')
model = model.to(device)

# 优化和损失
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)


# 用于检测准确率
def accuracy(preds, labels):
    return (preds == labels).mean()


# 加入 opacus 进行 DP 训练
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    target_delta=DELTA,
    target_epsilon=EPSILON,
    epochs=EPOCHS,
    max_grad_norm=MAX_GRAD_NORM
)

# noise_multiplier: 采样并添加到批次中梯度平均值的噪声量
print(f'using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}')


# 定义训练函数
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top_acc = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer
    ) as memory_safe_data_loader:
        for i, (images, target) in enumerate(memory_safe_data_loader):
            # for i, (images, target) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # 计算输出
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # 计算准确度和损失
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )


# 定义检测函数
def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            # 记录每一个 batch 的损失和准确度
            losses.append(loss.item)
            top_acc.append(acc)

    # 计算平均准确度
    acc_avg = np.mean(top_acc)

    print(f' test set: \n loss: {np.mean(losses):.6f}  acc: {acc_avg * 100:.6f}')

    return np.mean(top_acc)


# 训练网络
for epoch in tqdm(range(EPOCHS), desc='Epoch', unit='epoch'):
    train(model, train_loader, optimizer, epoch + 1, device)

# 在测试数据上测试网络
top_acc = test(model, test_loader, device)
print(f'test acc: {top_acc}')
