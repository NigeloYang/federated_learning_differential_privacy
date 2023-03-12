''' Opacus集成使用和部分函数使用
Abadi et.al提出：DP-SGD算法 & Mironov et.al：提出的高斯采样机制的瑞丽差分隐私
1、对整体批次梯度的单个贡献进行封顶。将每个样本的梯度值范数剪裁为某个值
2、标准的高斯噪声被添加到生成的批次梯度中，以隐藏单个贡献。
3、小批次应该通过统一采样形成，即在每个训练步骤中，来自数据集的每个样本都包含一定的概率p。
注意，这不同与数据集被洗牌并分成批次的标准方法：每个样本在给定的时间段内出现多次或根本不出现的概率为非零。

使用 opacus.privacyEngine.make_private() 可以这样表示
model: 对应梯度采样计算
optimizer：对应梯度的clip noise
dataloader：对应数据集的采样
'''

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from opacus import PrivacyEngine

import warnings
warnings.simplefilter("ignore")


class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


dataset = TensorDataset(torch.rand(100, 16), torch.randint(0, 2, (100,)))
privacy_engine = PrivacyEngine()
model = SampleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
data_loader = DataLoader(dataset, batch_size=10)

print(
    f"Before make_private(). "
    f"Model:{type(model)}, Optimizer:{type(optimizer)}, DataLoader:{type(data_loader)} \n"
)

'''集成使用opacus，完成差分隐私内容'''
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    max_grad_norm=1.0,
    noise_multiplier=1.0
)
print('=' * 20)
print(
    f"Before make_private(). "
    f"Model:{type(model)}, Optimizer:{type(optimizer)}, DataLoader:{type(data_loader)}"
)

# GradSampleModule acts like an underlying nn.Module, and additionally computes per sample gradient tensor (p.grad_sample) for its parameters
# DPOptimizer takes parameters with p.grad_sample computed and performs clipping and noise addition
# DPDataLoader takes a vanilla DataLoader and switches the sampling mechanism to Poisson sampling

'''自定义使用opacus，完成全部差分隐私内容'''
# 1.先初始模型
model = SampleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
data_loader = DataLoader(dataset, batch_size=10)

# 2.Model ===> GradSampleModule
from opacus import GradSampleModule

model = GradSampleModule(model)
y = model(torch.rand(10, 16))
y.sum().backward()

print('\n grad 和 grad_sample 在形状上的差距')
grad = model.fc1.weight.grad
grad_sample = model.fc1.weight.grad_sample
print(f'grad size: {grad.shape}')
print(f'grad_sample size: {grad_sample.shape}')

print('\n original grad 和 ave grad_sample 区别')
grad_sample_agg = grad_sample.mean(dim=0)
print("Average grad_sample over 1st dimension. Equal to original: ", torch.allclose(grad, grad_sample_agg))

# 3.DataLoader ===> DPDataLoader
from opacus.data_loader import DPDataLoader

dp_data_loader = DPDataLoader.from_data_loader(data_loader, distributed=False)

print("Is dataset the same: ", dp_data_loader.dataset == data_loader.dataset)
print(f"DPDataLoader length: {len(dp_data_loader)}, original: {len(data_loader)}")
print("DPDataLoader sampler: ", dp_data_loader.batch_sampler)

# 检测每次参与训练的数据大小
batch_size = []
for x,y in data_loader:
    batch_size.append(len(x))

dp_batch_size = []
for x,y in dp_data_loader:
    dp_batch_size.append(len(x))
print(f' original data_loader batchSize:{batch_size} \n dp_data_loader batchSize:{dp_batch_size}')

# 4. Optimizer ===> DPOptimizer
from opacus.optimizers import DPOptimizer

sample_rate = 1/len(data_loader)
expected_batch_size = int(len(data_loader.dataset) * sample_rate)

optimizer = DPOptimizer(
    optimizer=optimizer,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    expected_batch_size=expected_batch_size,
)

# 5. RDPAccountant 隐私预算值统计
from opacus.accountants import RDPAccountant

accountant = RDPAccountant()
optimizer.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=sample_rate))


def hook_fn(optim: DPOptimizer):
    accountant.step(
        noise_multiplier=optim.noise_multiplier,
        sample_rate=sample_rate * optim.accumulated_iterations,
    )
    
    
    
    
    
# 检测opacus之后的模型数据
print(
    f"\n After part opacus (). "
    f"Model:{type(model)}, Optimizer:{type(optimizer)}, DataLoader:{type(data_loader)}"
)




