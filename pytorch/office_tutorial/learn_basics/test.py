import torch
from torchvision.transforms import Lambda

# sample_idx = torch.randint(100, size=(1,)).item()
# print(sample_idx)

t = torch.tensor([[1, 2], [3, 4]])
t_r = t.gather(1, torch.tensor([[0, 0], [1, 0]]))
print(t_r)

# target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
# print(target_transform)
