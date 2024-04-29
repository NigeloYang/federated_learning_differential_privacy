# Step 1: Importing PyTorch and Opacus
import torch
import tqdm
from torchvision import datasets, transforms
import numpy as np
from opacus import PrivacyEngine

# Step 2: Loading MNIST Data
train_loader = torch.utils.data.DataLoader(
  datasets.MNIST(
    '../mnist',
    train=True,
    download=True,
    transform=transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ]
    ),
  ),
  batch_size=64,
  shuffle=True,
  num_workers=1,
  pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
  datasets.MNIST(
    '../mnist',
    train=False,
    transform=transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ]
    ),
  ),
  batch_size=1024,
  shuffle=True,
  num_workers=1,
  pin_memory=True
)

# Step 3: Creating a PyTorch Neural Network Classification Model and Optimizer
model = torch.nn.Sequential(
  torch.nn.Conv2d(1, 16, 8, 2, padding=3),
  torch.nn.ReLU(),
  torch.nn.MaxPool2d(2, 1),
  torch.nn.Conv2d(16, 32, 4, 2),
  torch.nn.ReLU(),
  torch.nn.MaxPool2d(2, 1),
  torch.nn.Flatten(),
  torch.nn.Linear(32 * 4 * 4, 32),
  torch.nn.ReLU(),
  torch.nn.Linear(32, 10)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Step 4: Attaching a Differential Privacy Engine to the Optimizer
privacy_engine = PrivacyEngine(
  model, batch_size=64, sample_size=60000,
  alphas=range(2, 32), noise_multiplier=1.3, max_grad_norm=1.0,
)

privacy_engine.attach(optimizer)


# Step 5: Training the private model over multiple epochs
def train(model, train_loader, optimizer, epoch, device, delta):
  model.train()
  criterion = torch.nn.CrossEntropyLoss()
  losses = []
  for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
  epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
  print(
    f"Train Epoch: {epoch} \t"
    f"Loss: {np.mean(losses):.6f} "
    f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
  )


for epoch in range(1, 11):
  train(model, train_loader, optimizer, epoch, device="cpu", delta=1e-5)
