import torch.nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar

if __name__ == '__main__':
  args = args_parser()
  # 根据传入的参数判别是否使用 cuda 进行运算
  if args.gpu and torch.cuda.is_available():
    device = 'cuda'
    print(f'device is {device}')
  else:
    device = 'cpu'
    print(f'device is {device}')
  
  # 加载数据集
  train_dataset, test_dataset, _ = get_dataset(args)
  
  # 建立模型
  if args.model == 'cnn':
    # CNN　模型
    if args.dataset == 'mnist':
      global_model = CNNMnist(args=args)
    elif args.dataset == 'fmnist':
      global_model = CNNFashion_Mnist(args=args)
    elif args.dataset == 'cifar':
      global_model = CNNCifar(args=args)
  elif args.model == 'mlp':
    # 多层感知机　模型
    img_size = train_dataset[0][0].shape
    len_in = 1
    for x in img_size:
      len_in *= x
      global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
  
  else:
    exit('没有适合的模型，需要创建一个模型')
  
  # 将模型设置为训练并将其发送到设备上
  global_model.to(device)
  global_model.train()
  print(global_model)
  
  # 开始训练
  if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=0.5)
  elif args.optimzier == 'adam':
    optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr, weight_decay=1e-4)
  
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  criterion = torch.nn.NLLLoss().to(device)
  epoch_loss = []
  
  for epoch in tqdm(range(args.epochs)):
    batch_loss = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)
      
      optimizer.zero_grad()
      outputs = global_model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      
      if batch_idx % 50 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(
          epoch + 1, batch_idx * len(images), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
          loss.item())
        )
      batch_loss.append(loss.item())
  
    loss_avg = sum(batch_loss) / len(batch_loss)
    print('train avg loss: ', loss_avg)
    epoch_loss.append(loss_avg)
    
  # Plot loss
  plt.figure()
  plt.plot(range(len(epoch_loss)), epoch_loss)
  plt.xlabel('epochs')
  plt.ylabel('Train loss')
  plt.savefig('./save/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))
  
  # testing
  test_acc, test_loss = test_inference(args, global_model, test_dataset)
  print('Test on: ', len(test_dataset), 'samples')
  print("Test Accuracy: {:.2f}%".format(100 * test_acc))
