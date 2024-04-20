import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

if __name__ == '__main__':
  start_time = time.time()
  
  # 定义路径
  path_project = os.path.abspath('..')
  logger = SummaryWriter('../logs')
  
  args = args_parser()
  exp_details(args)
  
  # 选择训练的方式 CUDA or CPU
  if args.gpu and torch.cuda.is_available():
    device = 'cuda'
    print(f'device is {device}')
  else:
    device = 'cpu'
    print(f'device is {device}')
  
  # 加载数据集和用户群组
  train_dataset, test_dataset, user_group = get_dataset(args)
  
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
  
  # 为模型选择适合训练的设备
  global_model.to(device)
  
  # model.train()的作用是启用 Batch Normalization 和 Dropout
  global_model.train()
  print(global_model)
  
  # 获取权重
  global_weight = global_model.state_dict()
  
  # 开始训练
  train_loss, train_acc = [], []
  val_acc_list, net_list = [], []
  cv_loss, cv_acc = [], []
  print_every = 2
  val_loss_pre, counter = 0, 0
  
  for epoch in tqdm(range(args.epochs)):
    local_weight, local_losses = [], []
    print(f'\n global training round: {epoch + 1} | \n')
    
    global_model.train()
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    
    # 跟据随机选择的客户端进行本地数据集的训练
    for idx in idxs_users:
      local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_group[idx], logger=logger)
      w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
      local_weight.append(copy.deepcopy(w))
      local_losses.append(copy.deepcopy(loss))
    
    # 更新全局权重
    global_weight = average_weights(local_weight)
    global_model.load_state_dict(global_weight)
    
    loss_avg = sum(local_losses) / len(local_losses)
    train_loss.append(loss_avg)
    
    # 计算每个时期本地所有用户的平均训练准确度
    list_acc, list_loss = [], []
    global_model.eval()
    # for c in range(args.num_users):
    for idx in idxs_users:
      local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_group[idx], logger=logger)
      acc, loss = local_model.inference(model=global_model)
      list_acc.append(acc)
      list_loss.append(loss)
    
    train_acc.append(sum(list_acc) / len(list_acc))
    
    # 打印每一个 every 'i' 之后的 全局训练的损失
    if (epoch + 1) % print_every == 0:
      print(f'\n avg training stats after {epoch + 1} global rounds: ')
      print(f'training loss: {np.mean(np.array(train_loss))}')
      print('Train Accuracy: {:.2f}% \n'.format(100 * train_acc[-1]))
  
  # 训练完成后,进行测试
  test_acc, test_loss = test_inference(args, global_model, test_dataset)
  
  print(f' \n Results after {args.epochs} global rounds of training:')
  print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_acc[-1]))
  print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
  
  # 保存对象 train_loss 和 train_accuracy:
  file_name = './save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(
    args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs
  )
  
  with open(file_name, 'wb') as f:
    pickle.dump([train_loss, train_acc], f)
  
  print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
  
  # PLOTTING(optional
  matplotlib.use('Agg')
  
  # Plot Loss curve
  plt.figure()
  plt.title('Training Loss vs Communication rounds')
  plt.plot(range(len(train_loss)), train_loss, color='r')
  plt.ylabel('Training loss')
  plt.xlabel('Communication Rounds')
  plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.format(
    args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs
  ))
  
  # Plot Average Accuracy vs Communication rounds
  plt.figure()
  plt.title('Average Accuracy vs Communication rounds')
  plt.plot(range(len(train_acc)), train_acc, color='k')
  plt.ylabel('Average Accuracy')
  plt.xlabel('Communication Rounds')
  plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.format(
    args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs
  ))
