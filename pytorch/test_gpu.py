''' 如何在GPU 版本下禁用 GPU 使用cpu
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # 指定此处为-1即可

CUDA_VISIBLE_DEVICES = "-1"  # 禁用gpu
CUDA_VISIBLE_DEVICES = "0"  # 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
CUDA_VISIBLE_DEVICES = "1"  # 设置当前使用的GPU设备仅为1号设备  设备名称为'/gpu:0'
CUDA_VISIBLE_DEVICES = "0,1"  # 设置当前使用的GPU设备为0,1号两个设备,名称依次为'/gpu:0'、'/gpu:1'
CUDA_VISIBLE_DEVICES = "1,0"  # 设置当前使用的GPU设备为1,0号两个设备,名称依次为'/gpu:0'、'/gpu:1'。表示优先使用1号设备,然后使用0号设备
'''

import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Get cpu or gpu device for training.
device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(torch.version.cuda)
