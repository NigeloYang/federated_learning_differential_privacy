# -*- coding: utf-8 -*-
# @Time : 2024/10/22 15:30
# @Author : Yang

import torch

print('torch version: ',torch.__version__)
print('CUDA: ',torch.cuda.is_available())
print("CUDA Version: ",torch.version.cuda)
print("cuDNN version is :",torch.backends.cudnn.version())
print('cuda count: ',torch.cuda.device_count())