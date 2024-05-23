# -*- coding: utf-8 -*-
# Time    : 2024/5/15
# By      : Yang

import torch
from model.models import LeNet,CNNMnist


class ClientSpa(object):
    def __init__(self, args, train_dataset, id=-1):
        self.args = args
        self.local_model = CNNMnist().to(args.device)
        # self.local_model = LeNet().to(args.device)
        
        self.client_id = id
        
        self.train_dataset = train_dataset
        
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / args.client_nums)
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.mask = {}
        for name, param in self.local_model.state_dict().items():
            p = torch.ones_like(param) * args.param_sparsity
            if torch.is_floating_point(param):
                self.mask[name] = torch.bernoulli(p)
            else:
                self.mask[name] = torch.bernoulli(p).long()
        
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=args.batch_size,
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))
    
    def local_train(self, global_model):
        
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        
        # print("\nlocal model train ... ... ")
        # for name, layer in self.local_model.named_parameters():
        #     print(name, "->", layer.size())
        
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.learn_rate, momentum=self.args.momentum)
        
        self.local_model.train()
        for e in range(self.args.local_epochs):
            for batch_id, (data, target) in enumerate(self.train_loader):
                # for name, layer in self.local_model.named_parameters():
                #     print(name, '->', torch.mean(self.local_model.state_dict()[name].data))
                
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                
                optimizer.step()
            
            print("Local Epoch %d Done." % e)
        
        diff = dict()
        diff_total = 0
        diff_mask_total = 0
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
            diff_total += diff[name].numel()
            # print(f"{name} total parameters:  {diff_total/(1024*1024):.3f}M")
            diff[name] = diff[name] * self.mask[name]
            diff_mask_total += diff[name].numel()
            # print(f"{name} maks total parameters:  {diff_mask_total/(1024*1024):.3f}M")
        # print(f"total parameters: {diff_total/(1024*1024):.3f}M")
        # print(f"mask total parameters:  {diff_mask_total/(1024*1024):.3f}M")
        return diff
