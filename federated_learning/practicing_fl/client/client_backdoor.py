# -*- coding: utf-8 -*-
# Time    : 2024/5/15
# By      : Yang

import torch
from model.models import LeNet, CNNMnist, model_norm


class ClientBack(object):
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
    
    def local_train(self, global_model, c_id):
        
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        
        # print("\nlocal model train ... ... ")
        # for name, layer in self.local_model.named_parameters():
        #     print(name, "->", layer.size())
        
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.learn_rate, momentum=self.args.momentum)
        
        self.local_model.train()
        correct = 0
        dataset_size = 0
        for e in range(self.args.local_epochs):
            for batch_id, (data, target) in enumerate(self.train_loader):
                # for name, layer in self.local_model.named_parameters():
                #     print(name, '->', torch.mean(self.local_model.state_dict()[name].data))
                
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                dataset_size += data.size()[0]
                
                optimizer.zero_grad()
                output = self.local_model(data)
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                
                optimizer.step()

            acc = 100.0 * (float(correct) / float(dataset_size))
            print("Client {:>2} | Local Train Malicious Epoch {:>3} Done. | Acc: {:>4.3f}".format(c_id, e, acc))
        
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        return diff
    
    def local_train_malicious(self, global_model, c_id):
        
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        
        # print("\nlocal model train ... ... ")
        # for name, layer in self.local_model.named_parameters():
        #     print(name, "->", layer.size())
        
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.learn_rate, momentum=self.args.momentum)
        
        pos = []
        for i in range(2, 28):
            pos.append([i, 3])
            pos.append([i, 4])
            pos.append([i, 5])
        
        self.local_model.train()
        correct = 0
        dataset_size = 0
        
        for e in range(self.args.local_epochs):
            for batch_id, (data, target) in enumerate(self.train_loader):
                # for name, layer in self.local_model.named_parameters():
                #     print(name, '->', torch.mean(self.local_model.state_dict()[name].data))
                dataset_size += data.size()[0]
                
                for k in range(int(self.args.poi_per_batch * self.args.batch_size)):
                    img = data[k].numpy()
                    for i in range(0, len(pos)):
                        img[0][pos[i][0]][pos[i][1]] = 1.0
                        # cifar-10 need three dim
                        # img[1][pos[i][0]][pos[i][1]] = 0
                        # img[2][pos[i][0]][pos[i][1]] = 0
                    
                    target[k] = self.args.poison_label
                
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                
                optimizer.zero_grad()
                output = self.local_model(data)
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                
                c_loss = torch.nn.functional.cross_entropy(output, target)
                model_dloss = model_norm(self.local_model, global_model)
                loss = (1 - self.args.alpha) * c_loss + self.args.alpha * model_dloss
                loss.backward()
                
                optimizer.step()
            acc = 100.0 * (float(correct) / float(dataset_size))
            print("Client {:>2} | Local Train Malicious Epoch {:>3} Done. | Acc: {:>4.3f}".format(c_id, e, acc))
        
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = self.args.eta * (data - global_model.state_dict()[name]) + global_model.state_dict()[name]
        return diff
