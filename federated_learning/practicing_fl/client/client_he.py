# -*- coding: utf-8 -*-
# @Time : 2024/10/18 14:43
# @Author : Yang

import time

import numpy as np
import torch

from model.models import CNNMnist


class ClientHE(object):
    def __init__(self, args, train_dataset, public_key, private_key, id=-1):
        self.args = args
        self.public_key = public_key
        self.private_key = private_key

        self.local_model = CNNMnist().to(args.device)
        # self.local_model = LeNet().to(args.device)

        self.parameter_shape = self.get_parameter_shape(self.local_model)

        self.client_id = id

        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / args.client_nums)
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=args.batch_size,
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))

    def get_parameter_shape(self, model):
        para_shape = {}
        for k, v in model.named_parameters():
            para_shape[k] = v.shape

        return para_shape

    def encrypt_weights(self, global_model):
        encrypted_weight = {}
        for name, data in global_model.named_parameters():
            en_parameters = data.flatten(0).detach().cpu().numpy().tolist()
            encrypted_weight[name] = self.encrypt_vector(self.public_key, en_parameters)
        return encrypted_weight

    def encrypt_vector(self, public_key, x):
        return [public_key.encrypt(i) for i in x]

    def decrypt_vector(self, private_key, x):
        return [private_key.decrypt(i) for i in x]

    def local_train(self, encrypt_gweights, c_id):
        print(f"----------------------Client {c_id} Decrypting----------------------")
        client_des = time.time()
        for name, enparam in encrypt_gweights.items():
            enparam = self.decrypt_vector(self.private_key, enparam)
            ori_params = torch.reshape(torch.Tensor(enparam), self.parameter_shape[name]).to(self.args.device)
            self.local_model.state_dict()[name].copy_(ori_params.clone())
        print('Client {:>2} Decrypt Time Cost: {:>4.2f}s'.format(c_id, time.time() - client_des))

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.learn_rate, momentum=self.args.momentum)

        self.local_model.train()
        correct = 0
        dataset_size = 0

        for epoch in range(self.args.local_epochs):
            for batch_id, (data, target) in enumerate(self.train_loader):
                data = data.to(self.args.device)
                target = target.to(self.args.device)

                dataset_size += data.size(0)

                optimizer.zero_grad()

                output = self.local_model(data)
                correct += (torch.sum(torch.argmax(output, dim=1) == target)).item()

                # pred = output.data.max(1)[1]  # get the index of the max log-probability
                # correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

                loss = torch.nn.functional.cross_entropy(output, target)

                loss.backward()

                optimizer.step()

            acc = 100.0 * (float(correct) / float(dataset_size))
            print("Client {:>2} | Local HE Train: {:>3}th Epoch Done | Acc: {:>4.3f}".format(c_id, epoch, acc))

        diff = dict()
        client_ens = time.time()
        print(f"----------------------Client {c_id} Encrypting----------------------")
        for name, data in self.local_model.state_dict().items():
            en_parameters = data.flatten(0).cpu().detach().numpy().tolist()
            en_params = self.encrypt_vector(self.public_key, en_parameters)

            diff[name] = np.subtract(en_params, encrypt_gweights[name])
        print('Client {} Encrypt Time Cost: {:>4.2f}s \n'.format(c_id, time.time() - client_ens))
        return diff
