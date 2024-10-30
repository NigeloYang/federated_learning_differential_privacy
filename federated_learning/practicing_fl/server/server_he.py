# -*- coding: utf-8 -*-
# @Time : 2024/10/18 14:44
# @Author : Yang
import time

import torch
from model.models import LeNet, CNNMnist


class ServerHE(object):
    def __init__(self, args, test_dataset, private_key, public_key):
        self.args = args
        self.private_key = private_key
        self.public_key = public_key
        self.global_model = CNNMnist().to(args.device)
        self.parameter_shape = self.get_parameter_shape(self.global_model)

        self.encrypt_global_weights = self.encrypt_weights(self.global_model)

        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    def get_parameter_shape(self, model):
        para_shape = {}
        for name, value in model.named_parameters():
            para_shape[name] = value.shape

        return para_shape

    def encrypt_weights(self,global_model):
        encrypted_weight = {}
        server_ens = time.time()
        print("----------------------Server Encrypting----------------------")
        for name, data in global_model.named_parameters():
            en_parameters = data.reshape(-1).detach().cpu().numpy().tolist()
            encrypted_weight[name] = self.encrypt_vector(self.public_key, en_parameters)

        print('Server encrypt time: {:>4.2f}s'.format(time.time() - server_ens))
        return encrypted_weight

    def encrypt_vector(self, public_key, x):
        return [public_key.encrypt(i) for i in x]

    def decrypt_vector(self, x):
        return [self.private_key.decrypt(i) for i in x]

    def model_aggregate(self, weight_accumulator):
        server_des = time.time()
        print("----------------------Server Decrypting----------------------")
        for name, data in self.global_model.state_dict().items():
            de_params = self.decrypt_vector(weight_accumulator[name])
            ori_params = torch.reshape(torch.Tensor(de_params), self.parameter_shape[name]).to(self.args.device)
            update_per_layer = ori_params * self.args.lambdas

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
        print('Server decrypt time: {:>4.2f}s'.format(time.time()-server_des))

    def model_eval(self):
        self.global_model.eval()
        # print("\n\nstart to model evaluation......")
        # for name, layer in self.global_model.named_parameters():
        #	print(name, "->", torch.mean(layer.data))

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.test_loader):
            data, target = batch
            dataset_size += data.size()[0]

            data = data.to(self.args.device)
            target = target.to(self.args.device)

            output = self.global_model(data)
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            #  sum up batch loss
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l
