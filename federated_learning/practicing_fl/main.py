# -*- coding: utf-8 -*-
# Time    : 2024/5/15
# By      : Yang

import argparse
import random
import time

from tqdm import tqdm
import numpy as np

import torch
from torchvision import datasets, transforms

from client.client_normal import ClientNormal
from server.server_normal import ServerNormal
from client.client_compression import ClientCom
from server.server_compression import ServerCom
from client.client_sparsity import ClientSpa
from server.server_sparsity import ServerSpa
from client.client_backdoor import ClientBack
from server.server_backdoor import ServerBack
from client.client_dp import ClientDP
from server.server_dp import ServerDP
from client.client_he import ClientHE
from server.server_he import ServerHE

from utils import paillier


def get_dataset(args):
    dir = './data/'

    if args.dataset == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(root=dir, train=True, download=True, transform=transform)

        test_dataset = datasets.FashionMNIST(root=dir, train=False, download=True, transform=transform)

    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-cn', '--client_nums', type=int, default=10)
    parser.add_argument('-ds', '--dataset', type=str, default="fmnist")
    parser.add_argument('-ge', '--global_epoch', type=int, default=10)
    parser.add_argument('-le', '--local_epochs', type=int, default=5)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learn_rate', type=float, default=0.01)
    parser.add_argument('-mt', '--momentum', type=float, default=0.0001)

    parser.add_argument('-m_t', '--model_type', type=str, default="he")

    parser.add_argument('--lambdas', type=float, default=0.5)
    parser.add_argument('--eta', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=0.001)
    parser.add_argument('--dp', type=bool, default=True)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--C', type=float, default=100.0)
    parser.add_argument('--w', type=int, default=1)
    parser.add_argument('-cr', '--compress_rate', type=float, default=0.8)
    parser.add_argument('-sc', '--sample_clients', type=int, default=2)
    parser.add_argument('-pr', '--param_sparsity', type=float, default=0.6)
    parser.add_argument('-ppb', '--poi_per_batch', type=float, default=0.2)
    parser.add_argument('-plb', '--poison_label', type=int, default=7)
    parser.add_argument('-dce', '--device', type=str, default="cpu")

    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.device = 'cuda'
    print('device: ', args.device)

    train_datasets, test_datasets = get_dataset(args)
    clients = []
    print('*' * 50, 'training', args.model_type, 'model', '*' * 50)
    if args.model_type == 'normal':
        server = ServerNormal(args, test_datasets)
        for c_id in range(args.client_nums):
            clients.append(ClientNormal(args, train_datasets, c_id))
    elif args.model_type == "compression":
        server = ServerCom(args, test_datasets)
        for c_id in range(args.client_nums):
            clients.append(ClientCom(args, train_datasets, c_id))
    elif args.model_type == "sparsity":
        server = ServerSpa(args, test_datasets)
        for c_id in range(args.client_nums):
            clients.append(ClientSpa(args, train_datasets, c_id))
    elif args.model_type == "backdoor":
        server = ServerBack(args, test_datasets)
        for c_id in range(args.client_nums):
            clients.append(ClientBack(args, train_datasets, c_id))
    elif args.model_type == "dp":
        server = ServerDP(args, test_datasets)
        for c_id in range(args.client_nums):
            clients.append(ClientDP(args, train_datasets, c_id))
    elif args.model_type == "he":
        public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
        server = ServerHE(args, test_datasets, private_key, public_key)
        for c_id in range(args.client_nums):
            clients.append(ClientHE(args, train_datasets, public_key, private_key, c_id))
    else:
        assert "no model"

    global_time = time.time()
    for ge in tqdm(range(args.global_epoch), desc='Processing'):
        print(f'\n--------------- Global training Round: {ge + 1}th ------------------------')
        ge_stime = time.time()
        candidates = random.sample(clients, args.sample_clients)

        client_weights = {}
        client_weights_name = {}
        if args.model_type == 'he':
            for name, params in server.encrypt_global_weights.items():
                client_weights[name] = np.zeros(len(params))
                client_weights_name[name] = 0

            for client in candidates:
                diff = client.local_train(server.encrypt_global_weights, client.client_id)
                for name, params in server.encrypt_global_weights.items():
                    if name in diff:
                        client_weights[name] = np.add(client_weights[name], diff[name])
                        client_weights_name[name] += 1
        else:
            for name, params in server.global_model.state_dict().items():
                client_weights[name] = torch.zeros_like(params)
                client_weights_name[name] = 0

            if args.model_type == 'normal':
                for client in candidates:
                    diff = client.local_train(server.global_model, client.client_id)

                    for name, params in server.global_model.state_dict().items():
                        if name in diff:
                            client_weights[name].add_(diff[name])
                            client_weights_name[name] += 1
            elif args.model_type == "compression":
                for client in candidates:
                    diff = client.local_train(server.global_model, client.client_id)

                    for name, params in server.global_model.state_dict().items():
                        if name in diff:
                            client_weights[name].add_(diff[name])
                            client_weights_name[name] += 1
            elif args.model_type == "sparsity":
                for client in candidates:
                    diff = client.local_train(server.global_model, client.client_id)

                    for name, params in server.global_model.state_dict().items():
                        if name in diff:
                            client_weights[name].add_(diff[name])
                            client_weights_name[name] += 1
            elif args.model_type == "backdoor":
                for client in candidates:
                    if client.client_id in [1, 4, 7]:
                        print("malicious client:", client.client_id)
                        diff = client.local_train_malicious(server.global_model, client.client_id)
                    else:
                        diff = client.local_train(server.global_model, client.client_id)

                    for name, params in server.global_model.state_dict().items():
                        if name in diff:
                            client_weights[name].add_(diff[name])
                            client_weights_name[name] += 1
            elif args.model_type == "dp":
                for client in candidates:
                    if args.dp:
                        print("DP Train")
                        diff = client.local_train_dp(server.global_model, client.client_id)
                    else:
                        print('No DP Train')
                        diff = client.local_train(server.global_model, client.client_id)

                    for name, params in server.global_model.state_dict().items():
                        if name in diff:
                            client_weights[name].add_(diff[name])
                            client_weights_name[name] += 1
            else:
                assert "no model"

        print('{}th global epoch time cost: {:>4.2f}'.format(ge, time.time() - ge_stime))
        if args.model_type == 'normal':
            server.model_aggregate(client_weights)
        elif args.model_type == "compression":
            server.model_aggregate(client_weights, client_weights_name)
        elif args.model_type == "sparsity":
            server.model_aggregate(client_weights)
        elif args.model_type == "backdoor":
            server.model_aggregate(client_weights)
        elif args.model_type == "dp":
            server.model_aggregate(client_weights)
        elif args.model_type == "he":
            server.model_aggregate(client_weights)
        else:
            assert "no type model_aggregate"

        acc, loss = server.model_eval()

        print('{:>2}th Global Epoch Done | Acc: {:.3f} | Loss: {:.3f}'.format(ge + 1, acc, loss))

    print('FL Train End, Time Cost:{:>4.2f}'.format(time.time() - global_time))