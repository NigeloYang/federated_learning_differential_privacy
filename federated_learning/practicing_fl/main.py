# -*- coding: utf-8 -*-
# Time    : 2024/5/15
# By      : Yang

import argparse
import random
from tqdm import tqdm

import torch
from torchvision import datasets, transforms

from server.server_compression import ServerCom
from server.server_sparsity import ServerSpa
from client.client_compression import ClientCom
from client.client_sparsity import ClientSpa


def get_dataset(args):
    dir = './data/'
    
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
        train_dataset = datasets.MNIST(root=dir, train=True, download=True, transform=transform)
    
        test_dataset = datasets.MNIST(root=dir, train=False, download=True, transform=transform)

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
    parser.add_argument('-ds', '--dataset', type=str, default="mnist")
    parser.add_argument('-ge', '--global_epoch', type=int, default=10)
    parser.add_argument('-le', '--local_epochs', type=int, default=5)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learn_rate', type=float, default=0.01)
    parser.add_argument('-mt', '--momentum', type=float, default=0.0001)

    parser.add_argument('-m_t', '--model_type', type=str, default="sparsity")
    
    parser.add_argument('-lma', '--lambdas', type=float, default=0.5)
    parser.add_argument('-cr', '--compress_rate', type=float, default=0.95)
    parser.add_argument('-sc', '--sample_clients', type=int, default=2)
    parser.add_argument('-pr', '--param_sparsity', type=float, default=0.6)
    parser.add_argument('-dce', '--device', type=str, default="cpu")
    
    args = parser.parse_args()
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ', args.device)
    
    train_datasets, test_datasets = get_dataset(args)
    clients = []
    if args.model_type == "compression":
        server = ServerCom(args, test_datasets)
        for c_id in range(args.client_nums):
            clients.append(ClientCom(args, train_datasets, c_id))
    elif args.model_type == "sparsity":
        server = ServerSpa(args, test_datasets)
        for c_id in range(args.client_nums):
            clients.append(ClientSpa(args, train_datasets, c_id))
    else :
        assert "no model"
        
    
    for ge in tqdm(range(args.global_epoch)):
        candidates = random.sample(clients, args.sample_clients)
        
        client_weights = {}
        client_weights_name = {}
        
        for name, params in server.global_model.state_dict().items():
            client_weights[name] = torch.zeros_like(params)
            client_weights_name[name] = 0
        
        for client in candidates:
            diff = client.local_train(server.global_model)
            
            for name, params in server.global_model.state_dict().items():
                if name in diff:
                    client_weights[name].add_(diff[name])
                    client_weights_name[name] += 1

        if args.model_type == "compression":
            server.model_aggregate(client_weights, client_weights_name)
        elif args.model_type == "sparsity":
            server.model_aggregate(client_weights)
        else:
            assert "no type model_aggregate"
        
        acc, loss = server.model_eval()
        
        print("Epoch %d, acc: %f, loss: %f\n" % (ge, acc, loss))
