# -*- coding: utf-8 -*-
# Time    : 2024/5/15
# By      : Yang

import torch
from model.models import LeNet, CNNMnist


class ServerDP(object):
    def __init__(self, args, test_dataset):
        self.args = args
        self.global_model = CNNMnist().to(args.device)
        # self.global_model = LeNet().to(args.device)
        
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * self.args.lambdas
            
            # if self.args.dp:
            #     sigma = self.args.sigma
            #     if torch.cuda.is_available():
            #         noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, sigma)
            #     else:
            #         noise = torch.FloatTensor(update_per_layer.shape).normal_(0, sigma)
            #     update_per_layer.add_(noise)
            # else:
            #     if data.type() != update_per_layer.type():
            #         data.add_(update_per_layer.to(torch.int64))
            #     else:
            #         data.add_(update_per_layer)
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)
    
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
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  #
        
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        
        return acc, total_l


if __name__ == "__main__":
    print()
