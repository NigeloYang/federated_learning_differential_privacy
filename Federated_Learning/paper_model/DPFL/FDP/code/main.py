from __future__ import print_function

import argparse

# import logging
from datetime import datetime
from opacus.utils import module_modification
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from opacus import PrivacyEngine
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from data import prepare_data
from nets import prepare_net
from sync import run_ma

# logging.basicConfig(
#     stream=sys.stdout,
#     format="%(asctime)s -  %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )
#
# logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup(
    num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        function handle to create `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return partial(lr_scheduler.LambdaLR, lr_lambda=lr_lambda, last_epoch=last_epoch)


def test_one_model(model, loss_func, test_loader, device):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for cnt, (data, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)
            output = model(data)
            # mean per batch
            test_loss += loss_func(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    test_loss /= cnt + 1
    return test_loss, accuracy


def local_test(input_model, loss_func, local_test_loaders, device, selected):

    num_clients = len(local_test_loaders)
    test_loss, test_acc = np.zeros(num_clients), np.zeros(num_clients)

    for ii in range(num_clients):
        if ii in selected:
            if isinstance(input_model, list):
                model = input_model[ii]
            else:
                model = input_model
            test_loss[ii], test_acc[ii] = test_one_model(
                model,
                loss_func=loss_func,
                test_loader=local_test_loaders[ii],
                device=device,
            )

    return (
        test_loss[selected],
        test_acc[selected],
        np.mean(test_loss[selected]),
        np.mean(test_acc[selected]),
    )


def local_train(model, loss_func, train_loader, optimizer, max_grad_norm, device):
    model.train()

    running_loss = 0.0
    correct = 0.0
    for cnt, (data, target) in enumerate(train_loader):

        # Training
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()

        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        optimizer.step()

    running_loss /= cnt + 1
    # uniform sampling: total data points might not be the same as dataset size
    train_acc = correct / (cnt + 1) / len(target)

    return running_loss, train_acc


def init_fl_training(
    num_clients,
    Net,
    OptimizerAlg,
    optimizer_kwargs,
    SchedulerAlg,
    scheduler_kwargs,
    device,
    num_epoch,
):

    if OptimizerAlg == "sgd":
        OptimizerAlg = optim.SGD
    elif OptimizerAlg == "adadelta":
        OptimizerAlg = optim.Adadelta
    elif OptimizerAlg == "adam":
        OptimizerAlg = optim.Adam
    elif OptimizerAlg == "rmsprop":
        OptimizerAlg = optim.RMSprop
    else:
        raise NotImplementedError

    if SchedulerAlg == "step":
        SchedulerAlg = lr_scheduler.StepLR
    elif SchedulerAlg == "multistep":
        SchedulerAlg = lr_scheduler.MultiStepLR
    elif SchedulerAlg == "lambda":
        SchedulerAlg = get_linear_schedule_with_warmup(
            num_warmup_steps=30, num_training_steps=num_epoch, last_epoch=-1
        )
    else:
        raise NotImplementedError

    # print(
    #     "Optimizer: {} Scheduler: {}".format(
    #         OptimizerAlg.__name__, SchedulerAlg.__name__
    #     )
    # )

    global_model = module_modification.convert_batchnorm_modules(Net()).to(device)
    local_models, local_optimizers, local_schedulers = [], [], []
    for ii in range(num_clients):
        model = module_modification.convert_batchnorm_modules(Net()).to(device)
        model.load_state_dict(global_model.state_dict())
        local_models.append(model)

        optimizer = OptimizerAlg(local_models[-1].parameters(), **optimizer_kwargs)
        local_optimizers.append(optimizer)
        if scheduler_kwargs:
            local_schedulers.append(SchedulerAlg(optimizer, **scheduler_kwargs))
        else:
            local_schedulers.append(SchedulerAlg(optimizer))

    return (global_model, local_models, local_optimizers, local_schedulers)


def train(
    Net,
    local_train_loaders,
    local_test_loaders,
    loss_func,
    num_clients,
    num_epoch,
    OptimizerAlg,
    optimizer_kwargs,
    SchedulerAlg,
    scheduler_kwargs,
    sync_gap=1,
    batch_size=16,
    max_grad_norm=1.0,
    noise_sigma=1.0,
    sampling_rate=1.0,
    device="cpu",
):
    if loss_func == "CrossEntropy":
        loss_func = nn.CrossEntropyLoss()
    elif loss_func == "NLL":
        loss_func = nn.NLLLoss()
    else:
        raise NotImplementedError

    print("Using model {} with {}".format(Net.__name__, type(loss_func).__name__))
    print("Privacy: noise std {}, max grad norm {}".format(noise_sigma, max_grad_norm))

    num_tr_samples = np.array([len(item.dataset) for item in local_train_loaders])
    num_te_samples = np.array([len(item.dataset) for item in local_test_loaders])
    print("Number of training samples: {}".format(num_tr_samples))
    print("Number of test samples: {}".format(num_te_samples))

    (global_model, local_models, local_optimizers, local_schedulers) = init_fl_training(
        num_clients=num_clients,
        Net=Net,
        OptimizerAlg=OptimizerAlg,
        optimizer_kwargs=optimizer_kwargs,
        SchedulerAlg=SchedulerAlg,
        scheduler_kwargs=scheduler_kwargs,
        device=device,
        num_epoch=num_epoch,
    )

    if noise_sigma > 0:
        for ii in range(num_clients):
            privacy_engine = PrivacyEngine(
                local_models[ii],
                batch_size=batch_size,
                sample_size=len(local_train_loaders[ii].dataset),
                # alpha is not important for us
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=noise_sigma,
                max_grad_norm=max_grad_norm,
            )
            privacy_engine.attach(local_optimizers[ii])

    sync_count = 0

    train_loss, train_acc, test_loss, test_acc = (
        np.zeros(num_clients),
        np.zeros(num_clients),
        np.zeros(num_clients),
        np.zeros(num_clients),
    )

    num_rounds = int(num_epoch / sync_gap)
    epoch = 0
    for rr in range(num_rounds):
        ### One Round Training ###
        while True:
            selected = np.where(np.random.uniform(0, 1, num_clients) <= sampling_rate)[
                0
            ].astype(int)
            if selected.size:
                break

        print("Selected {}/{} clients to sync.".format(selected.size, num_clients))
        for _ in range(sync_gap):
            ### One Epoch Inside Sync Round ###
            epoch += 1
            for ii in trange(num_clients):
                if ii in selected:
                    train_loss[ii], train_acc[ii] = local_train(
                        local_models[ii],
                        loss_func=loss_func,
                        train_loader=local_train_loaders[ii],
                        optimizer=local_optimizers[ii],
                        max_grad_norm=max_grad_norm,
                        device=device,
                    )
                    local_schedulers[ii].step()

                    test_loss[ii], test_acc[ii] = test_one_model(
                        local_models[ii],
                        loss_func=loss_func,
                        test_loader=local_test_loaders[ii],
                        device=device,
                    )

            avg_train_acc = np.mean(train_acc)
            avg_test_acc = np.mean(test_acc)
            writer.add_scalars(
                "Average Train Accuracy", {"local": avg_train_acc}, epoch
            )
            writer.add_scalars("Average Test Accuracy", {"local": avg_test_acc}, epoch)

            print(
                "===> Epoch {} Local models: Avg Train Accuracy: {:.2f}%, Avg Test"
                " Accuracy: {:.2f}%\n".format(
                    epoch, 100.0 * avg_train_acc, 100.0 * avg_test_acc
                )
            )

        ###  Synchronization ####
        sync_count += 1
        print("Syncing #{}... MA".format(sync_count))
        run_ma(global_model, local_models, selected=selected, device=device, alpha=0.1)
        _, test_acc_global, _, avg_test_acc_global = local_test(
            global_model,
            loss_func,
            local_test_loaders,
            device,
            selected=np.arange(num_clients),
        )
        writer.add_scalars(
            "Average Test Accuracy", {"global": avg_test_acc_global}, epoch
        )
        print(
            "===> After Sync: Global Model Avg Acc {:.2f}% \n".format(
                100.0 * avg_test_acc_global
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cuda", type=int, default=1)
    args = parser.parse_args()

    now = datetime.now()
    run_name = (
        str(now.month)
        + "_"
        + str(now.day)
        + "_"
        + str(now.hour)
        + "_"
        + str(now.minute)
        + "_"
        + args.config.split("/")[-1][:-5]
    )
    print(run_name)

    global writer
    writer = SummaryWriter("./runs/" + run_name)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(device)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    print("Loading {} Data...".format(config["dataset"]))

    local_train_loaders, local_test_loaders = prepare_data(
        dataset=config["dataset"],
        num_clients=config["federated"]["num_clients"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["dataloader"]["num_workers"],
    )

    num_clients = config["federated"]["num_clients"]

    Net = prepare_net(config["net"])

    train(
        Net,
        local_train_loaders,
        local_test_loaders,
        loss_func=config["train"]["loss"],
        num_clients=num_clients,
        OptimizerAlg=config["train"]["optimizer"],
        optimizer_kwargs=config["train"]["optimizer_kwargs"],
        SchedulerAlg=config["scheduler"]["type"],
        scheduler_kwargs=config["scheduler"]["scheduler_kwargs"],
        num_epoch=config["train"]["epoch"],
        sync_gap=config["federated"]["sync_gap"],
        batch_size=config["train"]["batch_size"],
        max_grad_norm=config["max_grad_norm"],
        noise_sigma=config["noise_sigma"],
        sampling_rate=config["sampling"],
        device=device,
    )


if __name__ == "__main__":
    main()
