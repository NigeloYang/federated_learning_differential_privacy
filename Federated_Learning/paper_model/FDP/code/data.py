import logging
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from sampler import UniformSampler

logging.basicConfig(
    format="%(asctime)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def prepare_mnist_non_iid(num_clients=100, batch_size=16, num_workers=2):
    class FixLabelMNIST(torch.utils.data.Dataset):
        def __init__(self, data, targets, transform=None):
            super(FixLabelMNIST, self).__init__()
            self.data = data
            self.targets = targets
            self.transform = transform

        def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], int(self.targets[index])

            if self.transform is not None:
                img = Image.fromarray(img.numpy(), mode="L")
                img = self.transform(img)
            else:
                img = img.float()

            return img, target

        def __len__(self):
            return len(self.data)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # prepare training data
    data_file = "../data/mnist/MNIST/processed/training.pt"
    data, targets = torch.load(data_file)
    ind = np.argsort(targets.numpy())
    data = data[ind]
    targets = targets[ind]

    shards = np.arange(400)
    shard_size = 60000 / 400
    shard_per_client = int(400 / num_clients)
    np.random.shuffle(shards)

    # prepare training data
    local_train_loaders = []
    train_labels = []
    for ii in range(num_clients):
        my_shards = shards[(ii * shard_per_client) : (ii + 1) * shard_per_client]
        ind = np.array([])
        for jj in my_shards:
            ind = np.append(
                arr=ind, values=np.arange((jj * shard_size), (jj + 1) * shard_size)
            )
        subset = FixLabelMNIST(
            data=data[ind], targets=targets[ind], transform=transform
        )

        ### Uniform sampling
        if batch_size == 8:
            niter = 76
        else:
            niter = np.ceil(len(subset) / batch_size).astype(int)
        sampler = UniformSampler(
            batch_size=batch_size, niter=niter, data_size=len(subset)
        )

        local_train_loaders.append(
            torch.utils.data.DataLoader(
                subset, sampler=sampler, batch_size=batch_size, num_workers=num_workers
            )
        )

        train_labels.append(np.unique(subset.targets.numpy()))
        logger.info("Clinet {} got data with labels: {}".format(ii, train_labels[ii]))

    # prepare test data
    data_file = "../data/mnist/MNIST/processed/test.pt"
    data, targets = torch.load(data_file)
    targets_numpy = targets.numpy()

    local_test_loaders = []
    test_labels = []
    for ii in range(num_clients):
        ind = np.array([]).astype(int)
        for label in train_labels[ii]:
            loc = np.where(targets_numpy == label)
            ind = np.append(ind, loc)

        # sample 200 data from the testing dataset
        ind = ind[np.random.choice(len(ind), 200)]

        subset = FixLabelMNIST(
            data=data[ind], targets=targets[ind], transform=transform
        )
        local_test_loaders.append(
            torch.utils.data.DataLoader(subset, batch_size=64, num_workers=num_workers)
        )
        test_labels.append(np.unique(local_test_loaders[ii].dataset.targets.numpy()))
        assert np.all(np.sort(test_labels[ii]) == np.sort(train_labels[ii]))

    return local_train_loaders, local_test_loaders


def prepare_cifar_non_iid(num_clients=100, batch_size=16, num_workers=2):
    class MyCIFAR(torch.utils.data.Dataset):
        def __init__(self, data, targets, transform=None):
            super(MyCIFAR, self).__init__()
            self.data = data
            self.targets = targets
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            return img, target

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load original data
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [["test_batch", "40351d587109b95175f43aff81a1287e"]]

    # now load the picked numpy arrays
    tr_data, tr_targets = [], []
    for file_name, checksum in train_list:
        file_path = os.path.join("../data/cifar-10-batches-py", file_name)
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            tr_data.append(entry["data"])
            if "labels" in entry:
                tr_targets.extend(entry["labels"])
            else:
                tr_targets.extend(entry["fine_labels"])

    tr_data = np.vstack(tr_data).reshape(-1, 3, 32, 32)
    tr_data = tr_data.transpose((0, 2, 3, 1))  # convert to HWC
    ind = np.argsort(np.array(tr_targets)).astype(int)
    tr_data = tr_data[ind]
    tr_targets = np.array(tr_targets)[ind]

    te_data, te_targets = [], []
    for file_name, checksum in test_list:
        file_path = os.path.join("../data/cifar-10-batches-py", file_name)
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            te_data.append(entry["data"])
            if "labels" in entry:
                te_targets.extend(entry["labels"])
            else:
                te_targets.extend(entry["fine_labels"])

    te_data = np.vstack(te_data).reshape(-1, 3, 32, 32)
    te_data = te_data.transpose((0, 2, 3, 1))  # convert to HWC
    # sort by labels
    ind = np.argsort(np.array(te_targets)).astype(int)
    te_data = te_data[ind]
    te_targets = np.array(te_targets)[ind]

    local_train_loaders, local_test_loaders = [], []

    # tr_available = np.ones(len(tr_data))
    nn = len(tr_data) / num_clients
    for ii in range(num_clients):
        p_class = np.random.dirichlet(np.ones(10) * 0.5)

        """ training """
        # if ii == num_clients - 1:
        #     tr_ind = np.where(tr_available > 0)[0][:int(nn)]
        # else:
        # nn = 2500
        count = np.random.multinomial(nn, p_class)
        print(count)
        tr_ind = np.array([]).astype(int)
        for label in range(10):
            # ind = np.where(np.logical_and(tr_targets == label, tr_available))[0]
            ind = np.where(tr_targets == label)[0]
            select = np.random.permutation(ind)[: count[label]]
            tr_ind = np.append(arr=tr_ind, values=select)
            # tr_available[select] = 0.0

        subset = MyCIFAR(
            data=tr_data[tr_ind], targets=tr_targets[tr_ind], transform=transform_train
        )
        sampler = UniformSampler(
            batch_size=batch_size,
            niter=np.ceil(len(subset) / batch_size).astype(int),
            data_size=len(subset),
        )
        local_train_loaders.append(
            torch.utils.data.DataLoader(
                subset, batch_size=batch_size, num_workers=num_workers, sampler=sampler
            )
        )

        """ testing """
        count = np.random.multinomial(200, p_class)
        te_ind = np.array([]).astype(int)
        for label in range(10):
            ind = np.where(te_targets == label)[0]
            select = np.random.permutation(ind)[: count[label]]
            te_ind = np.append(arr=te_ind, values=select)

        subset = MyCIFAR(
            data=te_data[te_ind], targets=te_targets[te_ind], transform=transform_test
        )
        local_test_loaders.append(
            torch.utils.data.DataLoader(subset, batch_size=128, num_workers=num_workers)
        )
    return local_train_loaders, local_test_loaders


def prepare_data(dataset, num_clients, batch_size, num_workers):
    if dataset == "mnist-non-iid":
        return prepare_mnist_non_iid(
            num_clients=num_clients, batch_size=batch_size, num_workers=num_workers
        )
    elif dataset == "cifar-non-iid":
        return prepare_cifar_non_iid(
            num_clients=num_clients, batch_size=batch_size, num_workers=num_workers
        )
    else:
        raise NotImplementedError
