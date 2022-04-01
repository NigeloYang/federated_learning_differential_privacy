import torch
from torch.utils.data import Sampler


class UniformSampler(Sampler):
    def __init__(self, batch_size, niter, data_size):
        self._batch_size = batch_size
        self._niter = niter
        self._data_size = data_size

    def __iter__(self):
        n = self._data_size
        ret = []
        for ii in range(self._niter):
            ret.extend(torch.randperm(n)[: self._batch_size])
        return iter(ret)
