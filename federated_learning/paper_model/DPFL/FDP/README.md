# federated-fdp

This repo contains demo code for the following paper:

> Qinqing Zheng, Shuxiao Chen, Qi Long, Weijie J. Su.  *Federated f-Differential Privacy*. AISTATS 2021. [[arXiv:2102.11158](https://arxiv.org/abs/2102.11158)]


### Abstract
Federated learning (FL) is a training paradigm where the clients collaboratively learn
models by repeatedly sharing information without compromising much on the privacy of
their local sensitive data. In this paper, we introduce federated f-differential
privacy, a new notion specifically tailored to the federated setting, based on the
framework of Gaussian differential privacy.  Federated f-differential privacy operates
on record level: it provides the privacy guarantee on each individual record of one
client’s data against adversaries. We then propose a generic private federated learning
framework PriFedSync that accommodates a large family of state-of-the-art FL algorithms,
which provably achieves federated f-differential privacy.


### Code organization
- `code`: code to run private FL training for non-iid MNIST and non-iid CIFAR. One might
need to change the data path in `data.py`.
- `config`: sample config files for the setup used in our experiments.

### Code Dependency
- [pytorch](https://github.com/pytorch/pytorch)
- [opacus](https://github.com/pytorch/opacus)

