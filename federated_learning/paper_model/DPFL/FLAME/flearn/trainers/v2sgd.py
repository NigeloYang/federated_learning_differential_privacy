import numpy as np
from tqdm import tqdm, trange
import math
from .fedbase import BaseFedarated
from ..utils.tf_utils import process_grad
from ..utils.utils import clip, sparsify
from ..utils.priv_utils import sampling_randomizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


class Server(BaseFedarated):
    '''
    SS-FL-V2
    '''
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train (SS-Double)')
        self.inner_opt = GradientDescentOptimizer(learning_rate=params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.clip_C = self.norm
        self.m_p = self.clients_per_round / self.mp_rate
        print("Setting the padding size for each dimension with ", self.m_p)
        self.em_s = self.clients_per_round /self.rate
        self.sample = int((self.dim_model + self.dim_y)/self.rate)
        print("Randomly sampling {} dimensions".format(self.sample))
        self.choice_list = []

    def train(self):
        '''Train using Federated Proximal'''
        self.train_grouping()

    def local_process(self, flattened):
        choices = np.random.choice(flattened.size, self.sample)
        self.choice_list.extend(choices)
        return sampling_randomizer(flattened, choices, self.clip_C, self.epsilon, self.delta, self.mechanism)

    def server_process(self, messages):
        '''
        basic aggregate, but enlarge the learning rate when Top-k is applied
        '''
        return self.aggregate_p(messages)
