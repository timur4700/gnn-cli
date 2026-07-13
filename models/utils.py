import torch
from torch.nn import Module, ReLU, SiLU, Sigmoid, Tanh
from torch_geometric.nn import BatchNorm, GraphNorm, LayerNorm, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import dropout_edge
from typing import Literal

import numpy as np



def split_data(train_set, frac=0.1, seed=42):

    train_len = len(train_set)
    val_len = int(train_len * frac)

    rng = np.random.default_rng(seed=seed)
    val_idx = rng.choice(train_len, size=val_len, replace=False)

    return val_idx



def make_optim_sheduler(model, lr, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    return optimizer, scheduler



def freeze_randomization(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def make_act_function(function_name: Literal['relu', 'silu', 'sigmoid', 'tanh']):

    act_functions = {'relu': ReLU,
                     'silu': SiLU,
                     'sigmoid': Sigmoid,
                     'tanh': Tanh}
    
    return act_functions[function_name]



def make_normalization(name: Literal['batchnorm', 'graphnorm', 'layernorm']):

    norm_functions = {'batchnorm': BatchNorm,
                      'graphnorm': GraphNorm,
                      'layernorm': LayerNorm}
    
    return norm_functions[name]


def cosine_cutoff(r, cutoff):
    return 0.5 * (torch.cos(torch.pi * r / cutoff) + 1.0) * (r < cutoff).float()


def make_pooling(glob_pool):

    if glob_pool == 'sum':
        return global_add_pool

    elif glob_pool == 'mean':
        return global_mean_pool

    elif glob_pool == 'max':
        return global_max_pool



class DropoutEdge(Module):

    def __init__(self, dropout_p, undirected=True):
        super().__init__()

        self.dropout_p = dropout_p
        self.undirected = undirected

    def forward(self, edge_index):
        return dropout_edge(edge_index, self.dropout_p, force_undirected=self.undirected, training=self.training)



class RBF(Module):

    def __init__(self, n_rbf, device, cutoff=10):
        super().__init__()

        self.n_rbf = n_rbf
        self.centers = torch.linspace(0, cutoff, n_rbf, device=device, dtype=torch.float32)
        self.gamma = cutoff

    def forward(self, norm):

        rbf = torch.exp(-self.gamma * (norm - self.centers) ** 2)
        return rbf