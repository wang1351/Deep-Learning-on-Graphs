import torch
import numpy as np

def loss_modularity(r, bi_adj, mod):
    size = bi_adj.shape[0]
    bi_adj_no_diag = bi_adj*(torch.ones(size, size) - torch.eye(size))
    return np.power(bi_adj_no_diag.sum(), -1) * (r.t() @ mod @ r).trace()

