import argparse
import torch
import os
import time

import utils as utils
import models as models
from copy import deepcopy
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    # (same arguments as in main or can be omitted if not called directly)
    return parser.parse_args()

def local_train_worker_inline(
    node_id,
    global_wts,
    local_data,
    local_label,
    inp_dim,
    out_dim,
    net_name,
    num_local_iters,
    batch_size,
    lr,
    device,
    sample_type,
    rr_indices
):
    """
    Inline version of local training (no multiprocessing).
    """
    # Rebuild local model from global weights
    local_model = utils.vec_to_model(global_wts, net_name, inp_dim, out_dim, device)
    local_model.train()

    # We'll iterate 'num_local_iters' times over mini-batches
    idx_start = rr_indices[node_id]
    for _ in range(num_local_iters):

        if sample_type == 'round_robin':
            # take the next batch of size `batch_size`, wrapping around the dataset
            N = local_data.shape[0]
            # build indices [idx_start, idx_start+1, ..., idx_start+batch_size-1] mod N
            indices = torch.tensor(
                [(idx_start + i) % N for i in range(batch_size)],
                device=local_data.device,
                dtype=torch.long
            )
            x = local_data[indices].to(device)
            y = local_label[indices].to(device)
            # advance start pointer by batch_size, modulo N
            idx_start = (idx_start + batch_size) % N

            
        else:
            # random
            indices = torch.randint(0, local_data.shape[0], (batch_size,), device=local_data.device)
            x = local_data[indices].to(device)
            y = local_label[indices].to(device)

        # one forward/backward pass
        
        loss_fn = torch.nn.CrossEntropyLoss()
        out = local_model(x)
        loss = loss_fn(out, y)
        local_model.zero_grad()
        loss.backward()
        for p in local_model.parameters():
            if p.grad is not None: 
                p.data -= lr * p.grad

    rr_indices[node_id] = idx_start
    # return new weights
    return utils.model_to_vec(local_model)
