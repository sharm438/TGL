import torch
import numpy as np
import pdb

def federated_aggregation(stack, weights):
    avg = torch.sum(stack * weights.view(-1,1), dim=0)
    return avg

def p2p_aggregation(leaf_wts, W):
    """
    W: shape [num_leaves, num_leaves]
    leaf_wts: shape [num_leaves, dim]
    returns: shape [num_leaves, dim]
    """
    return torch.mm(W, leaf_wts)

def p2p_local_aggregation(node_wts, outdegree, return_W=False, alive_mask=None):
    """
    node_wts: shape [num_nodes, dim]
    outdegree: int
    return_W: bool -> if True, also return the effective mixing matrix
    """
    num_nodes = node_wts.shape[0]
    device = node_wts.device
    
    # 1. Create the Topology Matrix (W)
    # We initialize W as zeros. 
    # Even if return_W is False, we use W internally for efficient matmul aggregation.
    W = torch.zeros((num_nodes, num_nodes), device=device)

    # We still loop to generate random neighbors (topology), 
    # but we DO NOT do heavy tensor math inside this loop.
    for i in range(num_nodes):
        # Create a pool of neighbors excluding the node itself
        all_neighbors = torch.arange(num_nodes, device=device)
        all_neighbors = all_neighbors[all_neighbors != i]  # Exclude self

        # Pick random neighbors
        if outdegree <= num_nodes - 1:
            chosen_neighbors = all_neighbors[torch.randperm(all_neighbors.size(0))[:outdegree]]
        else:
            chosen_neighbors = all_neighbors[torch.randint(0, all_neighbors.size(0), (outdegree,))]
        
        # Apply alive mask if needed
        if alive_mask is not None:
            chosen_neighbors = chosen_neighbors[alive_mask[chosen_neighbors]]
        
        # Set connection weights to 1.0 (Uniform averaging)
        # Self-connection
        W[i, i] = 1.0 
        # Neighbor connections
        W[i, chosen_neighbors] = 1.0

    # 2. Normalize Rows (Row-Stochastic)
    # This creates the "mean" effect. 
    # If a node connects to 3 neighbors + itself = 4 nodes, every weight becomes 0.25
    row_sums = W.sum(dim=1, keepdim=True)
    # Avoid division by zero if a node somehow has no connections (unlikely with self-loop)
    W = W / (row_sums + 1e-10)

    # 3. Aggregate (One-Shot Matrix Multiplication)
    # This replaces the slow loop of .mean() calls
    updated_wts = torch.mm(W, node_wts)

    if return_W:
        return updated_wts, W
    else:
        return updated_wts

def hsl_aggregation(node_wts, hub_indices, spoke_indices,
                    hub_degree, spoke_degree, return_W=False):
    """
    Asymmetric-degree gossip (Hub-Spoke-Local).

    hub_indices  : list or 1-D tensor of hub node indices (fixed, chosen once)
    spoke_indices: list or 1-D tensor of spoke node indices
    hub_degree   : int  — out-degree for hub nodes (excluding self)
    spoke_degree : int  — out-degree for spoke nodes (excluding self)

    Sampling is identical to p2p_local_aggregation: random without replacement
    from ALL nodes excluding self.  Self-loop always included.  Row-stochastic.
    """
    num_nodes = node_wts.shape[0]
    device    = node_wts.device

    W = torch.zeros((num_nodes, num_nodes), device=device)
    all_nodes = torch.arange(num_nodes, device=device)

    for i in hub_indices:
        pool   = all_nodes[all_nodes != i]
        degree = min(hub_degree, pool.size(0))
        chosen = pool[torch.randperm(pool.size(0), device=device)[:degree]]
        W[i, i]      = 1.0
        W[i, chosen] = 1.0

    for i in spoke_indices:
        pool   = all_nodes[all_nodes != i]
        degree = min(spoke_degree, pool.size(0))
        chosen = pool[torch.randperm(pool.size(0), device=device)[:degree]]
        W[i, i]      = 1.0
        W[i, chosen] = 1.0

    row_sums = W.sum(dim=1, keepdim=True)
    W = W / (row_sums + 1e-10)

    updated_wts = torch.mm(W, node_wts)

    if return_W:
        return updated_wts, W
    return updated_wts
            
def old_p2p_local_aggregation(node_wts, outdegree, return_W=False, alive_mask=None):
    """
    node_wts: shape [num_nodes, dim]
    outdegree: int
    return_W: bool -> if True, also return the effective mixing matrix
    """
    num_nodes = node_wts.shape[0]
    dim = node_wts.shape[1]
    updated_wts = torch.zeros_like(node_wts)

    # If returning the mixing matrix, create NxN zero matrix
    W_local = None
    if return_W:
        W_local = torch.zeros((num_nodes, num_nodes), device=node_wts.device)

    for i in range(num_nodes):
        # Create a pool of neighbors excluding the node itself
        all_neighbors = torch.arange(num_nodes, device=node_wts.device)
        all_neighbors = all_neighbors[all_neighbors != i]  # Exclude self

        # Pick exactly 'outdegree' random neighbors from the pool
        if outdegree <= num_nodes - 1:
            chosen_neighbors = all_neighbors[torch.randperm(all_neighbors.size(0))[:outdegree]]
        else:
            chosen_neighbors = all_neighbors[torch.randint(0, all_neighbors.size(0), (outdegree,))]
        
        if alive_mask is not None:
            chosen_neighbors = chosen_neighbors[alive_mask[chosen_neighbors]]
        
        # Include the node itself
        chosen = torch.cat((chosen_neighbors, torch.tensor([i], device=node_wts.device)))
        neighbors_models = node_wts[chosen]
        
        # Compute the mean vector and update weights
        mean_vec = neighbors_models.mean(dim=0)
        updated_wts[i] = mean_vec

        if return_W:
            # Each row i => 1/|chosen| for columns in chosen
            for c in chosen:
                W_local[i, c] = 1.0

    if return_W:
        # Convert each row i into row-stochastic
        for i in range(num_nodes):
            row_sum = torch.sum(W_local[i])
            if row_sum > 0:
                W_local[i] /= row_sum
        return updated_wts, W_local
    else:
        return updated_wts

