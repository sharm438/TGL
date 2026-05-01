import argparse
import torch
import torch.multiprocessing as mp
import os
import time
import json
from copy import deepcopy
import utils as utils
import aggregation as aggregation
import train_node as train_node
import models as models
import math
import pdb
import tqdm
from tqdm import tqdm

from simple_base_graph import SimpleBaseGraph
from base_graph import BaseGraph

import torch.multiprocessing as mp
import eval_worker as eval_worker

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default='experiment',
                        help='Experiment name for output files.')
    parser.add_argument("--dataset", type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'femnist', 'agnews'],
                        help='Dataset to use.')
    parser.add_argument("--fraction", type=float, default=1.0,
                        help='Fraction of the dataset to be used for training.')
    parser.add_argument("--bias", type=float, default=0.0,
                        help='Parameter to control non-IID data distribution.')
    parser.add_argument("--aggregation", type=str, default='fedsgd',
                        choices=['fedsgd', 'p2p', 'p2p_local', 'tgl'],
                        help='Aggregation protocol to use.')
    parser.add_argument("--topo", type=str, default=None,
                        choices=['ring', 'torus', 'erdos-renyi', 'base-graph', 'simple-base-graph', 'exponential', 'hsl'],
                        help='Topology to use for p2p graphs.')
    parser.add_argument("--budget", type=int, default=None,
                        help='Number of edges in an undirected p2p graph '
                             '(actual directed edges = 2*budget).')
    parser.add_argument("--num_leaves", type=int, default=10,
                        help='Number of leaves (worker nodes).')
    parser.add_argument("--num_relays", type=int, default=1,
                        help='Number of relays (for tgl).')
    parser.add_argument("--num_rounds", type=int, default=100,
                        help='Number of training or simulation rounds.')
    parser.add_argument("--num_local_iters", type=int, default=1,
                        help='Local training epochs/iterations per round.')
    parser.add_argument("--batch_size", type=int, default=None,
                        help='Mini-batch size for local training.')
    parser.add_argument("--eval_time", type=int, default=1,
                        help='Evaluate the model every "eval_time" rounds.')
    parser.add_argument("--gpu", type=int, default=0,
                        help='GPU ID; use -1 for CPU.')
    parser.add_argument("--k", type=int, default=2,
                        help='Number of neighbors in a k-regular graph.')
    parser.add_argument("--leaf_budget", type=int, default=1,
                        help='(Unused in this script) leaf connection budget.')
    parser.add_argument("--lr", type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument("--sample_type", type=str, default="round_robin",
                        choices=["round_robin","random"],
                        help='How data is sampled for local training.')
    parser.add_argument("--seed", type=int, default=108,
                        help='Random seed. If 0 or negative, uses a random seed.')

    # tgl parameters
    parser.add_argument("--b_lr", type=int, default=2,
                        help='Max leaf connections per relay in stage 1.')
    parser.add_argument("--b_rr", type=int, default=1,
                        help='Max neighbor connections per relay in stage 2.')
    parser.add_argument("--b_rl", type=int, default=2,
                        help='Max relay connections per leaf in stage 3.')

    # Monitoring flags
    parser.add_argument("--monitor_model_drift", action='store_true',
                        help="Compute and log average model drift among leaves.")
    parser.add_argument("--monitor_degree", action='store_true',
                        help="If set, compute and log node in/out degrees.")
    parser.add_argument("--graph_simulation_only", action='store_true',
                        help="If set, no actual training occurs; only simulate graphs.")

    parser.add_argument("--num_workers", type=int, default=0,
                    help="Number of parallel workers for P2P evaluation. 0 = Sequential (default).")

    # hsl (hub-spoke-local) parameters — used when --topo hsl
    parser.add_argument("--hub_degree", type=int, default=None,
                        help='Out-degree for hub nodes in hsl topology. '
                             'If None, computed from b_lr, b_rr, b_rl, num_relays, num_leaves.')
    parser.add_argument("--spoke_degree", type=int, default=None,
                        help='Out-degree for spoke nodes in hsl topology. '
                             'If None, computed from b_lr, b_rl, num_relays, num_leaves.')
                             
    return parser.parse_args()


def main(args):
    """
    Main function:
    1. Sets random seeds if specified.
    2. Prepares the device, output paths, and data (unless only simulating).
    3. Executes training or graph simulation based on "args.aggregation" and "args.graph_simulation_only".
    4. Logs metrics (accuracy, loss, etc.) for analysis.
    """

    # Set random seeds if desired
    if args.seed > 0:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Info] Using fixed seed={args.seed}")
    else:
        print("[Info] Using a random seed (non-reproducible).")

    # Set device for training
    aggregator_device = torch.device(
        f'cuda:{args.gpu}' if (args.gpu >= 0 and torch.cuda.is_available()) else 'cpu'
    )

    # Prepare output directory
    outputs_path = os.path.join(os.getcwd(), "outputs")
    if not os.path.isdir(outputs_path):
        os.makedirs(outputs_path)
    filename = os.path.join(outputs_path, args.exp)

    # Load data if we are doing real training
    if not args.graph_simulation_only:
        if args.dataset == 'femnist':
            trainObject, distributed_data, distributed_label = utils.load_data(args.dataset, 32, args.lr, args.fraction, num_leaves=args.num_leaves)
            num_clients = len(distributed_label)
            counts = torch.tensor([distributed_label[i].shape[0] for i in range(num_clients)],dtype=torch.float32)
            node_weights = counts / counts.sum()
            node_weights = node_weights.to(aggregator_device)
            out_dim = trainObject.num_outputs           
        else:
            trainObject = utils.load_data(args.dataset, args.batch_size, args.lr, args.fraction, num_leaves=args.num_leaves)
            data, labels = trainObject.train_data, trainObject.train_labels
            out_dim = trainObject.num_outputs            
            distributedData = utils.clustered_distribute_data(
                (data, labels),
                num_nodes=args.num_leaves,
                num_clusters=args.num_leaves,
                alpha=args.bias,
                out_dim=out_dim,
                device=torch.device("cpu"),
                seed=args.seed
            )
            distributed_data = distributedData.distributed_input
            distributed_label = distributedData.distributed_output
            
            node_weights = distributedData.wts.to(aggregator_device)

        lr = args.lr
        batch_size = trainObject.batch_size
        inp_dim = trainObject.num_inputs
        
        net_name = trainObject.net_name
        test_data = trainObject.test_data
        # Report basic statistics about data distribution
        num_samples_per_leaf = [len(distributed_data[i]) for i in range(args.num_leaves)]
        min_size = min(num_samples_per_leaf)
        max_size = max(num_samples_per_leaf)
        mean_size = sum(num_samples_per_leaf) / len(num_samples_per_leaf)
        var_size = sum((s - mean_size)**2 for s in num_samples_per_leaf) / len(num_samples_per_leaf)
        sd_size = math.sqrt(var_size)
        
        print(f"[Data Dist] Min={min_size}, Max={max_size}, Mean={mean_size:.1f}, SD={sd_size:.1f}")
        # Initialize the global model
        global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)
        global_wts = utils.model_to_vec(global_model)

        node_states = global_wts.detach().unsqueeze(0).repeat(args.num_leaves, 1)
        if args.aggregation == 'tgl':
            relay_states = global_wts.detach().unsqueeze(0).repeat(args.num_relays, 1)
        
        # For round-robin or random sampling within each leaf
        rr_indices = {}
        for node_id in range(args.num_leaves):
            ds_size = distributed_data[node_id].shape[0]
            if ds_size > 0:
                perm = torch.randperm(ds_size)
                distributed_data[node_id] = distributed_data[node_id][perm]
                distributed_label[node_id] = distributed_label[node_id][perm]
            rr_indices[node_id] = 0

        # Initialize MP Pool if needed
        worker_pool = None
        if args.num_workers > 0 and not args.graph_simulation_only:
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass 
            print(f"[Info] Initializing multiprocessing pool with {args.num_workers} workers...")
            worker_pool = mp.Pool(
                processes=args.num_workers, 
                initializer=eval_worker.init_worker,
                initargs=(args.dataset, net_name, inp_dim, out_dim, batch_size, args.gpu, args.fraction, args.num_leaves)
            )
    else:
        # If only simulating graph connectivity, create dummy data placeholders
        test_data = None
        inp_dim, out_dim = 1, 1
        net_name = 'lenet'
        batch_size = 32
        node_weights = torch.ones(args.num_leaves) / args.num_leaves
        global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)

    # Dictionary to log metrics over rounds
    metrics = {
        'round': [],
        'global_acc': [],
        'global_loss': [],
        'local_acc': [],
        'local_loss': [],
        'leaf_acc': []
    }
    if args.monitor_model_drift:
        metrics['pre_drift'] = []
        metrics['post_drift'] = []

    # For node degrees (only if monitor_degree is enabled)
    degree_dict = {'round': []}
    if args.monitor_degree:
        degree_dict['p2p_indegree'] = []
        degree_dict['p2p_outdegree'] = []
        degree_dict['tgl_stage1_leaf_out'] = []
        degree_dict['tgl_stage2_relay_in'] = []
        degree_dict['tgl_stage3_relay_out'] = []

    def compute_model_drift(stack):
        """
        Compute average distance of each leaf's parameter vector from the mean.
        Optimized to use a Tensor stack directly and torch.no_grad() to save memory.
        """
        with torch.no_grad():
            mean_model = torch.mean(stack, dim=0)
            # Calculate norm of differences efficiently
            dists = torch.norm(stack - mean_model, dim=1)
            return torch.mean(dists).item()

    # ----------------------------------------------------------------------------------
    # Main processing: either real training or graph simulation
    # ----------------------------------------------------------------------------------
    if not args.graph_simulation_only:
        # ------------------------------------------------------------------------
        # Real Training
        # ------------------------------------------------------------------------
        if args.topo == 'base-graph':
            graph = BaseGraph(args.num_leaves, args.k)
            W_base = graph.w_list
        elif args.topo == 'simple-base-graph':
            graph = SimpleBaseGraph(args.num_leaves, args.k)
            W_simple_base = graph.w_list
        elif args.topo == 'exponential':
            W = utils.create_exponential_graph(args.num_leaves, aggregator_device)
        elif args.topo == 'hsl':
            # Compute default degrees from TGL parameters
            n_l, n_r = args.num_leaves, args.num_relays
            default_hub_degree   = int(math.ceil(max(
                args.b_lr + args.b_rr,
                args.b_rr + n_l * args.b_rl / n_r
            )))
            default_spoke_degree = int(math.ceil(max(
                n_r * args.b_lr / n_l,
                args.b_rl
            )))
            hub_degree   = args.hub_degree   if args.hub_degree   is not None else default_hub_degree
            spoke_degree = args.spoke_degree if args.spoke_degree is not None else default_spoke_degree

            tgl_edges  = n_r * args.b_lr + n_r * args.b_rr + n_l * args.b_rl
            hsl_edges  = n_r * hub_degree + (n_l - n_r) * spoke_degree

            print(f"[HSL] Default hub_degree={default_hub_degree}  "
                  f"spoke_degree={default_spoke_degree}")
            if args.hub_degree is not None or args.spoke_degree is not None:
                print(f"[HSL] Overridden hub_degree={hub_degree}  "
                      f"spoke_degree={spoke_degree}")
            print(f"[HSL] Total directed edges => HSL: {hsl_edges}  "
                  f"TGL reference: {tgl_edges}")

            # Fix hub indices for the entire run using the same seed
            torch.manual_seed(args.seed)
            hub_perm    = torch.randperm(args.num_leaves)
            hub_indices   = hub_perm[:n_r].tolist()
            spoke_indices = hub_perm[n_r:].tolist()
            print(f"[HSL] Hub nodes (first {n_r}): {sorted(hub_indices)}")
        try:    
            for rnd in range(args.num_rounds):
                # Step 2: Each leaf trains locally and returns updated weights
                for node_id in range(args.num_leaves):
                    start_wts = node_states[node_id].detach().clone()
                    updated_wts = train_node.local_train_worker_inline(
                        node_id,
                        start_wts,
                        distributed_data[node_id],
                        distributed_label[node_id],
                        inp_dim, out_dim, net_name,
                        args.num_local_iters,
                        batch_size, args.lr,
                        aggregator_device,
                        args.sample_type,
                        rr_indices
                    )
                    node_states[node_id] = updated_wts.detach()

                # (Optional) Log pre-aggregation drift
                if args.monitor_model_drift and (rnd + 1) % args.eval_time == 0:
                    drift_val = compute_model_drift(node_states)
                    metrics['pre_drift'].append(drift_val)

                # Step 3: Aggregation
                if args.aggregation == 'fedsgd':
                    aggregated_wts = aggregation.federated_aggregation(node_states, node_weights)
                    # Update global model
                    global_model = utils.vec_to_model(
                        aggregated_wts.to(aggregator_device),
                        net_name, inp_dim, out_dim, aggregator_device
                    )
                    node_states = aggregated_wts.detach().unsqueeze(0).repeat(args.num_leaves, 1)
                    # (Optional) Log post-aggregation drift
                    if args.monitor_model_drift and (rnd + 1) % args.eval_time == 0:
                        drift_val = compute_model_drift(node_states)
                        metrics['post_drift'].append(drift_val)

                elif args.aggregation == 'p2p':
                    # Create adjacency matrix based on the chosen topology
                    if args.topo == 'hsl':
                        # Full aggregation handled internally — no W construction needed
                        node_states, W = aggregation.hsl_aggregation(
                            node_states,
                            hub_indices, spoke_indices,
                            hub_degree, spoke_degree,
                            return_W=True
                        )
                        node_states = node_states.detach()
                    else: 
                        if args.topo == 'ring':
                            W = utils.create_ring_graph(args.num_leaves, aggregator_device)
                        elif args.topo == 'torus':
                            W = utils.create_torus_graph(args.num_leaves, aggregator_device)
                        elif args.topo == 'erdos-renyi':
                            W = utils.create_erdos_renyi_graph(args.num_leaves, args.budget, aggregator_device)
                        elif args.topo == 'base-graph':
                            W = W_base[rnd % len(W_base)].cuda()
                        elif args.topo == 'simple-base-graph': 
                            W = W_simple_base[rnd % len(W_simple_base)].cuda()
                        
                        else:
                            W = utils.create_k_random_regular_graph(args.num_leaves, args.k, aggregator_device)

                        # Perform p2p averaging
                        node_states = aggregation.p2p_aggregation(node_states, W)
                        node_states = node_states.detach()

                    # (Optional) Log node degrees if requested
                    if args.monitor_degree:
                        indegs = (W.clone().cpu().sum(dim=0)).tolist()
                        outdegs = (W.clone().cpu().sum(dim=1)).tolist()
                        degree_dict['p2p_indegree'].append(indegs)
                        degree_dict['p2p_outdegree'].append(outdegs)

                    # Use the first node’s model as "global" for convenience
                    global_model = utils.vec_to_model(
                        node_states.mean(dim=0), net_name, inp_dim, out_dim, aggregator_device
                    )
                    if (rnd + 1) % args.eval_time == 0 and args.monitor_model_drift:
                        drift_val = compute_model_drift(node_states)
                        metrics['post_drift'].append(drift_val)

                elif args.aggregation == 'p2p_local':
                    # Each leaf merges with neighbors chosen locally in each round
                    node_states, W_local = aggregation.p2p_local_aggregation(
                        node_states, args.k, return_W=True
                    )
                    node_states = node_states.detach()
                    # (Optional) Log node degrees if requested
                    if args.monitor_degree:
                        indegs = (W_local.clone().cpu().sum(dim=0)).tolist()
                        outdegs = (W_local.clone().cpu().sum(dim=1)).tolist()
                        degree_dict['p2p_indegree'].append(indegs)
                        degree_dict['p2p_outdegree'].append(outdegs)

                    # Use the first node’s model as "global"
                    global_model = utils.vec_to_model(
                        node_states.mean(dim=0), net_name, inp_dim, out_dim, aggregator_device
                    )
                    if (rnd + 1) % args.eval_time == 0 and args.monitor_model_drift:
                        drift_val = compute_model_drift(node_states)
                        metrics['post_drift'].append(drift_val)

                elif args.aggregation == 'tgl':
                    """
                    relays and leaves Learning (tgl):
                    Stage 1: leaves -> relays
                    Stage 2: relays perform decentralized mixing (p2p_local)
                    Stage 3: relays -> leaves
                    """

                    # Stage 1: leaves -> relays
                    stage1_matrix = torch.zeros((args.num_relays, args.num_leaves), device=aggregator_device)
                    for relay_id in range(args.num_relays):
                        if args.b_lr <= args.num_leaves:
                            chosen_leaf_ids = torch.randperm(args.num_leaves, device=aggregator_device)[:args.b_lr]
                        else:
                            chosen_leaf_ids = torch.randint(0, args.num_leaves, (args.b_lr,), device=aggregator_device)
                        for s_id in chosen_leaf_ids:
                            stage1_matrix[relay_id, s_id] = 1.0
                    # Row-normalize
                    for row_i in range(args.num_relays):
                        row_sum = torch.sum(stage1_matrix[row_i])
                        if row_sum > 0:
                            stage1_matrix[row_i] /= row_sum

                    # Compute relay-level aggregation
                    for relay_id in range(args.num_relays):
                        indices = (stage1_matrix[relay_id] > 0).nonzero(as_tuple=True)[0]
                        if len(indices) > 0:
                            relay_states[relay_id] = node_states[indices].mean(dim=0).detach()

                    # Stage 2: relays perform local p2p mixing among themselves
                    relay_states, stage2_matrix = aggregation.p2p_local_aggregation(
                        relay_states, args.b_rr, return_W=True
                    )
                    relay_states = relay_states.detach()
                    # Stage 3: relays -> leaves
                    stage3_matrix = torch.zeros((args.num_leaves, args.num_relays), device=aggregator_device)
                    for leaf_id in range(args.num_leaves):
                        if args.b_rl <= args.num_relays:
                            chosen_relay_ids = torch.randperm(args.num_relays, device=aggregator_device)[:args.b_rl]
                        else:
                            chosen_relay_ids = torch.randint(0, args.num_relays, (args.b_rl,), device=aggregator_device)
                        for h_id in chosen_relay_ids:
                            stage3_matrix[leaf_id, h_id] = 1.0
                    # Row-normalize
                    for row_i in range(args.num_leaves):
                        row_sum = torch.sum(stage3_matrix[row_i])
                        if row_sum > 0:
                            stage3_matrix[row_i] /= row_sum

                    # Final leaf weights after receiving from relays
                    for leaf_id in range(args.num_leaves):
                        indices = (stage3_matrix[leaf_id] > 0).nonzero(as_tuple=True)[0]
                        if len(indices) > 0:
                            node_states[leaf_id] = relay_states[indices].mean(dim=0).detach()

                    # (Optional) Track degrees
                    if args.monitor_degree:
                        stage1_T = stage1_matrix.t()
                        leaf_outdeg_stage1 = (stage1_T > 0).sum(dim=1).tolist()
                        relay_indeg_stage2 = (stage2_matrix > 0).sum(dim=0).tolist()
                        relay_outdeg_stage3 = (stage3_matrix > 0).sum(dim=0).tolist()
                        degree_dict['tgl_stage1_leaf_out'].append(leaf_outdeg_stage1)
                        degree_dict['tgl_stage2_relay_in'].append(relay_indeg_stage2)
                        degree_dict['tgl_stage3_relay_out'].append(relay_outdeg_stage3)

                    # Update global model (simple average of final leaf weights)
                    global_model = utils.vec_to_model(
                        node_states.mean(dim=0), net_name, inp_dim, out_dim, aggregator_device
                    )
                    if (rnd + 1) % args.eval_time == 0 and args.monitor_model_drift:
                        drift_val = compute_model_drift(node_states)
                        metrics['post_drift'].append(drift_val)

                if (rnd + 1) % args.eval_time == 0:
                    if args.aggregation == 'fedsgd':
                        utils.evaluate_and_log(
                            current_round=rnd + 1,
                            metrics=metrics,
                            model_source=global_model,
                            mode='federated',
                            test_data=test_data,
                            device=aggregator_device
                        )

                    # Path 2: P2P / tgl (Multi-Model)
                    else:
                        # Sub-path A: Parallel
                        if args.num_workers > 0:
                            print(f"[Round {rnd+1}] Starting Parallel Evaluation...")
                            payload = [w.detach().cpu() for w in node_states]
                            results = worker_pool.map(eval_worker.evaluate_node, payload)
                            
                            g_losses, g_accs = zip(*results)
                            metrics['round'].append(rnd + 1)
                            metrics['global_acc'].append(list(g_accs))
                            metrics['global_loss'].append(list(g_losses))
                            
                            print(f"[Round {rnd+1}] Parallel {args.aggregation.upper()} => "
                                f"Acc range: [{min(g_accs):.4f}, {max(g_accs):.4f}]")

                        # Sub-path B: Sequential
                        else:
                            utils.evaluate_and_log(
                                current_round=rnd + 1,
                                metrics=metrics,
                                model_source=node_states,
                                mode='p2p',
                                test_data=test_data,
                                device=aggregator_device,
                                net_name=net_name, inp_dim=inp_dim, out_dim=out_dim
                            )
        finally:
            if worker_pool is not None:
                worker_pool.close()
                worker_pool.join()
    else:
        # ------------------------------------------------------------------------
        # Graph Simulation Only (no actual training)
        # ------------------------------------------------------------------------

        # Spectral gap and edge accumulators (unchanged)
        sum_spectral_gap   = 0.0
        sum_num_edges      = 0
        count_graph_rounds = 0

        # Degree accumulators — ELL / p2p / HSL
        outdeg_all_rounds  = []   # [n_nodes] tensor per round: out-degree per node
        indeg_all_rounds   = []   # [n_nodes] tensor per round: in-degree per node

        # Degree accumulators — TGL only
        leaf_outdeg_s1_rounds  = []   # [n_leaves]: how many relays sampled each leaf
        relay_indeg_s2_rounds  = []   # [n_relays]: relay in-degree from Stage 2 gossip
        relay_outdeg_s3_rounds = []   # [n_relays]: how many leaves sampled each relay

        # Topology pre-computation
        W_base = None
        if args.topo == 'base-graph':
            graph  = BaseGraph(args.num_leaves, args.k)
            W_base = graph.w_list
        elif args.topo == 'simple-base-graph':
            graph         = SimpleBaseGraph(args.num_leaves, args.k)
            W_simple_base = graph.w_list
        elif args.topo == 'hsl':
            n_l, n_r = args.num_leaves, args.num_relays
            default_hub_degree = int(math.ceil(max(
                args.b_lr + args.b_rr,
                args.b_rr + n_l * args.b_rl / n_r
            )))
            default_spoke_degree = int(math.ceil(max(
                n_r * args.b_lr / n_l,
                args.b_rl
            )))
            hub_degree   = args.hub_degree   if args.hub_degree   is not None else default_hub_degree
            spoke_degree = args.spoke_degree if args.spoke_degree is not None else default_spoke_degree
            torch.manual_seed(args.seed)
            hub_perm      = torch.randperm(args.num_leaves)
            hub_indices   = hub_perm[:n_r].tolist()
            spoke_indices = hub_perm[n_r:].tolist()
            print(f"[HSL Sim] hub_degree={hub_degree}  spoke_degree={spoke_degree}")
            tgl_edges = n_r * args.b_lr + n_r * args.b_rr + n_l * args.b_rl
            hsl_edges = n_r * hub_degree + (n_l - n_r) * spoke_degree
            print(f"[HSL Sim] Total edges => HSL: {hsl_edges}  TGL reference: {tgl_edges}")

        # --------------------------------------------------------------------
        # Simulation loop
        # --------------------------------------------------------------------
        for rnd in tqdm(range(args.num_rounds), desc="Simulating Graphs"):
            d        = 10
            leaf_wts = torch.randn(args.num_leaves, d, device=aggregator_device)

            # ----------------------------------------------------------------
            # p2p branch (all fixed-W topologies including HSL)
            # ----------------------------------------------------------------
            if args.aggregation == 'p2p':
                if args.topo == 'hsl':
                    _, W = aggregation.hsl_aggregation(
                        leaf_wts,
                        hub_indices, spoke_indices,
                        hub_degree, spoke_degree,
                        return_W=True
                    )
                elif args.topo == 'erdos-renyi':
                    W = utils.create_erdos_renyi_graph(
                        args.num_leaves, args.budget, aggregator_device)
                elif args.topo == 'ring':
                    W = utils.create_ring_graph(args.num_leaves, aggregator_device)
                elif args.topo == 'torus':
                    W = utils.create_torus_graph(args.num_leaves, aggregator_device)
                elif args.topo == 'base-graph':
                    W = W_base[rnd % len(W_base)].cuda()
                else:
                    W = utils.create_k_random_regular_graph(
                        args.num_leaves, args.k, aggregator_device)

                _ = aggregation.p2p_aggregation(leaf_wts, W)

                # Binary W excluding self-loops
                W_bin = (W > 0).float()
                W_bin.fill_diagonal_(0.0)

                # Out-degree of node i = row i nonzero count (nodes i pulls from)
                outdeg_all_rounds.append(W_bin.sum(dim=1).cpu())
                # In-degree of node j = col j nonzero count (nodes that pull from j)
                indeg_all_rounds.append(W_bin.sum(dim=0).cpu())

                num_edges = W_bin.sum().item()
                e         = torch.linalg.eigvals(W)
                e_abs     = torch.abs(e)
                e_sorted, _ = torch.sort(e_abs, descending=True)
                gap_rnd   = (1.0 - e_sorted[1]).item()

                sum_spectral_gap   += gap_rnd
                sum_num_edges      += num_edges
                count_graph_rounds += 1

            # ----------------------------------------------------------------
            # ELL (p2p_local) branch
            # ----------------------------------------------------------------
            elif args.aggregation == 'p2p_local':
                _, W_local = aggregation.p2p_local_aggregation(
                    leaf_wts, args.k, return_W=True)

                W_bin = (W_local > 0).float()
                W_bin.fill_diagonal_(0.0)

                outdeg_all_rounds.append(W_bin.sum(dim=1).cpu())
                indeg_all_rounds.append(W_bin.sum(dim=0).cpu())

                num_edges    = W_bin.sum().item()
                e            = torch.linalg.eigvals(W_local)
                e_abs        = torch.abs(e)
                e_sorted, _  = torch.sort(e_abs, descending=True)
                gap_rnd      = (1.0 - e_sorted[1]).item()

                sum_spectral_gap   += gap_rnd
                sum_num_edges      += num_edges
                count_graph_rounds += 1

            # ----------------------------------------------------------------
            # TGL branch
            # ----------------------------------------------------------------
            elif args.aggregation == 'tgl':

                # Stage 1: each relay samples b_lr leaves (relay controls its in-degree)
                stage1_matrix = torch.zeros(
                    (args.num_relays, args.num_leaves), device=aggregator_device)
                for relay_id in range(args.num_relays):
                    if args.b_lr <= args.num_leaves:
                        chosen = torch.randperm(
                            args.num_leaves, device=aggregator_device)[:args.b_lr]
                    else:
                        chosen = torch.randint(
                            0, args.num_leaves, (args.b_lr,), device=aggregator_device)
                    for s_id in chosen:
                        stage1_matrix[relay_id, s_id] = 1.0
                for row_i in range(args.num_relays):
                    row_sum = torch.sum(stage1_matrix[row_i])
                    if row_sum > 0:
                        stage1_matrix[row_i] /= row_sum

                # Stage 2: relay gossip (p2p_local among relays)
                relay_wts = torch.randn(args.num_relays, d, device=aggregator_device)
                _, stage2_matrix = aggregation.p2p_local_aggregation(
                    relay_wts, args.b_rr, return_W=True)

                # Stage 3: each leaf samples b_rl relays (leaf controls its in-degree)
                stage3_matrix = torch.zeros(
                    (args.num_leaves, args.num_relays), device=aggregator_device)
                for leaf_id in range(args.num_leaves):
                    if args.b_rl <= args.num_relays:
                        chosen = torch.randperm(
                            args.num_relays, device=aggregator_device)[:args.b_rl]
                    else:
                        chosen = torch.randint(
                            0, args.num_relays, (args.b_rl,), device=aggregator_device)
                    for h_id in chosen:
                        stage3_matrix[leaf_id, h_id] = 1.0
                for row_i in range(args.num_leaves):
                    row_sum = torch.sum(stage3_matrix[row_i])
                    if row_sum > 0:
                        stage3_matrix[row_i] /= row_sum

                # ---- Degree profiling ----

                # Stage 1: leaf out-degree = column sums of binary stage1_matrix
                # stage1_matrix[relay, leaf] > 0 means relay pulls from leaf
                # col j nonzero count = how many relays sampled leaf j
                s1_bin = (stage1_matrix > 0).float()        # [n_r, n_l]
                leaf_outdeg_s1_rounds.append(
                    s1_bin.sum(dim=0).cpu())                 # [n_l]

                # Stage 2: relay in-degree = column sums of binary stage2_matrix excl. diagonal
                # stage2_matrix[i, j] > 0 means relay i pulls from relay j
                # col j nonzero count excl. diagonal = how many relays pull from relay j
                s2_bin = (stage2_matrix > 0).float()
                s2_bin.fill_diagonal_(0.0)
                relay_indeg_s2_rounds.append(
                    s2_bin.sum(dim=0).cpu())                 # [n_r]

                # Stage 3: relay out-degree = column sums of binary stage3_matrix
                # stage3_matrix[leaf, relay] > 0 means leaf pulls from relay
                # col h nonzero count = how many leaves sampled relay h
                s3_bin = (stage3_matrix > 0).float()        # [n_l, n_r]
                relay_outdeg_s3_rounds.append(
                    s3_bin.sum(dim=0).cpu())                 # [n_r]

                # Edge count
                edges_stage1 = s1_bin.sum().item()
                edges_stage2 = s2_bin.sum().item()
                edges_stage3 = s3_bin.sum().item()
                total_edges  = edges_stage1 + edges_stage2 + edges_stage3

                # Spectral gap of effective leaf-to-leaf mixing matrix
                W_eff_round = torch.matmul(
                    stage3_matrix,
                    torch.matmul(stage2_matrix, stage1_matrix))
                e            = torch.linalg.eigvals(W_eff_round)
                e_abs        = torch.abs(e)
                e_sorted, _  = torch.sort(e_abs, descending=True)
                gap_rnd      = (1.0 - e_sorted[1]).item()

                sum_spectral_gap   += gap_rnd
                sum_num_edges      += total_edges
                count_graph_rounds += 1

        # --------------------------------------------------------------------
        # Degree statistics helper
        # --------------------------------------------------------------------
        def degree_stats(rounds_list, label, split_at=None, split_labels=None):
            """
            rounds_list  : list of 1-D float tensors [n_nodes], one per round
            split_at     : optional int — split tensor at this index to report
                           two node-type groups separately (e.g. hubs vs spokes)
            split_labels : tuple of two strings for the two groups
            Returns a dict saved into metrics.
            """
            if not rounds_list:
                return {}
            mat = torch.stack(rounds_list, dim=0)   # [R, N]

            def _report(m, name):
                flat          = m.flatten()
                per_node_mean = m.mean(dim=0)
                d = {
                    'avg':          round(flat.mean().item(), 4),
                    'min':          round(flat.min().item(), 4),
                    'max':          round(flat.max().item(), 4),
                    'std':          round(flat.std().item(), 4),
                    'var':          round(flat.var().item(), 4),
                    'min_node_avg': round(per_node_mean.min().item(), 4),
                    'max_node_avg': round(per_node_mean.max().item(), 4),
                    'std_node_avg': round(per_node_mean.std().item(), 4),
                }
                print(f"\n  [{name}]")
                print(f"    all (round,node) pairs: "
                      f"avg={d['avg']:.3f}  min={d['min']:.0f}  "
                      f"max={d['max']:.0f}  std={d['std']:.3f}  "
                      f"var={d['var']:.3f}")
                print(f"    per-node avg across rounds: "
                      f"min={d['min_node_avg']:.3f}  "
                      f"max={d['max_node_avg']:.3f}  "
                      f"std={d['std_node_avg']:.3f}")
                return d

            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            result = {'all': _report(mat, 'all nodes')}

            if split_at is not None and split_labels is not None:
                result[split_labels[0]] = _report(mat[:, :split_at],  split_labels[0])
                result[split_labels[1]] = _report(mat[:, split_at:], split_labels[1])

            return result

        # --------------------------------------------------------------------
        # Summary output
        # --------------------------------------------------------------------
        if count_graph_rounds > 0:
            avg_gap   = sum_spectral_gap / float(count_graph_rounds)
            avg_edges = sum_num_edges    / float(count_graph_rounds)
            metrics['avg_spectral_gap'] = avg_gap
            metrics['avg_num_edges']    = avg_edges
            print(f"\n[Info] Average spectral gap across rounds: {avg_gap:.6f}")
            print(f"[Info] Average directed edges per round:   {avg_edges:.2f}")

            # ELL / p2p / HSL degree stats
            if outdeg_all_rounds:
                split_at     = None
                split_labels = None
                if args.topo == 'hsl':
                    # Rearrange so hub columns come first, then spokes
                    all_idx = hub_indices + spoke_indices
                    outdeg_all_rounds = [t[all_idx] for t in outdeg_all_rounds]
                    indeg_all_rounds  = [t[all_idx] for t in indeg_all_rounds]
                    split_at     = len(hub_indices)
                    split_labels = ('hubs', 'spokes')

                metrics['outdeg_stats'] = degree_stats(
                    outdeg_all_rounds,
                    f"OUT-degree — {args.aggregation} / {args.topo}",
                    split_at, split_labels)
                metrics['indeg_stats'] = degree_stats(
                    indeg_all_rounds,
                    f"IN-degree  — {args.aggregation} / {args.topo}",
                    split_at, split_labels)

            # TGL degree stats
            if leaf_outdeg_s1_rounds:
                metrics['tgl_leaf_outdeg_s1'] = degree_stats(
                    leaf_outdeg_s1_rounds,
                    "TGL Stage 1 — leaf OUT-degree "
                    "(how many relays sampled each leaf)")

                metrics['tgl_relay_indeg_s2'] = degree_stats(
                    relay_indeg_s2_rounds,
                    "TGL Stage 2 — relay IN-degree "
                    "(how many relay peers each relay received from)")

                metrics['tgl_relay_outdeg_s3'] = degree_stats(
                    relay_outdeg_s3_rounds,
                    "TGL Stage 3 — relay OUT-degree "
                    "(how many leaves sampled each relay)")
    # ----------------------------------------------------------------------------------
    # Save final metrics to a JSON file
    # ----------------------------------------------------------------------------------
    with open(filename + '_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Training or simulation complete.")

    # Save degree dictionary if node-degree monitoring was enabled
    if args.monitor_degree:
        with open(filename + '_degree.json', 'w') as f:
            json.dump(degree_dict, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)