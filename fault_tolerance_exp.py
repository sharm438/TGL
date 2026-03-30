"""
fault_tolerance_exp.py
----------------------
TGL training with dynamic node failures. Mirrors main.py exactly.

Fault model (no compensation):
  - At the start of each round, crash_rate% of relays (or leaves) are sampled
    uniformly at random and marked as crashed for that round only.

  Stage 1 (leaf -> relay):
    Each relay samples exactly b_rl leaves via torch.randperm, identical to
    main.py. Crashed leaves in the sampled set are silently dropped. The relay
    averages only the survivors. If all b_rl sampled leaves are crashed, the
    relay retains its own current model unchanged.

  Stage 2 (relay <-> relay):
    p2p_local_aggregation is called on the full relay_states tensor, identical
    to main.py. After the result is returned, crashed relays have their rows
    restored to the pre-gossip state — they neither sent nor received anything.

  Stage 3 (relay -> leaf):
    Each leaf samples exactly b_lr relays via torch.randperm, identical to
    main.py. Crashed relays in the sampled set are silently dropped. The leaf
    averages only the survivors. If all b_lr sampled relays are crashed, the
    leaf retains its own current model unchanged.

  Leaf crash mode:
    Crashed leaves skip local training (stale model carried over) and are
    excluded from Stage 1 sampling — relays cannot pull from a crashed leaf.

Crashes are resampled independently every round (dynamic / transient).

Usage:
    python fault_tolerance_exp.py \
        --dataset cifar10 --num_leaves 100 \
        --num_relays 20 --b_rl 15 --b_rr 10 --b_lr 2 \
        --crash_type relay --crash_rate 20 \
        --lr 0.1 --bias 0.1 --num_local_iters 5 \
        --num_rounds 1000 --eval_time 10 --num_workers 10 \
        --monitor_model_drift --gpu 0 --seed 108 \
        --exp ft_relay20_cifar10_s100h20
"""

import argparse
import torch
import torch.multiprocessing as mp
import os
import json
import math
import random

import utils as utils
import models as models
import train_node as train_node
import aggregation as aggregation
import eval_worker as eval_worker


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", type=str, default="ft_experiment")

    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["mnist", "cifar10", "femnist", "agnews"])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0.1)

    # TGL topology
    parser.add_argument("--num_leaves",  type=int, default=100)
    parser.add_argument("--num_relays",  type=int, default=20)
    parser.add_argument("--b_rl", type=int, default=15)
    parser.add_argument("--b_rr", type=int, default=10)
    parser.add_argument("--b_lr", type=int, default=2)

    # Fault injection
    parser.add_argument("--crash_type", type=str, default="relay",
                        choices=["relay", "leaf"])
    parser.add_argument("--crash_rate", type=float, default=0.0,
                        help="Percentage of nodes crashed per round (0-100)")

    # Training — identical defaults to main.py paper configs
    parser.add_argument("--num_rounds",      type=int,   default=1000)
    parser.add_argument("--num_local_iters", type=int,   default=5)
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=0.1)
    parser.add_argument("--eval_time",       type=int,   default=10)
    parser.add_argument("--gpu",             type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=108)
    parser.add_argument("--sample_type",     type=str,   default="round_robin",
                        choices=["round_robin", "random"])
    parser.add_argument("--num_workers",     type=int,   default=10)
    parser.add_argument("--monitor_model_drift", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Seeds (identical to main.py) --------------------------------------
    if args.seed > 0:
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Info] Using fixed seed={args.seed}")

    aggregator_device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )

    os.makedirs("outputs", exist_ok=True)
    filename = os.path.join("outputs", args.exp)

    # ---- Data loading (identical to main.py) --------------------------------
    if args.dataset == "femnist":
        trainObject, distributed_data, distributed_label = utils.load_data(
            args.dataset, 32, args.lr, args.fraction,
            num_leaves=args.num_leaves)
        num_clients = len(distributed_label)
        counts = torch.tensor(
            [distributed_label[i].shape[0] for i in range(num_clients)],
            dtype=torch.float32)
        node_weights = (counts / counts.sum()).to(aggregator_device)
        out_dim = trainObject.num_outputs
    else:
        trainObject = utils.load_data(
            args.dataset, args.batch_size, args.lr, args.fraction,
            num_leaves=args.num_leaves)
        data, labels = trainObject.train_data, trainObject.train_labels
        out_dim = trainObject.num_outputs
        distributedData = utils.clustered_distribute_data(
            (data, labels),
            num_nodes=args.num_leaves,
            num_clusters=args.num_leaves,
            alpha=args.bias,
            out_dim=out_dim,
            device=torch.device("cpu"),
            seed=args.seed,
        )
        distributed_data  = distributedData.distributed_input
        distributed_label = distributedData.distributed_output
        node_weights      = distributedData.wts.to(aggregator_device)

    lr         = args.lr
    batch_size = trainObject.batch_size
    inp_dim    = trainObject.num_inputs
    net_name   = trainObject.net_name
    test_data  = trainObject.test_data

    # ---- Model init (identical to main.py) ----------------------------------
    global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)
    global_wts   = utils.model_to_vec(global_model)

    node_states  = global_wts.detach().unsqueeze(0).repeat(args.num_leaves, 1)
    relay_states = global_wts.detach().unsqueeze(0).repeat(args.num_relays, 1)

    # ---- Round-robin pointers (identical to main.py) ------------------------
    rr_indices = {}
    for node_id in range(args.num_leaves):
        ds_size = distributed_data[node_id].shape[0]
        if ds_size > 0:
            perm = torch.randperm(ds_size)
            distributed_data[node_id]  = distributed_data[node_id][perm]
            distributed_label[node_id] = distributed_label[node_id][perm]
        rr_indices[node_id] = 0

    # ---- Worker pool (identical to main.py) ---------------------------------
    worker_pool = None
    if args.num_workers > 0:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        print(f"[Info] Initialising pool with {args.num_workers} workers...")
        worker_pool = mp.Pool(
            processes=args.num_workers,
            initializer=eval_worker.init_worker,
            initargs=(args.dataset, net_name, inp_dim, out_dim,
                      batch_size, args.gpu, args.fraction, args.num_leaves),
        )

    # ---- Metrics (identical to main.py) -------------------------------------
    metrics = {
        "round": [], "global_acc": [], "global_loss": [],
        "local_acc": [], "local_loss": [], "leaf_acc": [],
        "n_crashed": [],
    }
    if args.monitor_model_drift:
        metrics["pre_drift"]  = []
        metrics["post_drift"] = []

    # ---- How many nodes crash per round
    n_total = args.num_relays if args.crash_type == "relay" else args.num_leaves
    n_crash = max(0, int(math.floor(n_total * args.crash_rate / 100.0)))

    print(f"[Info] crash_type={args.crash_type}  "
          f"crash_rate={args.crash_rate:.0f}%  "
          f"n_crash={n_crash}/{n_total} per round")

    # ---- Model drift helper (identical to main.py) --------------------------
    def compute_model_drift(stack):
        with torch.no_grad():
            mean_model = torch.mean(stack, dim=0)
            dists = torch.norm(stack - mean_model, dim=1)
            return torch.mean(dists).item()

    # =========================================================================
    # Training loop
    # =========================================================================
    try:
        for rnd in range(args.num_rounds):

            # --- Sample crashed nodes for this round -------------------------
            if args.crash_type == "relay":
                crashed_relays = set(random.sample(range(args.num_relays), n_crash)) \
                                 if n_crash > 0 else set()
                crashed_leaves = set()
            else:
                crashed_relays = set()
                crashed_leaves = set(random.sample(range(args.num_leaves), n_crash)) \
                                 if n_crash > 0 else set()

            alive_leaves = [l for l in range(args.num_leaves)
                            if l not in crashed_leaves]

            # -----------------------------------------------------------------
            # Local training — alive leaves only (identical to main.py 259-273)
            # -----------------------------------------------------------------
            for node_id in alive_leaves:
                start_wts = node_states[node_id].detach().clone()
                updated_wts = train_node.local_train_worker_inline(
                    node_id,
                    start_wts,
                    distributed_data[node_id],
                    distributed_label[node_id],
                    inp_dim, out_dim, net_name,
                    args.num_local_iters,
                    batch_size, lr,
                    aggregator_device,
                    args.sample_type,
                    rr_indices,
                )
                node_states[node_id] = updated_wts.detach()

            # Pre-aggregation drift (identical to main.py 275-278)
            if args.monitor_model_drift and (rnd + 1) % args.eval_time == 0:
                metrics["pre_drift"].append(compute_model_drift(node_states))

            # =================================================================
            # TGL aggregation with fault injection
            # Sampling is IDENTICAL to main.py — torch.randperm over the full
            # index set. Crashed nodes in the sample are dropped silently.
            # No resampling, no compensation.
            # =================================================================

            # -----------------------------------------------------------------
            # Stage 1: leaves -> relays (mirrors main.py lines 357-376)
            # -----------------------------------------------------------------
            stage1_matrix = torch.zeros(
                (args.num_relays, args.num_leaves), device=aggregator_device)

            for relay_id in range(args.num_relays):
                # Identical sampling to main.py
                if args.b_rl <= args.num_leaves:
                    sampled_leaf_ids = torch.randperm(
                        args.num_leaves, device=aggregator_device)[:args.b_rl]
                else:
                    sampled_leaf_ids = torch.randint(
                        0, args.num_leaves, (args.b_rl,), device=aggregator_device)

                # Drop crashed leaves from the sampled set — no resampling
                alive_sampled = [int(l) for l in sampled_leaf_ids
                                 if int(l) not in crashed_leaves]

                if len(alive_sampled) == 0:
                    # All sampled leaves crashed: relay keeps its own model
                    # (stage1_matrix row stays zero → no update applied below)
                    pass
                else:
                    for s_id in alive_sampled:
                        stage1_matrix[relay_id, s_id] = 1.0

            # Row-normalise (identical to main.py)
            for relay_id in range(args.num_relays):
                row_sum = torch.sum(stage1_matrix[relay_id])
                if row_sum > 0:
                    stage1_matrix[relay_id] /= row_sum

            # Apply Stage 1 (identical to main.py 373-376)
            for relay_id in range(args.num_relays):
                indices = (stage1_matrix[relay_id] > 0).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    relay_states[relay_id] = node_states[indices].mean(dim=0).detach()
                # else: relay had no alive sampled leaves → keeps its own model

            # -----------------------------------------------------------------
            # Stage 2: relay gossip (mirrors main.py lines 378-382)
            # Call p2p_local_aggregation on the FULL relay tensor — identical
            # to main.py. Then restore crashed relays to their pre-gossip state
            # so they neither contributed nor received anything.
            # -----------------------------------------------------------------
            pre_gossip_relay_states = relay_states.clone()

            relay_states, stage2_matrix = aggregation.p2p_local_aggregation(
                relay_states, args.b_rr, return_W=True)
            relay_states = relay_states.detach()

            # Restore crashed relays — they did not participate
            for relay_id in crashed_relays:
                relay_states[relay_id] = pre_gossip_relay_states[relay_id]

            # -----------------------------------------------------------------
            # Stage 3: relays -> leaves (mirrors main.py lines 383-402)
            # -----------------------------------------------------------------
            stage3_matrix = torch.zeros(
                (args.num_leaves, args.num_relays), device=aggregator_device)

            for leaf_id in range(args.num_leaves):
                # Identical sampling to main.py
                if args.b_lr <= args.num_relays:
                    sampled_relay_ids = torch.randperm(
                        args.num_relays, device=aggregator_device)[:args.b_lr]
                else:
                    sampled_relay_ids = torch.randint(
                        0, args.num_relays, (args.b_lr,), device=aggregator_device)

                # Drop crashed relays from the sampled set — no resampling
                alive_sampled = [int(r) for r in sampled_relay_ids
                                 if int(r) not in crashed_relays]

                if len(alive_sampled) == 0:
                    # All sampled relays crashed: leaf keeps its own model
                    # (stage3_matrix row stays zero → no update applied below)
                    pass
                else:
                    for h_id in alive_sampled:
                        stage3_matrix[leaf_id, h_id] = 1.0

            # Row-normalise
            for leaf_id in range(args.num_leaves):
                row_sum = torch.sum(stage3_matrix[leaf_id])
                if row_sum > 0:
                    stage3_matrix[leaf_id] /= row_sum

            # Apply Stage 3 (identical to main.py 399-402)
            for leaf_id in range(args.num_leaves):
                indices = (stage3_matrix[leaf_id] > 0).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    node_states[leaf_id] = relay_states[indices].mean(dim=0).detach()
                # else: leaf had no alive sampled relays → keeps its own model

            # Post-aggregation drift (identical to main.py 418-420)
            if args.monitor_model_drift and (rnd + 1) % args.eval_time == 0:
                metrics["post_drift"].append(compute_model_drift(node_states))

            # -----------------------------------------------------------------
            # Evaluation (identical to main.py 422-459 p2p parallel branch)
            # -----------------------------------------------------------------
            if (rnd + 1) % args.eval_time == 0:
                metrics["n_crashed"].append(n_crash)

                if args.num_workers > 0:
                    print(f"[Round {rnd+1}] Starting Parallel Evaluation...")
                    payload = [w.detach().cpu() for w in node_states]
                    results = worker_pool.map(eval_worker.evaluate_node, payload)

                    g_losses, g_accs = zip(*results)
                    metrics["round"].append(rnd + 1)
                    metrics["global_acc"].append(list(g_accs))
                    metrics["global_loss"].append(list(g_losses))

                    print(f"[Round {rnd+1}] TGL crash={args.crash_rate:.0f}% "
                          f"({args.crash_type}) => "
                          f"Acc range: [{min(g_accs):.4f}, {max(g_accs):.4f}]  "
                          f"Crashed: {n_crash}/{n_total}")
                else:
                    utils.evaluate_and_log(
                        current_round=rnd + 1,
                        metrics=metrics,
                        model_source=node_states,
                        mode="p2p",
                        test_data=test_data,
                        device=aggregator_device,
                        net_name=net_name, inp_dim=inp_dim, out_dim=out_dim,
                    )

    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()

    # ---- Save (identical to main.py) ----------------------------------------
    with open(filename + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {filename}_metrics.json")


if __name__ == "__main__":
    main()