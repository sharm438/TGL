"""
fault_tolerance_exp.py
----------------------
TGL training with dynamic node failures. Mirrors main.py exactly.

Fault model (clean, no contamination):
  - At the start of each round, crash_rate% of relays (or leaves) are sampled
    uniformly at random and marked as crashed for that round only.

  Stage 1 (leaf -> relay):
    Crashed relays do NOT sample leaves and do NOT update their model.
    Their relay_states entry is frozen at its previous value.
    Alive relays sample exactly b_lr leaves via torch.randperm over all
    num_leaves (identical to main.py). Crashed leaves in the sampled set
    are silently dropped. The relay averages only the alive survivors.
    If all b_lr sampled leaves happen to be crashed, the relay keeps its
    own current model. We track how many alive relays had 0 alive leaves
    in their sample (leaf_outdeg0_count per round).

  Stage 2 (relay <-> relay):
    Gossip runs only on the alive relay sub-tensor. Each alive relay
    samples b_rr peers from the alive relay pool only (clipped if
    fewer alive relays exist — unavoidable arithmetic). Crashed relays
    are completely excluded — they neither send nor receive. Their
    relay_states entry is unchanged.

  Stage 3 (relay -> leaf):
    Each leaf samples exactly b_rl relays via torch.randperm over all
    num_relays (identical to main.py). Crashed relays in the sampled
    set are silently dropped. The leaf averages only the alive survivors.
    If all b_rl sampled relays are crashed, the leaf keeps its own
    post-training model from this round's local SGD. We track how many
    leaves had 0 alive relays in their sample (relay_indeg0_count per
    round).

  Leaf crash mode:
    Crashed leaves skip local training (stale model carried over) and
    are excluded from the Stage 1 sampling pool.

Crashes are resampled independently every round (dynamic / transient).

At the end of training, we print:
  - Average fraction of alive relays with 0 alive leaves sampled (Stage 1)
  - Average fraction of leaves with 0 alive relays sampled (Stage 3)

Usage:
    python fault_tolerance_exp.py \\
        --dataset cifar10 --num_leaves 100 \\
        --num_relays 20 --b_lr 15 --b_rr 10 --b_rl 2 \\
        --crash_type relay --crash_rate 20 \\
        --lr 0.1 --bias 0.1 --num_local_iters 5 \\
        --num_rounds 1000 --eval_time 10 --num_workers 10 \\
        --monitor_model_drift --gpu 0 --seed 108 \\
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
    parser.add_argument("--bias",     type=float, default=0.1)

    # TGL topology
    parser.add_argument("--num_leaves", type=int, default=100)
    parser.add_argument("--num_relays", type=int, default=20)
    parser.add_argument("--b_lr", type=int, default=15,
                        help="Max leaves sampled per relay in Stage 1.")
    parser.add_argument("--b_rr", type=int, default=10,
                        help="Max relay neighbours in Stage 2.")
    parser.add_argument("--b_rl", type=int, default=2,
                        help="Max relays sampled per leaf in Stage 3.")

    # Fault injection
    parser.add_argument("--crash_type", type=str, default="relay",
                        choices=["relay", "leaf"])
    parser.add_argument("--crash_rate", type=float, default=0.0,
                        help="Percentage of nodes crashed per round (0-100).")

    # Training — identical to main.py
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
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available()
        else "cpu"
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

    # ---- Crash count --------------------------------------------------------
    n_total = args.num_relays if args.crash_type == "relay" else args.num_leaves
    n_crash = max(0, int(math.floor(n_total * args.crash_rate / 100.0)))

    print(f"[Info] crash_type={args.crash_type}  "
          f"crash_rate={args.crash_rate:.0f}%  "
          f"n_crash={n_crash}/{n_total} per round")

    # ---- Degree-zero trackers -----------------------------------------------
    # For relay crash: track alive relays with 0 alive leaves (Stage 1)
    #                  and leaves with 0 alive relays (Stage 3)
    relay_s1_zero_counts = []   # per round: # alive relays with 0 alive leaves
    leaf_s3_zero_counts  = []   # per round: # leaves with 0 alive relays

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

            # --- Sample crashed nodes ----------------------------------------
            if args.crash_type == "relay":
                crashed_relays = set(
                    random.sample(range(args.num_relays), n_crash)
                ) if n_crash > 0 else set()
                crashed_leaves = set()
            else:
                crashed_relays = set()
                crashed_leaves = set(
                    random.sample(range(args.num_leaves), n_crash)
                ) if n_crash > 0 else set()

            alive_relays = [r for r in range(args.num_relays)
                            if r not in crashed_relays]
            alive_leaves = [l for l in range(args.num_leaves)
                            if l not in crashed_leaves]

            # -----------------------------------------------------------------
            # Local training — alive leaves only (identical to main.py)
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

            # Pre-aggregation drift
            if args.monitor_model_drift and (rnd + 1) % args.eval_time == 0:
                metrics["pre_drift"].append(compute_model_drift(node_states))

            # =================================================================
            # Stage 1: leaf -> relay
            # Crashed relays are completely skipped — they do not sample,
            # do not update. Alive relays sample b_lr leaves from the full
            # pool (identical to main.py) then drop crashed leaves silently.
            # =================================================================
            rnd_relay_s1_zeros = 0

            for relay_id in alive_relays:
                # Sample identical to main.py
                if args.b_lr <= args.num_leaves:
                    sampled = torch.randperm(
                        args.num_leaves,
                        device=aggregator_device)[:args.b_lr]
                else:
                    sampled = torch.randint(
                        0, args.num_leaves,
                        (args.b_lr,), device=aggregator_device)

                # Drop crashed leaves — no resampling
                alive_sampled = [int(l) for l in sampled
                                 if int(l) not in crashed_leaves]

                if len(alive_sampled) == 0:
                    # Relay keeps its own current model — nothing written
                    rnd_relay_s1_zeros += 1
                else:
                    chosen = torch.tensor(
                        alive_sampled, device=aggregator_device)
                    relay_states[relay_id] = \
                        node_states[chosen].mean(dim=0).detach()
            # crashed relays: relay_states[relay_id] frozen — not touched

            relay_s1_zero_counts.append(rnd_relay_s1_zeros)

            # =================================================================
            # Stage 2: relay <-> relay gossip
            # Run p2p_local_aggregation ONLY on the alive relay sub-tensor.
            # Crashed relays are completely excluded — they are not in the
            # pool, cannot be sampled, and their state is untouched.
            # =================================================================
            if len(alive_relays) > 1:
                alive_relay_tensor = relay_states[alive_relays]
                # Clip b_rr to alive pool — unavoidable arithmetic only
                effective_b_rr = min(args.b_rr, len(alive_relays) - 1)
                updated_alive, _ = aggregation.p2p_local_aggregation(
                    alive_relay_tensor, effective_b_rr, return_W=True)
                updated_alive = updated_alive.detach()
                for idx, relay_id in enumerate(alive_relays):
                    relay_states[relay_id] = updated_alive[idx]
            # If only 1 alive relay: no gossip, state unchanged
            # crashed relays: relay_states[relay_id] still frozen

            # =================================================================
            # Stage 3: relay -> leaf
            # Each leaf samples b_rl relays from the full pool (identical to
            # main.py) then drops crashed relays silently. If all sampled
            # relays are crashed, leaf keeps its own post-training model.
            # =================================================================
            rnd_leaf_s3_zeros = 0

            for leaf_id in range(args.num_leaves):
                pool = alive_relays  # sample only from alive relays
                if len(pool) == 0:
                    # no alive relays at all — leaf keeps its own model
                    rnd_leaf_s3_zeros += 1
                    continue

                k = min(args.b_rl, len(pool))
                perm = torch.randperm(len(pool), device=aggregator_device)[:k]
                chosen_ids = torch.tensor(
                    [pool[i] for i in perm.tolist()],
                    device=aggregator_device)
                node_states[leaf_id] = \
                    relay_states[chosen_ids].mean(dim=0).detach()
                # S3-zeros only increments if NO alive relays exist at all

            leaf_s3_zero_counts.append(rnd_leaf_s3_zeros)

            # Post-aggregation drift
            if args.monitor_model_drift and (rnd + 1) % args.eval_time == 0:
                metrics["post_drift"].append(compute_model_drift(node_states))

            # -----------------------------------------------------------------
            # Evaluation (identical to main.py parallel branch)
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
                          f"Crashed: {n_crash}/{n_total}  "
                          f"S1-zeros: {rnd_relay_s1_zeros}  "
                          f"S3-zeros: {rnd_leaf_s3_zeros}")
                else:
                    utils.evaluate_and_log(
                        current_round=rnd + 1,
                        metrics=metrics,
                        model_source=node_states,
                        mode="p2p",
                        test_data=test_data,
                        device=aggregator_device,
                        net_name=net_name,
                        inp_dim=inp_dim,
                        out_dim=out_dim,
                    )

    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()

    # ---- Degree-zero summary ------------------------------------------------
    total_rounds = len(relay_s1_zero_counts)
    if total_rounds > 0:
        n_alive_relays = args.num_relays - n_crash

        avg_s1_zeros = sum(relay_s1_zero_counts) / total_rounds
        avg_s3_zeros = sum(leaf_s3_zero_counts)  / total_rounds

        # Fraction relative to alive relay count and total leaf count
        frac_s1 = avg_s1_zeros / max(n_alive_relays, 1)
        frac_s3 = avg_s3_zeros / args.num_leaves

        print(f"\n{'='*60}")
        print(f"Degree-zero summary over {total_rounds} rounds")
        print(f"  crash_type={args.crash_type}  "
              f"crash_rate={args.crash_rate:.0f}%  "
              f"n_crash={n_crash}")
        print(f"  Stage 1 — alive relays with 0 alive leaves sampled:")
        print(f"    avg per round = {avg_s1_zeros:.2f} / {n_alive_relays} "
              f"alive relays  ({100*frac_s1:.1f}%)")
        print(f"  Stage 3 — leaves with 0 alive relays sampled:")
        print(f"    avg per round = {avg_s3_zeros:.2f} / {args.num_leaves} "
              f"leaves  ({100*frac_s3:.1f}%)")
        print(f"{'='*60}")

        metrics["avg_relay_s1_zeros"]    = avg_s1_zeros
        metrics["avg_leaf_s3_zeros"]     = avg_s3_zeros
        metrics["frac_relay_s1_zeros"]   = frac_s1
        metrics["frac_leaf_s3_zeros"]    = frac_s3

    # ---- Save (identical to main.py) ----------------------------------------
    with open(filename + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {filename}_metrics.json")


if __name__ == "__main__":
    main()