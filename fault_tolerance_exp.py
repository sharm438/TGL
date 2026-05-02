"""
fault_tolerance_exp.py
----------------------
TGL training with dynamic node failures using a retry-with-timeout protocol.
Mirrors main.py exactly for all non-fault-tolerance logic.

Fault model: binomial independent crashes
-----------------------------------------
Each relay (or leaf) independently fails with probability crash_prob per
round. The number of crashed nodes follows Binomial(n_total, crash_prob),
varying naturally across rounds. This is more realistic than a fixed quota
and produces honest graceful degradation curves.

Protocol design (coordinator-free, retry-with-cap)
---------------------------------------------------
Stage 1:
  Crashed relays are skipped entirely — frozen state.
  Alive relays sample b_lr leaves from the full pool (identical to main.py)
  and drop crashed leaves silently. s1_zero tracked if no alive leaves found.

Stage 2:
  Gossip runs only on the alive relay sub-tensor. Crashed relays excluded.
  b_rr clipped to len(alive_relays) - 1 if needed (arithmetic only).

Stage 3 (retry-with-cap):
  Relay cap = ceil(relay_capacity_factor * n_l * b_rl / n_r).
    - Relay is unavailable if crashed OR contact count >= cap.
    - Crashed and overloaded relays are indistinguishable (both = timeout).
  Leaf retry budget = ceil(leaf_retry_factor * b_rl) total attempts.
    - Leaves probe relays in a random order from the full pool.
    - Leaves are processed in random shuffled order each round so no leaf
      is consistently disadvantaged by early vs late position.
    - Leaf collects until b_rl successes or retry budget exhausted.
    - If 0 collected: s3_zero (leaf keeps own post-training model).
    - If 0 < collected < b_rl: s3_partial (leaf averages what it got).

Defaults:
  relay_capacity_factor = 1.5  (50% headroom above expected load)
  leaf_retry_factor     = 3.0  (3x target collection as max attempts)

Summary statistics printed at end of training.

Usage:
    python fault_tolerance_exp.py \\
        --dataset cifar10 --num_leaves 100 \\
        --num_relays 20 --b_lr 15 --b_rr 10 --b_rl 2 \\
        --crash_type relay --crash_prob 0.2 \\
        --lr 0.1 --bias 0.1 --num_local_iters 5 \\
        --num_rounds 1000 --eval_time 10 --num_workers 10 \\
        --monitor_model_drift --gpu 0 --seed 108 \\
        --exp ft_relay_p20_cifar10_s100h20
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
                        help="Target relay models collected per leaf in Stage 3.")

    # Fault injection — binomial independent crashes
    parser.add_argument("--crash_type", type=str, default="relay",
                        choices=["relay", "leaf"])
    parser.add_argument("--crash_prob", type=float, default=0.0,
                        help="Per-node independent crash probability per round "
                             "(0.0 = no crashes, 0.4 = 40%% expected crash rate). "
                             "Crashes are drawn from Binomial(n_total, crash_prob) "
                             "independently each round.")

    # Retry-with-cap protocol parameters
    parser.add_argument("--relay_capacity_factor", type=float, default=1.5,
                        help="Relay cap = ceil(factor * n_l * b_rl / n_r). "
                             "Default 1.5 = 50%% headroom above expected load.")
    parser.add_argument("--leaf_retry_factor", type=float, default=3.0,
                        help="Max leaf attempts = ceil(factor * b_rl). "
                             "Default 3.0 = three times the target collection.")

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

    # ---- Seeds (identical to main.py) ---------------------------------------
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

    # ---- Protocol parameters ------------------------------------------------
    relay_cap = math.ceil(
        args.relay_capacity_factor * args.num_leaves * args.b_rl / args.num_relays
    )
    b_rl_max = math.ceil(args.leaf_retry_factor * args.b_rl)

    expected_crashes = args.crash_prob * (
        args.num_relays if args.crash_type == "relay" else args.num_leaves)

    print(f"[Info] crash_type={args.crash_type}  "
          f"crash_prob={args.crash_prob:.2f}  "
          f"expected_crashes={expected_crashes:.1f} per round")
    print(f"[Info] relay_cap={relay_cap} per relay per round  "
          f"(factor={args.relay_capacity_factor}x  "
          f"expected_load={args.num_leaves * args.b_rl / args.num_relays:.1f})")
    print(f"[Info] b_rl_max={b_rl_max} max attempts per leaf  "
          f"(factor={args.leaf_retry_factor}x  target={args.b_rl})")

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
        "n_crashed_per_round": [],        # actual crash count each eval round
    }
    if args.monitor_model_drift:
        metrics["pre_drift"]  = []
        metrics["post_drift"] = []

    # ---- Degree-zero and partial trackers -----------------------------------
    relay_s1_zero_counts   = []
    leaf_s3_zero_counts    = []
    leaf_s3_partial_counts = []
    actual_crash_counts    = []   # actual number crashed each round (varies)

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

            # --- Binomial independent crash sampling -------------------------
            # Each node independently crashes with probability crash_prob.
            # Number of crashes ~ Binomial(n_total, crash_prob) — varies each round.
            if args.crash_type == "relay":
                crashed_relays = set(
                    r for r in range(args.num_relays)
                    if random.random() < args.crash_prob
                )
                crashed_leaves = set()
                actual_crash_counts.append(len(crashed_relays))
            else:
                crashed_relays = set()
                crashed_leaves = set(
                    l for l in range(args.num_leaves)
                    if random.random() < args.crash_prob
                )
                actual_crash_counts.append(len(crashed_leaves))

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
            # Crashed relays skipped entirely — frozen state.
            # Alive relays sample b_lr leaves from full pool, drop crashed.
            # =================================================================
            rnd_s1_zeros = 0

            for relay_id in alive_relays:
                if args.b_lr <= args.num_leaves:
                    sampled = torch.randperm(
                        args.num_leaves,
                        device=aggregator_device)[:args.b_lr]
                else:
                    sampled = torch.randint(
                        0, args.num_leaves,
                        (args.b_lr,), device=aggregator_device)

                alive_sampled = [int(l) for l in sampled
                                 if int(l) not in crashed_leaves]

                if len(alive_sampled) == 0:
                    rnd_s1_zeros += 1
                else:
                    chosen = torch.tensor(
                        alive_sampled, device=aggregator_device)
                    relay_states[relay_id] = \
                        node_states[chosen].mean(dim=0).detach()

            relay_s1_zero_counts.append(rnd_s1_zeros)

            # =================================================================
            # Stage 2: relay <-> relay gossip
            # Only alive relays participate. Crashed relays fully excluded.
            # =================================================================
            if len(alive_relays) > 1:
                alive_relay_tensor = relay_states[alive_relays]
                effective_b_rr = min(args.b_rr, len(alive_relays) - 1)
                updated_alive, _ = aggregation.p2p_local_aggregation(
                    alive_relay_tensor, effective_b_rr, return_W=True)
                updated_alive = updated_alive.detach()
                for idx, relay_id in enumerate(alive_relays):
                    relay_states[relay_id] = updated_alive[idx]

            # =================================================================
            # Stage 3: relay -> leaf (retry-with-cap, coordinator-free)
            #
            # Relay contact counters reset each round.
            # Leaves processed in random shuffled order.
            # Each leaf probes relays in random order from full pool.
            # Relay unavailable if: crashed OR contact_count >= relay_cap.
            # Leaf stops when b_rl collected OR b_rl_max attempts exhausted.
            # =================================================================
            rnd_s3_zeros    = 0
            rnd_s3_partials = 0

            relay_contact_count = {r: 0 for r in range(args.num_relays)}

            leaf_order = list(range(args.num_leaves))
            random.shuffle(leaf_order)

            for leaf_id in leaf_order:
                probe_order = random.sample(
                    range(args.num_relays), args.num_relays)

                collected = []
                attempts  = 0

                for relay_id in probe_order:
                    if attempts >= b_rl_max:
                        break
                    if len(collected) >= args.b_rl:
                        break

                    attempts += 1

                    if relay_id in crashed_relays:
                        continue
                    if relay_contact_count[relay_id] >= relay_cap:
                        continue

                    collected.append(relay_states[relay_id])
                    relay_contact_count[relay_id] += 1

                if len(collected) == 0:
                    rnd_s3_zeros += 1
                    # leaf keeps its own post-training model
                else:
                    if len(collected) < args.b_rl:
                        rnd_s3_partials += 1
                    stacked = torch.stack(collected, dim=0)
                    node_states[leaf_id] = stacked.mean(dim=0).detach()

            leaf_s3_zero_counts.append(rnd_s3_zeros)
            leaf_s3_partial_counts.append(rnd_s3_partials)

            # Post-aggregation drift
            if args.monitor_model_drift and (rnd + 1) % args.eval_time == 0:
                metrics["post_drift"].append(compute_model_drift(node_states))

            # -----------------------------------------------------------------
            # Evaluation (identical to main.py parallel branch)
            # -----------------------------------------------------------------
            if (rnd + 1) % args.eval_time == 0:
                n_crashed_this_round = actual_crash_counts[-1]
                metrics["n_crashed_per_round"].append(n_crashed_this_round)

                if args.num_workers > 0:
                    print(f"[Round {rnd+1}] Starting Parallel Evaluation...")
                    payload = [w.detach().cpu() for w in node_states]
                    results = worker_pool.map(eval_worker.evaluate_node, payload)

                    g_losses, g_accs = zip(*results)
                    metrics["round"].append(rnd + 1)
                    metrics["global_acc"].append(list(g_accs))
                    metrics["global_loss"].append(list(g_losses))

                    print(f"[Round {rnd+1}] "
                          f"crash_prob={args.crash_prob:.2f} "
                          f"({args.crash_type}) "
                          f"Acc:[{min(g_accs):.4f},{max(g_accs):.4f}] "
                          f"crashed={n_crashed_this_round} "
                          f"S1-zeros={rnd_s1_zeros} "
                          f"S3-zeros={rnd_s3_zeros} "
                          f"S3-partial={rnd_s3_partials}")
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

    # ---- Summary statistics -------------------------------------------------
    total_rounds = len(relay_s1_zero_counts)

    if total_rounds > 0:
        avg_crashed     = sum(actual_crash_counts)    / total_rounds
        avg_s1_zeros    = sum(relay_s1_zero_counts)   / total_rounds
        avg_s3_zeros    = sum(leaf_s3_zero_counts)    / total_rounds
        avg_s3_partials = sum(leaf_s3_partial_counts) / total_rounds

        n_total = args.num_relays if args.crash_type == "relay" else args.num_leaves
        avg_alive = n_total - avg_crashed

        frac_s1     = avg_s1_zeros    / max(avg_alive, 1)
        frac_s3z    = avg_s3_zeros    / args.num_leaves
        frac_s3p    = avg_s3_partials / args.num_leaves

        print(f"\n{'='*60}")
        print(f"Retry-with-cap summary over {total_rounds} rounds")
        print(f"  crash_type={args.crash_type}  "
              f"crash_prob={args.crash_prob:.2f}  "
              f"relay_cap={relay_cap}  "
              f"b_rl_max={b_rl_max}")
        print(f"\n  Actual crashes per round:")
        print(f"    avg={avg_crashed:.2f}  "
              f"min={min(actual_crash_counts)}  "
              f"max={max(actual_crash_counts)}  "
              f"(expected={args.crash_prob * n_total:.1f})")
        print(f"\n  Stage 1 — alive relays with 0 alive leaves:")
        print(f"    avg/round={avg_s1_zeros:.2f} / {avg_alive:.1f} "
              f"alive relays  ({100*frac_s1:.1f}%)")
        print(f"\n  Stage 3 — leaves with 0 models collected (hard zero):")
        print(f"    avg/round={avg_s3_zeros:.2f} / {args.num_leaves} "
              f"leaves  ({100*frac_s3z:.1f}%)")
        print(f"\n  Stage 3 — leaves with partial collection (0 < n < b_rl):")
        print(f"    avg/round={avg_s3_partials:.2f} / {args.num_leaves} "
              f"leaves  ({100*frac_s3p:.1f}%)")
        print(f"{'='*60}")

        metrics["relay_cap"]             = relay_cap
        metrics["b_rl_max"]              = b_rl_max
        metrics["avg_crashed_per_round"] = avg_crashed
        metrics["avg_relay_s1_zeros"]    = avg_s1_zeros
        metrics["avg_leaf_s3_zeros"]     = avg_s3_zeros
        metrics["avg_leaf_s3_partials"]  = avg_s3_partials
        metrics["frac_relay_s1_zeros"]   = frac_s1
        metrics["frac_leaf_s3_zeros"]    = frac_s3z
        metrics["frac_leaf_s3_partials"] = frac_s3p

    # ---- Save (identical to main.py) ----------------------------------------
    with open(filename + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {filename}_metrics.json")


if __name__ == "__main__":
    main()