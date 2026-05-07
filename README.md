# Beyond Flat Gossip: Relay Gossip Learning for Scalable Collaborative AI

**TGL** (Relay Gossip Learning) is a communication-efficient decentralized collaborative learning framework. It introduces a two-tier push–gossip–pull protocol where resource-constrained leaf nodes communicate through a small relay layer, achieving strong global mixing without increasing the per-leaf communication burden as the network scales.

> **Paper:** *Beyond Flat Gossip: Relay Gossip Learning for Scalable Collaborative AI*  
> Under review at NeurIPS 2026.

---

## Repository Structure

```
main.py                  # Entry point — training, evaluation, graph simulation
aggregation.py           # Federated, P2P, ELL, TGL, HSL, Teleportation aggregation
train_node.py            # Inline local SGD per node
eval_worker.py           # Parallel evaluation worker (multiprocessing)
models.py                # ResNet-20, FEMNIST-CNN, SmallTransformer, LeNet, ViT
utils.py                 # Data loading, Dirichlet distribution, graph utilities
base_graph.py            # BaseGraph topology (structured k-peer mixing matrices)
simple_base_graph.py     # HyperHyperCube and SimpleBaseGraph topologies
dynamic_graph.py         # DynamicGraph base class (rotating mixing matrices)
fault_tolerance_exp.py   # Crash-aware TGL-FT experiment with retry-cap protocol
requirements.txt         # Python dependencies
```

---

## Installation

```bash
conda create -n tgl python=3.10 -y
conda activate tgl
pip install -r requirements.txt
```

**Key dependencies** (fill in exact versions from `requirements.txt`):

| Package | Version |
|---|---|
| torch | 2.3.1+cu121 |
| torchvision | 0.18.1 |
| torchtext | 0.18.0 |
| torchdata | 0.9.0 |
| torchaudio | 2.3.1+cu121 |
| numpy | 1.26.3 |
| portalocker | 3.1.1 |
| sympy | (see requirements.txt) |
| tqdm | (see requirements.txt) |

---

## Arguments

### Core

| Argument | Description |
|---|---|
| `--dataset` | `mnist`, `cifar10`, `femnist`, `agnews` |
| `--aggregation` | `fedsgd`, `p2p`, `p2p_local`, `tgl` |
| `--num_leaves` | Number of leaf (worker) nodes |
| `--bias` | Dirichlet concentration parameter α for non-IID split (lower = more heterogeneous) |
| `--lr` | Learning rate |
| `--num_local_iters` | Local SGD steps per round |
| `--num_rounds` | Total communication rounds |
| `--eval_time` | Evaluate every N rounds |
| `--num_workers` | Parallel eval workers (0 = sequential) |
| `--seed` | Random seed (108 used in all paper experiments) |
| `--exp` | Output name — results saved to `outputs/<exp>_metrics.json` |
| `--monitor_model_drift` | Log pre/post-aggregation L2 weight drift |
| `--graph_simulation_only` | Skip training; simulate graph and report spectral gap and degree statistics |

### TGL (`--aggregation tgl`)

| Argument | Description |
|---|---|
| `--num_relays` | Number of relay nodes |
| `--b_lr` | Leaves sampled per relay in Stage 1 (leaf-to-relay budget) |
| `--b_rr` | Relay-to-relay neighbours in Stage 2 |
| `--b_rl` | Relays sampled per leaf in Stage 3 (relay-to-leaf budget) |

### ELL (`--aggregation p2p_local`)

| Argument | Description |
|---|---|
| `--k` | Random neighbours per node per round |

### P2P fixed topology (`--aggregation p2p`)

| Argument | Description |
|---|---|
| `--topo` | `ring`, `torus`, `erdos-renyi`, `base-graph`, `exponential`, `hsl`, `teleportation` |
| `--budget` | Edge count for Erdős–Rényi graphs |
| `--k` | Degree for `base-graph` or `exponential` |
| `--num_relays` | Hub count for `hsl` topology |
| `--hub_degree` | Out-degree for hub nodes in `hsl` |
| `--spoke_degree` | Out-degree for spoke nodes in `hsl` |
| `--k_teleport` | Active nodes per round for `teleportation` |

---

## Reproducing Paper Results

All paper experiments use `--seed 108`. Outputs are written to `outputs/`.

### TGL — CIFAR-10, 100 nodes (G3 configuration, 400 directed edges)

```bash
python main.py --monitor_model_drift \
    --dataset cifar10 --aggregation tgl \
    --num_leaves 100 --num_relays 20 \
    --b_lr 10 --b_rr 5 --b_rl 1 \
    --lr 0.1 --bias 0.1 --num_local_iters 5 \
    --num_rounds 1000 --eval_time 10 --num_workers 10 \
    --gpu 0 --seed 108 \
    --exp tgl_cifar10_s100h20_G3
```

### ELL — CIFAR-10, 100 nodes (matched budget, k=4, 400 directed edges)

```bash
python main.py --monitor_model_drift \
    --dataset cifar10 --aggregation p2p_local --k 4 \
    --num_leaves 100 \
    --lr 0.1 --bias 0.1 --num_local_iters 5 \
    --num_rounds 1000 --eval_time 10 --num_workers 10 \
    --gpu 0 --seed 108 \
    --exp ell_cifar10_s100_k4
```

### FedSGD — CIFAR-10, 100 nodes

```bash
python main.py --monitor_model_drift \
    --dataset cifar10 --aggregation fedsgd \
    --num_leaves 100 \
    --lr 0.1 --bias 0.1 --num_local_iters 5 \
    --num_rounds 1000 --eval_time 10 --num_workers 10 \
    --gpu 0 --seed 108 \
    --exp fl_cifar10_s100
```

### TGL — FEMNIST, 175 nodes (G3 configuration, 885 directed edges)

```bash
python main.py --monitor_model_drift \
    --dataset femnist --aggregation tgl \
    --num_leaves 175 --num_relays 15 \
    --b_lr 20 --b_rr 4 --b_rl 3 \
    --lr 0.02 --num_local_iters 3 \
    --num_rounds 1000 --eval_time 10 --num_workers 10 \
    --gpu 0 --seed 108 \
    --exp tgl_femnist_s175h15_G3
```

---

## Fault Tolerance Experiments (TGL-FT)

The crash-aware variant (`fault_tolerance_exp.py`) models each relay independently failing with probability `crash_prob` per round and derives protocol parameters (relay capacity cap and leaf retry budget) analytically from the crash probability.

```bash
# 20% expected relay crash rate, G6 config (700 edges)
python fault_tolerance_exp.py \
    --dataset cifar10 --num_leaves 100 \
    --num_relays 20 --b_lr 15 --b_rr 10 --b_rl 2 \
    --crash_type relay --crash_prob 0.2 \
    --lr 0.1 --bias 0.1 --num_local_iters 5 \
    --num_rounds 1000 --eval_time 10 --num_workers 10 \
    --monitor_model_drift --gpu 0 --seed 108 \
    --exp ft_relay_p20_cifar10_s100h20_G6
```

---

## Graph Simulation

To profile spectral gap and degree statistics without running training:

```bash
# TGL spectral gap simulation
python main.py --graph_simulation_only \
    --aggregation tgl --num_leaves 100 --num_relays 20 \
    --b_lr 10 --b_rr 5 --b_rl 1 \
    --num_rounds 1000 --gpu 0 --seed 108 \
    --exp sim_tgl_G3

# HSL (heterogeneous P2P baseline) spectral gap
python main.py --graph_simulation_only \
    --aggregation p2p --topo hsl \
    --num_leaves 100 --num_relays 20 \
    --hub_degree 10 --spoke_degree 2 \
    --num_rounds 1000 --gpu 0 --seed 108 \
    --exp sim_hsl_G3
```

---

## Datasets and Training Configurations

| Dataset | Model | Params | Nodes | lr | Local steps | Batch | Rounds |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | ResNet-20 | 0.27M | 100, 200 | 0.1 | 5 | 128 | 1000 |
| FEMNIST | CNN | 6.6M | 175, 350 | 0.02 | 3 | 32 | 1000 |
| AG News | Tiny Transformer | 12.9M | 100, 200 | 0.04 | 4 | 64 | 1000 |

FEMNIST uses natural writer-level non-IID structure. CIFAR-10 and AG News use a Dirichlet concentration parameter of α=0.1 for 100-node experiments and α=0.5 for 200-node experiments, with uniform sample counts enforced across nodes to isolate label heterogeneity from data imbalance.

---

## Output Format

Each run writes `outputs/<exp>_metrics.json` with keys `round`, `global_acc` (list of per-node accuracies at each eval round), `global_loss`, `pre_drift`, `post_drift` (if `--monitor_model_drift`). Fault tolerance runs additionally include `avg_relay_s1_zeros`, `avg_leaf_s3_zeros`, `avg_leaf_s3_partials`, and `relay_cap`.

---
