Below is a sample **README** for this code base. It contains an overview of the purpose, lists the important command-line arguments (with emphasis on the “store_true” type arguments), and provides example runs for various aggregation modes—**Federated Learning** (`fedsgd`), **EL Oracle** (`p2p`), **EL Local** (`p2p_local`), and **Hubs-and-Spokes Learning** (`hsl`).

---

# README

## Overview

This repository implements a variety of distributed/federated learning paradigms with *non-IID* data distribution, including:

- **Federated Learning** (`fedsgd`): A traditional server-based approach where a global parameter server averages updates from the nodes (spokes).
- **Peer-to-Peer** (`p2p`): Decentralized baselines - Exponential Graph, BaseGraph, Erdos-Renyi, k-redular.
- **EL Local** (`p2p_local`): A truly distributed peer-to-peer method where each node randomly picks \(k\) neighbors per round (without a global graph).
- **Hubs-and-Spokes Learning** (`hsl`): A three-step process involving multiple hubs and spokes exchanging their models in different stages.

### Key Features

- **Non-IID data** distribution among spokes  
- **Multiple datasets** (MNIST, CIFAR-10)  
- **Flexible aggregator** choices (FL, P2P, P2P Local, HSL)  
- **Seed control** for reproducibility  
- **Optional** monitoring of model drift, node degrees, and graph simulation

---

## Installation

1. Install [PyTorch](https://pytorch.org/) (version >= 1.8 recommended).
2. Install torchvision or other dependencies if needed:
   ```bash
   pip install torchvision
   ```
3. Clone this repository and ensure you have a `python` environment with `numpy`, `argparse`, etc.

---

## Usage

All scripts can be run via:

```bash
python main.py [options]
```

### Core Arguments

| **Argument**                 | **Type**     | **Description**                                                                                                                                                                                                    |
|------------------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--exp`                      | `str`        | **Experiment name** (used for output JSON files, e.g., `<exp>_metrics.json`).                                                                                                                                      |
| `--dataset`                  | `str`        | Dataset to use: `mnist` or `cifar10`.                                                                                                                                                                              |
| `--fraction`                 | `float`      | Fraction of the dataset to use (e.g., `1.0` = entire training set, `0.5` = half).                                                                                                                                  |
| `--bias`                     | `float`      | Non-IIDness level. `0.0` → effectively IID, larger → more skewed distribution.                                                                                                                                     |
| `--aggregation`              | `str`        | Aggregation method: `fedsgd`, `p2p`, `p2p_local`, or `hsl`.                                                                                                                                                        |
| `--num_spokes`               | `int`        | Number of “spoke” nodes (clients).                                                                                                                                                                                 |
| `--num_hubs`                 | `int`        | Number of hubs (applies only to `hsl`).                                                                                                                                                                           |
| `--num_rounds`               | `int`        | Number of global communication rounds.                                                                                                                                                                            |
| `--num_local_iters`          | `int`        | Number of local gradient steps each node performs per round.                                                                                                                                                       |
| `--batch_size`               | `int`        | Local batch size on each node.                                                                                                                                                                                    |
| `--eval_time`                | `int`        | Evaluate metrics every `eval_time` rounds.                                                                                                                                                                        |
| `--gpu`                      | `int`        | GPU index to use (e.g. `0`). Use `-1` for CPU.                                                                                                                                                                     |
| `--lr`                       | `float`      | Learning rate used in local training.                                                                                                                                                                             |
| `--sample_type`              | `str`        | Local mini-batch sampling method: `round_robin` or `random`.                                                                                                                                                      |
| `--seed`                     | `int`        | Random seed for reproducibility. `<=0` means no fixed seed.                                                                                                                                                       |

#### Peer-to-Peer (P2P) Arguments

| **Argument**  | **Type** | **Description**                                                                                                   |
|---------------|----------|-------------------------------------------------------------------------------------------------------------------|
| `--k`         | `int`    | - **P2P** (`p2p`) uses a centrally generated \(k\)-regular graph.  <br>- **EL Local** (`p2p_local`) each node chooses \(k\) neighbors. |

#### Hubs-and-Spokes (HSL) Arguments

| **Argument**  | **Type** | **Description**                                                                                                                                                                                              |
|---------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--b_lr`      | `int`    | **Stage 1**: Each hub randomly selects `b_lr` spokes and averages them. (Hub “indegree” from spokes)                                                                                                         |
| `--b_rr`      | `int`    | **Stage 2**: Hubs mix with each other using an “EL Local” style aggregator with outdegree = `b_rr`.                                                                                                         |
| `--b_rl`      | `int`    | **Stage 3**: Each spoke randomly selects `b_rl` hubs to average (Spoke “indegree” from hubs).                                                                                                               |

### **store_true** Arguments

| **Argument**                | **Type**       | **Description**                                                                                                                                                                                                                |
|-----------------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--monitor_model_drift`     | `store_true`   | If set, computes the **model drift** among spokes (distance from the spoke models to their mean) **before** and **after** the aggregator step, each time we do an evaluation (`eval_time` intervals).                                                                 |
| `--monitor_degree`          | `store_true`   | If set, computes and logs node in/out degrees every round (e.g., how many neighbors chose each node in P2P or HSL’s stages). This is saved separately (e.g., `<exp>_degree.json`).                                               |
| `--graph_simulation_only`   | `store_true`   | If set, **no actual local training** is performed. Instead, the code only simulates the aggregator step (P2P / P2P Local / HSL) to obtain random mixing matrices each round. Useful for analyzing mixing properties, spectral gap, etc. |

---

## Example Runs

Below are example commands showing how to run the different modes:

1. **Federated Learning (FL / `fedsgd`):**
   ```bash
   python main.py \
       --exp fed_mnist \
       --dataset mnist \
       --fraction 1.0 \
       --bias 0.0 \
       --aggregation fedsgd \
       --num_spokes 10 \
       --num_rounds 50 \
       --num_local_iters 1 \
       --batch_size 32 \
       --eval_time 5 \
       --gpu -1 \
       --lr 0.01
   ```
   This runs standard federated averaging on MNIST with 10 spokes, using CPU (`--gpu -1`) for 50 rounds.

2. ** (P2P / `p2p`):**
   ```bash
   python main.py \
       --exp p2p_mnist \
       --dataset mnist \
       --aggregation p2p \
       --topo exponential \
       --num_spokes 10 \
       --num_rounds 50 \
       --num_local_iters 5 \
       --batch_size 32 \
       --eval_time 5 \
       --gpu 0 \
       --lr 0.01 \
       --bias 0.3
   ```
   - **EL Oracle** uses a globally generated *\(k\)-regular* matrix with `k=2`.  
   - Each round, the aggregator re-samples or reuses a random permutation to build that matrix.

3. **EL Local (P2P Local / `p2p_local`):**
   ```bash
   python main.py \
       --exp p2p_local_mnist \
       --dataset mnist \
       --aggregation p2p_local \
       --k 3 \
       --num_spokes 10 \
       --num_rounds 50 \
       --num_local_iters 2 \
       --batch_size 32 \
       --eval_time 5 \
       --gpu 0 \
       --lr 0.01 \
       --bias 0.5
   ```
   - **EL Local** has each node choose `k=3` random neighbors (plus itself) every round, with no global coordinator.

4. **TGL (Tiered Gossip Learning / `hsl`):**
   ```bash
   python main.py \
       --exp hsl_cifar \
       --dataset cifar10 \
       --aggregation hsl \
       --num_leaves 12 \
       --num_relays 3 \
       --b_lr 2 \
       --b_rr 1 \
       --b_rl 2 \
       --num_rounds 80 \
       --num_local_iters 2 \
       --batch_size 64 \
       --eval_time 10 \
       --gpu 0 \
       --lr 0.005 \
       --bias 0.4
   ```
   Here, each round has:
   1. **Stage 1** (`b_hs=2`): Relays each pick 2 spokes.
   2. **Stage 2** (`b_hh=1`): Relays mix among themselves (each picks 1 relay).
   3. **Stage 3** (`b_sh=2`): Leaves each pick 2 relays to update from.

---

## Additional Notes

- Use `--monitor_model_drift` to track and log how much node models deviate from each other each round.
- Use `--monitor_degree` to track and log how many connections each node has in P2P or each stage in HSL.
- For debugging or faster iteration, you can run **graph simulation only** by adding `--graph_simulation_only`. This **skips all data loading and local training** and only simulates the aggregator steps:
  ```bash
  python main.py \
      --exp sim_p2p \
      --aggregation p2p_local \
      --num_leaves 10 \
      --k 2 \
      --num_rounds 30 \
      --graph_simulation_only
  ```
  This will generate random mixing matrices, compute their average, spectral gap, etc., and save them to JSON, **without** doing any real training.

- **Reproducibility**: You can fix a seed for data partitioning and model initialization using `--seed <positive_int>`. For example:
  ```bash
  python main.py --seed 108 --aggregation p2p --k 2 ...
  ```

- **Output Files**:  
  - **`<exp>_metrics.json`**: Contains training/test metrics, spectral gap, average mixing matrix, etc., depending on aggregator.  
  - **`<exp>_degree.json`**: If `--monitor_degree` is set, this contains arrays of node in/out degrees per round.

---

## Contact / Contributing

- Contributions or pull requests are welcome!  
- For further issues or questions, please open an Issue on this repository.

Enjoy exploring **Federated**, **P2P**, and **Hubs-and-Spokes** learning methods in a single code base!
