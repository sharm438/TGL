import torch
import utils as utils

# Global variables that persist in the worker process
_global_test_loader = None
_device = None
_net_name = None
_dims = None

def init_worker(dataset_name, net_name, inp_dim, out_dim, batch_size, gpu_id, fraction, num_leaves):
    """
    Initializes the worker process.
    Loads the global test set once so it persists for the process lifetime.
    """
    global _global_test_loader, _device, _net_name, _dims
    
    # 1. Setup Device
    if gpu_id >= 0 and torch.cuda.is_available():
        _device = torch.device(f"cuda:{gpu_id}")
    else:
        _device = torch.device("cpu")
        
    _net_name = net_name
    _dims = (inp_dim, out_dim)
    
    # 2. Load Data
    # We use load_data to get the TrainObject, which contains the test_data loader.
    # We assume utils.load_data returns (TrainObject, distributed_data, distributed_label)
    # or just TrainObject depending on the dataset. The TGL utils returns a tuple for most.
    print(f"[Worker] Initializing on {_device} for dataset {dataset_name}...")
    
    # Set lr=0.0 as it's not needed for loading
    loaded_stuff = utils.load_data(dataset_name, batch_size, lr=0.0, fraction=fraction, num_leaves=num_leaves)
    
    if isinstance(loaded_stuff, tuple):
        train_obj = loaded_stuff[0]
    else:
        train_obj = loaded_stuff
        
    _global_test_loader = train_obj.test_data

def evaluate_node(wts):
    """
    Evaluates a single node's weights on the global test set.
    
    Args:
        wts (torch.Tensor): The flattened model weights (on CPU).
    
    Returns:
        (float, float): (global_loss, global_accuracy)
    """
    # 1. Move weights to the worker's device
    wts = wts.to(_device)
    
    # 2. Reconstruct Model
    inp_dim, out_dim = _dims
    model = utils.vec_to_model(wts, _net_name, inp_dim, out_dim, _device)
    
    # 3. Evaluate using existing TGL utility
    loss, acc = utils.evaluate_global_metrics(model, _global_test_loader, _device)
    
    # Return floats to detach from graph/device
    return float(loss), float(acc)