"""Initialize random seeds and save the state needed to resume an epoch boundary."""
import json
import os
import random

import numpy as np
import torch


def setup(seed, threads):
    """Seed Python, NumPy and PyTorch and select deterministic numerical kernels."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(_):
    """Give each DataLoader worker reproducible Python and NumPy random streams."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def save_rng(train_loader, val_loader, device):
    """Capture image-augmentation, model, and DataLoader random states."""
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                train=train_loader.generator.get_state(), val=val_loader.generator.get_state())


def restore_rng(state, train_loader, val_loader, device):
    """Continue from the same random streams when restoring last.pt."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)
    train_loader.generator.set_state(state['train'])
    val_loader.generator.set_state(state['val'])


def save_json(path, value):
    """Write readable numeric output next to the run's checkpoints."""
    path.write_text(json.dumps(value, indent=2) + '\n')
