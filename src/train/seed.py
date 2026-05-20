# Global seeding for reproducibility across all random sources.

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic ops -> small overhead
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
