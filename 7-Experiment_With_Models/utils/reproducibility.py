import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

def set_global_seed(seed: int = 42):
    """Set random seed across python, numpy, torch, and tensorflow for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if tf is not None:
        try:
            # TF 2.x
            tf.random.set_seed(seed)
        except AttributeError:
            # Older TF
            tf.set_random_seed(seed) 