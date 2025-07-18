import os
import random
import torch
import numpy as np


def set_seed_everywhere(seed: int):
    print(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    print("Set torch seed")
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        print("Set torch cuda seed")
    np.random.seed(seed)
    print("Set numpy seed")
    random.seed(seed)
    print("Set random seed")
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("Set PYTHONHASHSEED")
