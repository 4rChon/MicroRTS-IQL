import random
import torch
import numpy as np


def set_seed_everywhere(seed: int):
    print(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    print(f"Set torch seed to {seed}")
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        print(f"Set torch cuda seed to {seed}")
    np.random.seed(seed)
    print(f"Set numpy seed to {seed}")
    random.seed(seed)
    print(f"Set random seed to {seed}")

    # os.environ["PYTHONHASHSEED"] = str(seed)
