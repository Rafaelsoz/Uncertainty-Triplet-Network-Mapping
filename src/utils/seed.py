import random
import numpy as np

from torch import manual_seed, cuda

def define_seed(seed: int):
    manual_seed(seed)
    cuda.manual_seed_all(seed)
    cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)