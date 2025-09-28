import os
import random
import numpy as np
import pandas as pd

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def makedirs(path):
    os.makedirs(path, exist_ok=True)
