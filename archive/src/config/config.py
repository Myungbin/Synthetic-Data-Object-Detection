import os
import random

import numpy as np
import torch


class Config:
    TRAIN_PATH = "./data/train"
    TEST_PATH = "./data/test"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASS = 34
    IMG_SIZE = 800  # (1300, 800)
    EPOCH = 10
    LR = 3e-4
    BATCH_SIZE = 8
    SEED = 1103


cfg = Config()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
