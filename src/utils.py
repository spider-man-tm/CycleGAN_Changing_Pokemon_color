import os
import random
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def path_conf(path):
    if not os.path.exists(path):
        os.makedirs(path)


def df_to_log(df_path):
    log = []
    df = pd.read_csv(df_path)
    for i in range(df.shape[0]):
        log_epoch = {
            'epoch': df.iloc[i][0],
            'train_G_loss': df.iloc[i][1],
            'train_D_loss': df.iloc[i][2],
            'g_lr': df.iloc[i][3],
            'd_lr': df.iloc[i][4],
        }
        log.append(log_epoch)

    return log
