from config.config import Config
from src.dataset import augmentation, PokemonTrainDataset
# from src.loss import HingeLoss
from src.model import Generator, Discriminator, load_model, save_model
from src.scheduler import GradualWarmupScheduler
from src.nn_base import train, valid
from src.utils import (
    seed_everything,
    path_conf,
    df_to_log,
)

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import pytz
import itertools
import warnings

from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True, help='this name directory created.')
parser.add_argument('-f', '--fold', type=int, required=True, help='train fold.')
args = parser.parse_args()
No = args.name
FOLD = args.fold

config = Config()

# Argument Parser
No = args.name
FOLD = args.fold

# Variables that is changed every time
LOAD_MODEL_EPOCH = config.load_model_epoch
DATA_LOADER_SEED = config.data_loader_seed
N_EPOCHS = config.n_epochs

# Semi Regular Variables
DEBUG = config.debug
TOTAL_EPOCHS = config.total_epochs
WARMUP_EPOCHS = config.warmup_epochs
IMAGE_SIZE = config.image_size
MODEL_IMG_SAVE_EPOCH = config.model_img_save_epoch

# Regular Variables
BATCH_SIZE = config.batch_size
N_SPLIT = config.n_split
D_MAX_LR = config.d_max_lr
D_MIN_LR = config.d_min_lr
G_MAX_LR = config.g_max_lr
G_MIN_LR = config.g_min_lr
LAMBDA_CYCLE = config.lambda_cycle
SEED = config.seed
ROOT_DIR = config.root_dir

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# data directory
TRAIN_DATA_A_DIR = config.domain_a_dir
TRAIN_DATA_B_DIR = config.domain_b_dir

# save directory
MODEL_G_DIR = f'{ROOT_DIR}/checkpoint/G/{str(No).zfill(3)}'
MODEL_F_DIR = f'{ROOT_DIR}/checkpoint/F/{str(No).zfill(3)}'
MODEL_DA_DIR = f'{ROOT_DIR}/checkpoint/D1/{str(No).zfill(3)}'
MODEL_DB_DIR = f'{ROOT_DIR}/checkpoint/D2/{str(No).zfill(3)}'
LOG_DIR = f'{ROOT_DIR}/output/log/{str(No).zfill(3)}'
FIGURE_DIR = f'{ROOT_DIR}/output/figure/{str(No).zfill(3)}'
PRED_VAL_A_TO_B_DIR = f'{ROOT_DIR}/output/pred_val_a_to_b/{str(No).zfill(3)}'
PRED_VAL_B_TO_A_DIR = f'{ROOT_DIR}/output/pred_val_b_to_a/{str(No).zfill(3)}'

seed_everything(SEED)
kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)
img_ids_A = np.array(sorted(os.listdir(TRAIN_DATA_A_DIR)))
tr_ix, va_ix = list(kf.split(img_ids_A, img_ids_A))[FOLD]
train_A, valid_A = img_ids_A[tr_ix], img_ids_A[va_ix]

img_ids_B = np.array(sorted(os.listdir(TRAIN_DATA_B_DIR)))
train_B, valid_B = img_ids_B, img_ids_B

seed_everything(DATA_LOADER_SEED)

if DEBUG:
    train_A = train_A[:2]
    valid_A = valid_A[:2]

train_dataset = PokemonTrainDataset(
    train_A,
    train_B,
    TRAIN_DATA_A_DIR,
    TRAIN_DATA_B_DIR,
    IMAGE_SIZE,
    transform=augmentation(mode='train')
)
valid_dataset = PokemonTrainDataset(
    valid_A,
    valid_B,
    TRAIN_DATA_A_DIR,
    TRAIN_DATA_B_DIR,
    IMAGE_SIZE,
    transform=augmentation(mode='valid')
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

mkdirs = [
    MODEL_G_DIR,
    MODEL_F_DIR,
    MODEL_DA_DIR,
    MODEL_DB_DIR,
    LOG_DIR,
    FIGURE_DIR,
    PRED_VAL_A_TO_B_DIR,
    PRED_VAL_B_TO_A_DIR,
]

for p in mkdirs:
    path_conf(p)

if LOAD_MODEL_EPOCH:
    logs = df_to_log(os.path.join(LOG_DIR, 'log.csv'))
else:
    logs = []

G = Generator()
F = Generator()
Da = Discriminator()
Db = Discriminator()

GF_optimizer = optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=G_MAX_LR, betas=(0.5, 0.999))
GF_scheduler_cos = CosineAnnealingWarmRestarts(GF_optimizer, T_0=TOTAL_EPOCHS, T_mult=1, eta_min=G_MIN_LR)
GF_scheduler = GradualWarmupScheduler(GF_optimizer, multiplier=1, total_epoch=WARMUP_EPOCHS, after_scheduler=GF_scheduler_cos)

Da_optimizer = optim.Adam(Da.parameters(), lr=D_MAX_LR, betas=(0.5, 0.999))
Da_scheduler_cos = CosineAnnealingWarmRestarts(Da_optimizer, T_0=TOTAL_EPOCHS, T_mult=1, eta_min=D_MIN_LR)
Da_scheduler = GradualWarmupScheduler(Da_optimizer, multiplier=1, total_epoch=WARMUP_EPOCHS, after_scheduler=Da_scheduler_cos)

Db_optimizer = optim.Adam(Db.parameters(), lr=D_MAX_LR, betas=(0.5, 0.999))
Db_scheduler_cos = CosineAnnealingWarmRestarts(Db_optimizer, T_0=TOTAL_EPOCHS, T_mult=1, eta_min=D_MIN_LR)
Db_scheduler = GradualWarmupScheduler(Db_optimizer, multiplier=1, total_epoch=WARMUP_EPOCHS, after_scheduler=Db_scheduler_cos)

if LOAD_MODEL_EPOCH:
    G, GF_optimizer, GF_scheduler = load_model(G, GF_optimizer, GF_scheduler, MODEL_G_DIR, LOAD_MODEL_EPOCH, DEVICE)
    F, GF_optimizer, GF_scheduler = load_model(F, GF_optimizer, GF_scheduler, MODEL_F_DIR, LOAD_MODEL_EPOCH, DEVICE)
    Da, Da_optimizer, Da_scheduler = load_model(Da, Da_optimizer, Da_scheduler, MODEL_DA_DIR, LOAD_MODEL_EPOCH, DEVICE)
    Db, Db_optimizer, Db_scheduler = load_model(Db, Db_optimizer, Db_scheduler, MODEL_DB_DIR, LOAD_MODEL_EPOCH, DEVICE)

G.to(DEVICE)
F.to(DEVICE)
Da.to(DEVICE)
Db.to(DEVICE)

# criterion_gan_g = HingeLoss('gen')
# criterion_gan_d_real = HingeLoss('dis_real')
# criterion_gan_d_fake = HingeLoss('dis_fake')
criterion_gan_g = nn.BCEWithLogitsLoss()
criterion_gan_d_real = nn.BCEWithLogitsLoss()
criterion_gan_d_fake = nn.BCEWithLogitsLoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()


def main():
    num_train_imgs = len(train_dataset)

    for epoch in range(N_EPOCHS):
        epoch += LOAD_MODEL_EPOCH
        now_time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        start_time = time.time()

        GF_scheduler.step()
        Da_scheduler.step()
        Db_scheduler.step()

        print('-----------------------------------------------------------')
        print(f'Epoch {epoch + 1} / {N_EPOCHS + LOAD_MODEL_EPOCH}')
        print(f'Time: {now_time}')
        print('-----------------------------------------------------------\n')

        g_loss, d_loss = train(
            train_loader, G, F, Da, Db, GF_optimizer, Da_optimizer, Db_optimizer,
            criterion_gan_g, criterion_gan_d_real, criterion_gan_d_fake,
            criterion_cycle, criterion_identity, DEVICE, LAMBDA_CYCLE, BATCH_SIZE
        )

        if (epoch + 1) % MODEL_IMG_SAVE_EPOCH == 0 or epoch == 0:
            valid(valid_loader, G, F, PRED_VAL_A_TO_B_DIR, PRED_VAL_B_TO_A_DIR, DEVICE, epoch)

        g_loss /= num_train_imgs
        d_loss /= num_train_imgs

        end_time = time.time()
        print(f'epoch: {epoch+1}')
        print(f'time: {(end_time - start_time):.3f}sec.\n')

        print(f'train_G_loss: {g_loss:.5f}')
        print(f'train_D_loss: {d_loss:.5f}\n')

        g_lr = GF_scheduler.get_lr()[0]
        print(f'G lr: {g_lr:.9f}')
        d_lr = Da_scheduler.get_lr()[0]
        print(f'D lr: {d_lr:.9f}\n\n')

        log_epoch = {
            'epoch': epoch + 1,
            'train_G_loss': g_loss,
            'train_D_loss': d_loss,
            'g_lr': g_lr,
            'd_lr': d_lr,
        }
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(f'{LOG_DIR}/log.csv', index=False)

        if (epoch + 1) % MODEL_IMG_SAVE_EPOCH == 0 or epoch == 0:
            save_model(epoch, G, GF_optimizer, GF_scheduler, MODEL_G_DIR)
            save_model(epoch, F, GF_optimizer, GF_scheduler, MODEL_F_DIR)
            save_model(epoch, Da, Da_optimizer, Da_scheduler, MODEL_DA_DIR)
            save_model(epoch, Db, Db_optimizer, Db_scheduler, MODEL_DB_DIR)

        df = pd.read_csv(f'{LOG_DIR}/log.csv')

        plt.figure()
        plt.plot(df['train_D_loss'], label='Discriminator Loss', color='blue')
        plt.plot(df['train_G_loss'], label='Generator Loss', color='red')
        plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0)
        plt.savefig(f'{FIGURE_DIR}/loss.png', bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(df['d_lr'], label='Discriminator LR', color='blue')
        plt.plot(df['g_lr'], label='Generator LR', color='red')
        plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0)
        plt.savefig(f'{FIGURE_DIR}/lr.png', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
