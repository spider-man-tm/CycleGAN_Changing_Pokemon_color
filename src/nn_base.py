from .utils import path_conf

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch


def train(
    loader,
    G,
    F,
    Da,
    Db,
    GF_optimizer,
    Da_optimizer,
    Db_optimizer,
    criterion_gan_g,
    criterion_gan_d_real,
    criterion_gan_d_fake,
    criterion_cycle,
    criterion_identity,
    device,
    lambda_cycle,
    batch_size,
):
    torch.backends.cudnn.benchmark = True
    G_loss, D_loss = 0, 0
    G.train()
    F.train()
    Da.train()
    Db.train()

    for (A_real, B_real), (A_id, B_id) in tqdm(loader):
        # Train Generator
        A_real = A_real.to(device)
        B_real = B_real.to(device)
        B_fake = G(A_real)
        A_fake = F(B_real)
        B_rec = G(A_fake)
        A_rec = F(B_fake)
        
        Da_out_fake = Da(A_fake)
        Db_out_fake = Db(B_fake)
        
        A_fake_copy = A_fake.detach()
        B_fake_copy = B_fake.detach()

        Da_out_size_fake = Da_out_fake.size()
        Da_ones_fake = torch.ones(Da_out_size_fake[0], 1, Da_out_size_fake[2], Da_out_size_fake[3]).to(device)
        Da_zeros_fake = torch.zeros(Da_out_size_fake[0], 1, Da_out_size_fake[2], Da_out_size_fake[3]).to(device)

        Db_out_size_fake = Db_out_fake.size()
        Db_ones_fake = torch.ones(Db_out_size_fake[0], 1, Db_out_size_fake[2], Db_out_size_fake[3]).to(device)
        Db_zeros_fake = torch.zeros(Db_out_size_fake[0], 1, Db_out_size_fake[2], Db_out_size_fake[3]).to(device)

        # Generator Adversarial Loss
        G_adv_loss = criterion_gan_g(Db_out_fake, Db_ones_fake)
        F_adv_loss = criterion_gan_g(Da_out_fake, Da_ones_fake)
        Generator_adv_loss = (G_adv_loss + F_adv_loss) / 2

        # Generator Cycle Loss
        G_cycle_loss = criterion_cycle(A_rec, A_real)
        F_cycle_loss = criterion_cycle(B_rec, B_real)
        Generator_cycle_loss = (G_cycle_loss + F_cycle_loss) / 2

        # Generator Identity Loss
        G_identity_loss = criterion_identity(G(B_real), B_real)
        F_identity_loss = criterion_identity(F(A_real), A_real)
        Generator_identity_loss = (G_identity_loss + F_identity_loss) / 2

        # Genarator Loss Total
        Generator_loss = Generator_adv_loss + lambda_cycle * Generator_cycle_loss + 0.5 * lambda_cycle * Generator_identity_loss

        GF_optimizer.zero_grad()
        Generator_loss.backward()
        GF_optimizer.step()

        G_loss += Generator_loss.item()

        # Train Discriminator
        Da_out_real = Da(A_real)
        Da_out_fake2 = Da(A_fake_copy)
        Db_out_real = Db(B_real)
        Db_out_fake2 = Db(B_fake_copy)

        Da_out_size_real = Da_out_real.size()
        Da_ones_real = torch.ones(Da_out_size_real[0], 1, Da_out_size_real[2], Da_out_size_real[3]).to(device)

        Db_out_size_real = Db_out_real.size()
        Db_ones_real = torch.ones(Da_out_size_real[0], 1, Db_out_size_real[2], Db_out_size_real[3]).to(device)

        Da_loss_real = criterion_gan_d_real(Da_out_real, Da_ones_real)
        Da_loss_fake = criterion_gan_d_fake(Da_out_fake2, Da_zeros_fake)
        Db_loss_real = criterion_gan_d_real(Db_out_real, Db_ones_real)
        Db_loss_fake = criterion_gan_d_fake(Db_out_fake2, Db_zeros_fake)

        Discriminator_loss = (Da_loss_real + Da_loss_fake + Db_loss_real + Db_loss_fake) / 2
        D_loss += Discriminator_loss.item()
        
        Da_optimizer.zero_grad()
        Db_optimizer.zero_grad()
        Discriminator_loss.backward()
        Da_optimizer.step()
        Db_optimizer.step()

    return G_loss * batch_size, D_loss * batch_size


def valid_img_save(real, fake, ids, path, epoch):
    for r, f, fname in zip(real, fake, ids):
        r = r.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        f = np.where(f < 0, 0, f)
        f = np.where(f > 1, 1, f)
        r *= 255.
        f *= 255.
        r = np.squeeze(r).transpose((1, 2, 0)).astype(np.uint8)
        f = np.squeeze(f).transpose((1, 2, 0)).astype(np.uint8)
        
        pred_path = path + f'/{str(epoch+1).zfill(3)}'
        path_conf(pred_path)

        img = np.hstack([r, f])
        pilImg = Image.fromarray(np.uint8(img))
        pilImg.save(os.path.join(pred_path, fname + '.jpg'))


def valid(loader, G, F, a_to_b_dir, b_to_a_dir, device, epoch):
    torch.backends.cudnn.benchmark = True
    G.eval()
    F.eval()
    with torch.no_grad():
        for (A_real, B_real), (A_id, B_id) in tqdm(loader):
            A_real = A_real.to(device)
            B_real = B_real.to(device)
            
            B_fake = G(A_real)
            A_fake = F(B_real)

            valid_img_save(A_real, B_fake, A_id, a_to_b_dir, epoch)
            valid_img_save(B_real, A_fake, B_id, b_to_a_dir, epoch)
