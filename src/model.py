import glob
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_bn_relu(3, 32, kernel_size=5)
        self.enc2 = self.conv_bn_relu(32, 64, kernel_size=3, pool_kernel=4)
        self.enc3 = self.conv_bn_relu(64, 128, kernel_size=3, pool_kernel=2)
        self.enc4 = self.conv_bn_relu(128, 256, kernel_size=3, pool_kernel=2)
        
        self.dec1 = self.conv_bn_relu(256, 128, kernel_size=3, pool_kernel=-2)
        self.dec2 = self.conv_bn_relu(128 + 128, 64, kernel_size=3, pool_kernel=-2)
        self.dec3 = self.conv_bn_relu(64 + 64, 32, kernel_size=3, pool_kernel=-4)
        self.dec4 = nn.Sequential(
            nn.Conv2d(32 + 32, 3, kernel_size=5, padding=2),
            nn.Tanh()
        )

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None):
        layers = []
        if pool_kernel is not None:
            if pool_kernel > 0:
                layers.append(nn.AvgPool2d(pool_kernel))
            elif pool_kernel < 0:
                layers.append(nn.UpsamplingNearest2d(scale_factor=-pool_kernel))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) // 2))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        out = self.dec1(x4)
        out = self.dec2(torch.cat([out, x3], dim=1))
        out = self.dec3(torch.cat([out, x2], dim=1))
        out = self.dec4(torch.cat([out, x1], dim=1))
        return out


class Discriminator(nn.Module):
    def __init__(self, norm='batch', pool_kernel_size=[4, 2, 2, 2]):
        super().__init__()
        self.norm = norm
        self.conv1 = self.conv_bn_relu(3, 16, kernel_size=5, reps=1)
        self.conv2 = self.conv_bn_relu(16, 32, pool_kernel=pool_kernel_size[0], normalization=False)
        self.conv3 = self.conv_bn_relu(32, 64, pool_kernel=pool_kernel_size[1])
        self.conv4 = self.conv_bn_relu(64, 128, pool_kernel=pool_kernel_size[2])
        self.conv5 = self.conv_bn_relu(128, 256, pool_kernel=pool_kernel_size[3])
        self.out_patch = nn.Conv2d(256, 1, kernel_size=1)

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None, reps=2, normalization=True):
        layers = []
        for i in range(reps):
            if i == 0 and pool_kernel is not None:
                layers.append(nn.AvgPool2d(pool_kernel))
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch,
                                    out_ch, kernel_size, padding=(kernel_size - 1) // 2))
            if normalization:
                if self.norm=='batch':
                    layers.append(nn.BatchNorm2d(out_ch))
                elif self.norm=='instance':
                    layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out_patch(x)
        return x


def load_model(model, opt, sche, dir, epoch, device):
    path = glob.glob(f'{dir}/model_epoch{str(epoch).zfill(3)}.pth')[0]
    print(f'\nmodel load: {path}\n')
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    opt.load_state_dict(checkpoint['optimizer'])
    sche.load_state_dict(checkpoint['scheduler'])
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return model, opt, sche


def save_model(epoch, model, opt, sche, save_path):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': sche.state_dict(),
    }
    torch.save(state, f'{save_path}/model_epoch{str(epoch+1).zfill(3)}.pth')
