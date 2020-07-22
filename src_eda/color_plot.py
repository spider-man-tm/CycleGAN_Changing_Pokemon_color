import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import cv2

sys.path.append('../config')
from config.config import Config

DATA_DIR = 'data'
config = Config()

PLOT_COLOR_PATHES = config.plot_color_pathes
PLOT_COLOR_PATHES = [os.path.join(DATA_DIR, p) for p in PLOT_COLOR_PATHES]
PLOT_HIST_PATHES = config.plot_hist_pathes
PLOT_HIST_PATHES = [os.path.join(DATA_DIR, p) for p in PLOT_HIST_PATHES]
FIGURE_DIR = 'eda'

def get_img_mean_helper(img):
    a, b, c = cv2.split(img)
    a = a[np.nonzero(a)]
    b = b[np.nonzero(b)]
    c = c[np.nonzero(c)]
    a = a[np.nonzero(-1 * (a-255))]
    b = b[np.nonzero(-1 * (b-255))]
    c = c[np.nonzero(-1 * (c-255))]
    a_mean, b_mean, c_mean = a.mean(), b.mean(), c.mean()
    
    return (a_mean, b_mean, c_mean)


def get_img_mean(img_dir):
    img_names = sorted(os.listdir(img_dir))
    h_values, s_values, v_values = [], [], []
    r_values, g_values, b_values = [], [], []
    
    for name in img_names:
        path = os.path.join(img_dir, name)
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        r, g, b = get_img_mean_helper(img_rgb)
        
        r_values.append(r)
        g_values.append(g)
        b_values.append(b)
        
    return (r_values, g_values, b_values)


def create_pixel_color(r, g, b):
    pixel_color = np.zeros((r.shape[0], 3))
    pixel_color[:, 0] = r
    pixel_color[:, 1] = g
    pixel_color[:, 2] = b
    
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_color)
    pixel_color = norm(pixel_color).tolist()
    
    return pixel_color
        

def rgb_3d_plot(img_dir, title):
    rgb = get_img_mean(img_dir)
    
    r = np.array(rgb[0])
    g = np.array(rgb[1])
    b = np.array(rgb[2])

    pixel_colors = create_pixel_color(r, g, b)

    fig = plt.figure(figsize=(7, 7))
    axis = fig.add_subplot(1, 1, 1, projection='3d', zorder=0)
    
    axis.scatter(r, g, b, facecolors=pixel_colors, marker='o', zorder=1)
    axis.set_xlabel('Red')
    axis.set_ylabel('Green')
    axis.set_zlabel('Blue')
    axis.set_xlim([0, 255])
    axis.set_ylim([0, 255])
    axis.set_zlim([0, 255])
    axis.text(5, 5, 255, f'R mean: {r.mean():.1f}', None, zorder=2)
    axis.text(5, 5, 230, f'G mean: {g.mean():.1f}', None, zorder=2)
    axis.text(5, 5, 205, f'B mean: {b.mean():.1f}', None, zorder=2)
    
    plt.title(f'{title}')
    plt.savefig(f'{FIGURE_DIR}/rgb_plot_{title}.png', bbox_inches='tight')
    plt.close()


def color_hist(path):
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) 
    sns.set()
    sns.set_style(style='ticks')
    
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)

    ab_2d = img_lab[:, :, 1:3].reshape(-1, 2)
    a = ab_2d[:, 0]
    b = ab_2d[:, 1]

    aedges = np.arange(0, 255)
    bedges = np.arange(0, 255)

    H, xedges, yedges = np.histogram2d(a, b, bins=(aedges, bedges))
    H = H.T

    H_log = np.log(H + 1)
    H_norm = H_log / H_log.max() * 255

    x = H_norm.shape[1]
    y = H_norm.shape[0]
    a_xy = np.repeat(xedges[:-1], y).reshape(x, y).T
    b_xy = np.repeat(yedges[:-1], x).reshape(y, x)

    ab_hist = np.dstack((H_norm, a_xy, b_xy)).astype('uint8')
    ab_hist_im = cv2.cvtColor(ab_hist, cv2.COLOR_Lab2RGB)
    ab_hist_im = cv2.resize(ab_hist_im, (255, 255))
    
    fig = plt.figure(figsize=[9, 4])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(img_rgb)
    ax1.set_title('image')

    ax2.set_title('a*b* Histgram')
    ax2.set_xlabel('G ← a* → R')
    ax2.set_ylabel('B ← b* → Y')
    ax2.axvline(255 / 2, color='white', linestyle='solid', linewidth=1)
    ax2.axhline(255 / 2, color='white', linestyle='solid', linewidth=1)
    ax2.imshow(ab_hist_im, origin='low', interpolation='bicubic')

    _, dirname, fname = path.split('/')
    plt.savefig(f'{FIGURE_DIR}/color_hist_{dirname}_{fname}.png', bbox_inches='tight')
    plt.close()


def main():
    for p in PLOT_COLOR_PATHES:
        _, title = p.split('/')
        rgb_3d_plot(img_dir=p, title=f'{title} RGB plot')
    for p in PLOT_HIST_PATHES:
        color_hist(p)


if __name__ == '__main__':
    main()
