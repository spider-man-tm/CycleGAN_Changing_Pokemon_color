Cycle GAN -Changing Pokemon's color-
====

<br>

## Usage
```
# setup
# kaggle/python-gpu-build image build.
# https://github.com/Kaggle/docker-python

# Launch the original docker container
# based on the official kaggle docker image.

$ docker build -t pokemon .
$ docker run -v `pwd`:/pokemon -p 8888:8888 -d --name pokemon --restart=always --gpus all pokemon
$ docker exec -it pokemon /bin/bash

# set config.yml

# output eda directory
$ sh eda.sh

# start train & valid
$ sh train.sh
```

<br>

## Directory
```
# checkpoint -> saved model’s directory.
# output/figure -> loss & learning rate figure
# output/log -> loss & score
# output/pred_val_a_to_b -> output images (domain_A -> domain_B)
# output/pred_val_b_to_a -> output images (domain_B -> domain_A)

/pokemon/
　├ checkpoint/
　│　├ {directoryname1}
　│　│　├ D
　│　│　│　├ num_epoch
　│　│　│　├  ...
　│　│　├ G
　│　├ {directoryname2}
　│　├  ...
　├ config/
　├ data/
　├ output/
　│　├ figure
　│　│　├ num_epoch
　│　│　├  ...
　│　├ log
　│　│　├ num_epoch
　│　│　├  ...
　│　└ pred_val_a_to_b
　│　│　├ num_epoch
　│　│　├  ...
　│　└ pred_val_b_to_a
　│　 　├ num_epoch
　│　 　├  ...
　├ src/
　├ src_eda/
　├ Dockerfile
　└ train.py
```

<br>

## EDA

![fig1](https://github.com/spider-man-tm/CycleGAN_Changing_Pokemon_color/blob/master/eda/color_hist_04_Grass_b_3.png.png)
![fig2](https://github.com/spider-man-tm/CycleGAN_Changing_Pokemon_color/blob/master/eda/color_hist_01_Fire_b_6.png.png)
![fig3](https://github.com/spider-man-tm/CycleGAN_Changing_Pokemon_color/blob/master/eda/color_hist_02_Water_b_9.png.png)
![fig4](https://github.com/spider-man-tm/CycleGAN_Changing_Pokemon_color/blob/master/eda/rgb_plot_04_Grass%20RGB%20plot.png)
![fig5](https://github.com/spider-man-tm/CycleGAN_Changing_Pokemon_color/blob/master/eda/rgb_plot_01_Fire%20RGB%20plot.png)
![fig6](https://github.com/spider-man-tm/CycleGAN_Changing_Pokemon_color/blob/master/eda/rgb_plot_02_Water%20RGB%20plot.png)
![fig7](https://github.com/spider-man-tm/CycleGAN_Changing_Pokemon_color/blob/master/eda/rgb_plot_03_Electric%20RGB%20plot.png)

<br>

## Output Image

### Water -> Grass
![fig8](https://github.com/spider-man-tm/readme_figure/blob/master/CycleGAN_Changing_Pokemon_color/w_to_g.png)

### Water -> Fire
![fig9](https://github.com/spider-man-tm/readme_figure/blob/master/CycleGAN_Changing_Pokemon_color/w_to_f.png)

### Grass -> Water
![fig10](https://github.com/spider-man-tm/readme_figure/blob/master/CycleGAN_Changing_Pokemon_color/g_to_w.png)

### Fire -> Water
![fig11](https://github.com/spider-man-tm/readme_figure/blob/master/CycleGAN_Changing_Pokemon_color/f_to_w.png)