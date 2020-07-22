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

<br>

## Output Image

### Water -> Grass

### Grass -> Water
