FROM kaggle/python-gpu-build
ENV NVIDIA_VISIBLE_DEVICES all
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
EXPOSE 8888
RUN mkdir -p /pokemon && \
    pip install -U pip && \
    pip install segmentation-models-pytorch && \
    pip install efficientnet-pytorch
WORKDIR /pokemon
