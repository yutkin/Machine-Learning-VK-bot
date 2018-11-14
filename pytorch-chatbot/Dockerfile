FROM nvidia/cuda:9.2-base-ubuntu18.04

ARG CHECKPOINT_URL=https://storage.googleapis.com/amiable-evening-221409.appspot.com/10000_checkpoint.tar

RUN apt-get update && apt-get install -y python3 python3-pip \
&& rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip flask \
    numpy torchvision_nightly cython cachetools \
    torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html

ADD . /app

ADD $CHECKPOINT_URL /app/10000_checkpoint.tar

CMD ["python3", "/app/app.py"]