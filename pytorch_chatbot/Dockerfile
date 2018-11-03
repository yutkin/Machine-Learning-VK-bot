FROM python:3.7

ARG CHECKPOINT_URL=https://storage.googleapis.com/amiable-evening-221409.appspot.com/8000_checkpoint.tar

RUN pip install -U pip flask \
    numpy torchvision_nightly cython cachetools \
    torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

ADD . /app

RUN curl $CHECKPOINT_URL > /app/8000_checkpoint.tar

CMD ["python", "/app/app.py"]
