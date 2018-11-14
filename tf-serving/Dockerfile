FROM tensorflow/serving:latest-gpu

ARG TF_MODEL=https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz
ARG MODEL_PATH=/models/resnet
ENV MODEL_NAME=resnet

RUN apt-get update && apt-get -y install curl

RUN mkdir -p $MODEL_PATH
RUN curl -s $TF_MODEL | tar --strip-components=2 -C $MODEL_PATH -xvz
