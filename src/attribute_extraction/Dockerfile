# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
FROM daangn/faiss:1.6.3-gpu
RUN apt-get update && apt-get install -y git
RUN pip install torch==1.9.0
RUN pip install transformers
RUN pip install mlflow
RUN pip install seqeval
RUN pip install fugashi
RUN pip install ipadic

RUN git clone https://github.com/NVIDIA/apex
RUN pip install -v --disable-pip-version-check --no-cache-dir ./apex


ENV DATA_PATH /src

RUN mkdir /workspace
RUN mkdir /workspace/models
ENV MODEL_PATH /workspace/models

WORKDIR /workspace

