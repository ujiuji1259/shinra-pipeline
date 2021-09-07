FROM nvcr.io/nvidia/pytorch:21.08-py3
# FROM daangn/faiss:1.6.3-gpu
RUN apt-get update && apt-get install -y git
# RUN pip install torch==1.9.0
RUN pip install transformers
RUN pip install mlflow
RUN pip install seqeval
RUN pip install fugashi
RUN pip install ipadic

# RUN git clone https://github.com/NVIDIA/apex
# RUN pip install -v --disable-pip-version-check --no-cache-dir ./apex

RUN conda install -c pytorch faiss-gpu

RUN mkdir /data
RUN mkdir /models
# ENV MODEL_PATH /models
# ENV DATA_PATH /src

WORKDIR /workspace

