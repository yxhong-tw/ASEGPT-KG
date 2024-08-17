FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

WORKDIR /root

COPY . /root/ASEGPT-KG

RUN apt update \
    && apt install -y wget

# Install Miniconda for Faiss
RUN mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm -rf ~/miniconda3/miniconda.sh \
    && ~/miniconda3/bin/conda init bash

# Create conda environment and install dependencies
RUN ~/miniconda3/bin/conda create --name venv python=3.10 -y \
    && ~/miniconda3/envs/venv/bin/pip install -r /root/ASEGPT-KG/requirements.txt \
    && ~/miniconda3/bin/conda install -n venv -c pytorch -c nvidia faiss-gpu=1.8.0

RUN echo "conda activate venv" >> ~/.bashrc

RUN ~/miniconda3/envs/venv/bin/python ASEGPT-KG/pre_process/bertopic/download_ckiptagger_model.py
