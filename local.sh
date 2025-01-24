#!/bin/bash

# Create environment
conda create --name mamba python=3.10 -y
conda activate test

# Upgrade pip
python3 -m pip install --upgrade pip
python3 -m pip install -U setuptools

# PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
python3 -c 'import torch; print(torch.__version__); print(torch.tensor([1,2,3]).cuda())'

# Mamba
python3 -m pip install git+https://github.com/Dao-AILab/causal-conv1d
python3 -m pip install git+https://github.com/state-spaces/mamba

python3 test.py
