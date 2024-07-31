#!/bin/bash
conda create --name cscnns python=3.10 -y
source activate cscnns

# PyTorch + escnn
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install escnn -q --no-cache

# Data generation [Maxwell 2d]
pip install pycharge

# JAX
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax

# Other packages
pip install matplotlib
pip install wandb
pip install cliffordlayers
pip install neuraloperator

# [WARNING] GNU Parallel might get handy for data generation. Uncomment the following lines to install it.
# sudo apt-get install parallel

echo "Environment 'cscnns' created and packages installed."




