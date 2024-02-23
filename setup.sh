#!/bin/bash
conda create --name cscnns python=3.11 -y
source activate cscnns

pip install matplotlib

# JAX
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax

# PyTorch (CPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
pip install git+https://github.com/AMLab-Amsterdam/lie_learn
pip install escnn

echo "Environment 'cscnns' created and packages installed."