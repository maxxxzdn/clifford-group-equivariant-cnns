<div align="center">

<p align="center">
  <a href="https://arxiv.org/abs/2402.14730"><img src="./figures/main_fig.png?raw=True" alt="Figure 1" width="400px"></a>
</p>

<h3>Clifford-Steerable Convolutional Neural Networks</h3>

<b> Authors: </b>Maksim Zhdanov, David Ruhe, Maurice Weiler, and Ana Lucic, Johannes Brandstetter, Patrick Forré 

[ArXiv](https://arxiv.org/abs/2402.14730) | [Blog](https://maxxxzdn.github.io/blog/cscnns.html) | [Playbook](playbook.ipynb) | [Google Colab](https://colab.research.google.com/drive/1M196l6XWi56WJRCVJjd-VCWqzrtsboN2?usp=sharing)

</div>

## Abstract

We present Clifford-Steerable Convolutional Neural Networks (CS-CNNs), a novel class of $\mathrm{E}(p, q)$-equivariant CNNs. CS-CNNs process multivector fields on pseudo-Euclidean spaces $\mathbb{R}^{p,q}$. They cover, for instance, $\mathrm{E}(3)$-equivariance on $\mathbb{R}^3$ and Poincaré-equivariance on Minkowski spacetime $\mathbb{R}^{1,3}$. Our approach is based on an implicit parametrization of $\mathrm{O}(p,q)$-steerable kernels via Clifford group equivariant neural networks. We significantly and consistently outperform baseline methods on fluid dynamics as well as relativistic electrodynamics forecasting tasks.


## Requirements

To install all the necessary requirements, including JAX and PyTorch, run:
```sh
bash setup.sh
```

## Example code
Below is a simple example of initializing and applying a CS-ResNet to a random multivector input:
```python
import jax

from algebra.cliffordalgebra import CliffordAlgebra
from models.resnets import CSResNet

algebra = CliffordAlgebra((1, 1))

config = dict(
    algebra=algebra,
    time_history=4,
    time_future=1,
    hidden_channels=16,
    kernel_num_layers=4,
    kernel_hidden_dim=12,
    kernel_size=7,
    bias_dims=(0,),
    product_paths_sum=algebra.geometric_product_paths.sum().item(),
    make_channels=1,
    blocks=(2, 2, 2, 2),
    norm=True,
    padding_mode="symmetric",
)

csresnet = CSResNet(**config)

# random input for initialization
rng = jax.random.PRNGKey(42)
mv_field = jax.random.normal(rng, (16, config.time_history, 64, 64, algebra.n_blades))
params = csresnet.init(rng, mv_field)

# compute the output
out = csresnet.apply(params, mv_field)
```
Note that the field must come in shape `(Batch, Channels, ..., Blades)`, where `...` indicates grid dimensions (depth, width, etc.).

## Experiments

### Navier-Stokes (PDEarena)
The instructions for the data generation can be found in [datasets/data/ns/README.md](datasets/data/ns/README.md). 
```bash
cd datasets/data/ns
bash download.sh
python preprocess.py
```

To reproduce the experiment, run:

#### CS-ResNet
```bash
python experiment.py --experiment ns --model gcresnet --metric 1 1 --time_history 4 --time_future 1 --num_data 64 --batch_size 8 --norm 1 --hidden_channels 48
```

#### ResNet
```bash
python experiment.py --experiment ns --model resnet --metric 1 1 --time_history 4 --time_future 1 --num_data 64 --batch_size 8 --norm 1 --hidden_channels 96
```

### Maxwell 3D (PDEarena)
The instructions for the data generation can be found in [datasets/data/maxwell3d/README.md](datasets/data/maxwell3d/README.md). 
```bash
cd datasets/data/maxwell3d
bash download.sh
python preprocess.py
```

To reproduce the experiment, run:

#### CS-ResNet
```bash
python experiment.py --experiment maxwell3d --model gcresnet --metric 1 1 1 --time_history 4 --time_future 1 --num_data 64 --batch_size 2 --norm 1 --hidden_channels 12 --scheduler cosine
```

#### ResNet
```bash
python experiment.py --experiment maxwell3d --model resnet --metric 1 1 1 --time_history 4 --time_future 1 --num_data 64 --batch_size 2 --norm 1 --hidden_channels 12 --scheduler cosine
```

### Maxwell 2D+1 (spacetime)
The instructions for the data generation can be found in [datasets/data/maxwell2d/datagen/README.md](datasets/data/maxwell2d/datagen/README.md). 
```bash
cd datasets/datagen/maxwell2d
bash generate.sh --num_points 512 --partition train
```

To reproduce the experiment, run:

#### CS-ResNet
```bash
python experiment.py --experiment maxwell2d --model gcresnet --metric -1 1 1 --time_history 32 --time_future 32 --num_data 512 --batch_size 16 --norm 0 --hidden_channels 12
```
#### ResNet
```bash
python experiment.py --experiment maxwell2d --model resnet --metric -1 1 1 --time_history 32 --time_future 32 --num_data 512 --batch_size 16 --norm 0 --hidden_channels 13
```

## TODO list
The repository is incomplete at the moment, below is the roadmap:

- [x] [implementation](modules) of Clifford-steerable kernels/convolutions (in JAX)
- [x] [implementation](models) of Clifford-steerable ResNet and basic ResNet (in JAX)
- [x] [demonstrating example](playbook.ipynb) + test equivariance (escnn + PyTorch required)
- [x] code for the [data generation](datasets/datagen/maxwell2d/README.md) (Maxwell on spacetime)
- [x] replicating experimental results
  - [x] Navier-Stokes (PDEarena)
  - [x] Maxwell 3D (PDEarena)
  - [x] Maxwell 2D+1 (PyCharge)
- [x] implementation of Clifford ResNet and Steerable ResNet (in PyTorch)

## Citation

If you find this repository useful in your research, please consider citing us:

```
@inproceedings{Zhdanov2024CliffordSteerableCN,
    title = {Clifford-Steerable Convolutional Neural Networks},
    author = {Maksim Zhdanov and David Ruhe and Maurice Weiler and Ana Lucic and Johannes Brandstetter and Patrick Forr'e},
    booktitle = {International {Conference} on {Machine} {Learning} ({ICML})},
    year = {2024},
}
```