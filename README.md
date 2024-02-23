<p align="center">
  <img src="./figures/main_fig.png?raw=True" width="400px">
</p>
<h1 align="center">Clifford-Steerable Convolutional Neural Networks</h1>

[![Paper](https://img.shields.io/badge/arXiv-2402.14730-blue)](https://arxiv.org/abs/2402.14730) 

This repository contains code of the paper [**Clifford-Steerable Convolutional Neural Networks**](https://arxiv.org/abs/2402.14730).

---

### Abstract

We present Clifford-Steerable Convolutional Neural Networks (CS-CNNs), a novel class of $\mathrm{E}(p, q)$-equivariant CNNs. CS-CNNs process multivector fields on pseudo-Euclidean spaces $\mathbb{R}^{p,q}$. They cover, for instance, $\mathrm{E}(3)$-equivariance on $\mathbb{R}^3$ and Poincaré-equivariance on Minkowski spacetime $\mathbb{R}^{1,3}$. Our approach is based on an implicit parametrization of $\mathrm{O}(p,q)$-steerable kernels via Clifford group equivariant neural networks. We significantly and consistently outperform baseline methods on fluid dynamics as well as relativistic electrodynamics forecasting tasks.


### Requirements

To install all the necessary requirements, including JAX and PyTorch (CPU), run:
```sh
chmod +x setup.sh
./setup.sh
```

### TODO list
The repository is incomplete at the moment, below is the roadmap:

- [x] [implementation](modules) of Clifford-steerable kernels/convolutions (in JAX)
- [x] [implementation](models) of Clifford-steerable ResNet and basic ResNet (in JAX)
- [x] [demonstrating example](playbook.ipynb) + test equivariance (escnn + PyTorch required)
- [ ] implementation of Clifford ResNet and Steerable ResNet (in PyTorch)
- [ ] code for the data generation (Maxwell on spacetime)
- [ ] replicating experimental results

### Citation

If you find this repository useful in your research, please consider citing us:

```bibtex
@misc{zhdanov2024cliffordsteerable,
      title={Clifford-Steerable Convolutional Neural Networks}, 
      author={Maksim Zhdanov and David Ruhe and Maurice Weiler and Ana Lucic and Johannes Brandstetter and Patrick Forré},
      year={2024},
      eprint={2402.14730},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```