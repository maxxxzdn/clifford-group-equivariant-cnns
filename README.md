<div align="center">

<p align="center">
  <a href="https://arxiv.org/abs/2212.06096"><img src="./figures/main_fig.png?raw=True" alt="Figure 1" width="400px"></a>
</p>

<h3>Clifford-Steerable Convolutional Neural Networks</h3>

<b> Authors: </b>Maksim Zhdanov, David Ruhe, Maurice Weiler, and Ana Lucic, Johannes Brandstetter, Patrick Forré 

[ArXiv](https://arxiv.org/abs/2402.14730) | [Playbook](playbook.ipynb)

</div>

## Abstract

We present Clifford-Steerable Convolutional Neural Networks (CS-CNNs), a novel class of $\mathrm{E}(p, q)$-equivariant CNNs. CS-CNNs process multivector fields on pseudo-Euclidean spaces $\mathbb{R}^{p,q}$. They cover, for instance, $\mathrm{E}(3)$-equivariance on $\mathbb{R}^3$ and Poincaré-equivariance on Minkowski spacetime $\mathbb{R}^{1,3}$. Our approach is based on an implicit parametrization of $\mathrm{O}(p,q)$-steerable kernels via Clifford group equivariant neural networks. We significantly and consistently outperform baseline methods on fluid dynamics as well as relativistic electrodynamics forecasting tasks.


## Requirements

To install all the necessary requirements, including JAX and PyTorch (CPU), run:
```sh
chmod +x setup.sh
./setup.sh
```

## TODO list
The repository is incomplete at the moment, below is the roadmap:

- [x] [implementation](modules) of Clifford-steerable kernels/convolutions (in JAX)
- [x] [implementation](models) of Clifford-steerable ResNet and basic ResNet (in JAX)
- [x] [demonstrating example](playbook.ipynb) + test equivariance (escnn + PyTorch required)
- [ ] implementation of Clifford ResNet and Steerable ResNet (in PyTorch)
- [ ] code for the data generation (Maxwell on spacetime)
- [ ] replicating experimental results

## Citation

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