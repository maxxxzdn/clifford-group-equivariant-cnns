# Maxwell's equation on spacetime $\mathbb{R}^{1,2}$
We simulate data for Maxwell’s equations on spacetime $\mathbb{R}^{1,2}$ using [PyCharge](https://pycharge.readthedocs.io/en/latest/). A typical simulation looks as follows:

<p align="center">
  <img src="../../../../figures/pycharge_maxwell2d.gif" alt="PyCharge">
</p>

To generate the data, run:
```sh
bash generate.sh --num_points 32 --partition train
```

It might be a good idea to use GNU Parallel to parallelize the data generation process. To install:
```sh
sudo apt-get install parallel
```

## Description
Electromagnetic fields are emitted by point sources that move, orbit and oscillate at relativistic speeds. 


The spacetime grid has a resolution of 256 points in both spatial and the temporal dimension.
Its spatial extent are $50$ nm and the temporal extent are $3.77 \cdot 10^{-14}$ s.


Sampled simulations contain between 2 to 4 oscillating charges and 1 to 2 orbiting charges. 
The sources have charges sampled uniformly as integer values between $−3e$ and $3e$. 
Their positions are sampled uniformly on the grid, with a predefined minimum initial distance between them.
Each charge has a random linear velocity and either oscillates in a random direction or orbits with a random radius.
Oscillation and rotation frequencies, as well as velocities are sampled such that the overall particle velocity does not
exceed $0.85c$, which is necessary since the PyCharge simulation becomes unstable beyond this limit.

### Normalization

As the field strengths span many orders of magnitude, we normalize the generated fields by dividing bivectors by their Minkowski norm and multiplying them by the logarithm of this norm. This step is non-trivial since Minkowski-norms can be zero or negative, however, we found that they are always positive in the generated data. 

We filter out numerical artifacts by removing outliers with a standard deviation greater than 20. The final dataset comprises 8192 training, 256 validation and 256 test simulations.

## Using for your own project

The folder is self-sufficient for generating data, you can safely take it for your own project. We only use Clifford algebra utilites (written in PyTorch) to detect and filter out data with large norm. Don't forget to cite PyCharge:

```
@article{filipovich2022PyCharge,
    title={PyCharge: an open-source Python package for self-consistent electrodynamics simulations of Lorentz oscillators and moving point charges},
    author={Filipovich, Matthew J and Hughes, Stephen},
    journal={Computer Physics Communications},
    volume={274},
    pages={108291},
    year={2022},
    publisher={Elsevier}
}
```
and us:

```
@inproceedings{Zhdanov2024CliffordSteerableCN,
    title = {Clifford-Steerable Convolutional Neural Networks},
    author = {Maksim Zhdanov and David Ruhe and Maurice Weiler and Ana Lucic and Johannes Brandstetter and Patrick Forr'e},
    booktitle = {International {Conference} on {Machine} {Learning} ({ICML})},
    year = {2024},
}
```