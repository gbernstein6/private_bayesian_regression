# private_bayesian_regression

This is a code release as part of the paper Differentially Private Bayesian Linear Regression.[1] (http://arxiv.org/abs/1910.13153)

The main file is `driver.py`, which generates toy data, calculates the posterior for each method, and plots them. Note that the purpose of this release is to provide code for the methods in the paper and does not replicate the experimental harness to produce the results in the paper.

* **Code Requirements**
  - Python 3.6+
  - Numpy, Scipy, Matplotlib
  - [PyMC3](https://docs.pymc.io/) (if you wish to use the MCMC-IND method)


[1] Garrett Bernstein and Daniel Sheldon. Differentially Private Bayesian Linear Regression. NeurIPS 2019. [http://arxiv.org/abs/1910.13153]
