# Probabilistic Programming for Embedding Theory and Quantifying Uncertainty in Econometric Analysis

---
  - Hugo Storm (hugo.storm@ilr.uni-bonn.de)
  - Thomas Heckelei
  - Kathy Baylis
---

This repository contains the replication package for the paper "Probabilistic Programming for Embedding Theory and Quantifying Uncertainty in Econometric Analysis" by Hugo Storm, Thomas Heckelei, and Kathy Baylis. 


## Overview

Code for the three examples in chapter 3 can be found under the folder `/examples`. 

1) `1_linear_regression.py`: Code accombining section __3.1 Simple linear model to predict winter wheat yield__. The file produces figures 2 and 3 in the paper.

2) `2_prospect_theory.py`: Code accompanying section __3.2 Cumulative prospect theory model__. The file produces figures 4 and 5 in the paper.

2) `3_potential_outcome.py`: Code accompanying section __3.3 3.3.	Potential outcome framework (with non-linear treatment effects)__. The file produces figures 6 and 7 in the paper.


## Computational requirements

1) Install pipenv dependencies ```pipenv install --dev```

2) In order to use GPU support it is required to install numpyro[cuda] manually using wheels, currently seems to be not supported in pipenv

    ```
    pipenv shell

    pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    ```
    Optionally install tensorflow-proability (this is only required for the truncated distributions in
    example 1) 
    ```pip install --upgrade tensorflow-probability```
    
    Optionally install flax manually (required for the potential outcome example):
    ```pip install flax```

[Alternativly use VS Code devcontainer](https://code.visualstudio.com/docs/remote/containers): using provided docker file and devcontainer.json (jax and flax need to be manually installed in container as with pipenv)