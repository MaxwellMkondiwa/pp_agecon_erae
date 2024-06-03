Probabilistic Programming for Embedding Theory and Quantifying Uncertainty in Econometric Analysis

==============================

---
  - Hugo Storm (hugo.storm@ilr.uni-bonn.de)
  - Thomas Heckelei
  - Kathy Baylis
---

This repository contains the replication package for the paper "Probabilistic Programming for Embedding Theory and Quantifying Uncertainty in Econometric Analysis" by Hugo Storm, Thomas Heckelei, and Kathy Baylis. 


## Overview

Code for the three examples in chapter 3 can be found under the folder `/examples`. 

1) `python examples/1_linear_regression.py`: Code accombining section __3.1 Simple linear model to predict winter wheat yield__. The file produces figures 2 and 3 in the paper.

2) `2_prospect_theory.py`: Code accompanying section __3.2 Cumulative prospect theory model__. The file produces figures 4 and 5 in the paper.

2) `3_potential_outcome.py`: Code accompanying section __3.3 3.3.	Potential outcome framework (with non-linear treatment effects)__. The file produces figures 6 and 7 in the paper.


## Computational requirements

The repository is set up to run in a Docker container. Pull the
repository and open it in VS Code with the Remote-Containers extension.
This requires that you have a) a Docker Engine installed (https://docs.docker.com/engine/install/) and b) the VS Code Dev-Containers extension installed (Extension identifier: `ms-vscode-remote.remote-containers`). 

With this in place follow the instructions to create the development container in VS Code:

1. Clone the repository: `git clone https://github.com/hstorm/pp_agecon_erae.git`
2. Open the clone folder in VS Code and hit `Ctrl+Shift+P` and select `Remote-Containers: Reopen in Container`. 

All the necessary dependencies are then automatically installed in the Docker container.

---

### Alternative manual approach without docker (Note: the Docker approach is the recommended approach)

1. Install Pipenv from https://pipenv.pypa.io/en/latest/

2. Clone the repository ```git clone https://github.com/hstorm/pp_agecon_erae.git```

3. Create pipenv environment by running ```pipenv install --dev```

4) (Optional) In order to use GPU support it is required to install numpyro[cuda] manually using wheels, currently seems to be not supported in pipenv

    ```
    pipenv shell

    pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    ```
    (Optional) Install tensorflow-proability (this is only required for the truncated distributions in
    example 1) 
    ```pip install --upgrade tensorflow-probability```
    
    (Optional) Install flax manually (required for the potential outcome example):
    ```pip install flax```