

## Code PP in AgEcon




## Notes to setup environment using pipenv or vscode devcontainer

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