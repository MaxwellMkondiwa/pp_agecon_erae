# %%
"""Example of a potential outcome model

Hugo Storm Feb 2024

"""
import os
import time
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import flax.linen as nn
import jax
from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive, SVI, autoguide, init_to_feasible
import numpyro.optim as optim
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.contrib.module import  random_flax_module

if len(jax.devices(backend='gpu'))>0:
    numpyro.set_platform("gpu")
else:
    numpyro.set_platform("cpu")

az.style.use("arviz-darkgrid")

# Make sure that numpyro is the correct version
assert numpyro.__version__.startswith("0.12.1")

# %%
rng_key = random.PRNGKey(123)
np.random.seed(seed=123)

# %%
def model_POF(Z, X, T=None, Y=None):
    """Define a simple potential outcome model"""
    alpha = numpyro.sample("alpha", dist.Normal(0.,1).expand([X.shape[1]]))
    beta = numpyro.sample("beta", dist.Normal(0.,1).expand([Z.shape[1]]))
    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1))

    Y0 = X @ alpha  
    tau = Z @ beta  
    Y1 = Y0 + tau 
    T = numpyro.sample("T", dist.Bernoulli(logits=tau), obs=T)
    numpyro.sample("Y", dist.Normal(Y1*T + Y0*(1-T), sigma_Y), obs=Y)
    
    # Collect Y0 and Y1 as deterministic for later use
    numpyro.deterministic("Y0", Y0)
    numpyro.deterministic("Y1", Y1)
        
# %%
def model_POF_poly(Z, X, polyDegree=1, stepFunction=False, T=None, Y=None):
    """Extented potential outcome model with polynomial terms and 
    step function for the treatment effect

    Args:
        Z (_type_): Explanatory variables (standardized)
        X (_type_): Explanatory variables (standardized)
        polyDegree (int, optional): Degree of polynomial. Defaults to 1.
        stepFunction (bool, optional): Indicator to use a step function for 
            first explanatory variable. Defaults to False.
        T (np array, optional): Treatment status. Defaults to None.
        Y (np array, optional): Observed outcome. Defaults to None.
    """
    alpha = numpyro.sample("alpha", dist.Normal(0.,1).expand([X.shape[1]]))
    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1))
    
    beta = numpyro.sample("beta", dist.Normal(0.,1).expand([Z.shape[1]]))
    tau = Z @ beta 
    if polyDegree>1:
        betaSq = numpyro.sample("betaSq", dist.Normal(0.,1).expand([Z.shape[1]-1]))
        tau = tau + Z[:,1:]**2 @ betaSq # "1:" to exclude constant term

        if polyDegree>2:
            betaCub = numpyro.sample("betaCub", dist.Normal(0.,1).expand([Z.shape[1]-1]))
            tau = tau + Z[:,1:]**3 @ betaCub # "1:" to exclude constant term

    if stepFunction:
        betaStep = numpyro.sample("betaStep", dist.Normal(0.,5)) 
        tau = tau + betaStep * (Z[:,1]>0.0)

    Y0 = X @ alpha 
    Y1 = Y0 + tau 
    T = numpyro.sample("T", dist.Bernoulli(logits=Y1 - Y0), obs=T)
    numpyro.sample("Y", dist.Normal(Y1*T + Y0*(1-T), sigma_Y), obs=Y)
    
    # Collect Y0 and Y1 as deterministic for later use
    numpyro.deterministic("Y0", Y0)
    numpyro.deterministic("Y1", Y1)

# %%
# Define a Flax NN model class which can be used within the DGP
# Useful source: https://omarfsosa.github.io/bayesian_nn
from typing import Sequence
class MLP(nn.Module):
    """
    Flexible MLP module, allowing different number of layer and layer size, as
    well as dropout.
    # Run and inspect model
    root_key = jax.random.PRNGKey(seed=0)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

    model = MLP([12, 8, 4, 1], [0.0, 0.2, 0.3])
    batch = jnp.ones((32, 10))
    variables = model.init(jax.random.PRNGKey(0), batch,is_training=False)
    output = model.apply(variables, batch,is_training=True, rngs={'dropout': dropout_key})
    print(output.shape)  # (32, 1)

    # inspect model
    jax.tree_util.tree_map(jnp.shape, variables)
    """  
    
    lst_layer: Sequence[int]
    dropout_rates: Sequence[float]
    use_bias: Sequence[float]

    @nn.compact
    def __call__(self, x, is_training:bool):
        assert len(self.lst_layer) == len(self.dropout_rates) + 1
        assert len(self.lst_layer) == len(self.use_bias) + 1
        
        for iLayer in range(0,len(self.lst_layer[:-1])):
            x = nn.leaky_relu(nn.Dense(self.lst_layer[iLayer],use_bias=self.use_bias[iLayer])(x))
        
            if self.dropout_rates[iLayer] > 0.0:
                x = nn.Dropout(self.dropout_rates[iLayer], 
                    deterministic=not is_training)(x)
        
        # FIXME batch norm not implemented and not yet easily working with numpyro
        # x = nn.BatchNorm(
        #     use_bias=False,
        #     use_scale=False,
        #     momentum=0.9,
        #     use_running_average=not is_training,
        # )(x)
        
        x = nn.Dense(self.lst_layer[-1])(x).squeeze()
        return x



# %%
def model_POF_NN(hyperparams, Z, X, T=None, Y=None, is_training=False):
    """Potential outcome model using neural networks (dense MPL) for 
    the treatment effect and the not treated outcome

    Args:
        hyperparams (dict): dict to specify number of layers, dropout, batchsize,
        X (_type_): Explanatory variables (standardized)
        T (np array, optional): Treatment status. Defaults to None.
        Y (np array, optional): Observed outcome. Defaults to None.
        is_training (bool, optional): Set to true for inference. Defaults to False.
    """
    lst_lay_Y0 = hyperparams['lst_lay_Y0']
    lst_drop_Y0 = hyperparams['lst_drop_Y0']
    lst_bias_Y0 = hyperparams['lst_bias_Y0']
    
    lst_lay_tau = hyperparams['lst_lay_tau']
    lst_drop_tau = hyperparams['lst_drop_tau']
    lst_bias_tau = hyperparams['lst_bias_tau']
    
    assert len(lst_lay_Y0) == len(lst_drop_Y0) + 1
    assert len(lst_lay_Y0) == len(lst_bias_Y0) + 1
    assert len(lst_lay_tau) == len(lst_drop_tau) + 1
    assert len(lst_lay_tau) == len(lst_bias_tau) + 1

    # Specify a NN for the potential outcomes without the treatment effect
    prior_MLP_Y0 = {**{f"Dense_{i}.bias":dist.Cauchy() for i in range(0,len(lst_lay_Y0))},
                    **{f"Dense_{i}.kernel":dist.Normal(0.,1) for i in range(0,len(lst_lay_Y0))}}
    MLP_Y0 = random_flax_module("MLP_Y0",
                MLP(lst_lay_Y0, lst_drop_Y0,lst_bias_Y0),
                input_shape=(1, X.shape[1]), 
                prior=prior_MLP_Y0,
                apply_rng=["dropout"],is_training=True)

    # Specify the treatment effect parameter as a NN
    prior_MPL_tau = {**{f"Dense_{i}.bias":dist.Cauchy() for i in range(0,len(lst_lay_tau))},
                    **{f"Dense_{i}.kernel":dist.Normal(0.,1) for i in range(0,len(lst_lay_tau))}}
    MLP_tau = random_flax_module("MLP_tau",
                MLP(lst_lay_tau, lst_drop_tau,lst_bias_tau),
                input_shape=(1, Z.shape[1]), 
                prior=prior_MPL_tau,
                apply_rng=["dropout"],is_training=True)

    sigma_Y = numpyro.sample("sigma_Y", dist.Exponential(1))
    
    rng_key = hyperparams['rng_key']
    with numpyro.plate("samples", X.shape[0], subsample_size=hyperparams["batch_size"]):
        batch_X = numpyro.subsample(X, event_dim=1)
        batch_Z = numpyro.subsample(Z, event_dim=1)
        batch_Y = numpyro.subsample(Y, event_dim=0) if Y is not None else None
        batch_T = numpyro.subsample(T, event_dim=0) if T is not None else None

        rng_key, _rng_key = jax.random.split(key=rng_key)
        Y0 = MLP_Y0(batch_X, is_training, rngs={"dropout": _rng_key})

        rng_key, _rng_key = jax.random.split(key=rng_key)
        tau = MLP_tau(batch_Z, is_training, rngs={"dropout": _rng_key})

        Y1 = Y0 + tau
        T = numpyro.sample("T", dist.Bernoulli(logits=Y1 - Y0), obs=batch_T)
        numpyro.sample("Y", dist.Normal(Y1*T + Y0*(1-T), sigma_Y), obs=batch_Y)
        
        # Collect tau, Y0 and Y1 as deterministic for later use
        numpyro.deterministic("tau", tau)
        numpyro.deterministic("Y0", Y0)
        numpyro.deterministic("Y1", Y1)


# %%        
def data_generating(rng_key=rng_key,
                    modelTypeDataGen = 'linear',
                    N = 10000,
                    K = 5,
                    X=None): 
    """Helper function to generate data using different model types.
    Creates some basic plots to illustrate DGP

    Args:
        rng_key (_type_, optional): numpyro rng key. Defaults to rng_key.
        modelTypeDataGen (str, optional): Name of model type. Defaults to 'linear'.
        N (int, optional): number of samples. Defaults to 10000.
        K (int, optional): number of explanatory variables (ignored if X is provided). Defaults to 5.
        X (_type_, optional): Predefined exlanatory variables. Defaults to None.

    Raises:
        ValueError: if modelTypeDataGen is not recognized

    """
    # %
    # Generate X if not provided    
    if X is None:
        X = np.random.normal(0, 1.0, size=(N,K-1))
        X = np.hstack([np.ones((N,1)),X]) # add a constant
        Z = np.random.normal(0, 1.0, size=(N,K-1))
        Z = np.hstack([np.ones((N,1)),Z]) # add a constant
        

    if modelTypeDataGen == 'linear':
        model = model_POF
        datX_conditioned = {'Z':Z,'X':X}
    elif modelTypeDataGen == 'poly2':
        model = model_POF_poly
        datX_conditioned = {'Z':Z, 'X':X, 'polyDegree':2}
    elif modelTypeDataGen == 'poly3':
        model = model_POF_poly
        datX_conditioned = {'Z':Z, 'X':X, 'polyDegree':3}
    elif modelTypeDataGen == 'poly3_step':
        model = model_POF_poly
        datX_conditioned = {'Z':Z, 'X':X,'stepFunction':True, 'polyDegree':3}
    elif modelTypeDataGen == 'NN':
        model = model_POF_NN
        
        hyperparams = {}
        hyperparams['N'] = N
        hyperparams['K'] = 5
        hyperparams['rng_key'] = rng_key
        hyperparams['batch_size'] = hyperparams['N']
        hyperparams['lst_lay_Y0'] = [512,64,1]
        hyperparams['lst_drop_Y0'] = [0.0,0.0]
        hyperparams['lst_bias_Y0'] = [True,True]
        hyperparams['lst_lay_tau'] = [512,64,32,1]
        hyperparams['lst_drop_tau'] = [0.0,0.0,0.0]
        hyperparams['lst_bias_tau'] = [True,True,True]
        
        datX_conditioned = {'Z':Z, 'X':X, 'hyperparams':hyperparams}
    else:
        raise ValueError('modelTypeDataGen not recognized')
    # %
    # Run the DGP  once to get values for latent variables
    rng_key, rng_key_ = random.split(rng_key)
    lat_predictive = Predictive(model, num_samples=1)
    lat_samples = lat_predictive(rng_key_,**datX_conditioned)
    lat_samples['Y0'].shape

    coefTrue = {s:lat_samples[s][0] for s in 
                lat_samples.keys() if s not in ['Y','T','Y0', 'Y1']}
    coefTrue.keys()
    # %
    if modelTypeDataGen == 'poly3':
        coefTrue['beta'] = jnp.array([0, 0.5],dtype='float32')
        coefTrue['betaSq'] = jnp.array([-0.1],dtype='float32')
        coefTrue['betaCub'] = jnp.array([0.5],dtype='float32')
    if modelTypeDataGen == 'poly3_step':
        coefTrue['beta'] = jnp.array([0, 0.5],dtype='float32')
        coefTrue['betaSq'] = jnp.array([-0.1],dtype='float32')
        coefTrue['betaCub'] = jnp.array([0.05],dtype='float32')
        coefTrue['betaStep'] = jnp.array([3],dtype='float32')
    
    # %
    # Condition the model and get predictions for Y
    condition_model = numpyro.handlers.condition(model, data=coefTrue)
    conditioned_predictive = Predictive(condition_model, num_samples=1)
    prior_samples = conditioned_predictive(rng_key_,**datX_conditioned)
    Y_unscaled = prior_samples['Y'].squeeze()
    T = prior_samples['T'].squeeze()
    Y0 = prior_samples['Y0'].squeeze()
    Y1 = prior_samples['Y1'].squeeze()
    print('avg treatment effect',np.mean(Y1-Y0))
    plt.hist(Y1-Y0,bins=100);
        
    # Standardize Y
    Y_mean = Y_unscaled.mean(axis=0)
    Y_std = Y_unscaled.std(axis=0)
    Y = (Y_unscaled - Y_mean)/Y_std
    
    print('Share treated',np.mean(T))
    print(f'Mean(Y)={np.mean(Y):.4f}; std(Y)={np.std(Y):.4f}')
    
    # %
    if modelTypeDataGen != 'NN':
        beta_true = prior_samples['beta'].squeeze()
        alpha_true = prior_samples['alpha'].squeeze()
    else:
        beta_true = {key:val for key, val in prior_samples.items() if 'MLP_tau' in key}
        alpha_true = {key:val for key, val in prior_samples.items() if 'MLP_Y0' in key}

    # Plot true Treatment heterogneity, for first covariate
    k = 1
    x_percentile = np.percentile(Z[:,k],q=[2.5,97.5])
    x_range = np.linspace(x_percentile[0],x_percentile[1],100)
    x_mean = X.mean(axis=0)
    x_plot = np.repeat(x_mean.reshape(1,-1),100,axis=0)
    x_plot[:,k] = x_range
    
    datX_plot = datX_conditioned.copy()
    datX_plot['X'] = x_plot
    datX_plot['Z'] = x_plot
    
    if modelTypeDataGen == 'NN':
        datX_plot['hyperparams']['batch_size'] = 100
    
    # Get prediction from the "true" conditioned model
    true_predict = conditioned_predictive(rng_key_,**datX_plot)
    # %
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    # Plot "true" effect in red    
    ax.plot(x_plot[:,k],(true_predict['Y1']-true_predict['Y0'])[0,:],color='r',alpha=1);

    ax.set_xlabel(f'Z[{k}]', fontsize=20)
    ax.set_ylabel('tau', fontsize=20)
    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
        
    sns.rugplot(data=Z[T==1,1], 
            ax=ax, color='black',lw=1, alpha=.005)    
    ax.set_xlim([-3,3])
    # %
    # Plot hist of Y0 and Y1
    fig, ax = plt.subplots()
    ax.hist(Y[T==0][:10000],bins=100,density=True,color='green',alpha=0.5,label='T=0');
    ax.hist(Y[T==1][:10000],bins=100,density=True,color='red',alpha=0.5,label='T=1');
    
    # Plot scatter of tau vs Z    
    mu_diff = Y1-Y0
    aa = pd.DataFrame(np.hstack([Z,mu_diff[:,None]]),
                      columns=[f'Z{i}' for i in range(0,Z.shape[1])]+['tau'])
    x_vars = [f'Z{i}' for i in range(0,Z.shape[1])]
    y_vars = ["tau"]
    
    g = sns.PairGrid(aa,x_vars=x_vars, y_vars=y_vars)
    g.map(sns.scatterplot,s=0.1)
    # %
    return Y, Y_unscaled, Y_mean, Y_std, T, Z, X, Y0, Y1, beta_true, alpha_true, conditioned_predictive, datX_conditioned

# %%
if __name__ == '__main__':

    # %%
    # =====================================================
    # Generate the data, set modelType to desired name
    # =====================================================
    for modelTypeDataGen in ['poly3','poly3_step']:
        # %%
        # modelTypeDataGen = 'linear'
        # modelTypeDataGen = 'poly2'
        # modelTypeDataGen = 'poly3'
        # modelTypeDataGen = 'poly3_step'
        # modelTypeDataGen = 'NN'
        N = 200000
        K = 2
        # Set a seed for reproducibility
        # rng_key = jnp.array([0, 1], dtype='uint32')
         
        rng_key, rng_key_ = random.split(rng_key)
        (Y, Y_unscaled, Y_mean, Y_std, T, Z, X, Y0_true, 
        Y1_true, beta_true, alpha_true, 
        conditioned_predictive, datX_conditioned) = data_generating(
            rng_key=rng_key,
            modelTypeDataGen = modelTypeDataGen,
            N = N,
            K = K) 
        
        # %%
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.hist(Z[T==0,1],bins=100, label='T=0', color='black', alpha=0.5);
        ax.hist(Z[T==1,1],bins=100, label='T=1', color='darkgrey', alpha=0.5);
        ax.set_xlabel(f'Z[1]', fontsize=20)
        ax.legend()

        # %%
        # =====================================================
        # Estimate model, set inference model to desired model
        # =====================================================
        rng_key, rng_key_ = random.split(rng_key)
        
        for modelTypeInference in ['linear','poly2','poly3','NN']:
            # %%
            # modelTypeInference = 'linear'
            # modelTypeInference = 'poly2'
            # modelTypeInference = 'poly3'
            # modelTypeInference = 'NN'
            if modelTypeInference == 'linear':
                model = model_POF
                datXY = {'Z':Z, 'X':X, 'Y':Y_unscaled, 'T':T}
                datX = {'Z':Z, 'X':X, 'T':T}
            elif modelTypeInference == 'poly2':
                model = model_POF_poly
                datXY = {'Z':Z, 'X':X, 'Y':Y_unscaled, 'T':T, 'polyDegree':2}
                datX = {'Z':Z, 'X':X, 'T':T, 'polyDegree':2}
            elif modelTypeInference == 'poly3':
                model = model_POF_poly
                datXY = {'Z':Z, 'X':X, 'Y':Y_unscaled, 'T':T, 'polyDegree':3}
                datX = {'Z':Z, 'X':X, 'T':T, 'polyDegree':3}
            elif modelTypeInference == 'NN':
                model = model_POF_NN
                
                hyperparams = {}
                hyperparams['N'] = N
                hyperparams['K'] = K
                hyperparams['rng_key'] = rng_key
                hyperparams['batch_size'] = 512
                hyperparams['lst_lay_Y0'] = [512,64,1]
                hyperparams['lst_drop_Y0'] = [0.2,0.2]
                hyperparams['lst_bias_Y0'] = [True,True]
                hyperparams['lst_lay_tau'] = [1028,512,64,1]
                hyperparams['lst_drop_tau'] = [0.2,0.2,0.2]
                hyperparams['lst_bias_tau'] = [True,True,True]
                
                datXY = {'Z':Z, 'X':X, 'Y':Y_unscaled, 'T':T, 'hyperparams':hyperparams,'is_training':True}
                datX = {'Z':Z, 'X':X, 'T':T,  'hyperparams':hyperparams, 'is_training':False}
            else:
                raise ValueError('modelTypeInference not recognized')
            
            # %%
            # Estimate with SVI
            start = time.time()
            rng_key, rng_key_ = random.split(rng_key)
            guide = autoguide.AutoNormal(model, 
                                init_loc_fn=init_to_feasible)

            svi = SVI(model,guide,optim.Adam(0.005),Trace_ELBO())
            svi_result = svi.run(rng_key_, 15000,**datXY)
            print("\nInference elapsed time:", time.time() - start)
            plt.plot(svi_result.losses)
            svi_params = svi_result.params
            
            # %%
            # Get samples from the posterior
            predictive = Predictive(guide, params=svi_params, num_samples=500)
            samples_svi = predictive(random.PRNGKey(1), **datX)
            samples_svi.keys()
            # %%
            # Get posterior predictions using samples from the posterior
            predictivePosterior = Predictive(model, posterior_samples=samples_svi)
            post_predict = predictivePosterior(random.PRNGKey(1), **datX)
            post_predict.keys()
            
            tau_mean_true = np.mean(Y1_true-Y0_true)
            print('True: avg treatment effect',tau_mean_true)
            tau_mean_hat_scaled = np.mean(post_predict['Y1']-post_predict['Y0'])
            tau_mean_hat = np.mean((post_predict['Y1']*Y_std+Y_mean)-(post_predict['Y0']*Y_std+Y_mean))
            print('Estimated (scaled): avg treatment effect',tau_mean_hat_scaled)
            print('Estimated: avg treatment effect',tau_mean_hat)
            
            if modelTypeInference != 'NN':    
                print('alpha_true',alpha_true)
                print('alpha_hat',np.mean(samples_svi['alpha'],axis=0))
                
                print('beta_true',beta_true)
                print('beta_hat',np.mean(samples_svi['beta'],axis=0))

            # %%
            k = 1
            x_percentile = np.percentile(Z[:,k],q=[0.1,99])
            x_range = np.linspace(x_percentile[0],x_percentile[1],100)
            x_mean = X.mean(axis=0)
            x_plot = np.repeat(x_mean.reshape(1,-1),100,axis=0)
            x_plot[:,k] = x_range
            
            datX_plot = datX.copy()
            datX_plot['X'] = x_plot
            datX_plot['Z'] = x_plot
            datX_plot['T'] = jnp.zeros(100)
            if modelTypeInference == 'NN':
                datX_plot['hyperparams']['batch_size'] = 100
            
            datX_plot_conditioned = datX_conditioned.copy()
            datX_plot_conditioned['X'] = x_plot
            datX_plot_conditioned['Z'] = x_plot
            datX_plot_conditioned['T'] = jnp.zeros(100)
            # Get posterior predictions
            post_predict = predictivePosterior(random.PRNGKey(1), **datX_plot)
            # Get prediction from the "true" conditioned model
            true_predict = conditioned_predictive(rng_key_,**datX_plot_conditioned)
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            for i in range(1,300):
                tau_i = ((post_predict['Y1'])-(post_predict['Y0']))[i,:]
                ax.plot(x_plot[:,k],tau_i,color='k',alpha=0.2);
            # Add "true" effect in red    
            ax.plot(x_plot[:,k],(true_predict['Y1']-true_predict['Y0'])[0,:],color='r',alpha=1);

            ax.set_xlabel(f'Z[{k}]', fontsize=20)
            ax.set_ylabel(r'$E[\tau]$', fontsize=20)
            # Set tick font size
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(20)
            
            sns.rugplot(data=Z[T==1,1], ax=ax, color='black',lw=1, alpha=.005)   
            ax.set_xlim([x_percentile[0],x_percentile[1]])
            fig.savefig(f'../figures/POF_{modelTypeDataGen}_{modelTypeInference}.png',dpi=300)    
    # %%
    
        

    

