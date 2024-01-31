# %%
# %%
"""Example Prospect Theory model in PP

Hugo Storm Feb 2024

"""


# Paper links
# Blog for ML estimation https://www.thegreatstatsby.com/posts/2021-03-08-ml-prospect/ 

# Source for prior on lamda and rho 
# https://www.pnas.org/doi/full/10.1073/pnas.0806761106

import os
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import softmax

import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

# %%
def load_data():
    # import data
    df = pd.read_csv('https://raw.githubusercontent.com/paulstillman/thegreatstatsby/main/_posts/2021-03-08-ml-prospect/data_all_2021-01-08.csv')
    
    # transform study in categorical variable, and get numeric categories
    df['study_cat'] = pd.Categorical(df['study']).codes
    
    # count unique subjects by study
    df.groupby('study')['subject'].nunique()
    N = df.groupby('study')['subject'].nunique().sum()
    N_study = 3
    N_gainOnly = 165 # From https://www.thegreatstatsby.com/posts/2021-03-08-ml-prospect/
    N_gainLoss = 50 # From https://www.thegreatstatsby.com/posts/2021-03-08-ml-prospect/
    return df, N, N_study, N_gainOnly, N_gainLoss
# %%
def sample_prior(rng_key, model, num_samples=1000, **kwargs):
    rng_key, rng_key_ = random.split(rng_key)
    prior_predictive = Predictive(model, num_samples=num_samples)
    prior_samples = prior_predictive(rng_key, **kwargs)
    return prior_samples

#%%
def pyro_inference(rng_key,model, num_samples=200, num_warmup=1000, 
                   num_chains=2, **kwargs):
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains)
    mcmc.run(rng_key_, **kwargs)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    
    return mcmc, samples

# %%
def plot_utility(lam, rho, ax=None, addTitle=True):
    x_range = jnp.linspace(-10,10,20)
    util = utility(x_range,  lam=lam, rho=rho)

    if ax is None:
        f, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_range,util, color='blue',linewidth=0.5);
    ax.plot([-12, 12], [-12, 12], ls="--", c="lightgray");
    
    ax.set_ylabel('Utility, u(x)')
    ax.set_xlabel('Payout (x)')
    if addTitle:
        ax.set_title(f'$\lambda$ = {lam:.2f}, $\\rho$ = {rho:.2f}, $\\mu$ = 1')
    if ax is None:
        plt.xlim([-12, 12])
        plt.ylim([-12, 12])
        plt.show()
        
# %%
def utility(x, lam, rho):
    # Note the additional jnp.where are required to make sure that there are no nans in 
    # gradient calculations, see https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
    util = jnp.where(x > 0, 
                     jnp.where(x>0,x,0)**rho, 
                     -lam * (-jnp.where(x>0,0,x))**rho)
    return util
# %%
# define a model for prospect theory in numpyro
def model_PT(gain, loss, cert, took_gamble=None):
    # Define priors
    lam = numpyro.sample('lam', dist.TruncatedNormal(loc=2, scale=1.0,low=1., high=4.))
    # rho = numpyro.sample('rho', dist.TruncatedNormal(loc=1, scale=1.0,low=0.5, high=1.5))
    rho = numpyro.sample('rho', dist.TruncatedNormal(loc=1, scale=1.0,low=0.5, high=1.))
    mu = numpyro.sample('mu', dist.Uniform(0.5, 1.5))
    
    # Calculate utility of gamble and certain option
    util_reject =  utility(cert, lam, rho)
    util_accept = 0.5 * utility(gain, lam, rho) + 0.5 * utility(loss, lam, rho)
    util_diff =  util_accept - util_reject
    
    # Calculate probability of accepting gamble
    p_accept = 1/(1+jnp.exp(-mu*util_diff))
    
    # Choice of took_gamble
    numpyro.sample('took_gamble', dist.BernoulliProbs(p_accept), obs=took_gamble)

# %%
if __name__ == "__main__":    
    

    # %%
    # load data
    df, N, N_study, N_gainOnly, N_gainLoss = load_data()
    # %%
    rng_key = random.PRNGKey(0)
    
    # %%
    dat_X_train = dict(gain=df['gain'].values,
                       loss=df['loss'].values,
                       cert=df['cert'].values,
                        )
    dat_XY_train = dict(gain=df['gain'].values,
                       loss=df['loss'].values,
                       cert=df['cert'].values,
                       took_gamble=df['took_gamble'].values
                        )
    # %%
    # Illustrate utility function
    fig, ax = plt.subplots(figsize=(4, 3))
    plot_utility(2.6, .65, ax=ax)
    # plot_utility(1., 0.5, ax=ax) # use prior extremes
    # plot_utility(4, 1, ax=ax) # use prior extremes
    ax.set_ylim([-10, 10])
    ax.set_xlim([-10, 10])
    
    # %%
    # sample for prior
    prior_sam = sample_prior(rng_key, model_PT,**dat_X_train)

    # %%
    f, ax = plt.subplots(figsize=(6, 6))
    for i in range(0,200):
        plot_utility(lam=prior_sam['lam'][i] , rho=prior_sam['rho'][i],ax=ax, addTitle=False );
    plt.xlim([-12, 12])
    plt.ylim([-12, 12])
    plt.show()
    # %%
    print('Shape of rho',prior_sam['rho'].shape)
    plt.hist(prior_sam['rho'], bins=100);
    # %%
    print('Shape of lam',prior_sam['lam'].shape)
    plt.hist(prior_sam['lam'], bins=100);
    # %%
    print('Shape of took_gamble',prior_sam['took_gamble'].shape)
    plt.hist(prior_sam['took_gamble'][0,:10000], bins=100);
    
    # %%
    mcmc_M1, post_sam = pyro_inference(rng_key,model_PT,**dat_XY_train)

    # %%
    f, ax = plt.subplots(figsize=(6, 6))
    for i in range(0,200):
        plot_utility(lam=post_sam['lam'][i] , rho=post_sam['rho'][i],ax=ax,addTitle=False );
    plt.xlim([-12, 12])
    plt.ylim([-12, 12])
    plt.show()
    
    # %%
    azMCMC = az.from_numpyro(mcmc_M1)
    # %%
    az.plot_posterior(azMCMC);

    # %%
    ax = az.plot_pair(
        azMCMC,
        # centered,
        var_names=["rho", "lam"],
        kind="kde",
        textsize=22,
        kde_kwargs={
            "hdi_probs": [0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
            "contourf_kwargs": {"cmap": "Blues"},
        },
    )

    # ax.set_aspect("equal")
    plt.show()
# %%
