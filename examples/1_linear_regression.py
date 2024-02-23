# %%
"""Example linear model in PP

Hugo Storm Feb 2024

"""
import os

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import jax 
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

if len(jax.devices(backend='gpu'))>0:
    numpyro.set_platform("gpu")
else:
    numpyro.set_platform("cpu")
    
az.style.use("arviz-darkgrid")

os.chdir(os.path.dirname(os.getcwd()))

# Make sure that numpyro is the correct version
assert numpyro.__version__.startswith("0.12.1")

# %%
# Set seed for reproducibility
rng_key = random.PRNGKey(1)
np.random.seed(0)

# %%
# Load data
from util.load_yield_data import getData
dfL_train, dfL_test, lstCatCrop, lstCatNUTS3, lstSmi25, lstSmi180, scale_train = getData()   

lstColX = ['bodenzahl_scaled'] 

dfWheat_train = dfL_train.loc[dfL_train['crop']=='Winterweizen',:]
    
Soil = dfL_train.loc[dfL_train['crop']=='Winterweizen','bodenzahl_scaled'].values 
Yield = dfL_train.loc[dfL_train['crop']=='Winterweizen','yield_scaled'].values    

# %%
print(f"SoilRating [0-100]: Mean={scale_train['bodenzahl_mean']:.2f}, Std={scale_train['bodenzahl_std']:.2f}")
print(f"WinterWheatYield: Mean={scale_train['Winterweizen_yield_mean']:.2f}dt, Std={scale_train['Winterweizen_yield_std']:.2f}dt")

# %%
# =============================================================================
# Define most basic linear regression model
# =============================================================================
def model(Soil, Yield=None):
    beta = numpyro.sample('beta', dist.Normal(0,1))
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    numpyro.sample('Yield',dist.Normal(Soil*beta,sigma), obs=Yield)
    
# Example of a linear regression in matrix notation, 
# same as "model()" but suitable for more then one explanatory variable     
def model_matrix(X, Y=None):
    beta = numpyro.sample('beta', dist.Normal(0,1).expand([X.shape[1]]))
    sigma = numpyro.sample('sigma', dist.Exponential(4))
    numpyro.sample('Y',dist.Normal(X @ beta,sigma), obs=Y)

# Same model as above, but with std of beta prior as a parameter
def model_sigma_b(Soil, sigma_b, Yield=None):
    beta = numpyro.sample('beta', dist.Normal(0,sigma_b))
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    numpyro.sample('Yield',dist.Normal(Soil*beta,sigma), obs=Yield)
    
# Same model as above, but with yield as a student-t distribution and truncated at zero
lowtrunc_scale = (0-scale_train['Winterweizen_yield_mean'])/scale_train['Winterweizen_yield_std']
def model_trunc(Soil, sigma_b, Yield=None):
    beta = numpyro.sample('beta', dist.Normal(0,sigma_b))
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    df = 5 # degrees of freedom for student-t distribution
    # Truncate studentT distribution
    numpyro.sample('Yield',dist.LeftTruncatedDistribution(
        dist.StudentT(df,Soil*beta,sigma),low=lowtrunc_scale), obs=Yield)
    # Alternative use a truncated normal instead
    # numpyro.sample('Yield',dist.TruncatedNormal(Soil*beta,sigma,low=lowtrunc_scale), obs=Yield)


# %%
# =============================================================================
# Prior sampling
# =============================================================================
model = model_sigma_b
# model = model_trunc # Change here to use the truncated model

nPriorSamples = 1000 # Number of prior samples
# Perform prior sampling for different values of sigma_b
for sigma_b in [1,5]:
    # %
    rng_key, rng_key_ = random.split(rng_key)
    prior_predictive = Predictive(model, num_samples=nPriorSamples)
    prior_samples = prior_predictive(rng_key_,Soil=Soil, sigma_b=sigma_b)

    # %
    # Plot prior samples
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist((prior_samples['Yield'].flatten()[~np.isinf(prior_samples['Yield'].flatten())]
                *scale_train['Winterweizen_yield_std']
                +scale_train['Winterweizen_yield_mean'])/10,
            bins=100, density=True, color='grey');
    ax.set_title(fr'$\beta$~Normal(0,{sigma_b})', fontsize=20)
    ax.set_xlabel('Yield [t/ha]', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)
    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)

    # Plot regression lines
    x_range_scaled = np.linspace(-5,5,100)
    x_mean_scaled = Soil.mean(axis=0)
    x_plot = np.repeat(x_mean_scaled.reshape(1,-1),100,axis=0)
    x_plot[:,0] = x_range_scaled
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x_range = x_range_scaled*scale_train['bodenzahl_std']+scale_train['bodenzahl_mean']
    for i in range(1,300):
        y_hat_scaled = x_plot @ prior_samples['beta'][i].reshape(-1,1) 
        
        y_hat = y_hat_scaled*scale_train['Winterweizen_yield_std']+scale_train['Winterweizen_yield_mean']

        ax.plot(x_range,y_hat/10,color='k',alpha=0.2)

    ax.set_title(fr'$\beta$~Normal(0,{sigma_b})', fontsize=20)    
    ax.set_xlabel('Soil Rating', fontsize=20)
    ax.set_ylabel('Yield [t/ha]', fontsize=20)
    ax.set_xlim([30,70])
    if sigma_b==1:
        ax.set_ylim([0,15])
    else:
        ax.set_ylim([-20,40])
    plt.tight_layout()
    sns.rugplot(data=Soil*scale_train['bodenzahl_std']+scale_train['bodenzahl_mean'], 
            ax=ax, color='grey')    


    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
    # %
    # =============================================================================
    # Estimate model using numpyro MCMC
    # =============================================================================
    print(f"Estimate model with sigma_b={sigma_b}")
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=800, num_warmup=1000, num_chains=2)
    mcmc.run(rng_key_, Soil=Soil, sigma_b=sigma_b, Yield=Yield)
    mcmc.print_summary()

    # Inspect MCMC sampling using arviz    
    azMCMC = az.from_numpyro(mcmc)
    azMCMC= azMCMC.assign_coords({'b_dim_0':lstColX})
    # az.summary(azMCMC)
    az.plot_trace(azMCMC);

    # Get posterior samples
    post_samples = mcmc.get_samples()
    
    # Plot posterior samples
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(prior_samples['beta'],bins=100,density=True, label='prior', color='grey');
    ax.hist(post_samples['beta'],bins=100,density=True, label='posterior', color='black');
    ax.set_title(fr'$\beta$~Normal(0,{sigma_b})', fontsize=20)
    ax.set_xlabel(fr"$\beta$", fontsize=20)
    ax.set_xlim([-1,1])
    ax.set_ylabel('Density', fontsize=20)
    ax.legend()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(15)

    # Plot regression lines
    x_range_scaled = np.linspace(-5,5,100)
    x_mean_scaled = Soil.mean(axis=0)
    x_plot = np.repeat(x_mean_scaled.reshape(1,-1),100,axis=0)
    x_plot[:,0] = x_range_scaled
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    x_range = x_range_scaled*scale_train['bodenzahl_std']+scale_train['bodenzahl_mean']
    for i in range(1,300):
        y_hat_scaled = x_plot @ post_samples['beta'][i].reshape(-1,1) 
        y_hat = y_hat_scaled*scale_train['Winterweizen_yield_std']+scale_train['Winterweizen_yield_mean']
        ax.plot(x_range,y_hat/10,color='k',alpha=0.2)

    ax.set_title(fr'$\beta$~Normal(0,{sigma_b})', fontsize=20)    
    ax.set_xlabel('Soil Rating', fontsize=20)
    ax.set_ylabel('Yield [t/ha]', fontsize=20)
    ax.set_xlim([30,70])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
        
    sns.rugplot(data=Soil*scale_train['bodenzahl_std']+scale_train['bodenzahl_mean'], 
                ax=ax, color='grey')    

    # %%
