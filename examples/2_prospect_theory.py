# %%
"""Example Prospect Theory model in PP

Hugo Storm Feb 2024

"""


# Paper links
# Blog for ML estimation https://www.thegreatstatsby.com/posts/2021-03-08-ml-prospect/ 

# Source for prior on lamda and rho 
# https://www.pnas.org/doi/full/10.1073/pnas.0806761106
import os
import sys
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib as mpl

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

az.style.use("arviz-darkgrid")
plt.style.use('default')
mpl.rcParams['figure.dpi'] = 300

wd = '/workspaces/pp_agecon_erae'
os.chdir(wd)
sys.path.append(wd)

# %%
def load_data():
    """Wrapper to load data from thegreatstatsby.com"""
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
    """Helper function to sample from prior predictive distribution"""
    rng_key, rng_key_ = random.split(rng_key)
    prior_predictive = Predictive(model, num_samples=num_samples)
    prior_samples = prior_predictive(rng_key, **kwargs)
    return prior_samples

#%%
def pyro_inference(rng_key,model, num_samples=200, num_warmup=1000, 
                   num_chains=2, **kwargs):
    """Wrapper for NUTS sampling in numpyro"""
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains)
    mcmc.run(rng_key_, **kwargs)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    
    return mcmc, samples

# %%
def plot_utility(lam, rho, ax=None, addTitle=True, color='blue'):
    """ Helper function to plot utility function"""
    x_range = jnp.linspace(-10,10,20)
    util = utility(x_range,  lam=lam, rho=rho)

    if ax is None:
        f, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_range,util, color=color,linewidth=0.5);
    ax.plot([-12, 12], [-12, 12], ls="--", c="lightgray");
    
    ax.set_ylabel('Utility, u(x)', fontsize=20)
    ax.set_xlabel('Payout (x)', fontsize=20)
    if addTitle:
        ax.set_title(f'$\lambda$ = {lam:.2f}, $\\rho$ = {rho:.2f}, $\\mu$ = 1', fontsize=20)
    if ax is None:
        plt.xlim([-12, 12])
        plt.ylim([-12, 12])
        plt.show()
        
# %%
def utility(x, lam, rho):
    # Note the additional jnp.where are required to make sure that there are no nans in 
    # gradient calculations, see https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
    return jnp.where(x > 0, 
                     jnp.where(x>0,x,0)**rho, 
                     -lam * (-jnp.where(x>0,0,x))**rho)
# %%
# define a model for prospect theory in numpyro
def model_PT(gain, loss, certain, took_gamble=None):
    # Define priors
    lam = numpyro.sample('lam', dist.TruncatedNormal(loc=2, scale=1.0,low=1., high=4.))
    # rho = numpyro.sample('rho', dist.TruncatedNormal(loc=1, scale=1.0,low=0.5, high=1.5))
    rho = numpyro.sample('rho', dist.TruncatedNormal(loc=1, scale=1.0,low=0.5, high=1.))
    
    # Calculate utility of gamble and certain option
    util_reject =  utility(certain, lam, rho)
    util_accept = 0.5 * utility(gain, lam, rho) + 0.5 * utility(loss, lam, rho)
    util_diff =  util_accept - util_reject
    
    # Calculate probability of accepting gamble
    p_accept = 1/(1+jnp.exp(-util_diff))
    
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
    # Get data
    dat_X_train = dict(gain=df['gain'].values,
                       loss=df['loss'].values,
                       certain=df['cert'].values,
                        )
    dat_XY_train = dict(gain=df['gain'].values,
                       loss=df['loss'].values,
                       certain=df['cert'].values,
                       took_gamble=df['took_gamble'].values
                        )
    # %%
    # =============================
    # Create Figure Appendix A-1
    # =============================
    # Illustrate utility function
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15, 5),tight_layout=True)
    for lam, rho, ax in [(1.,0.5,ax1),(2.6,.65,ax2),(4.,1.,ax3)]:
        # fig, ax = plt.subplots(figsize=(4, 3))
        plot_utility(lam, rho, ax=ax) # use prior extremes
        ax.set_ylim([-10, 10])
        ax.set_xlim([-10, 10])
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(20)
        
    fig.savefig(f'figures/PT_utilPriorExtrem_AppendixA1.png',dpi=300,
        bbox_inches='tight')
    # %%
    # =============================
    # Sample for prior
    # =============================
    prior_sam = sample_prior(rng_key, model_PT,**dat_X_train)

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
    # =============================
    # Perform posterior sampling
    # =============================
    mcmc_M1, post_sam = pyro_inference(rng_key,model_PT,**dat_XY_train)
    
    # %%
    # Transform mcmc object to arviz object (for plotting posterior samples)
    azMCMC = az.from_numpyro(mcmc_M1)
    # %%
    # =============================
    # Create figure 4
    # =============================
    f, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 5),tight_layout=True)

    # Figure 4 (left): prior samples    
    for i in range(0,200):
        plot_utility(lam=prior_sam['lam'][i] , 
                     rho=prior_sam['rho'][i],ax=ax1, 
                     addTitle=False,
                     color='black');
    ax1.set_xlim([-10, 10])
    ax1.set_ylim([-12, 10])
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(20)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.set_title('Prior samples', fontsize=20)
    
    # Figure 4 (right): posterior samples    
    for i in range(0,200):
        plot_utility(lam=post_sam['lam'][i] , 
                     rho=post_sam['rho'][i],
                     ax=ax2,
                     addTitle=False,
                     color='black');
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([-12, 10])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontsize(20)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
    
    ax2.set_title('Posterior samples', fontsize=20)
    
    f.savefig(f'figures/PT_figure4.png',dpi=300,
        bbox_inches='tight')
    plt.show()
    # %%
    # =============================
    # Create figure 5
    # =============================
    # Figure 4 left and middle: Marginal posterior samples
    f, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5),tight_layout=True)
    az.plot_posterior(azMCMC,
                      var_names=["rho", "lam"],
                      textsize=20,
                      round_to=3,
                      figsize=(10,5),
                      ax=(ax1,ax2));
    for ax in (ax1,ax2):
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(17)
    # Figure 4 right: Pairplot    
    az.plot_pair(
        azMCMC,
        # centered,
        var_names=["rho", "lam"],
        kind="kde",
        textsize=20,
        figsize=(5,5),
        kde_kwargs={
            "hdi_probs": [0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
            "contourf_kwargs": {"cmap": "Blues"},
        },
        ax = ax3
    )
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    plt.show()
    f.savefig(f'figures/PT_figure5.png',dpi=300,
        bbox_inches='tight')
# %%
