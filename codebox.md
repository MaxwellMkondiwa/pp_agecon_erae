
Formatted using https://hackmd.io for syntax highlighting and 
line numbers.

Code box 3.1
```python=
def model(Soil,Yield=None):
	beta = numpyro.sample('beta', 
                  dist.Normal(0,1))
	sigma = numpyro.sample('sigma', 
                  dist.Exponential(1))
	numpyro.sample('Yield', 
           dist.Normal(Soil*beta, sigma), obs=Yield)
```

Code box 3.2
```python=
def utility(x, lam, rho):
    return jnp.where(x > 0, 
             jnp.where(x>0,x,0)**rho, 	
            -lam * (-jnp.where(x>0,0,x))**rho)
def model_PT(gain, loss, certain, took_gamble=None):
    # Define priors
    lam=numpyro.sample('lam',
               dist.TruncatedNormal(
                   loc=2,scale=1.0,low=1.,high=4.))
    rho=numpyro.sample('rho',
               dist.TruncatedNormal(
                   loc=1,scale=1.0,low=0.5,high=1.))
    # Calculate utility of gamble and certain option
    util_reject = utility(certain, lam, rho)
    util_accept = (0.5*utility(gain, lam, rho) 
                   + 0.5 * utility(loss, lam, rho))
    util_diff = util_accept - util_reject
    # Calculate probability of accepting gamble
    p_accept = 1/(1+jnp.exp(-util_diff))
    # Choice to take gamble
    numpyro.sample('took_gamble', 
                   dist.BernoulliProbs(p_accept), 
                   obs=took_gamble)
```


Code box 3.3
```python=
def model_POF(Z, X, T=None, Y=None):
    # Define priors
    alpha_out = numpyro.sample('alpha_out', 
                dist.Normal(0.,1).expand([X.shape[1]]))
    beta_treat = numpyro.sample('beta_treat',
                dist.Normal(0.,1).expand([X.shape[1]]))
    sigma_Y = numpyro.sample('sigma_Y', 
                dist.Exponential(1))
    # Calculate expected outcomes
    Y0 = X @ alpha_out 
    tau = Z @ beta_treat 
    Y1 = Y0 + tau 
    # Define distribution of treatment
    T = numpyro.sample('T', 
               dist.Bernoulli(logits=tau), obs=T)
    # Define distribution of outcome
    numpyro.sample('Y', 
               dist.Normal(Y1*T + Y0*(1-T), sigma_Y), 
               obs=Y)
```