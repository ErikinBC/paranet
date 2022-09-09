"""
Check that the elastic net model can solve along the solution path
"""

# External modules
import os
import numpy as np
import pandas as pd
import plotnine as pn
from scipy import stats

# Internal modules
from paranet.models import parametric
from paranet.utils import dist_valid

# Set up folders for saving figures
dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'tests', 'figures')
assert os.path.exists(dir_figures), 'figures folder not found'

di_msr = {'l2':'L2', 's':'# sparse'}

n = 100
p = 250
p0 = 5
k = 3
seed = 1
beta_thresh = 1e-6
beta_ratio = 500
x_raw = stats.norm().rvs([n,p],seed)
alpha = np.atleast_2d(np.repeat(0.5,k))  # Make weibull different from exponential
beta = np.zeros([p+1, k])
# Add p0 non-zero coefficients plus an intercept
beta[:p0+1] = stats.uniform(-1,2).rvs([p0+1,k], seed)


# Set up oracle model to generate data
enc_oracle = parametric(dist_valid, x_raw, alpha=alpha, beta=beta, scale_x=True, scale_t=False, add_int=True)
t, d = enc_oracle.rvs(1, seed=seed)

# Initialize the high-dimensional model
enc_elnet = parametric(dist_valid, x_raw, scale_x=True, scale_t=True)

# (i) Check lambda-max calculations & solution path
n_gamma = 10
gamma, beta_thresh = enc_elnet.find_lambda_max(x_raw, t, d, rho=0.5)
gamma_seq = np.linspace(0, gamma.max(0), n_gamma)[::-1]


# (ii) Calculate ridge
gamma_seq1 = np.exp(np.linspace(np.log(0.01), np.log(1),25))
# Get results for ridge
holder1 = []
for gamma in gamma_seq1:
    gamma_vec = np.repeat(gamma, p+1)
    enc_elnet.fit(None, t, d, gamma=gamma_vec, rho=0, beta_thresh=beta_thresh, beta_ratio=1)
    l2_beta = np.sum(enc_elnet.beta**2, 0)
    n_zero = np.sum(enc_elnet.beta == 0, 0)
    df1 = pd.DataFrame({'dist':dist_valid, 'gamma':gamma,'l2':l2_beta, 's':n_zero})
    holder1.append(df1)
# Plot L2-results
res_ridge = pd.concat(holder1).melt(['dist','gamma'],None,'msr')
gg_ridge = (pn.ggplot(res_ridge, pn.aes(x='gamma',y='value',color='dist')) + 
    pn.theme_bw() + pn.geom_line() + 
    pn.scale_x_log10() + 
    pn.facet_wrap('~msr',labeller=pn.labeller(msr=di_msr)))
gg_ridge.save(os.path.join(dir_figures, 'gg_ridge.png'), width=7, height=4)


# (iii) Get results for lasso
gamma_seq2 = np.exp(np.linspace(np.log(0.01), np.log(20), 10))
holder2 = []
for i, gamma in enumerate(gamma_seq2):
    gamma_vec = np.repeat(gamma, p+1)
    print(f'Iteration {i+1} of {len(gamma_seq2)}')
    enc_elnet.fit(None, t, d, gamma=gamma_vec, rho=1, beta_thresh=beta_thresh, beta_ratio=1, grad_tol=0.005)
    l1_beta = np.sum(np.abs(enc_elnet.beta),0)
    n_zero = np.mean(enc_elnet.beta == 0, 0)
    df2 = pd.DataFrame({'dist':dist_valid, 'gamma':gamma,'l1':l1_beta, 's':n_zero})
    holder2.append(df2)
# Plot L1-results
res_lasso = pd.concat(holder2).melt(['dist','gamma'],None,'msr')

gg_lasso = (pn.ggplot(res_lasso, pn.aes(x='gamma',y='value',color='dist')) + 
    pn.theme_bw() + pn.geom_line() + pn.scale_x_log10() + 
    pn.theme(subplots_adjust={'wspace': 0.25}) + 
    pn.facet_wrap('~msr',scales='free_y',labeller=pn.labeller(msr=di_msr)))
gg_lasso.save(os.path.join(dir_figures, 'gg_lasso.png'), width=7, height=3)


