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


def test_elnet_solution_path(n:int=100, p:int=250, p0:int=5, seed:int=1):
    """
    Checks that the Lasso solution path works as expected

    Inputs
    ------
    n:                  Number of observations
    p:                  Number of covariates
    p0:                 Number of "signal" covariates (others are noise)
    """
    k = len(dist_valid)
    x_raw = stats.norm().rvs([n,p],seed)
    alpha = np.atleast_2d(np.repeat(0.5,k))  # Make weibull different from exponential
    beta = np.zeros([p+1, k])
    # Add p0 non-zero coefficients plus an intercept
    beta[:p0+1] = stats.uniform(-1,2).rvs([p0+1,k], seed)

    # Set up oracle model to generate data
    enc_oracle = parametric(dist_valid, x_raw, alpha=alpha, beta=beta, scale_x=True, scale_t=False, add_int=True)
    t, d = enc_oracle.rvs(1, seed=seed)

    # Initialize the high-dimensional model
    enc_elnet = parametric(dist_valid, x_raw, scale_x=True, scale_t=False)

    # Check lambda-max calculations & solution path
    rho_seq = [1/3, 2/3, 1]
    rho_lbl = ['1/3','2/3','1']
    di_rho = dict(zip(rho_seq, rho_lbl))
    n_gamma = 20
    gamma_lb = 1e-2
    # Select first ten covariates + shape/scale
    n_beta = 10
    cn = ['gamma','shape','scale']+[f'b{j}' for j in range(1,n_beta+1)]

    holder = []
    for rho in rho_seq:
        gamma, beta_thresh = enc_elnet.find_lambda_max(x_raw, t, d, rho=rho)
        gamma_ub = gamma.max(0)
        gamma_seq = np.exp(np.linspace(np.log(gamma_lb), np.log(gamma_ub), n_gamma))[::-1]
        for g in gamma_seq:
            print(f'rho={rho:.2f}, gamma={g.mean():.3f}')
            g_mat = np.tile(g,[p+1,1])
            alpha_beta_init = np.vstack([enc_elnet.alpha, enc_elnet.beta])
            enc_elnet.fit(x_raw, t, d, g_mat, rho, beta_thresh=beta_thresh, alpha_beta_init=alpha_beta_init)
            ab_g = np.vstack([g,enc_elnet.alpha, enc_elnet.beta[:11]])
            df_g = pd.DataFrame(ab_g, columns=dist_valid, index=cn).assign(rho=rho)
            # Tidy up data
            df_g = df_g.rename_axis('cn').set_index('rho',append=True).melt(ignore_index=False,var_name='dist').set_index('dist',append=True).reset_index('cn').pivot(None,'cn','value').set_index('gamma',append=True).melt(ignore_index=False).reset_index()
            holder.append(df_g)
    # Merge and plot
    df_path = pd.concat(holder).reset_index(drop=True)
    df_path['rho'] = pd.Categorical(df_path['rho'],rho_seq).map(di_rho)
    # Set up variable types for plotting in colour
    di_var = {**{'shape':'shape', 'scale':'scale'}, 
            **{f'b{j}':'signal' for j in range(1,p0+1)},
            **{f'b{j}':'noise' for j in range(p0+1,n_beta+1)}}
    df_path['colz'] = pd.Categorical(df_path['cn'].map(di_var),['shape','scale','signal','noise'])
    df_path = df_path.query("~(dist=='exponential' & cn=='shape')")
    gg_path = (pn.ggplot(df_path, pn.aes(x='gamma', y='value', color='colz', group='cn')) + 
        pn.theme_bw() + pn.geom_line() + 
        pn.facet_wrap('~dist + rho',labeller=pn.label_both,scales='free') + 
        pn.scale_color_discrete(name='Variable type') + 
        pn.theme(subplots_adjust={'hspace': 0.4, 'wspace':0.2}) + 
        pn.labs(y='Coefficient', x='gamma'))
    gg_path.save(os.path.join(dir_figures, 'gg_path.png'), width=11, height=8)


if __name__ == '__main__':
    # Run the solution path
    n, p, p0, seed = 100, 250, 5, 1
    test_elnet_solution_path(n, p, p0, seed)