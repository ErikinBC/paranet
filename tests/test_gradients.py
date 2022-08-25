"""
CHECKS THAT ANALYTICAL GRADIENTS ALIGN WITH EXPECTED RESULTS
i) ANALYTICAL GRADIENT ALIGNS WITH FINITE DIFFERENCES
ii) SOLVER ACHIEVES HIGHEST LOG-LIKELIHOOD
"""

# External modules
import numpy as np
import pandas as pd

# Internal modules
from paranet.dists import surv_dist
from paranet.gradient import log_lik, grad_ll
from paranet.utils import dist_valid, vstack_pd
from paranet.solvers_grad import wrapper_grad_solver

# Scale and shape parameters shared by functions
lam = np.array([0.5, 1, 1.5])
alph = np.array([0.5, 1, 1.5])
k = len(lam)

def test_solver(n_sim:int=100000, tol:float=1e-3) -> None:
    """
    Test that a large enough sample size convergences to close to the oracle parameters

    Inputs
    ------
    nsim:               Number of points to sample from quantile range
    tol:                Tolerance for largest absolute value between MLE to oracle coefficients
    """
    for dist in dist_valid:
        gen_dist = surv_dist(dist, scale=lam, shape=alph)
        T_sim = gen_dist.quantile(p=np.linspace(1/n_sim,1-1/n_sim, n_sim))
        D_sim = np.zeros([n_sim, k],dtype=int) + 1
        hat_coef = wrapper_grad_solver(T_sim, D_sim, dist)
        alph_lam = np.vstack([gen_dist.shape, gen_dist.scale])
        print(f'-- Dist {dist} --')
        print(alph_lam.round(5))
        assert np.abs(hat_coef - alph_lam).max() < tol, f'MLE did not converge to tolerance {tol} for {dist}'


def test_finite_differences(n_sim:int=100, seed:int=1, tol:float=1e-4, epsilon:float=1e-10) -> None:
    """
    Test that the analytical gradient aligns with the analytical one
    
    Inputs
    ------
    nsim:               Number of points to sample from quantile range
    seed:               Controls reproducability of rvs()
    tol:                Tolerance for largest absolute value between gradients
    epsilon:            How much to add to the parameter vector to evaluate gradients
    """
    # Generate data
    di_TD_dist = dict.fromkeys(dist_valid)
    for dist in dist_valid:
        gen_dist = surv_dist(dist, scale=lam, shape=alph)
        di_TD_dist[dist] = gen_dist.rvs(n_sim, seed)

    # Perturb parameter vectors for finite differences
    lam_high, lam_low = lam + epsilon, lam - epsilon
    alph_high, alph_low = alph + epsilon, alph - epsilon
    # Loop over distributions
    for dist in dist_valid:
        T_dist, D_dist = di_TD_dist[dist]
        # (i) Gradient for lambda (0th position is for alpha)
        grad_lam_eps = (log_lik(T_dist, D_dist, lam_high, alph, dist) - log_lik(T_dist, D_dist, lam_low, alph, dist)) / (2*epsilon)
        grad_lam_fun = grad_ll(T_dist, D_dist, lam_high, alph, dist)[1]

        # (ii) Gradient for alpha (0th position is for alpha)
        grad_alph_eps = (log_lik(T_dist, D_dist, lam, alph_high, dist) - log_lik(T_dist, D_dist, lam, alph_low, dist)) / (2*epsilon)
        grad_alph_fun = grad_ll(T_dist, D_dist, lam_high, alph, dist)[0]
        
        # (iii) Print
        df_grad_lam = pd.DataFrame({'param':'lambda','finite_diff':grad_lam_eps, 'analytical':grad_lam_fun})
        df_grad_alph = pd.DataFrame({'param':'alpha','finite_diff':grad_alph_eps, 'analytical':grad_alph_fun})
        df_grad = vstack_pd(df_grad_lam, df_grad_alph)
        print(f'-- Dist {dist} --')
        print(df_grad)
        assert np.abs(grad_lam_eps - grad_lam_fun).max() < tol, f'Gradient difference is larger than tol: {tol}'
        assert np.abs(grad_alph_eps - grad_alph_fun).max() < tol, f'Gradient difference is larger than tol: {tol}'
        print(f'Functional gradient aligns with finite differences for: {dist}')
        

if __name__ == "__main__":
    # (i) Check that MLL solver works
    n = 100000
    tol_param = 1e-3
    test_solver(n, tol=tol_param)

    # (ii) Check that gradients align with finite differences
    n_sim = 100
    seed = 1
    tol = 1e-4
    epsilon = 1e-10
    test_finite_differences(n_sim, seed, tol, epsilon)

    print('~~~ test_gradients completed without errors ~~~')    