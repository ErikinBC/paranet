"""
DO UNIT TESTS
"""

# External modules
import numpy as np

# Internal modules
from dists import surv_dist
from gradient import log_lik, grad_ll_scale, grad_ll_shape
from utils import dist_valid

n_sim, seed = 100, 1
lam = np.array([0.5, 1.5])
alph = np.array([0.5, 1.5])
epsilon = 1e-10
tol = 1e-4
# Censoring
D_dist = np.zeros([n_sim, len(lam)],dtype=int) + 1
# Test the graduent different distributions
for dist in dist_valid:
    gen_dist = surv_dist(dist, scale=lam, shape=alph)
    T_dist = gen_dist.rvs(n_sim, seed)
    lam_high, lam_low = lam + epsilon, lam - epsilon
    alph_high, alph_low = alph + epsilon, alph - epsilon
    # (i) Gradient for lambda
    grad_lam_eps = (log_lik(T_dist, D_dist, lam_high, alph, dist) - log_lik(T_dist, D_dist, lam_low, alph, dist)) / (2*epsilon)
    grad_lam_fun = grad_ll_scale(T_dist, D_dist, lam_high, alph, dist)
    assert np.abs(grad_lam_eps - grad_lam_fun).max() < tol, f'Gradient difference is larger than tol: {tol}'

    # (ii) Gradient for alpha
    grad_alph_eps = (log_lik(T_dist, D_dist, lam, alph_high, dist) - log_lik(T_dist, D_dist, lam, alph_low, dist)) / (2*epsilon)
    grad_alph_fun = grad_ll_shape(T_dist, D_dist, lam_high, alph, dist)
    assert np.abs(grad_alph_eps - grad_alph_fun).max() < tol, f'Gradient difference is larger than tol: {tol}'

