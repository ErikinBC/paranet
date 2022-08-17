"""
CHECKS THAT ANALYTICAL GRADIENTS ALIGN WITH EXPECTED RESULTS
i) ANALYTICAL GRADIENT ALIGNS WITH FINITE DIFFERENCES
ii) SOLVER ACHIEVES HIGHEST LOG-LIKELIHOOD
"""

# External modules
import numpy as np

# Internal modules
from paranet.dists import surv_dist
from paranet.gradient import log_lik, grad_ll
from paranet.utils import dist_valid

# Simulation number and seed for reproducability
n_sim, seed = 100, 1
# Scale and shape parameters
lam = np.array([0.5, 1, 1.5])
alph = np.array([0.5, 1, 1.5])
# Set up parameters for finite differences
epsilon = 1e-10
lam_high, lam_low = lam + epsilon, lam - epsilon
alph_high, alph_low = alph + epsilon, alph - epsilon

# Absolute tolerance difference
tol = 1e-4
# Censoring
D_dist = np.zeros([n_sim, len(lam)],dtype=int) + 1
di_T_dist = dict.fromkeys(dist_valid)
for dist in dist_valid:
    gen_dist = surv_dist(dist, scale=lam, shape=alph)
    di_T_dist[dist] = gen_dist.rvs(n_sim, seed)

def test_solver():
    for dist in dist_valid:
        T_dist = di_T_dist[dist]
        grad_ll(T_dist, D_dist, lam, alph, dist)
    
#test_solver()

def test_finite_differences():
    """Test that the analytical gradient aligns with the analytical one"""
    for dist in dist_valid:
        T_dist = di_T_dist[dist]
        # (i) Gradient for lambda (0th position is for alpha)
        grad_lam_eps = (log_lik(T_dist, D_dist, lam_high, alph, dist) - log_lik(T_dist, D_dist, lam_low, alph, dist)) / (2*epsilon)
        grad_lam_fun = grad_ll(T_dist, D_dist, lam_high, alph, dist)[1]
        assert np.abs(grad_lam_eps - grad_lam_fun).max() < tol, f'Gradient difference is larger than tol: {tol}'

        # (ii) Gradient for alpha (0th position is for alpha)
        grad_alph_eps = (log_lik(T_dist, D_dist, lam, alph_high, dist) - log_lik(T_dist, D_dist, lam, alph_low, dist)) / (2*epsilon)
        grad_alph_fun = grad_ll(T_dist, D_dist, lam_high, alph, dist)[0]
        assert np.abs(grad_alph_eps - grad_alph_fun).max() < tol, f'Gradient difference is larger than tol: {tol}'
        
        print(f'Functional gradient aligns with finite differences for: {dist}')
