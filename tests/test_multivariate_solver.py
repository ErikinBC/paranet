"""
Test the gradient solver for the non-regularized models
"""

# External modules
import numpy as np

# Internal modules
from paranet.utils import dist_valid
from paranet.models import parametric

def test_param_consistency(n:int=10000, p:int=10, lst_dist:list=dist_valid, n_sim:int=10, bhat_tol:float=0.01):
    """
    For a fixed p, increasing n should achieve asymtotic convergence of the parameters
    """
    k = len(lst_dist)
    seed_range = np.arange(1, n_sim+1)
    for s in seed_range:
        np.random.seed(s)
        # Generate ground-truth data
        if s % 1 == 0:
            print(f'Seed {s}')
        alpha = np.random.rand(1,k)+0.5
        alpha[:,np.where(np.array(lst_dist) == 'exponential')[0][0]] = 1
        beta = np.random.uniform(-1,1,[p+1,k])
        alpha_beta = np.vstack([alpha, beta])
        x = np.random.randn(n,p)
        # Draw data from covariates
        enc_dgp = parametric(lst_dist, x, alpha=alpha, beta=beta, add_int=True, scale_x=False, scale_t=False)
        t, d = enc_dgp.rvs(censoring=0, n_sim=1, seed=s)
        t, d = np.squeeze(t), np.squeeze(d)
        # Fit model
        enc_dgp.fit(x, t, d)
        alpha_beta_hat = np.vstack([enc_dgp.alpha, enc_dgp.beta])
        bhat_err = np.abs(alpha_beta - alpha_beta_hat).max()
        assert bhat_err < bhat_tol, f'Largest discrepancy between actual and expected is greater than {bhat_tol}: {bhat_err:.6f} for simulation {s}'


if __name__ == '__main__':
    # Check that with large enough sample size we can arbitraily close to expected solution
    n, p = 250000, 10
    n_sim, bhat_tol = 10, 0.01
    test_param_consistency(n, p, dist_valid, n_sim, bhat_tol)