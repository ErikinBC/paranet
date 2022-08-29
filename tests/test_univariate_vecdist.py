"""
Check that the vectorization of distribution vector works as expected
"""

# External modules
import numpy as np

# Internal modules
from paranet.univariate.dists import surv_dist
from paranet.utils import dist_valid
from paranet.univariate.solvers_grad import wrapper_grad_solver


def test_one2many(n_sim:int=10000):
    """
    Check that a single survival vector can fit multiple distributions and that the empirical estimates align with the ground truth
    """
    t_seq = np.linspace(0.5,1,10)
    for i, dist in enumerate(dist_valid):
        dist_gt = surv_dist(dist=dist, scale=2, shape=3)
        t, d = dist_gt.rvs(n_sim, censoring=0.25, seed=1)
        beta_hat = wrapper_grad_solver(t, d, dist_valid)
        dist_all = surv_dist(dist=dist_valid, scale=beta_hat[1], shape=beta_hat[0])
        haz_hat = dist_all.hazard(t_seq)
        haz_act = dist_gt.hazard(t_seq)
        idx_min = np.argmin(np.mean((haz_hat - haz_act)**2,0))
        assert i == idx_min, 'expected closest approximation to be aligned with ground truth distribution'
    print('~ End of test_one2many() ~')


def test_solver(n_sim:int=1000000, tol:float=1e-2, seed:int=1) -> None:
    """
    Test that a large enough sample size convergences to close to the oracle parameters

    Inputs
    ------
    nsim:               Number of points to sample from quantile range
    tol:                Tolerance for largest absolute value between MLE to oracle coefficients
    """
    shape = np.array([1,0.5,0.5])
    scale = np.array([1,1,1])
    # Create multiple survival distributions
    dist_all = surv_dist(dist = dist_valid, scale=scale, shape=shape)
    # Loop over censoring
    censoring_seq = [0, 1/4, 1/2]
    for censoring in censoring_seq:
        print(f'Fitting with {censoring:.2f} censoring')
        # Draw from different distributions
        t, d = dist_all.rvs(n_sim, censoring, seed)
        hat_coef = wrapper_grad_solver(t, d, dist_all.dist)
        alph_lam = np.vstack([dist_all.shape, dist_all.scale])
        err = np.abs(hat_coef - alph_lam).max()
        assert err < tol, f'MLE did not converge to tolerance {tol} for with censoring {censoring}: {err}'
    print('~ End of test_solver() ~')



if __name__ == "__main__":
    # (i) Check that MLL solver works
    n = 1000000
    tol_param = 0.01
    test_solver(n, tol=tol_param, seed=1)

    # (ii) That that we can fit multiple distributions to a single array
    n_sim = 10000
    test_one2many(n_sim)

    print('~~~ test_vec_dist completed without errors ~~~')    