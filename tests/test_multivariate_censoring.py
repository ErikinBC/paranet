"""
Check that the multivariate censoring approach works when the covariates come from a normal distribution
"""

# External modules
import numpy as np
# Internal modules
from paranet.utils import dist_valid
from paranet.models import parametric


def test_check_censoring(n:int=250000, p:int=5, n_sim:int=1, lst_dist:list=dist_valid, n_points:int=250, upper_constant:float=20, tol:float=0.01, seed:int=1) -> None:
    """
    Make sure that different censoring targets are met
    
    Inputs
    ------
    n:                  Number of rows for the design matrix
    p:                  Number of columns (i.e. covariates) for the design matrix
    n_sim:              Number of simulations to run for each observation
    lst_dist:           Distribution list to loop over
    n_points:           Number of points to pass into brute-force integration
    upper_constant:     How many multiples above median of 1/n_points risk quantile to perform integration over
    tol:                How big can discrepancy be between empirical and expected censoring to throw error?
    seed:               Reproducability seed
    """
    # Censoring values to check
    censor_seq = np.arange(0.25, 1, 0.25)
    l2_seq = np.array([1, 2, 3])
    shape_seq = np.array([0.5, 1, 2])
    # Will likely need to adjust the formula for including an intercept
    b0_seq = np.array([0])
    n_perm = len(lst_dist) * len(shape_seq) * len(l2_seq) * len(censor_seq)
    n_perm -= 2*(len(l2_seq) * len(censor_seq))

    # (i) Generate covariate data
    np.random.seed(seed)
    x = np.random.randn(n,p)
    j = 0
    for dist in lst_dist:
        for l2 in l2_seq:
            # Set up beta
            beta = np.repeat(np.sqrt(l2/p), p).reshape([p,1])
            for shape in shape_seq:
                for censoring in censor_seq:
                    if dist == 'exponential' and (shape != 1):
                        # Shape parameter is redundant for exponential                
                        continue
                    j += 1
                    print(f'Iteration {j} of {n_perm}')
                    enc_dist = parametric(dist, x=x, alpha=shape, beta=beta, add_int=False)
                    _, d = enc_dist.rvs(censoring, n_sim, n_points=n_points, upper_constant=upper_constant)
                    emp_censor = 1-d.mean()
                    err_censor = np.abs(emp_censor - censoring)
                    assert err_censor < tol, f'Empirical ({emp_censor:.3f}) and expected: ({censoring:.3f}) censoring did not align by: {err_censor:.3f} for dist={dist}, l2={l2:.2f}, shape={shape:.2f}, censoring={censoring}'


if __name__ == "__main__":
    # (i) Check that find_exp_scale_censoring_multi() from rvs_multi() works as expected
    n, p = 250000, 5
    n_sim = 1
    tol = 0.01
    n_points, upper_constant = 250, 20
    seed = 1
    test_check_censoring(n, p, n_sim, dist_valid, n_points, upper_constant, tol, seed)
    