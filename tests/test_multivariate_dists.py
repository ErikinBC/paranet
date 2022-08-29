"""
Ensure that the hazard/survival/pdf functions work as expected for the multivariate distributions
"""

# External modules
import numpy as np
# Internal
from paranet.utils import dist_valid
from paranet.multivariate.dists import hazard_multi, survival_multi, pdf_multi

def test_broadcast_dists(n:int=20, p:int=5, seed:int=1, zero_tol:float=1e-10):
    """
    Check that the time vector and distribution list are broadcast so that we get back a hazard matrix of shape (n,k) with expected properties
    """
    # 
    np.random.seed(seed)
    k = len(dist_valid)
    x = np.random.randn(n,p)
    t = np.random.rand(n, k)
    alpha_beta = np.random.rand(p+1, k)

    # Loop over each of the functions
    multi_funs = {'hazard':hazard_multi, 'survival':survival_multi, 'pdf':pdf_multi}

    for tt, funs in multi_funs.items():
        for i, dist in enumerate(dist_valid):
            print(f'- Checking {dist} for {tt} -')
            # (i) Check that t is broadcast for the different distributions
            mat_broadcast = funs(alpha_beta, x, t[:,i], dist_valid)
            assert mat_broadcast.shape == (n, k), 'Output should be (n,k)'
            mat_wide = funs(alpha_beta, x, t, dist_valid)
            assert np.all(mat_broadcast[:,i] == mat_wide[:,i]), 'First column should align'
            assert not np.all(np.delete(mat_broadcast,i,1) == np.delete(mat_wide,i,1)), 'No other columns should align'

            # (ii) Check that dist is broadcast
            mat_broadcast = funs(alpha_beta, x, t, dist)
            assert mat_broadcast.shape == (n, k), 'Output should be (n,k)'
            assert np.all(mat_broadcast.var(1) > 0), 'There should be differences between columns driven by different t'
            

            # (iii) Check that t and dist are broadcast
            mat_broadcast = funs(alpha_beta, x*0, t[:,i]*0+1, dist)
            assert mat_broadcast.shape == (n, k), 'Output should be (n,k)'
            assert np.all(mat_broadcast.var(axis=0) < zero_tol), 'Column wise variation should be zero'

if __name__ == "__main__":
    # (i) Check that broadcasting works as expected
    test_broadcast_dists()