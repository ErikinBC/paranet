"""
Ensure that the hazard/survival/pdf functions work as expected for the multivariate distributions
"""

# External modules
import numpy as np
# Internal
from paranet.utils import dist_valid, close2zero
from paranet.multivariate.dists import hazard_multi, survival_multi, pdf_multi, quantile_multi


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
            assert close2zero(mat_broadcast.var(axis=0), zero_tol), 'Column wise variation should be zero'


def test_broadcast_quantile(n:int=20, p:int=5, seed:int=1, zero_tol:float=1e-10):
    np.random.seed(seed)
    k = len(dist_valid)
    x = np.random.randn(n,p)
    alpha_beta = np.random.rand(p+1, k)

    # (i) Check that the float and vector works
    di_p = {'val':0.5, 'vec':np.arange(0.1,1,0.1)}
    for tt, val in di_p.items():
        print(f'- Checking quantile for {tt} -')
        # No broadcasting
        quant_all = quantile_multi(val, alpha_beta, x, dist_valid)
        assert np.all(quant_all.var(1) > 0), 'Different dists should have different quantiles'
        if isinstance(val, float):
            assert quant_all.shape == (n, k), 'Should have dimensions (n,k)'
        else:
            assert quant_all.shape == (n, k, len(val)), 'Should have dimensions (n,k)'
        
        for i, dist in enumerate(dist_valid):
            # Broadcasting distribution
            quant_broadcast = quantile_multi(val, alpha_beta, x, dist)
            assert np.all(quant_broadcast[:,i] == quant_all[:,i]), 'Matching distributions should have the same quantile'
            assert np.all(np.delete(quant_broadcast, i, 1) != np.delete(quant_all, i, 1)), 'Non-matching distributions should have different quantiles'
            # Broadcasted distribtuion should have zero variation
            quant_zero_covar = quantile_multi(val, alpha_beta*0+1, x*0, dist)
            assert close2zero(quant_zero_covar.var(1), zero_tol) or close2zero(quant_zero_covar.var(0), zero_tol), 'At least one dimension should have zero variation'



if __name__ == "__main__":
    # (i) Check that broadcasting works as expected
    test_broadcast_dists()

    # (ii) Check the quantile function
    test_broadcast_quantile()