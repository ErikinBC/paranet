"""
Check that integration works as expected
"""

# External functions
import numpy as np
from scipy.integrate import quad

# Internal functions
from paranet.univariate.dists import quantile, survival, pdf, rvs
from paranet.utils import dist_valid, grad_finite_differences, t_wide

# Set up parameters to test
scale = t_wide([0.5, 1, 2])
shape = scale[:,::-1]


def test_pdf_survival(tol:float=1e-5):
    """Check that finite differences of survival is -f(t)"""
    xseq = np.arange(0.5,3,0.5,dtype=float)
    for dist in dist_valid:
        f_t = pdf(xseq, scale=scale, shape=shape, dist=dist)
        dS_t = grad_finite_differences(survival, xseq, scale=scale, shape=shape, dist=dist)
        assert np.abs(f_t + dS_t).max() < tol, f'f(t) != -dS(t)/dt for {dist}'
    print('~ test_pdf_survival(): success ~')


def test_pdf_intergration(tol:float=1e-4):
    """Check that f(t) integrates to one"""
    for k in range(scale.shape[1]):
        scale_k, shape_k = scale[:,k][0], shape[:,k][0]
        for dist in dist_valid:
            b = 2*quantile(0.999, scale_k, shape_k, dist)[0][0]
            y, _ = quad(func=pdf,a=0,b=b, args=(scale_k, shape_k, dist))
            err = np.abs(1 - y)
            assert err < tol , f'pdf integration is not close enough to one: {err}'
    print('~ test_pdf_intergration(): success ~')


def test_rvs_quantile(n_sim:int=1000000, tol:float=1e-2, seed:int=1):
    """Check that empirical quantiles are close to theoretical"""
    for dist in dist_valid:
        alpha_seq = np.arange(0.1, 1, 0.1)
        t, _ = rvs(n_sim, scale, shape, dist, seed=seed)
        q_emp = np.quantile(t, alpha_seq, axis=0)
        q_theory = quantile(alpha_seq, scale, shape, dist)
        q_err = np.abs(q_emp - q_theory).max()
        assert q_err < tol, f'Empirical quantiles do not align with theoretical for {dist}: {q_err}'
    print('~ test_rvs_quantile(): success ~')


if __name__ == "__main__":
    # (i) Check that (S(t+e)-S(t-e))/(2e) ~ -f(t)
    tol_dSt = 1e-5
    test_pdf_survival(tol_dSt)

    # (ii) Check that f(t) integrates to one
    tol_int = 1e-4
    test_pdf_intergration(tol_int)

    # (iii) Check that density of rvs() aligns with F^{-1}(t)
    n_sim = 1000000
    tol_rvs = 1e-2
    seed = 1
    test_rvs_quantile(n_sim, tol_rvs, seed)

    print('~~~ test_integration completed without errors ~~~')    
