"""
Test the main parametric model class and its associated methods
"""

# External modules
import numpy as np

# Internal modules
from paranet.models import parametric
from paranet.utils import should_fail, dist_valid, close2zero


def test_rvs_quantile(n:int=20, p:int=5, lst_dist:list=dist_valid, n_sim:int=250, tol:float=1e-2, seed:int=1):
    """Check that empirical quantiles are close to theoretical"""
    # Percentiles to check
    alpha_seq = np.arange(0.1, 1, 0.1)  

    # Generate data
    np.random.seed(seed)
    k = len(lst_dist)
    alpha = np.random.rand(1,k) + 0.5
    beta = np.random.rand(p+1,k) + 0.5
    x = np.random.randn(n, p)

    # Generate data to compare to quantiles
    enc_dist = parametric(lst_dist, x=x, alpha=alpha, beta=beta, add_int=True)
    t, _ = enc_dist.rvs(censoring=0, n_sim=n_sim, seed=seed)
    q_emp = np.quantile(t, alpha_seq, axis=2).transpose([1,2,0])
    q_theory = enc_dist.quantile(alpha_seq, x)
    assert q_emp.shape == q_theory.shape, 'Quantile output shapes not as expected'
    q_err = np.abs(q_emp - q_theory).max()
    assert q_err < tol, f'Empirical quantiles do not align with theoretical: {q_err}'
    print('~ test_rvs_quantile(): success ~')



def test_check_dists(n:int=20, p:int=5, lst_dist:list=dist_valid, seed:int=1):
    """
    Check that the different distributions (hazard/survival/pdf) work as expected for parametric
    """
    # (i) Create data
    np.random.seed(seed)
    x_mat = np.random.randn(n, p)
    k = len(lst_dist)
    alpha = np.random.rand(k)
    beta = np.random.rand(p+1,k)
    t = np.random.exponential(1, n)
    d = np.ones(t.shape)
    t_k = np.random.exponential(1, [n,k])
    
    # (ii) Check that class can be initialized with only a distribution
    methods = ['hazard', 'survival', 'pdf']
    for method in methods:
        print(f'--- Check for method {method} ---')
        enc_para = parametric(lst_dist)
        mat = getattr(enc_para, method)(t=t,x=x_mat,alpha=alpha,beta=beta)
        assert mat.shape == (n, k), 'Output does not match expected size (n,k)'
        mat_k = getattr(enc_para, method)(t=t_k,x=x_mat,alpha=alpha,beta=beta)
        assert mat_k.shape == (n, k), 'Output does not match expected size (n,k)'
        if method == 'hazard':
            # t does not matter for the hazard for the exponential distribution
            assert close2zero(mat[:,0] - mat_k[:,0]), 'First column should align with broadcast'
            assert np.all(np.delete(mat, 0, 1) != np.delete(mat_k, 0, 1)), 'No other columns should match is lst_dist is different'
        # Check that trying two columns of t will fail for k=3
        should_fail(getattr(enc_para, method), t=np.c_[t,t],x=x_mat,alpha=alpha,beta=beta)

        # (iii) Check that method can be calculated with initialized x and supplied alpha/beta
        enc_para = parametric(lst_dist, x=x_mat, scale_t=False)
        mat_x1 = getattr(enc_para, method)(t=t, alpha=alpha, beta=beta)[1:]
        # Removing one row should change the scaling factor
        enc_para = parametric(lst_dist, scale_t=False)
        mat_x2 = getattr(enc_para, method)(t=t[1:], x=x_mat[1:], alpha=alpha, beta=beta)
        assert np.all(mat_x1 != mat_x2), 'Expected scaling factor to be differenct'

        # (iv) Check that method can be calculated with initialized initialized alpha/beta and supplied x
        enc_para = parametric(lst_dist, alpha=alpha, beta=beta)
        getattr(enc_para, method)(t=t, x=x_mat).shape == (n, k), 'Output does not match expected size (n,k)'

        # (v) Check that providing all input parameters works
        enc_para = parametric(lst_dist, x=x_mat, t=t, d=d, alpha=alpha, beta=beta)
        mat_x1 = getattr(enc_para, method)()[1:]
        mat_x2 = getattr(enc_para, method)(t=t[1:], x=x_mat[1:])
        assert close2zero(mat_x1 - mat_x2), 'Removing row should not impact normlaization'


if __name__ == "__main__":
    # (i) Check that MLL solver works
    n, p, seed = 20, 4, 1
    test_check_dists(n, p, dist_valid, seed)

    # (ii) Check that quantile works as expected
    n_sim = 250000
    tol = 0.1
    seed = 1
    test_rvs_quantile(n=n, p=p, lst_dist=dist_valid, n_sim=n_sim, tol=tol, seed=seed)

    # (x) CHECK THAT WE CAN GENERATE WITH X==1