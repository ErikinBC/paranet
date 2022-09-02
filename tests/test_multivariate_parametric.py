"""
Test the main parametric model class and its associated methods
"""

# External modules
import numpy as np

# Internal modules
from paranet.models import parametric
from paranet.utils import should_fail, dist_valid, close2zero


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
        #if method == 'survival':
        #breakpoint()
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
    n, p, seed = 20, 5, 1
    test_check_dists(n, p, dist_valid, seed)

