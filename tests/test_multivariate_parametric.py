"""
Test the main parametric model class and its associated methods
"""

# External modules
import numpy as np

# Internal modules
from paranet.utils import should_fail, dist_valid
from paranet.models import parametric


# n=20;p=5;seed=1;lst_dist=dist_valid
def check_dists(n:int=20, p:int=5, lst_dist:list=dist_valid, seed:int=1):
    # (i) Create data
    np.random.seed(seed)
    x_mat = np.random.rand(n, p)
    k = len(lst_dist)
    alpha = np.random.rand(k)
    beta = np.exp(np.random.randn(p+1,k))
    t = np.random.exponential(1, n)
    t_k = np.random.exponential(1, [n,k])
    
    # (ii) Check that class can be initialized with only a distribution
    enc_para = parametric(lst_dist)
    haz_mat = enc_para.hazard(t=t,x=x_mat,alpha=alpha,beta=beta)
    assert haz_mat.shape == (n, k), 'Output does not match expected size (n,k)'
    haz_mat_k = enc_para.hazard(t=t_k,x=x_mat,alpha=alpha,beta=beta)
    assert haz_mat_k.shape == (n, k), 'Output does not match expected size (n,k)'
    assert np.all(haz_mat[:,0] == haz_mat_k[:,0]), 'First column should align with broadcast'
    assert np.all(np.delete(haz_mat, 0, 1) != np.delete(haz_mat_k, 0, 1)), 'No other columns shuold match is lst_dist is different'
    # Check that trying two columns of t will fail for k=3
    should_fail(enc_para.hazard, t=np.c_[t,t],x=x_mat,alpha=alpha,beta=beta)

    # (iii) Check that hazard can be calculated with initialized x and supplied alpha/beta
    enc_para = parametric(lst_dist, x=x_mat)
    enc_para.hazard(t=t, alpha=alpha, beta=beta)

    # (iv) Check that hazard can be calculated with initialized initialized alpha/beta and supplied x
    enc_para = parametric(lst_dist, alpha=alpha, beta=beta)

    # (v) Check that hazard can be calculated with initialized x, alpha, beta
    enc_para = parametric(lst_dist, x=x_mat, alpha=alpha, beta=beta)


    # When intercept is not default, then beta of (p,k) should error out



check_dists()