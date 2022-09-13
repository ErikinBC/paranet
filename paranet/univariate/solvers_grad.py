"""
WRAPPER TO PERFORM GRADIENT OPTIMIZATION
"""

# https://hastie.su.domains/Papers/glmnet.pdf
# https://www.cs.cmu.edu/~ggordon/10725-F12/slides/25-coord-desc.pdf
# https://github.com/pmelchior/proxmin
# https://github.com/lowks/gdprox
# https://pypi.org/project/pyproximal/

# External modules
import numpy as np
from scipy.optimize import minimize
from scipy.optimize.optimize import OptimizeResult

# Internal modules
from paranet.univariate.gradient import grad_ll, log_lik
from paranet.utils import di_bounds, is_vector, shape_scale_2vec, _get_p_k, broadcast_td_dist, t_long



def log_lik_vec(shape_scale:np.ndarray, t:np.ndarray, d:np.ndarray, dist:str or list) -> float:
    """
    MAPS A [p*k,1] PARAMETER VECTOR/MATRIX TO A SINGLE LOG-LIKELIHOOD

    Inputs
    ------
    scale_shape:            A [p*k,1] vector where the p+1'st entry exists if k>1
    t:                      A vector/matrix of the time-to-event
    d:                      A vector/matrix of the censoring indicator
    dist:                   One of the valid parametric distributions

    Returns
    -------
    A float which is the sum of the different log-likelihoods
    """
    # Input checks/transform
    is_vector(shape_scale)  # Should be a flattened vector
    p, k = _get_p_k(t)
    # Reshape into order needed for log-likelihood function
    shape_scale = shape_scale.reshape([k,p]).T
    assert shape_scale.shape[0] >= 2, 'There should be at least two rows (first is shape, second is scale)'
    shape, scale = shape_scale_2vec(shape_scale)
    lls = log_lik(t, d, scale, shape, dist).sum()
    return lls


def grad_ll_vec(shape_scale:np.ndarray, t:np.ndarray, d:np.ndarray, dist:str) -> float:
    """
    MAPS THE [p*k,1] TO A [p,k] PARAMETER VECTOR/MATRIX TO CALCULATE GRADIENTS AND THE RETURNS TO THE ORIGINAL [p*k,1] SIZE

    Inputs
    ------
    scale_shape:            A (p+1 x k) matrix that contains the scale, shape parameters. The first row corresponds to the shape parameter, and the rows are the different scale parameters. For the intercept-only only, p+1 = 2
    t:                      A vector/matrix of the time-to-event
    d:                      A vector/matrix of the censoring indicator
    dist:                   One of the valid parametric distributions

    Returns
    -------
    A [(p+1)*k,1] vector where the p+2'th exists if k>1
    """
    # Input checks/transform
    is_vector(shape_scale)  # Should be a flattened vector
    p, k = _get_p_k(t)
    # Reshape into order needed for log-likelihood function
    shape_scale = shape_scale.reshape([k,p]).T
    assert shape_scale.shape[0] >= 2, 'There should be at least two rows (first is shape, second is scale)'
    shape, scale = shape_scale_2vec(shape_scale)
    grad_mat = grad_ll(t, d, scale, shape, dist)
    grad_vec = grad_mat.T.flatten()
    return grad_vec


def opt_success(opt:OptimizeResult, tol_grad:float=1e-3) -> bool:
    """
    Checks whether the minimize() result aligns with expectations

    Inputs
    ------
    opt:                dsdfs
    tol_grad:           sdfs

    Returns
    -------
    A boolean indicating whether both conditions have been met
    """
    assert all([hasattr(opt, a) for a in ['jac','success']]), 'Optimize result does not have expected attributes'
    grad_success = np.abs(opt.jac).max() < tol_grad
    res = opt.success and grad_success
    return res


def wrapper_grad_solver(t:np.ndarray, d:np.ndarray, dist:str or list, x0:None or np.ndarray=None, tol_grad:float=1e-3) -> np.ndarray:
    """
    Carries out gradient-based optimization for univariate parameteric survival distributions

    Returns
    -------
    A (2,k) matrix of shape (first row) and scale (second row) parameters
    """
    # Input transforms
    t, d = t_long(t), t_long(d)
    t, d, dist = broadcast_td_dist(t, d, dist)
    # Univariate optimization has two rows (p=# of scale + 1 for shape)
    p, k  = 2, t.shape[1]
    if x0 is None:  # Initialize parameters if not specified
        # Use closed-form solution for exponential scale
        x0 = np.vstack([np.ones(k), d.sum(0) / t.sum(0)])
    assert x0.shape == (p, k), 'x0 needs to be a (p,k) matrix'
    # Define the bounds (scale/shape must be positive)
    
    # -- Run the optimizer w/ backups -- #
    shape_scale = np.ones(x0.shape)
    for i in range(k):
        x0_i, t_i, d_i, dist_i = x0[:,[i]], t[:,[i]], d[:,[i]], [dist[i]]
        # (i) Try unconstrained optimization
        opt_i = minimize(fun=log_lik_vec, jac=grad_ll_vec, x0=x0_i,  method='L-BFGS-B', args=(t_i, d_i, dist_i))
        first_success = opt_success(opt_i, tol_grad)
        if not first_success:
            # (ii) Try with bounded optimization
            bnds_i = di_bounds[dist[i]]
            opt_i = minimize(fun=log_lik_vec, jac=grad_ll_vec, x0=x0_i, method='L-BFGS-B', args=(t_i, d_i, dist_i), bounds=bnds_i)
            second_success = opt_success(opt_i, tol_grad)
            if not second_success:
                for f in [0.1, 10]:
                    # (iii) Try bounded optimzation with different initializations
                    opt_i = minimize(fun=log_lik_vec, jac=grad_ll_vec, x0=f*x0_i, method='L-BFGS-B', args=(t_i, d_i, dist_i), bounds=bnds_i)
                    third_success = opt_success(opt_i, tol_grad)
                    if not third_success:
                        opt_i = minimize(fun=log_lik_vec, x0=x0_i, method='nelder-mead', args=(t_i, d_i, dist_i), bounds=bnds_i)
                        assert opt_i.success, 'Unable to obtain convergence!'
        shape_scale[:,i] = opt_i.x
    return shape_scale
