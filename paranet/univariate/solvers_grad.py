"""
WRAPPER TO PERFORM GRADIENT OPTIMIZATION
"""

# https://docs.scipy.org/doc/scipy/tutorial/optimize.html#trust-region-nearly-exact-algorithm-method-trust-exact
# https://hastie.su.domains/Papers/glmnet.pdf
# https://www.cs.cmu.edu/~ggordon/10725-F12/slides/25-coord-desc.pdf
# https://github.com/pmelchior/proxmin
# https://github.com/lowks/gdprox
# https://pypi.org/project/pyproximal/

# External modules
import numpy as np
from scipy.optimize import minimize

# Internal modules
from paranet.univariate.gradient import grad_ll, log_lik
from paranet.utils import is_vector, shape_scale_2vec, _get_p_k, broadcast_td_dist, t_long

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


def wrapper_grad_solver(t:np.ndarray, d:np.ndarray, dist:str or list, x0:None or np.ndarray=None):
    """
    CARRIES OUT OPTIMIZATION FOR GRADIENT-BASED METHOD
    """
    # Input transforms
    t, d = t_long(t), t_long(d)
    t, d, dist = broadcast_td_dist(t, d, dist)
    # Currently we support two rows (p=# of scale + 1 for shape)
    p = 2
    k = t.shape[1]
    # Initialize parameters
    if x0 is None:
        x0 = np.zeros([p,k]) + 1
    else:
        assert x0.shape == (p, k), 'x0 needs to be a (p,k) matrix'
    # Flatten
    x0 = x0.T.flatten()
    # Define the bounds (scale/shape must be positive)
    bnds = tuple([(0, None) for j in range(len(x0))])
    # Check the log-likelihood
    try:
        log_lik_vec(x0, t, d, dist)
    except:
        print('log_lik_vec failed')
    # Check the gradient call
    try:
        grad_ll_vec(x0, t, d, dist)
    except:
        print('log_lik_vec failed')
    # Run the optimizer
    opt = minimize(fun=log_lik_vec, x0=x0, method='L-BFGS-B', jac=grad_ll_vec, args=(t, d, dist), bounds=bnds)
    # Check for convergence
    assert opt.success, f'Optimization did not converge: {opt.message}'
    # Return in (p,k) format
    shape_scale = opt.x.reshape([k,p]).T
    return shape_scale










