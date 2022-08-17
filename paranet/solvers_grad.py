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
from paranet.gradient import grad_ll, log_lik
from paranet.dists import surv_dist
from paranet.utils import t_wide, format_t_d, is_vector, shape_scale_2vec, get_p_k

def log_lik_vec(shape_scale:np.ndarray, t:np.ndarray, d:np.ndarray, dist:str) -> float:
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
    p, k = get_p_k(t)
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
    p, k = get_p_k(t)
    # Reshape into order needed for log-likelihood function
    shape_scale = shape_scale.reshape([k,p]).T
    assert shape_scale.shape[0] >= 2, 'There should be at least two rows (first is shape, second is scale)'
    shape, scale = shape_scale_2vec(shape_scale)
    grad_mat = grad_ll(t, d, scale, shape, dist)
    grad_vec = grad_mat.T.flatten()
    return grad_vec


# t=T_dist;d=D_dist;x0=init_coef.copy()
def wrapper_grad_solver(t:np.ndarray, d:np.ndarray, dist:str, x0:None or np.ndarray=None):
    """
    CARRIES OUT OPTIMIZATION FOR GRADIENT-BASED METHOD
    """
    # Input checks
    p, k = get_p_k(t)
    # Initialize parameters
    if x0 is None:
        x0 = np.zeros([p,k]) + 1
    else:
        assert x0.shape == (p, k), 'x0 needs to be a (p,k) matrix'
    # Flatten
    x0 = x0.T.flatten()
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
    breakpoint()
    minimize(fun=log_lik_vec, x0=x0, method='L-BFGS-B', jac=grad_ll_vec, args=(t, d, dist))


# GENERATE A REPRESENTATIVE SAMPLE (I.E. QUANTILES) OR DATA
n_sim, seed = 99, 1
dist = 'exponential'
lam = np.array([0.5, 1, 1.5])
k = len(lam)
D_dist = np.zeros([n_sim, k],dtype=int) + 1
gen_dist = surv_dist(dist, scale=lam)
T_dist = gen_dist.quantile(p=np.arange(0.01,1,0.01))
init_coef = np.vstack([np.zeros(k), lam])
wrapper_grad_ll(T_dist, D_dist, dist, init_coef)









