"""
Gradients for multivariate models
"""

# External modules
import numpy as np
from scipy.optimize import minimize

# Internal modules
from paranet.utils import di_bounds, dist2idx, broadcast_long, broadcast_dist, t_long
from paranet.multivariate.dists import check_multi_input
from paranet.univariate.solvers_grad import wrapper_grad_solver


def process_alphabeta_x_t_dist(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Convenience wrapper for putting arguments used by all gradient functions"""
    check_multi_input(alpha_beta, x, t, dist)
    k = alpha_beta.shape[1]
    t = broadcast_long(t, k)
    d = broadcast_long(d, k)
    dist = broadcast_dist(dist, k)
    # Calculate risks
    alpha = alpha_beta[[0]]
    beta = alpha_beta[1:]
    risk = np.exp(x.dot(beta))
    assert risk.shape[1] == k, 'Risk should have column length of k'
    # Return shape, scale, time, and censoring
    return alpha, risk, t, d, k


def log_lik(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculate the (negative) log-likelihood for a given shape/scale, covariate, time, and censoring value

    Inputs
    ------
    alpha_beta:             First row is shape parameters (alpha), second row onwards in scale parameters (should be (p+1,k))
    x:                      The (n,p) design matrix
    t:                      An (n,k) array of time values
    d:                      An (n,k) array of censoring values
    dist:                   The string or list of distributions

    Returns
    -------
    Returns an (k,) array of negative log-likelihoods, where k is the number of distributions
    """
    # Input checks and transforms
    alpha_beta = t_long(alpha_beta)
    alpha, risk, t, d, k = process_alphabeta_x_t_dist(alpha_beta, x, t, d, dist)

    # negative log-likelihood
    didx = dist2idx(dist)
    ll_vec = np.zeros(k)
    for s, i in didx.items():
        if s == 'exponential':
            ll_vec[i] = -np.mean(d[:,i] * np.log(risk[:,i]) - risk[:,i]*t[:,i], axis=0)
        if s == 'weibull':
            ll_vec[i] = -np.mean(d[:,i]*(np.log(alpha[:,i]*risk[:,i]) + (alpha[:,i]-1)*np.log(t[:,i])) - risk[:,i]*t[:,i]**alpha[:,i], axis=0)
        if s == 'gompertz':
            ll_vec[i] = -np.mean(d[:,i]*(np.log(risk[:,i])+alpha[:,i]*t[:,i]) - risk[:,i]/alpha[:,i]*(np.exp(alpha[:,i]*t[:,i])-1), axis=0)
    return ll_vec


def grad_ll_X(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculate the gradient for the covariates.

    Inputs
    ------
    See log_lik()

    Returns
    -------
    A (p,k) array of gradients for the different scale parameters. Assumes that if there is an intercept it is assigned a column in x
    """
    # Input checks and transforms
    alpha, risk, t, d, k = process_alphabeta_x_t_dist(alpha_beta, x, t, d, dist)
    p = x.shape[1]

    # gradient of the negative log-likelihood
    didx = dist2idx(dist)
    grad_mat = np.zeros([p,k])
    for s, i in didx.items():
        if s == 'exponential':
            grad_i = -np.mean(x*(d[:,i] - t[:,i]*risk[:,i]),0)
        if s == 'weibull':
            grad_i = -np.mean(x*(d[:,i] - t[:,i]**alpha[:,i]*risk[:,i]),0)
        if s == 'gompertz':
            grad_i = -np.mean(x*(d[:,i] - risk[:,i]*(np.exp(alpha[:,i]*t[:,i]) - 1)/alpha[:,i]),0)
        grad_mat[:,i] = grad_i.reshape(grad_mat[:,i].shape)
    return grad_mat


def grad_ll_shape(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculate the gradient for the shape parameters.

    Inputs
    ------
    See log_lik()

    Returns
    -------
    A (1,p) array of gradients for the different shape parameters
    """
    # Input checks and transforms
    alpha, risk, t, d, k = process_alphabeta_x_t_dist(alpha_beta, x, t, d, dist)

    # gradient of the negative log-likelihood
    didx = dist2idx(dist)
    grad_shape = np.zeros([1,k])
    for s, i in didx.items():
        if s == 'exponential':
            grad_i = -np.repeat(0, len(i))
        if s == 'weibull':
            grad_i = -np.mean( d[:,i]*(1/alpha[:,i] + np.log(t[:,i])) - risk[:,i]*t[:,i]**alpha[:,i]*np.log(t[:,i]), axis=0)
        if s == 'gompertz':
            grad_i = -np.mean( d[:,i]*t[:,i] - (risk[:,i]/alpha[:,i]**2)*(np.exp(alpha[:,i]*t[:,i])*(alpha[:,i]*t[:,i]-1) +1), axis=0)
        grad_shape[:,i] = grad_i.reshape(grad_shape[:,i].shape)
    return grad_shape


def grad_ll(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculate the gradient for the shape and covariates parameters.

    Inputs
    ------
    See log_lik()

    Returns
    -------
    A (p+1,k) array of gradients, where the first row is the shape parameters, and the 1:p+1 rows are for the different scale parameters. Assumes that if there is an intercept it is assigned a column in x.
    """
    alpha_beta = t_long(alpha_beta)
    g_shape = grad_ll_shape(alpha_beta, x, t, d, dist)
    g_scale = grad_ll_X(alpha_beta, x, t, d, dist)
    g = np.vstack([g_shape, g_scale])
    return g


def nll_solver(x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str, has_int:bool=False, grad_tol:float=1e-3, n_perm:int=10) -> np.ndarray:
    """
    Wrapper to find the coefficient vector which minimizes the negative log-likelihood for the different parameter survival distributions

    Inputs
    ------
    has_int:            Whether the first column of x is an intercept (default=False)
    grad_tol:           Post-convergence checks for largest gradient size allowable
    n_perm:             Number of random pertubations to do around "optimal" coefficient vector to check that lower log-likelihood is not possible
    See log_lik()

    Returns
    -------
    A (p,k) array of scale and shape coefficients.
    """
    # Input checks
    assert len(x.shape) == 2, 'x should have two dimensions (n,p)'
    assert len(t.shape) == 2, 't should have two dimensions (n,k)'
    assert len(d.shape) == 2, 'd should have two dimensions (n,k)'
    n, p = x.shape
    k = t.shape[1]
    assert n == t.shape[0] == d.shape[0], 'x, t, & d need to have the same number of rows'
    if has_int:
        assert np.all(x[:,0] == 1), 'If has_int==True, expected x[:,0] to be all ones!'

    # Set up optimization bounds
    bnds_p = tuple([(None, None) for j in range(p)])
    
    # Initialize vector with shape/scale from the univariate instance
    x0_intercept = wrapper_grad_solver(t, d, dist)
    # Set shape
    alpha_beta = np.zeros([p+1, k])
    alpha_beta[0] = x0_intercept[0]
    # Set intercept scale
    if has_int:
        # Because risk is exp(b0), we take the log to return to the level
        alpha_beta[1] = np.log(x0_intercept[1])

    # Run optimization for each distribution
    for i in range(k):
        x0_i = alpha_beta[:,[i]]  # Needs to be a column vector
        dist_i = [dist[i]]
        t_i, d_i = t[:,i], d[:,i]
        bnds_i = (di_bounds[dist[i]][0],) + bnds_p
        opt_i = minimize(fun=log_lik, jac=grad_ll, x0=x0_i, args=(x, t_i, d_i, dist_i), method='L-BFGS-B', bounds=bnds_i)
        # Check for convergence
        assert opt_i.success, f'Optimization was unsuccesful for {i}'
        grad_max_i = np.abs(opt_i.jac.flat).max()
        assert grad_max_i < grad_tol, f'Largest gradient after convergence > {grad_tol}: {grad_max_i}'
        # Do slight permutation
        np.random.seed(n_perm)
        dist_perm_i = list(np.repeat(dist_i, n_perm))
        x_alt = t_long(opt_i.x) + np.random.uniform(-0.01,0.01,[p+1,n_perm])
        assert np.all(opt_i.fun < log_lik(x_alt, x, t_i, d_i, dist_perm_i)), 'Small permutation around x_star yielded a lower negative log-likelihood!'
        # Store
        alpha_beta[:,i] = opt_i.x
    # Return
    return alpha_beta



