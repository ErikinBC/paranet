"""
Gradients for multivariate models
"""

# External modules
import warnings
import numpy as np
from scipy.optimize import minimize

# Internal modules
from paranet.utils import di_bounds, dist2idx, broadcast_long, broadcast_dist, str2lst, t_long
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


def loss_g(alpha_beta:np.ndarray, gamma:np.ndarray or float, rho:float, eps:float=1e-8) -> np.ndarray:
    """
    Gradient for the elastic net regularization parameter with the l1-approximation: 
    |x_i| ~ sqrt(x_i**2 + eps)

    Inputs
    ------
    See loss_g()

    Returns
    -------
    A (k,) array of elastic-net penalties
    """
    beta = alpha_beta[1:]
    l1_approx = np.sqrt(beta**2 + eps)
    ll = np.sum(gamma * (rho*l1_approx + 0.5*(1-rho)*beta**2), 0)
    return ll


def loss_g_l1(alpha_beta:np.ndarray, gamma:np.ndarray or float, rho:float) -> np.ndarray:
    """
    Implements loss_g() with the actual L1-norm
    """
    beta = alpha_beta[1:]
    l1_exact = np.abs(beta)
    ll = np.sum(gamma * (rho*l1_exact + 0.5*(1-rho)*beta**2), 0)
    return ll



def grad_g(alpha_beta:np.ndarray, gamma:np.ndarray or float, rho:float, eps:float=1e-8)  -> np.ndarray:
    """
    Gradient for the elastic net regularization parameter with the l1-approximation: 
    |x_i| ~ sqrt(x_i**2 + eps)

    Inputs
    ------
    See loss_g()

    Returns
    -------
    A (p+1,k) array of regularization values where the first row are always zeros for the alpha/shape parameter
    """
    beta = alpha_beta[1:]
    l1_approx = beta/np.sqrt(beta**2 + eps)
    grad_beta = gamma * (rho * l1_approx + (1-rho)*beta)
    # Alpha has a regularization penalty of zero
    grad = np.vstack([np.zeros([1,grad_beta.shape[1]]), grad_beta])
    return grad


def log_lik(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str, gamma:np.ndarray or float, rho:float, eps:float=1e-8) -> np.ndarray:
    """
    Calculate the (negative) log-likelihood for a given shape/scale, covariate, time, and censoring value

    Inputs
    ------
    alpha_beta:             First row is shape parameters (alpha), second row onwards in scale parameters (should be (p+1,k))
    x:                      The (n,p) design matrix
    t:                      An (n,k) array of time values
    d:                      An (n,k) array of censoring values
    dist:                   The string or list of distributions
    gamma:                  Regularization strength (0 is un-regularized)
    rho:                    Percent weighting towards L1 (1 is Lasso)
    eps:                    Approximation for the L1-norm (default=1e-8)
    
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
    ll_reg = loss_g(alpha_beta, gamma, rho, eps)
    for s, i in didx.items():
        if s == 'exponential':
            ll_vec_i = -np.mean(d[:,i] * np.log(risk[:,i]) - risk[:,i]*t[:,i], axis=0)
        if s == 'weibull':
            ll_vec_i = -np.mean(d[:,i]*(np.log(alpha[:,i]*risk[:,i]) + (alpha[:,i]-1)*np.log(t[:,i])) - risk[:,i]*t[:,i]**alpha[:,i], axis=0)
        if s == 'gompertz':
            ll_vec_i = -np.mean(d[:,i]*(np.log(risk[:,i])+alpha[:,i]*t[:,i]) - risk[:,i]/alpha[:,i]*(np.exp(alpha[:,i]*t[:,i])-1), axis=0)
        # Add on regularization
        ll_vec_i += ll_reg[i]
        ll_vec[i] = ll_vec_i
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
            grad_i = x.T.dot(d[:,i] - t[:,i]*risk[:,i])
        if s == 'weibull':
            grad_i = x.T.dot(d[:,i] - t[:,i]**alpha[:,i]*risk[:,i])
        if s == 'gompertz':
            grad_i = x.T.dot(d[:,i] - risk[:,i]*(np.exp(alpha[:,i]*t[:,i]) - 1)/alpha[:,i])
        # Get the "mean" and negative
        grad_i = -grad_i / len(x)
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


def grad_ll(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str, gamma:np.ndarray or float, rho:float, eps:float=1e-8) -> np.ndarray:
    """
    Calculate the gradient for the shape and covariates parameters.

    Inputs
    ------
    See log_lik()

    Returns
    -------
    A (p+1,k) array of gradients, where the first row is the shape parameters, and the 1:p+1 rows are for the different scale parameters. Assumes that if there is an intercept it is assigned a column in x.
    """
    # (i) Calculate the gradient for the data likelihood
    alpha_beta = t_long(alpha_beta)
    g_shape = grad_ll_shape(alpha_beta, x, t, d, dist)
    g_scale = grad_ll_X(alpha_beta, x, t, d, dist)
    g_data = np.vstack([g_shape, g_scale])
    # (ii) Calculate the gradient for the regularization terms
    g_reg = grad_g(alpha_beta, gamma, rho, eps)
    # (iii) Combine and return
    g = g_data + g_reg
    return g



def nll_solver(x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str, gamma:np.ndarray or float, rho:float, eps:float=1e-8, has_int:bool=False, grad_tol:float=1e-3, n_perm:int=10, alpha_beta_init=None, maxiter:int=15000) -> np.ndarray:
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
    dist = str2lst(dist)
    gamma = broadcast_long(gamma, k)

    # Set up optimization bounds
    bnds_p = tuple([(None, None) for j in range(p)])
    
    # Initialize vector with shape/scale from the univariate instance
    alpha_beta = np.zeros([p+1, k])
    if alpha_beta_init is None:
        x0_intercept = wrapper_grad_solver(t, d, dist)
        # Set shape
        alpha_beta[0] = x0_intercept[0]
        # Set intercept scale
        if has_int:
            # Because risk is exp(b0), we take the log to return to the level
            alpha_beta[1] = np.log(x0_intercept[1])
    else:
        assert alpha_beta.shape == alpha_beta_init.shape, f'alpha_beta_init needs to be of size {alpha_beta.shape}'
        alpha_beta = alpha_beta_init

    # Run optimization for each distribution
    msgs = ['STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT', 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT']
    for i in range(k):
        x0_i = alpha_beta[:,[i]]  # Needs to be a column vector
        dist_i = [dist[i]]
        t_i, d_i = t[:,i], d[:,i]
        gamma_i = gamma[:,[i]]
        bnds_i = (di_bounds[dist[i]][0],) + bnds_p
        opt_i = minimize(fun=log_lik, jac=grad_ll, x0=x0_i, args=(x, t_i, d_i, dist_i, gamma_i, rho, eps), method='L-BFGS-B', bounds=bnds_i, options={'maxiter':maxiter})
        # Check for convergence
        if opt_i.message in msgs:
            wmsg = f'Number of iterations exceeded, returning vector as is for {dist_i} ({i})'
            warnings.warn(wmsg)
            alpha_beta[:,i] = opt_i.x
        else:
            if not opt_i.success:
                breakpoint()
                log_lik(x0_i, x, t_i, d_i, dist_i, gamma_i, rho, eps)
                grad_ll(x0_i, x, t_i, d_i, dist_i, gamma_i, rho, eps)
            assert opt_i.success, f'Optimization was unsuccesful for {i}: {opt_i.message}'
            grad_max_i = np.abs(opt_i.jac.flat).max()
            if grad_max_i > grad_tol:
                print('Running SLSQP backup')
                opt_i = minimize(fun=log_lik, jac=grad_ll, x0=x0_i, args=(x, t_i, d_i, dist_i, gamma_i, rho, eps), method='SLSQP', bounds=bnds_i, options={'maxiter':maxiter})
                grad_max_i = np.abs(opt_i.jac.flat).max()
            assert grad_max_i < grad_tol, f'Largest gradient after convergence > {grad_tol}: {grad_max_i}'
            # Do slight permutation
            np.random.seed(n_perm)
            dist_perm_i = list(np.repeat(dist_i, n_perm))
            x_alt = t_long(opt_i.x) + np.random.uniform(-0.01,0.01,[p+1,n_perm])
            assert np.all(opt_i.fun < log_lik(x_alt, x, t_i, d_i, dist_perm_i, gamma_i, rho, eps)), 'Small permutation around x_star yielded a lower negative log-likelihood!'
            # Store
            alpha_beta[:,i] = opt_i.x
    # Return
    return alpha_beta
