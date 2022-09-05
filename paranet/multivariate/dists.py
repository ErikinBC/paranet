"""
Creates distribution/gradient specific functions for the multivariate parametric likelihoods
"""

# External modules
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import dblquad

# Internal modules
from paranet.univariate.dists import pdf as pdf_uni
from paranet.univariate.dists import survival as survival_uni
from paranet.utils import broadcast_dist, broadcast_long, dist2idx, check_interval, t_long, t_wide, try_squeeze


def check_multi_input(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, dist:list) -> None:
    """Check that inputs align to dimensional expectations"""
    p_ab, k_ab = alpha_beta.shape
    n, p = x.shape
    dist = broadcast_dist(dist, k_ab)
    n_t = len(t)
    assert k_ab == len(dist), 'Number of columns of alpha_beta needs to align with the length of dist'
    assert p_ab - 1 == p, 'Number of columns of x should be one less that the number of rows of alpha_beta (as the first row is the alpha/shape parameter)'
    assert n_t == n, 'The number of rows of x need to align with t'
    if len(t.shape) == 2:
        k_t = t.shape[1]
        if k_t != k_ab:
            assert k_t == 1, 'If length of dist is not equal to number of columns of t, then t must have a single column so it can be broadcast'


def hazard_multi(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculates the hazard function for the relevant classes with a matrix of covariates

    Inputs
    ------
    alpha_beta:         Matrix of coefficients ~ (p+1,k), where first row is the shape/alpha
    x:                  Matrix of covariates x~(n,p), where the first column is usually a one to indicate an intercept
    t:                  Time vector t~(n,1) or (n,k)
    dist:               A list of distributions to fit covariates for (length 1 or k)

    Returns
    -------
    An (n,k) matrix of hazards, where each column corresponds to a different distribution
    """
    # Input checks and transforms
    check_multi_input(alpha_beta, x, t, dist)
    k = alpha_beta.shape[1]
    n = x.shape[0]
    t = broadcast_long(t, k)
    dist = broadcast_dist(dist, k)

    # Calculate risks
    alpha = alpha_beta[[0]]
    beta = alpha_beta[1:]
    risk = np.exp(x.dot(beta))

    # Calculate hazard
    didx = dist2idx(dist)
    h_mat = np.zeros([n, k])
    
    for d, i in didx.items():
        if d == 'exponential':
            h_mat[:,i] = risk[:,i]
        if d == 'weibull':
            h_mat[:,i] = risk[:,i] * alpha[:,i] * t[:,i]**(alpha[:,i]-1)
        if d == 'gompertz':
            h_mat[:,i] = risk[:,i] * np.exp(alpha[:,i]*t[:,i])
    return h_mat



def survival_multi(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculates the survival function for the relevant classes with a matrix of covariates; see hazard_multi() for more details
    """
    # Input checks and transforms
    check_multi_input(alpha_beta, x, t, dist)
    k = alpha_beta.shape[1]
    n = x.shape[0]
    t = broadcast_long(t, k)
    dist = broadcast_dist(dist, k)

    # Calculate risks
    alpha = alpha_beta[[0]]
    beta = alpha_beta[1:]
    risk = np.exp(x.dot(beta))

    # Calculate survival
    didx = dist2idx(dist)
    s_mat = np.zeros([n, k])
    for d, i in didx.items():
        if d == 'exponential':
            s_mat[:,i] = np.exp(-risk[:,i] * t[:,i])
        if d == 'weibull':
            s_mat[:,i] = np.exp(-risk[:,i] * t[:,i]**alpha[:,i])
        if d == 'gompertz':
            s_mat[:,i] = np.exp(-risk[:,i]/alpha[:,i] * (np.exp(alpha[:,i]*t[:,i])-1))
    return s_mat


def pdf_multi(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculates the density function for the relevant classes with a matrix of covariates; see hazard_multi() for more details
    """
    h = hazard_multi(alpha_beta, x, t, dist)
    s = survival_multi(alpha_beta, x, t, dist)
    f = h * s
    return f


def broadcast_percentile(percentile:np.ndarray or float, n:int, k:int) -> np.ndarray:
    """
    Broadcast a float or (q,) array to an (n,k,q) array

    Inputs
    ------
    p:              Percentile input argument
    n:              Length
    k:              Number of columns

    Returns
    -------
    (n,k,q) array of percentiles
    """
    # Input checks
    percentile = t_long(percentile)
    check_interval(percentile, 0, 1, equals=False)
    # Broadcast
    q = len(percentile)
    p_mat = np.tile(np.tile(percentile,[1,n]),[k,1,1]).transpose([2,0,1])
    # Return checks
    assert len(p_mat.shape) == 3, 'Expected p_mat to be of (n,k,q)'
    assert p_mat.shape == (n, k, q), 'Expected p_mat to be of (n,k,q)'
    return p_mat


def quantile_multi(percentile:np.ndarray, alpha_beta:np.ndarray, x:np.ndarray, dist:list or str, squeeze:bool=True) -> np.ndarray:
    """
    Calculates the quantiles for the relevant classes with a matrix of covariates.
    
    Inputs
    ------
    squeeze:            Should the third axis be squeezed out if len(percentile)==1? (default=True)
    See hazard_multi()

    Returns
    -------
    An (n,k,q) array where n is the number of rows of x, k is the number of distributions (i.e. columns of alpha_beta), and q is the length of the quantile array being evaluated
    """
    # Input checks and transforms
    k, n = alpha_beta.shape[1], x.shape[0]
    percentile = broadcast_percentile(percentile, n, k)
    q = percentile.shape[2]
    check_multi_input(alpha_beta, x, percentile, dist)
    dist = broadcast_dist(dist, k)

    # Calculate risks
    alpha = alpha_beta[[0]]
    beta = alpha_beta[1:]
    risk = np.exp(x.dot(beta))
    # Add on dimension for quantile
    alpha = np.expand_dims(alpha, 2)
    risk = np.expand_dims(risk, 2)
    
    # Calculate quantile
    nlp = -np.log(1 - percentile)
    q_mat = np.zeros(percentile.shape)
    didx = dist2idx(dist)
    for d, i in didx.items():
        if d == 'exponential':
            q_mat[:,i,:] = nlp[:,i,:] / risk[:,i,:]
        if d == 'weibull':
            q_mat[:,i,:] = (nlp[:,i,:] / risk[:,i,:]) ** (1/alpha[:,i,:])
        if d == 'gompertz':
            q_mat[:,i,:] = 1/alpha[:,i,:] * np.log(1 + alpha[:,i,:]/risk[:,i,:]*nlp[:,i,:])
    if squeeze:
        q_mat = try_squeeze(q_mat, axis=2)
    return q_mat


def integral_for_censoring_multi(time:float or np.ndarray, risk:float or np.ndarray, scale_C:float, shape_T:float or np.ndarray, dist_T:str, l2_beta:float) -> float:
    """
    Returns the integral for int_0^infty int_0^infty F_C(time) f_{risk}(time;shape) f_{lambda}(risk); 0; l2) du di

    i) F_C(time) is the CDF of an Exponential(scale_C)
    ii) f_{risk}(time; shape) is the pdf of the target dist with scale (i) & shape
    iii) f_{lambda}(risk) is the pdf of the lognormal distribution with mean 0 and variance l2

    Inputs
    ------
    t:          Time value 
    r:          Risk (aka scale parameter)
    """
    # Calculate densitities and CDF
    f_dist = pdf_uni(time, risk, shape_T, dist_T)
    # Shape parameter is irrelevant
    F_exp = 1 - survival_uni(time, scale_C, scale_C, 'exponential')
    f_lam = lognorm(s=1,loc=0,scale=l2_beta).pdf(risk)
    # Return integral
    f_int = f_dist * F_exp * f_lam
    return f_int


def rvs_T_multi(n_sim:int, alpha_beta:np.ndarray, x:np.ndarray, dist:list or str, seed:None or int=None):
    """
    Generate n_sim samples from each (n,k) distribution

    Inputs
    ------
    n_sim:              Integer indicating the number of samples to generate
    seed:               Reproducibility seed (default=None)
    See hazard_multi() for remaining parameters

    Returns
    -------
    (n, k, n_sim) array of (uncensored) time-to-event measurements
    """
    # Input checks and transforms
    n, p = x.shape
    p_ab, k = alpha_beta.shape
    assert p == p_ab-1, 'Number of columns of x should be equal to number of rows of alpha_beta less one'
    dist = broadcast_dist(dist, k)

    # Calculate risks
    alpha = alpha_beta[[0]]
    beta = alpha_beta[1:]
    risk = np.exp(x.dot(beta))

    # Expand dimensions for broadcasting
    alpha = np.expand_dims(alpha, 2)
    risk = np.expand_dims(risk, 2)

    # Generate randomness
    if seed is not None:
        np.random.seed(seed)
    nlU = -np.log(np.random.rand(n, k, n_sim))
    
    # Calculate quantile
    T_act = np.zeros(nlU.shape)
    didx = dist2idx(dist)
    for d, i in didx.items():
        if d == 'exponential':
            T_act[:,i,:] = nlU[:,i,:] / risk[:,i,:]
        if d == 'weibull':
            T_act[:,i,:] = (nlU[:,i,:] / risk[:,i,:]) ** (1/alpha[:,i,:])
        if d == 'gompertz':
            T_act[:,i,:] = 1/alpha[:,i,:] * np.log(1 + alpha[:,i,:]/risk[:,i,:]*nlU[:,i,:])
    return T_act



def rvs_multi(censoring:float, n_sim:int, alpha_beta:np.ndarray, x:np.ndarray, dist:list or str, seed:None or int=None):
    """
    Generate n_sim samples from each distribution with censoring

    Inputs
    ------
    n_sim:              Integer indicating the number of samples to generate
    seed:               Reproducibility seed (default=None)
    See rvs_T_multi() for remaining parameters

    Returns
    -------
    (n_sim x k, n_sim x k) np.ndarray's of observed time-to-event measurements and censoring indicators (0==censored)
    """
    # Input checks
    check_interval(censoring, 0, 1)

    # Calculate the "actual" time-to-event
    T_act = rvs_T_multi(n_sim, alpha_beta, x, dist, seed)
    if censoring == 0:  # Return actual if there is not censoring
        D_cens = np.ones(T_act.shape)
        return T_act, D_cens

    # Determine the "scale" from an exponential needed to obtain censoring
    scale_C = find_exp_scale_censoring_multi()
    # k = T_act.shape[1]
    # scale_C = np.random.rand(1, k, 1)

    # Generate data from exponential distribution
    T_cens = -np.log(np.random.rand(*T_act.shape)) / scale_C
    D_cens = np.where(T_cens <= T_act, 0, 1)
    T_obs = np.where(D_cens == 1, T_act, T_cens)
    return T_obs, D_cens