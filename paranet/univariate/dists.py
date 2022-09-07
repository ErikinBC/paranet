"""
Classes to support parametric survival probability distributions
"""

# https://en.wikipedia.org/wiki/Exponential_distribution
# https://en.wikipedia.org/wiki/Log-logistic_distribution
# https://en.wikipedia.org/wiki/Log-normal_distribution
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3546387/pdf/sim0031-3946.pdf
# https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html
# https://square.github.io/pysurvival/models/parametric.html

# External modules
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# Internal modules
from paranet.utils import broadcast_long, param2array, len_of_none, str2lst, t_long, t_wide, check_dist_str, check_interval, dist2vec, dist2idx, broadcast_dist

# Supress warning about overflow
warnings.filterwarnings("ignore", category=RuntimeWarning)

def hazard(t:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:list or str) -> np.ndarray:
    """
    CALCULATES THE HAZARD FUNCTION FOR THE RELEVANT CLASSES

    Inputs
    ------
    t:                  time
    scale:              lambda
    shape:              alpha
    dist:               One of dist_valid
    """
    # Input cheks
    scale, shape = t_wide(scale), t_wide(shape)
    k = scale.shape[1]
    dist = broadcast_dist(dist, k)
    didx = dist2idx(dist)
    t_vec = t_long(t)
    n = t_vec.shape[0]
    h_mat = np.zeros([n, k])
    # Calculate hazard
    for d, i in didx.items():
        if d == 'exponential':
            h_mat[:,i] = scale[:,i] * (t_vec*0 + 1)
        if d == 'weibull':
            h_mat[:,i] = shape[:,i] * scale[:,i] * t_vec**(shape[:,i]-1)
        if d == 'gompertz':
            h_mat[:,i] = scale[:,i] * np.exp(shape[:,i]*t_vec)
    return h_mat


def survival(t:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES THE SURVIVAL FUNCTION FOR THE RELEVANT CLASSES (SEE HAZARD)
    """
    # Input cheks
    scale, shape = t_wide(scale), t_wide(shape)
    k = scale.shape[1]
    dist = broadcast_dist(dist, k)
    didx = dist2idx(dist)
    t_vec = t_long(t)
    n = t_vec.shape[0]
    s_mat = np.zeros([n, k])
    # Calculate survival
    for d, i in didx.items():
        if d == 'exponential':
            s_mat[:,i] = np.exp(-scale[:,i] * t_vec)
        if d == 'weibull':
            s_mat[:,i] = np.exp(-scale[:,i] * t_vec**shape[:,i])
        if d == 'gompertz':
            s_mat[:,i] = np.exp(-scale[:,i]/shape[:,i] * (np.exp(shape[:,i]*t_vec)-1))
    return s_mat


def pdf(t:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES THE DENSITY FUNCTION FOR THE RELEVANT CLASSES (SEE HAZARD)
    """
    check_dist_str(dist)
    h = hazard(t, scale, shape, dist)
    s = survival(t, scale, shape, dist)
    f = h * s
    return f


def quantile(p:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES THE QUANTILE FUNCTION FOR THE RELEVANT CLASSES (SEE HAZARD)
    """
    # Input checks
    scale, shape = t_wide(scale), t_wide(shape)
    k = scale.shape[1]
    dist = broadcast_dist(dist, k)
    didx = dist2idx(dist)
    nlp = -np.log(1 - broadcast_long(p, k))
    n = nlp.shape[0]    
    q_mat = np.zeros([n, k])
    # Calculate quantile
    for d, i in didx.items():
        if d == 'exponential':
            q_mat[:,i] = nlp[:,i] / scale[:,i]
        if d == 'weibull':
            q_mat[:,i] = (nlp[:,i] / scale[:,i]) ** (1/shape[:,i])
        if d == 'gompertz':
            q_mat[:,i] = 1/shape[:,i] * np.log(1 + shape[:,i]/scale[:,i]*nlp[:,i])
    return q_mat


def integral_for_censoring(t:float or np.ndarray, scale_C:float, scale_T:float, shape_T:float, dist_T:str) -> float:
    f_dist = pdf(t, scale_T, shape_T, dist_T)
    # Note the shape input doesn't matter here
    F_exp = 1 - survival(t, scale_C, scale_C, 'exponential')
    f_int = f_dist * F_exp
    return f_int


def censoring_exponential(scale_C:float, scale_T:float, shape_T:float, dist_T:str, alpha:float=0.001) -> float:
    """
    Function to calculate the probability that P(C < T), where T is the target distribution of interest (defined by scale/shape), and C is an exponential distribution that will act as the censoring distribution where T_obs = T if T<C, and C if T>C
    
    Inputs
    ------
    scale_C:                    The exponential scale for the censoring distribution
    scale_T:                    Scale of the target distribution
    shape_T:                    Shape parameter of the target distribution
    dist_T:                     Distribution of T        
    alpha:                      Use 2*F^{-1}(1-alpha) for the upper bound of the integral
    
    Returns
    -------
    A 1xk vector of vector of censoring probabilities P(C < T)
    """
    # (i) Calculate a reasonable upper-bound to integrate over
    b_upper = 2 * quantile(1-alpha, scale_T, shape_T, dist_T)
    # (ii) Loop over the columns
    censoring, _ = quad(func=integral_for_censoring, a=0, b=b_upper, args=(scale_C, scale_T, shape_T, dist_T))
    return censoring


def err2_censoring_exponential(scale_C:float, censoring:float, scale_T:float, shape_T:float, dist_T:str, ret_squared:bool=True) -> float:
    """Calculates squared error between target censoring and expected value"""
    expected_censoring = censoring_exponential(scale_C, scale_T, shape_T, dist_T)
    if ret_squared:
        err = np.sum((censoring - expected_censoring)**2)
    else:
        err = np.sum(censoring - expected_censoring)
    return err


def find_exp_scale_censoring(censoring:float, scale_T:np.ndarray, shape_T:np.ndarray or None, dist_T:str) -> np.ndarray:
    """
    Finds the scale parameter for an exponential distribution to achieve the target censoring for a given target distribution

    Inputs
    ------
    censoring:                  Probability that censoring RV will be less that actual
    
    Returns
    -------
    1xk vector of scale parameters for censoring exponential distribution
    """
    # (i) Input chekcs
    check_interval(censoring, 0, 1, equals_low=True)
    check_dist_str(dist_T)
    # (ii) Use the quantiles from each distribution
    scale_C = np.zeros(scale_T.shape)
    for i in range(scale_C.shape[1]):
        opt = minimize_scalar(fun=err2_censoring_exponential, bracket=(1,2),args=(censoring, scale_T[:,i], shape_T[:,i], dist_T[i]),method='brent')
        assert opt.success, 'Brent minimization was not successful'
        scale_C[:,i] = opt.x
    return scale_C


def rvs_T(n_sim:int, scale:np.ndarray, shape:np.ndarray or None, dist:str, seed:None or int=None) -> np.ndarray:
    """
    GENERATES n_sim RANDOM SAMPLES FROM A GIVEN DISTRIBUTION

    Inputs
    ------
    n_sim:              Integer indicating the number of samples to generate
    seed:               Reproducibility seed (default=None)
    See hazard() for remaining parameters

    Returns
    -------
    (n_sim x k) np.ndarray of time-to-event measurements
    """
    # Input cheks
    k = scale.shape[1]  # Assign the dimensionality
    dist = dist2vec(dist, scale)
    k = len(dist)
    # Generate randomness
    if seed is not None:
        np.random.seed(seed)
    nlU = -np.log(np.random.rand(n_sim, k))
    T_act = np.zeros([n_sim, k])
    # Calculate quantile
    didx = dist2idx(dist)
    for d, i in didx.items():
        if d == 'exponential':
            T_act[:,i] = nlU[:,i] / scale[:,i]
        if d == 'weibull':
            T_act[:,i] = (nlU[:,i] / scale[:,i]) ** (1/shape[:,i])
        if d == 'gompertz':
            T_act[:,i] = 1/shape[:,i] * np.log(1 + shape[:,i]/scale[:,i]*nlU[:,i])
    return T_act


def rvs(n_sim:int, scale:np.ndarray, shape:np.ndarray or None, dist:str, seed:None or int=None, censoring:float=0) -> tuple[np.ndarray,np.ndarray]:
    """
    GENERATES n_sim RANDOM SAMPLES FROM A GIVEN DISTRIBUTION

    Inputs
    ------
    See rvs_T()
    censoring:          Fraction (in expectation) of observations that should be censored
    n_points:           Number of ponits to use for the integral calculation 
    method:             Numerical method integration method

    Returns
    -------
    2*(n_sim x k) np.ndarray's of observed time-to-event measurements and censoring indicator
    """
    # Input checks
    check_interval(censoring, 0, 1, equals_low=True)
    scale, shape = t_wide(scale), t_wide(shape)
    k = scale.shape[1]
    assert scale.shape == shape.shape, 'scale and shape need to be the same'
    # (i) Calculate the "actual" time-to-event
    T_act = rvs_T(n_sim=n_sim, scale=scale, shape=shape, dist=dist, seed=seed)
    if censoring == 0:  # Return actual if there is not censoring
        D_cens = np.zeros(T_act.shape) + 1
        return T_act, D_cens
    # (ii) Determine the "scale" from an exponential needed to obtain censoring
    scale_C = find_exp_scale_censoring(censoring=censoring, scale_T=scale, shape_T=shape, dist_T=dist)

    # Generate data from exponential distribution
    T_cens = -np.log(np.random.rand(n_sim, k)) / scale_C
    D_cens = np.where(T_cens <= T_act, 0, 1)
    T_obs = np.where(D_cens == 1, T_act, T_cens)
    return T_obs, D_cens


class univariate_dist():
    def __init__(self, dist:str, scale:float or np.ndarray or None=None, shape:float or np.ndarray or None=None) -> None:
        """
        Backbone class for univriate parametric survival distributions. Choice of distribution will determine the call of other functions.

        Inputs
        ------
        dist:           A string for a valid distribution: exponential, weibull, gompertz 
        scale:          Scale parameter (float or array)
        shape:          Shape parameter (float of array)
        """
        # Do input checks
        check_dist_str(dist)
        # Assign to attributes
        self.dist = str2lst(dist)
        self.scale = param2array(scale)
        self.shape = param2array(shape)
        n_scale = len_of_none(self.scale)
        n_shape = len_of_none(self.shape)
        has_scale, has_shape = n_scale > 0, n_shape > 0
        assert has_scale + has_shape > 0, 'There must be at least one scale and shape parameter'
        self.n = n_scale
        if has_scale and has_shape:
            assert n_scale == n_shape, 'Array size of scale and shape needs to align'
        else:
            assert dist == 'exponential', 'Exponential distribution is the only one that does not need a shape parameter; leave as default (None)'
        # Put the shape/scales as 1xK
        self.scale = t_wide(self.scale)
        self.shape = t_wide(self.shape)
        self.k = self.scale.shape[1]
        # Broadcast dist if k > 1 and len(dist)==1
        self.dist = broadcast_dist(self.dist, self.k)
        # If distribution is expontial force the shape to be unity
        self.didx = dist2idx(self.dist)
        if 'exponential' in self.didx:
            idx_exp = self.didx['exponential']
            self.shape[:,idx_exp] = self.shape[:,idx_exp]*0 + 1
    
    
    def hazard(self, t):
        return hazard(t=t, scale=self.scale, shape=self.shape, dist=self.dist)

    def survival(self, t):
        return survival(t=t, scale=self.scale, shape=self.shape, dist=self.dist)

    def pdf(self, t):
        return pdf(t=t, scale=self.scale, shape=self.shape, dist=self.dist)

    def rvs(self, n_sim:int, censoring:float=0, seed:None or int=None):
        """INVERTED-CDF APPROACH"""
        return rvs(n_sim=n_sim, censoring=censoring, scale=self.scale, shape=self.shape, dist=self.dist, seed=seed)
 
    def quantile(self, p:float or np.ndarray):
        return quantile(p, self.scale, self.shape, self.dist)

