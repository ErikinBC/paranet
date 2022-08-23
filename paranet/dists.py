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
import numpy as np
from scipy.optimize import root, minimize_scalar, minimize

# Internal modules
from paranet.utils import param2array, len_of_none, t_long, t_wide, check_dist_str, check_interval, fast_auroc


def hazard(t:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES THE HAZARD FUNCTION FOR THE RELEVANT CLASSES

    Inputs
    ------
    t:                  time
    scale:              lambda
    shape:              alpha
    dist:               One of dist_valid
    """
    check_dist_str(dist)
    t_vec = t_long(t)
    if dist == 'exponential':
        h = t_vec * scale
    if dist == 'weibull':
        h = shape * scale * t_vec**(shape-1)
    if dist == 'gompertz':
        h = scale * np.exp(shape*t)
    return h


def survival(t:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES THE SURVIVAL FUNCTION FOR THE RELEVANT CLASSES (SEE HAZARD)
    """
    check_dist_str(dist)
    t_vec = t_long(t)
    if dist == 'exponential':
        s = np.exp(-scale * t_vec)
    if dist == 'weibull':
        s = np.exp(-scale * t_vec**shape)
    if dist == 'gompertz':
        s = np.exp(-scale/shape * (np.exp(shape*t_vec)-1))
    return s


def pdf(t:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES THE DENSITY FUNCTION FOR THE RELEVANT CLASSES (SEE HAZARD)
    """
    check_dist_str(dist)
    f = hazard(t, scale, shape, dist) * survival(t, scale, shape, dist)
    return f


def rvs_T(n_sim:int, k:int, scale:np.ndarray, shape:np.ndarray or None, dist:str, seed:None or int=None) -> np.ndarray:
    """
    GENERATES n_sim RANDOM SAMPLES FROM A GIVEN DISTRIBUTION

    Inputs
    ------
    n_sim:              Integer indicating the number of samples to generate
    k:                  The dimensionality of the scale/shape parameter
    seed:               Reproducibility seed (default=None)
    See hazard() for remaining parameters

    Returns
    -------
    (n_sim x k) np.ndarray of time-to-event measurements
    """
    if seed is not None:
        np.random.seed(seed)
    nlU = -np.log(np.random.rand(n_sim, k))
    if dist == 'exponential':
        T_act = nlU / scale
    if dist == 'weibull':
        T_act = (nlU / scale) ** (1/shape)
    if dist == 'gompertz':
        T_act = 1/shape * np.log(1 + shape/scale*nlU)
    return T_act


def quantile(p:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES THE QUANTILE FUNCTION FOR THE RELEVANT CLASSES (SEE HAZARD)
    """
    check_dist_str(dist)
    nlp = -np.log(1 - t_long(p))
    if dist == 'exponential':
        T = nlp / scale
    if dist == 'weibull':
        T = (nlp / scale) ** (1/shape)
    if dist == 'gompertz':
        T = 1/shape * np.log(1 + shape/scale*nlp)
    return T



def rvs(n_sim:int, k:int, scale:np.ndarray, shape:np.ndarray or None, dist:str, censoring:float=0, seed:None or int=None, n_censor_sim:int=100) -> tuple[np.ndarray,np.ndarray]:
    """
    GENERATES n_sim RANDOM SAMPLES FROM A GIVEN DISTRIBUTION

    Inputs
    ------
    See rvs_T
    censoring:          Fraction (in expectation) of observations that should be censored
    n_censor_sim:       How many quantile points to use to determine lambda which achieves the censoring target

    Returns
    -------
    2*(n_sim x k) np.ndarray's of observed time-to-event measurements and censoring indicator
    """
    check_interval(censoring, 0, 1)
    # (i) Calculate the "actual" time-to-event
    T_act = rvs_T(n_sim=n_sim, k=k, scale=scale, shape=shape, dist=dist, seed=seed)
    if censoring == 0:  # Return actual if there is not censoring
        D_cens = np.zeros(T_act.shape) + 1
        return T_act, D_cens
    # (ii) Determine the "scale" from an exponential needed   
    # Generate n_censor_sim observations from the actual distribution
    T_dist = rvs_T(n_sim=n_censor_sim, k=k, scale=scale, shape=shape, dist=dist, seed=seed)
    scale_D = np.zeros(scale.shape)  # Create a holder
    for j in range(scale.shape[1]):
        breakpoint()
        opt = minimize_scalar(compare_lam_auroc, args=(censoring, T_dist[:,j], n_censor_sim, seed), bracket=(1,2),method='brent')
        assert opt.success, 'Brent optimization failed'
        scale_D[:,j] = opt.x
    # Generate data from exponential distribution
    T_cens = -np.log(np.random.rand(n_sim, k)) / scale_D
    D_cens = np.where(T_cens <= T_act, 0, 1)
    T = np.where(D_cens == 1, T, T_cens)
    return T, D_cens


def compare_lam_auroc(scale:float, censoring:float, T_dist_target:np.ndarray, n_censor_sim:int, seed:int or None=None) -> float:
    """
    For a given scale parameter, compare the probability that a given exponential distribution is larger than a comparison

    Inputs
    ------
    scale:              Scale parameter to test for exponential
    censoring:          The targeted censoring rate
    T_dist_target:      Sample from the distribution we want to censor
    n_censor_sim:       Number of samples to draw from for exponential censoring distribution


    Returns
    -------
    The MSE between the empirical AUROC and 1-target AUROC
    """
    T_dist_cens = rvs_T(n_sim=n_censor_sim, k=1, scale=scale, shape=None, dist='exponential', seed=seed)
    auroc = fast_auroc(T_dist_cens, T_dist_target)
    err = auroc - (1-censoring)
    err2 = err**2
    return err2


class surv_dist():
    def __init__(self, dist:str, scale:float or np.ndarray or None=None, shape:float or np.ndarray or None=None) -> None:
        """
        Backbone class for parametric survival distributions. Choice of distribution will determine the call of other functions.

        Inputs
        ------
        dist:           A string for a valid distribution: exponential, weibull, gompertz 
        scale:          Scale parameter (float or array)
        shape:          Shape parameter (float of array)
        """
        # Do input checks
        check_dist_str(dist)
        # Assign to attributes
        self.dist = dist
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
        # If distribution is expontial force the shape to be unity
        if self.dist == 'exponential':
            self.shape = self.shape*0 + 1

    def check_t(self, t):
        assert len(t) == self.n, f't needs to be the same size the input parameter: {self.n}'

    def hazard(self, t):
        return hazard(t=t, scale=self.scale, shape=self.shape, dist=self.dist)

    def survival(self, t):
        return survival(t=t, scale=self.scale, shape=self.shape, dist=self.dist)

    def pdf(self, t):
        return pdf(t=t, scale=self.scale, shape=self.shape, dist=self.dist)

    def rvs(self, n_sim:int, censoring:float=0, seed:None or int=None, n_censor_sim:int=100):
        """INVERTED-CDF APPROACH"""
        return rvs(n_sim=n_sim, k=self.k, scale=self.scale, shape=self.shape, censoring=censoring, dist=self.dist, seed=seed)

    def quantile(self, p:float or np.ndarray):
        return quantile(p, self.scale, self.shape, self.dist)



