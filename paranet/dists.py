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
from scipy.optimize import minimize_scalar

# Internal modules
from paranet.num_methods import get_intergral
from paranet.utils import param2array, len_of_none, t_long, t_wide, check_dist_str, check_interval


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
        h = scale * (t_vec*0 + 1)
    if dist == 'weibull':
        h = shape * scale * t_vec**(shape-1)
    if dist == 'gompertz':
        h = scale * np.exp(shape*t_vec)
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
    h = hazard(t, scale, shape, dist)
    s = survival(t, scale, shape, dist)
    f = h * s
    return f


def quantile(p:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES THE QUANTILE FUNCTION FOR THE RELEVANT CLASSES (SEE HAZARD)
    """
    check_dist_str(dist)
    nlp = -np.log(1 - t_long(p))
    if dist == 'exponential':
        T_q = nlp / scale
    if dist == 'weibull':
        T_q = (nlp / scale) ** (1/shape)
    if dist == 'gompertz':
        T_q = 1/shape * np.log(1 + shape/scale*nlp)
    return T_q

def censoring_exponential(scale_C:np.ndarray, scale_T:np.ndarray, shape_T:np.ndarray or None, dist_T:str, xseq:np.ndarray, method:str='trapezoidal') -> np.ndarray:
    """
    Function to calculate the probability that P(C < T), where T is the target distribution of interest (defined by scale/shape), and C is an exponential distribution that will act as the censoring distribution where T_obs = T if T<C, and C if T>C
    
    Inputs
    ------
    scale_C:                    The exponential scale for the censoring distribution
    scale_T:                    Scale of the target distribution
    shape_T:                    Shape parameter of the target distribution
    dist_T:                     Distribution of T        
    xseq:                       Range of f(x) to evaluate over
    
    Returns
    -------
    A 1xk vector of vector of censoring probabilities P(C < T)
    """
    # Calculate function
    f1 = pdf(xseq, scale_T, shape_T, dist_T)
    if xseq.shape != f1.shape:
        xseq = xseq.reshape(f1.shape)
    f2 = np.exp(-scale_C*xseq)
    ff =  f1 * f2
    censoring = 1 - get_intergral(ff, xseq, method=method)
    return censoring


def err2_censoring_exponential(scale_C:np.ndarray, censoring:float, scale_T:np.ndarray, shape_T:np.ndarray or None, dist_T:str, xseq:np.ndarray, method:str='trapezoidal', ret_squared:bool=True):
    """Calculates squared error between target censoring and expected value"""
    expected_censoring = censoring_exponential(scale_C, scale_T, shape_T, dist_T, xseq, method)
    if ret_squared:
        err = np.sum((censoring - expected_censoring)**2)
    else:
        err = np.sum(censoring - expected_censoring)
    return err


def find_exp_scale_censoring(censoring:float, scale_T:np.ndarray, shape_T:np.ndarray or None, dist_T:str, n_points:int=100, method:str='trapezoidal') -> np.ndarray:
    """
    Finds the scale parameter for an exponential distribution to achieve the target censoring for a given target distribution

    Inputs
    ------
    censoring:                  Probability that censoring RV will be less that actual
    n_steps:                    Number of point to use for numerical integration
    method:                     Integration method
    n_points:                   Number of points to use to determine integral

    Returns
    -------
    1xk vector of scale parameters for censoring exponential distribution
    """
    # (i) Input chekcs
    check_interval(censoring, 0, 1)
    check_dist_str(dist_T)
    # (ii) Use the quantiles from each distribution
    pseq = np.arange(1/n_points, 1, 1/n_points)
    xseq = quantile(pseq, scale_T, shape_T, dist_T)
    scale_C = np.zeros(scale_T.shape)
    for i in range(scale_C.shape[1]):
        # opt = root_scalar(err2_censoring_exponential, args=(censoring, scale_T[:,i], shape_T[:,i], dist_T, xseq, method, False), method='brentq', bracket=(0.1,5),x0=0.1,x1=5)
        # assert opt.flag == 'converged', 'Brent minimization was not successful'
        # scale_C[:,i] = opt.root
        opt = minimize_scalar(fun=err2_censoring_exponential, bracket=(1,2),args=(censoring, scale_T[:,i], shape_T[:,i], dist_T, xseq, method),method='brent')
        assert opt.success, 'Brent minimization was not successful'
        scale_C[:,i] = opt.x
    return scale_C


def rvs_T(n_sim:int, k:int, scale:np.ndarray, shape:np.ndarray or None, dist:str, seed:None or int=None) -> np.ndarray:
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


def rvs(n_sim:int, scale:np.ndarray, shape:np.ndarray or None, dist:str, seed:None or int=None, censoring:float=0, n_points:int=1000, method:str='trapezoidal') -> tuple[np.ndarray,np.ndarray]:
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
    check_interval(censoring, 0, 1)
    if (not hasattr(scale, 'shape')) or (len(scale.shape) <= 1):
        scale, shape = t_wide(scale), t_wide(shape)
    assert scale.shape == shape.shape, 'scale and shape need to be the same'
    k = scale.shape[1]  # Assign the dimensionality
    # (i) Calculate the "actual" time-to-event
    T_act = rvs_T(n_sim=n_sim, k=k, scale=scale, shape=shape, dist=dist, seed=seed)
    if censoring == 0:  # Return actual if there is not censoring
        D_cens = np.zeros(T_act.shape) + 1
        return T_act, D_cens
    # (ii) Determine the "scale" from an exponential needed to obtain censoring
    scale_C = find_exp_scale_censoring(censoring=censoring, scale_T=scale, shape_T=shape, dist_T=dist, n_points=n_points, method=method)

    # Generate data from exponential distribution
    T_cens = -np.log(np.random.rand(n_sim, k)) / scale_C
    D_cens = np.where(T_cens <= T_act, 0, 1)
    T_obs = np.where(D_cens == 1, T_act, T_cens)
    return T_obs, D_cens


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

    def rvs(self, n_sim:int, censoring:float=0, seed:None or int=None, n_points:int=1000, method:str='trapezoidal'):
        """INVERTED-CDF APPROACH"""
        return rvs(n_sim=n_sim, censoring=censoring, scale=self.scale, shape=self.shape, dist=self.dist, seed=seed, n_points=n_points, method=method)
 
    def quantile(self, p:float or np.ndarray):
        return quantile(p, self.scale, self.shape, self.dist)

