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

# Internal modules
from utils import param2array, len_of_none, t_long, t_wide

# List of currently supported distributions
dist_valid = ['exponential', 'weibull', 'gompertz']


def check_dist_str(dist:str) -> None:
    """CHECK THAT STRING BELONGS TO VALID DISTRIBUTION"""
    assert isinstance(dist, str)
    assert dist in dist_valid, f'dist must be one of: {", ".join(dist_valid)}'


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


def rvs(n_sim:int, k:int, scale:np.ndarray, shape:np.ndarray or None, dist:str, seed:None or int) -> np.ndarray:
    """
    GENERATES n_sim RANDOM SAMPLES FROM A GIVEN DISTRIBUTION

    Inputs
    ------
    n_sim:              Integer indicating the number of samples to generate
    k:                  The dimensionality of the scale/shape parameter
    seed:               Reproducibility seed
    See hazard() for remaining parameters

    Returns
    -------
    (n_sim x k) np.ndarray
    """
    if seed is not None:
        np.random.seed(seed)
    nlU = -np.log(np.random.rand(n_sim, k))
    if dist == 'exponential':
        T = nlU / scale
    if dist == 'weibull':
        T = (nlU / scale) ** (1/shape)
    if dist == 'gompertz':
        T = 1/shape * np.log(1 - (shape*nlU)/(scale))
    return T


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
        T = 1/shape * np.log(1 - (shape*nlp)/(scale))
    return T


class base_dist():
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

    def check_t(self, t):
        assert len(t) == self.n, f't needs to be the same size the input parameter: {self.n}'

    def hazard(self, t):
        return hazard(t=t, scale=self.scale, shape=self.shape, dist=self.dist)

    def survival(self, t):
        return survival(t=t, scale=self.scale, shape=self.shape, dist=self.dist)

    def pdf(self, t):
        return pdf(t=t, scale=self.scale, shape=self.shape, dist=self.dist)

    def rvs(self, n_sim:int, seed:None or int=None):
        """INVERTED-CDF APPROACH"""
        return rvs(n_sim=n_sim, k=self.k, scale=self.scale, shape=self.shape, dist=self.dist, seed=seed)

    def quantile(self, p:float or np.ndarray):
        return quantile(p, self.scale, self.shape, self.dist)



