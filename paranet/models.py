"""
Contains the main
"""

# Externel modules
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Internal modules
from paranet.utils import broadcast_dist, check_dist_str, all_or_None, t_long, str2lst, check_type, not_none, t_wide, dist2idx
from paranet.multivariate.dists import hazard_multi, survival_multi, pdf_multi, quantile_multi


class parametric():
    def __init__(self, dist:list or str, x:np.ndarray or None=None, t:np.ndarray or None=None, d:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None, scale_x:bool=True, scale_t:bool=True, add_int:bool=True) -> None:
        """
        Backbone class for parametric survival distributions with covariate. Choice of distribution will determine the call of other functions.

        Inputs
        ------
        dist:           A string for a valid distribution: 'exponential', 'weibull', 'gompertz' (only argument that needs to be specified)
        """
        # --- (i) Input checks --- #
        [check_type(z, bool) for z in [scale_x, scale_t, add_int]]
        self.scale_x, self.scale_t = scale_x, scale_t
        self.add_int = True
        self.dist = dist

        # --- (ii) Pre-processing --- #
        # ~ (a) Process pre-defined covariates ~ #
        self.process_x(x)
        # ~ (b) Process pre-defined time/censoring ~ #
        self.process_td(t, d)
        # ~ (c) Process pre-defined shape/scale parameters ~ #
        self.process_alpha_beta(alpha, beta)
        # ~ (d) Process distribution ~ #
        self.process_dist()


    def process_dist(self) -> None:
        """Will ensure that distribution list matches expected size; requires that dist has already been assigned as an attribute"""
        check_dist_str(self.dist) # Check for valid distribution
        self.dist = str2lst(self.dist)
        if not_none(self.k):
            self.dist = broadcast_dist(self.dist, self.k)
        self.didx = dist2idx(self.dist)


    def process_x(self, x:np.ndarray or None=None) -> None:
        """
        When x is provided, processes the data to ensure that x is properly formatted and scaled
        """
        self.has_x = not_none(x)
        self.enc_x, self.p_x = None, None
        if self.has_x:
            self.x = t_long(x)
            self.p_x = x.shape[1]
            if self.scale_x:
                self.enc_x = StandardScaler().fit(self.x)
            else:
                # Training on a vector of zeros ensures that data is returned as-is
                self.enc_x = StandardScaler(np.zeros([1,self.p_x]))


    def process_td(self, t:np.ndarray or None=None, d:np.ndarray or None=None) -> None:
        di_dt = {'t':t, 'd':d}
        assert all_or_None(di_dt.values()), 'if x or t or d is specified, then all need to be specified'
        self.has_xdt = not_none(di_dt['x'])
        self.enc_t = None
        if self.has_xdt:
            self.t, self.d = t_long(t), t_long(d)
            self.k_t = self.t.shape[1]
            assert self.k_t == self.d.shape[1], 'Number of columns of t/d need to align'
            if self.scale_t:
                self.enc_t = MinMaxScaler().fit(self.t)
            else:
                self.enc_t = MinMaxScaler().fit(self.t)


    def process_alpha_beta(self, alpha:np.ndarray or None=None, beta:np.ndarray or None=None) -> None:
        """Processes the shape/scale parameters"""
        di_alpha_beta = {'alpha':alpha, 'beta':beta}
        assert all_or_None(di_alpha_beta.values()), 'if alpha or beta is specified, then all need to be specified'
        self.has_alpha_beta = not_none(di_alpha_beta['alpha'])
        self.alpha, self.beta = None, None
        if self.has_alpha_beta:
            self.alpha, self.beta = t_wide(alpha), t_wide(beta)
            self.p, self.k = self.beta.shape
            assert self.k == self.alpha.shape[1], 'Number of columns/length of alpha and beta need to be the same'
            if not_none(self.p_x):
                assert self.p_x + int(self.add_int) == self.p, 'Number of rows of beta should be equal to number of columns of x plus 1 if an intercept is to be added'
            # Broadcast the distribution list
            self.dist = broadcast_dist(self.dist)


    def fit(self, x:np.ndarray or None=None, t:np.ndarray or None=None, d:np.ndarray or None=None, l2:float or np.ndarray=0, l1:float or np.ndarray=0):
        """
        Fit the parametric survival model for different l2/l1 regularization stregnths (defaults to un-regularized)
        """


    def hazard(self, t:np.ndarray, x:np.ndarray, alpha:np.ndarray or None=None, beta:np.ndarray or None=None) -> np.ndarray:
        if self.has_alpha_beta:
            alpha_beta = np.vstack(self.alpha, self.beta)
        else:
            assert not_none(alpha) and not_none(beta), 'If alpha/beta have not been pre-specified, then they must be specified'
            alpha_beta = np.vstack(alpha, beta)
        # WORDS.....
        haz_mat = hazard_multi(alpha_beta, x, t, self.dist)
        return haz_mat


