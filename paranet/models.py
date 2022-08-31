"""
Contains the main
"""

# Externel modules
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
# Internal modules
from paranet.utils import broadcast_dist, broadcast_long, check_dist_str, all_or_None, t_long, str2lst, check_type, not_none, t_wide, dist2idx
from paranet.multivariate.dists import hazard_multi#, survival_multi, pdf_multi, quantile_multi
from paranet.multivariate.utils import args_alpha_beta, has_args_init


class parametric():
    def __init__(self, dist:list or str, x:np.ndarray or None=None, t:np.ndarray or None=None, d:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None, scale_x:bool=True, scale_t:bool=True, add_int:bool=True) -> None:
        """
        Backbone class for parametric survival distributions with covariate. Choice of distribution will determine the call of other functions.

        Inputs
        ------
        dist:           A string for a valid distribution: 'exponential', 'weibull', 'gompertz' (only argument that is required)
        x:              An (n,p) matrix of covariates. This can include an intercept (a column of ones).
        t:              An (n,k) matrix of time values. If array is (n,) or (n,1), then it will be broadcast (i.e. duplicated columns) if len(dist) > 1
        d:              An (n,k) matrix of censoring values. Must match the same size as t and will be equivalently broadcasted
        alpha:          A (k,) array of shape parameters which should align with length of dist
        beta:           A (p,k) matrix of scale parameters; number of rows should align with number of columns of x
        scale_x:        Whether the design matrix x should be normalized before model fitting and inference
        scale_t:        Whether the matrix of time values should be scaled by the maximum value during training and inference (although values will be returned to original scale for the purposes of output)
        """
        # --- (i) Input checks --- #
        [check_type(z, bool) for z in [scale_x, scale_t, add_int]]
        self.scale_x, self.scale_t = scale_x, scale_t
        self.add_int = add_int
        self.dist = str2lst(dist)
        # Initial attribute value of k, may get overwritten by process_alpha_beta() if k == 1
        self.k = len(self.dist)

        # --- (ii) Pre-processing --- #
        # ~ (a) Process pre-defined covariates ~ #
        self.process_x(x)
        # ~ (b) Process pre-defined time/censoring ~ #
        self.process_td(t, d)
        # ~ (c) Process pre-defined shape/scale parameters ~ #
        self.process_alpha_beta(alpha, beta)
        # ~ (d) Process distribution ~ #
        self.process_dist()

        # --- (iii) Attribute check --- #
        attrs = ['dist','x','t','d','alpha','beta','scale_x','scale_t','add_int','k','p','p_x']
        for attr in attrs:
            assert hasattr(self, attr), f'{attr} has failed to assigned as an attribute with one of the process methods or during the input checks'


    def process_x(self, x:np.ndarray or None=None) -> None:
        """
        When x is provided assigns the array to the following attribute:
        i) A (n,k) matrix
        ii) The dimensionality of the array x and final dimensionality after (possibly) adding the intercept
        iii) An encoding if scale_x=True
        """
        self.has_x = not_none(x)
        self.x, self.enc_x, self.p_x, self.p = None, None, None, None
        if self.has_x:
            self.x = t_long(x)
            self.p_x = x.shape[1]
            self.p = self.p_x + int(self.add_int)
            if self.scale_x:
                self.enc_x = StandardScaler().fit(self.x)


    def process_td(self, t:np.ndarray or None=None, d:np.ndarray or None=None) -> None:
        """
        Explain....
        """
        assert all_or_None([t, d]), 'if x or t or d is specified, then all need to be specified'
        self.t, self.d = None, None
        self.has_dt = not_none(t)
        self.enc_t = None
        if self.has_dt:
            self.t, self.d = t_long(t), t_long(d)
            self.k_t = self.t.shape[1]
            assert self.t.shape == self.d.shape[1], 't and d need to be the same shape'
            if self.scale_t:
                self.enc_t = MaxAbsScaler().fit(self.t)
            if self.has_x:
                assert self.t.shape[0] == self.x.shape[0], 'x and t need to have the same number of rows'


    def process_alpha_beta(self, alpha:np.ndarray or None=None, beta:np.ndarray or None=None) -> None:
        """
        Processes the shape/scale parameters. If alpha/beta are supplied:
        i) Dimensionality checks will be performed against x is also initialized
        ii) Attributes p and p_x will be assigned if x is not initialized
        iii) Attribute k will be (possibly) updated if dist is a list of length 1 but alpha has more than one value
        iv) alpha/beta are in the correct row-wise format
        """
        di_alpha_beta = {'alpha':alpha, 'beta':beta}
        assert all_or_None(di_alpha_beta.values()), 'if alpha or beta is specified, then all need to be specified'
        self.has_alpha_beta = not_none(di_alpha_beta['alpha'])
        self.alpha, self.beta = None, None
        if self.has_alpha_beta:
            self.alpha, self.beta = t_wide(alpha), t_wide(beta)
            p_beta, k_beta = self.beta.shape
            assert self.alpha.shape[1] == k_beta, 'Number of columns/length of alpha and beta need to be the same'
            # Check for dimensionality alignment with dist and ovewrite
            assert (self.k==1 and k_beta>1) or (self.k==k_beta), 'Dimensionality of beta needs to align with len(dist) OR len(dist) needs to be 1 to allow for broadcasting'
            self.k = max(self.k, k_beta)
            # Check for dimensionality alignment with x
            if not_none(self.p_x):
                assert p_beta == self.p, 'Number of rows of beta should be equal to number of columns of x plus 1 if an intercept is to be added'
            else:
                self.p, self.p_x = p_beta, p_beta - int(self.add_int)


    def process_dist(self) -> None:
        """
        Checks that the distribution attribute list:
        i) Contains only valid distributions
        ii) Will be broadcasted to match (k) if alpha/beta are specified
        iii) Will have the column indices stores in didx for the different distributions
        """
        check_dist_str(self.dist) # Check for valid distribution
        self.dist = broadcast_dist(self.dist, self.k)
        self.didx = dist2idx(self.dist)


    def process_t_x(self, t:np.ndarray or None=None, x:np.ndarray or None=None) -> tuple[np.ndarray, np.ndarray, int, int]:
        """
        Process the time and covariate matrix (if specified) and return the number of features/distributions we should expect to see

        Inputs
        ------
        t:              An (n,k) matrix of time values. If array is (n,) or (n,1), then it will be broadcast (i.e. duplicated columns) if len(dist) > 1
        x:              An (n,p) matrix of covariates. This can include an intercept (a column of ones).

        Returns
        -------
        A (possibly) transformed array or time values (t), covariates (x), number of distributions (k), and number of covariates (p)
        """
        has_args = has_args_init(t, x, self.t, self.x)
        if has_args:  # Use user-supplied parameters
            t, x = t_long(t), t_long(x)
            assert t.shape[0] == x.shape[0], 't and x need to have the same number of rows'
        else:  # Use the inhereted attributes
            t, x = self.t.copy(), self.x.copy()
        # (possibly) scale covariates
        if self.scale_x:
            if not_none(self.enc_x):  # Use the existing encoder
                x = self.enc_x.transform(x)
            else:  # Transform x 
                x = StandardScaler().fit_transform(x)
        # (possibly) scale time measurements
        if self.scale_t:
            if not_none(self.enc_x):  # Use the existing encoder
                t = self.enc_t.transform(t)
            else:  # Transform x 
                t = StandardScaler(with_mean=False).fit_transform(t)
        # (possibly) broadcast t if (t.shape[1] == 1) and (k > 1)
        t = broadcast_long(t, self.k)
        # (possibly) add an intercept to x
        if self.add_int:
            x = np.c_[np.ones(len(x)), x]
        # calclate final dimensionality
        k, p = t.shape[1], x.shape[1]
        return t, x, k, p


    def hazard(self, t:np.ndarray or None=None, x:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None) -> np.ndarray:
        """
        Calculate the hazard rate for different covariate/time combinations. Note that even if alpha/beta have been initialized, assigning them as arguments will over-ride existing existing values for calculating hazard

        Inputs
        ------
        t:              An 

        Outputs
        -------
        Returns an (n,k) matrix of hazards
        """
        t_trans, x_trans, k, p = self.process_t_x(t, x)
        alpha_beta = args_alpha_beta(k, p, alpha, beta, self.alpha, self.beta)
        haz_mat = hazard_multi(alpha_beta, x_trans, t_trans, self.dist)
        return haz_mat


n, p = 20, 5
np.random.seed(1)
x_mat = np.random.rand(n, p)
lst_dist = ['exponential','weibull','gompertz']
k = len(lst_dist)
alpha = np.random.rand(k)
beta = np.exp(np.random.randn(p+1,k))
# (i) Check that class can be initialized with onlly distributions
enc_para = parametric(lst_dist)

# (ii) Check that hazard can be calculated with initialized x and supplied alpha/beta
enc_para = parametric(lst_dist, x=x_mat)

# (iii) Check that hazard can be calculated with initialized initialized alpha/beta and supplied x
enc_para = parametric(lst_dist, alpha=alpha, beta=beta)

# (iv) Check that hazard can be calculated with initialized x, alpha, beta
enc_para = parametric(lst_dist, x=x_mat, alpha=alpha, beta=beta)


# When intercept is not default, then beta of (p,k) should error out




