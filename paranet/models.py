"""
Contains the main
"""

# Externel modules
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
# Internal modules
from paranet.univariate.solvers_grad import wrapper_grad_solver
from paranet.multivariate.gradient import nll_solver, grad_ll
from paranet.multivariate.multi_utils import args_alpha_beta, has_args_init
from paranet.multivariate.dists import hazard_multi, survival_multi, pdf_multi, quantile_multi, rvs_multi
from paranet.utils import broadcast_dist, broadcast_long, check_dist_str, all_or_None, t_long, str2lst, check_type, not_none, t_wide, dist2idx, find_nearest_decimal, check_d, check_t


class parametric():
    def __init__(self, dist:list or str, x:np.ndarray or None=None, t:np.ndarray or None=None, d:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None, scale_x:bool=True, scale_t:bool=True, add_int:bool=True) -> None:
        """
        Class for fitting parametric survival distributions with covariates. Choice of distribution will determine the call of other functions.

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
        # Initial attribute value of k, may get overwritten by _process_alpha_beta() if k == 1
        self.k = len(self.dist)

        # --- (ii) Pre-processing --- #
        # ~ (a) Process pre-defined covariates ~ #
        self._process_init_x(x)
        # ~ (b) Process pre-defined time/censoring ~ #
        self._process_td(t, d)
        # ~ (c) Process pre-defined shape/scale parameters ~ #
        self._process_alpha_beta(alpha, beta)
        # ~ (d) Process distribution ~ #
        self._process_dist()

        # --- (iii) Attribute check --- #
        attrs = ['dist','x','t','d','alpha','beta','scale_x','scale_t','add_int','k','p','p_x']
        for attr in attrs:
            assert hasattr(self, attr), f'{attr} has failed to assigned as an attribute with one of the process methods or during the input checks'


    def _process_init_x(self, x:np.ndarray or None=None) -> None:
        """
        Internal method for processing x on initialization if provided. Assigns the array to the following attributes:

        i) A (n,p) matrix (self.x)
        ii) The dimensionality of the array x and final dimensionality after (possibly) adding the intercept (self.p_x, self.p)
        iii) A normalizer if scale_x=True (self.enc_x)
        """
        self.has_x = not_none(x)
        self.x, self.enc_x, self.p_x, self.p = None, None, None, None
        if self.has_x:
            self.x = t_long(x)
            self.p_x = self.x.shape[1]
            self.p = self.p_x + int(self.add_int)
            if self.scale_x:
                self.enc_x = StandardScaler().fit(self.x)


    def _process_td(self, t:np.ndarray or None=None, d:np.ndarray or None=None) -> None:
        """
        Internal method for processing t and d on initialization if provided. Assigns the array to the following attributes:

        i) Two (n,k) matrices (self.t, self.d)
        ii) The number of columns of t (self.k_t)
        iii) A normalized if scale_t=True (self.enc_t)
        """
        assert all_or_None([t, d]), 'if x or t or d is specified, then all need to be specified'
        self.t, self.d = None, None
        self.has_dt = not_none(t)
        self.enc_t = None
        if self.has_dt:
            self.t, self.d = t_long(t), t_long(d)
            check_d(self.d)
            check_t(self.t)
            self.k_t = self.t.shape[1]
            assert self.t.shape == self.d.shape, 't and d need to be the same shape'
            if self.scale_t:
                self.enc_t = MaxAbsScaler().fit(self.t)
            if self.has_x:
                assert self.t.shape[0] == self.x.shape[0], 'x and t need to have the same number of rows'


    def _process_alpha_beta(self, alpha:np.ndarray or None=None, beta:np.ndarray or None=None) -> None:
        """
        Internal method to process the shape (alpha) & scale (beta) parameters. If alpha/beta are supplied:

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


    def _process_dist(self) -> None:
        """
        An internal method that checks the "dist" attribute list:

        i) Contains only valid distributions
        ii) Will be broadcasted to match (k) if alpha/beta are specified
        iii) Will have the column indices stores in didx for the different distributions
        """
        check_dist_str(self.dist) # Check for valid distribution
        self.dist = broadcast_dist(self.dist, self.k)
        self.didx = dist2idx(self.dist)


    def _get_p_k(self, x:np.ndarray or None=None, check_int:bool=False) -> tuple[int, int]:
        """
        Intenral method to get the number of covariates and distributions

        Inputs
        ------
        x:              An (n,p) matrix of covariates. This can include an intercept (a column of ones).
        check_int:      Whether p should be incremented by one if add_int==True (only set to True if x has not been processed to include an intercept)

        Returns
        -------
        The # of covariates (p) and the # distributions (k)
        """
        if x is None:
            assert self.x is not None, 'If x is none, x must be initialized'
            p = self.x.shape[1]
        else:
            p = x.shape[1]
        if self.add_int and check_int:
            p += 1
        return p, self.k

    def _process_t(self, t:np.ndarray or None=None) -> np.ndarray:
        """See _proccess_t_x"""
        has_args = has_args_init(t, self.t)
        if has_args:  # Use user-supplied parameters
            t = t_long(t)
        else:
            t = self.t.copy()
        check_t(t)
        # (possibly) scale time measurements
        if self.scale_t:
            if not_none(self.enc_t):  # Use the existing encoder
                t = self.enc_t.transform(t)
            else:  # Transform x 
                t = StandardScaler(with_mean=False).fit_transform(t)
        # (possibly) broadcast t if (t.shape[1] == 1) and (k > 1)
        t = broadcast_long(t, self.k)
        return t

    def _process_d(self, d:np.ndarray or None=None) -> np.ndarray:
        """For transforming d when inherited"""
        has_args = has_args_init(d, self.d)
        if has_args:  # Use user-supplied parameters
            d = t_long(d)
        else:
            d = self.d.copy()
        check_d(d)
        # (possibly) scale time measurements
        d = broadcast_long(d, self.k)
        return d


    def _process_x(self, x:np.ndarray or None=None) -> np.ndarray:
        """See _proccess_t_x"""
        has_args = has_args_init(x, self.x)
        if has_args:  # Use user-supplied parameters
            x = t_long(x)
        else:
            x = self.x.copy()
        # (possibly) scale covariates
        if self.scale_x:
            if not_none(self.enc_x):  # Use the existing encoder
                x = self.enc_x.transform(x)
            else:  # Transform x 
                x = StandardScaler().fit_transform(x)
        # (possibly) add an intercept to x
        if self.add_int:
            x = np.c_[np.ones(len(x)), x]
        return x


    def _process_t_x(self, t:np.ndarray or None=None, x:np.ndarray or None=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Internal method to process the time and covariate matrix (if specified) and return the number of features/distributions we should expect to see

        Inputs
        ------
        t:              An (n,k) matrix of time values. If array is (n,) or (n,1), then it will be broadcast (i.e. duplicated columns) if len(dist) > 1
        x:              An (n,p) matrix of covariates. This can include an intercept (a column of ones).

        Returns
        -------
        A (possibly) transformed array or time values (t) and covariates (x)
        """
        t = self._process_t(t)
        x = self._process_x(x)
        assert t.shape[0] == x.shape[0], 't and x need to have the same number of rows'
        p, k = self._get_p_k(x)
        x_p, t_k = x.shape[1], t.shape[1]
        assert x_p == p, f'Number of columns of x ({x_p}) needs to align with p ({p})'
        assert t_k == k, f'Number of columns of t ({t_k}) needs to align with k ({k})'
        return t, x


    def _trans_x_alpha_beta(self, x:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None) -> tuple[np.ndarray, np.ndarray]:
        """See _trans_t_x_alpha_beta()"""
        x_trans = self._process_x(x)
        p, k = self._get_p_k(x_trans)
        alpha_beta = args_alpha_beta(k, p, alpha, beta, self.alpha, self.beta)
        return x_trans, alpha_beta


    def _trans_t_x_alpha_beta(self, t:np.ndarray or None=None, x:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal method to process data to be passed in the hazard, survival, and pdf functions

        Returns
        -------
        t_trans:        (n,k) transformed time values
        x_trans:        (n,p) transformed covariate values
        alpha_beta:     (p,k) matrix of [shape ; scale] parameters
        """
        t_trans, x_trans = self._process_t_x(t, x)
        p, k = self._get_p_k(x_trans)
        alpha_beta = args_alpha_beta(k, p, alpha, beta, self.alpha, self.beta)
        return t_trans, x_trans, alpha_beta

    def _process_t_d_x(self, t:np.ndarray or None=None, d:np.ndarray or None=None, x:np.ndarray or None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process t, d, & x for the fit() method"""
        t_trans, x_trans = self._process_t_x(t, x)
        d_trans = self._process_d(d)
        k_t = t_trans.shape[1]
        assert t_trans.shape == d_trans.shape, 't_trans and d_trans need to be the same shape'
        assert k_t == len(self.dist), 'number of columns of t_trans needs to align with self.dist'
        return t_trans, d_trans, x_trans

    def _process_gamma(self, gamma:np.ndarray or float, p:int) -> np.ndarray:
        """Ensure the regularization parameter gamma aligns with number of columns of p"""
        gamma = np.atleast_1d(gamma)
        if gamma.shape[0] == 1:  # Broadcast
            gamma = np.repeat(gamma, p)
        gamma = t_long(gamma)
        assert gamma.shape[0] == p, 'length of gamma needs to align with the number of columns of x including the intercept'
        if self.add_int:  # do not regularize the intercept
            gamma[0] = 0
        return gamma


    def find_lambda_max(self, x:np.ndarray or None=None, t:np.ndarray or None=None, d:np.ndarray or None=None, rho:float=1) -> tuple[np.ndarray, float]:
        """
        Find the gamma in which coefficients are guaranteed to be zero (which can useful for doing a hyperparameter search)

        Returns
        -------
        [gamma_mat, thresh_min]
        gamma_mat:          A (p,k) array of distribution specific parameters which will achieve total sparsity for a given rho
        thresh_min:         The minimum threshold needed to get all zeros for the non-shape/scale parameters at gamma_mat
        """
        # - (i) Process inputs and do input checks - #
        t_trans, d_trans, x_trans = self._process_t_d_x(t, d, x)
        # - (ii) Solve the univariate optimization problem - #
        shape_scale0 = wrapper_grad_solver(t_trans, d_trans, self.dist)
        # - (iii) Find the (absolute) gradient at the intercept-values only - #
        p, k = x_trans.shape[1], t_trans.shape[1]
        alpha_beta = np.zeros([p+1, k])
        alpha_beta[0] = shape_scale0[0]
        if self.add_int:  # Put in log scale since exp(b0) is the new scale
            alpha_beta[1] = np.log(shape_scale0[1])
        # Note that the values of gamma/rho do not matter because alpha_beta is zero outside of the shape/scale rows
        idx_beta = 1 + int(self.add_int)
        grad0 = grad_ll(alpha_beta, x_trans, t_trans, d_trans, self.dist, gamma=0, rho=1)[idx_beta:]
        # Calculate the largest gradient for each distribution
        lam_max0 = np.max(np.abs(grad0),0)
        # Divide by rho
        lam_max_rho = lam_max0 / rho
        # - (iv) Fit model to get smallest value - #
        gamma_mat = np.zeros([p, k]) + np.atleast_2d(lam_max_rho)
        if self.add_int:  # Set intercept to zero
            gamma_mat[0] = 0
        self.fit(x, t, d, gamma_mat, rho, beta_thresh=1e-32)
        idx_beta = int(self.add_int)
        # Find threshold to set values to zero
        thresh_min = find_nearest_decimal(np.abs(self.beta[idx_beta:]).max())
        self.beta[idx_beta:] = 0
        # Return values
        return gamma_mat, thresh_min        


    def fit(self, x:np.ndarray or None=None, t:np.ndarray or None=None, d:np.ndarray or None=None, gamma:np.ndarray or float=0, rho:float=1, beta_thresh:float=1e-6, beta_ratio:float=1, eps:float=1e-8, grad_tol:float=0.005, n_perm:int=10, alpha_beta_init=None, maxiter:int=15000) -> None:
        """
        Defines a fitting procedure to learn alpha_beta which can then be used as an inhereted attribute in methods like hazard(), rvs(), or predict()
        
        Inputs
        ------
        x:                      A (n,p) array of covariates (if None, assumes x was assigned during initialization)
        t:                      A (n,k) array of time values (if None, assumes t was assigned during initialization)
        d:                      A (n,k) array of censoring values (if None, assumes d was assigned during initialization)
        dist:                   The string or list of distributions
        gamma:                  Regularization strength (0 is un-regularized)
        rho:                    Percent weighting towards L1 (1 is Lasso)
        beta_thresh:            If the absolute value of the (non-intercept) coefficients is less than beta_thresh, set to zero
        beta_ratio:             If the absolute ratio of the (non-intercept) coefficients to the largest coefficient is less than 1/beta_ratio, set to zero
        eps:                    Approximation for the L1-norm (default=1e-8)
        grad_tol:               Post-convergence checks for largest gradient size allowable
        n_perm:                 Number of random pertubations to do around "optimal" coefficient vector to check that lower log-likelihood is not possible
        alpha_beta_init:        Values to initialize the solver at
        

        Returns
        -------
        The (p+1,k) shape/scale covariate matrix as self.alpha_beta
        """
        # - (i) Process inputs and do input checks - #
        t_trans, d_trans, x_trans = self._process_t_d_x(t, d, x)
        p = x_trans.shape[1]        
        gamma = self._process_gamma(gamma, p)

        # - (ii) Run solver - #
        alpha_beta = nll_solver(x=x_trans, t=t_trans, d=d_trans, dist=self.dist, rho=rho, gamma=gamma, eps=eps, has_int=self.add_int, grad_tol=grad_tol, n_perm=n_perm, alpha_beta_init=alpha_beta_init, maxiter=maxiter)
        self.alpha = alpha_beta[[0]]
        self.beta = alpha_beta[1:]
        
        # - (iii) Apply full "sparsity" measures - #
        idx_beta = int(self.add_int)
        idx_thresh = np.abs(self.beta) < beta_thresh
        idx_ratio = np.abs(self.beta.max() / self.beta) >= beta_ratio
        idx_drop = idx_thresh & idx_ratio
        self.beta[idx_beta:] = np.where(idx_drop[idx_beta:], 0, self.beta[idx_beta:])
        
        # - (iv) output checks - #
        assert self.beta.shape[0] == x_trans.shape[1], 'Number of rows of beta should align with the number of columns of (transformed) x'
        assert t_trans.shape[1] == self.beta.shape[1], 'Number of columns of beta should be the same as k'


    def hazard(self, t:np.ndarray or None=None, x:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None) -> np.ndarray:
        """
        Calculate the hazard rate for different covariate/time combinations. Note that even if alpha/beta have been initialized, assigning them as arguments will over-ride existing existing values for calculating hazard

        Inputs
        ------
        t:              A (n,k) array of time values
        x:              A (n,p) array of covariates
        alpha:          A (k,1) array of shape values
        beta:           A (p,k) array of scale parameters

        Returns
        -------
        Returns an (n,k) matrix of hazards
        """
        t_trans, x_trans, alpha_beta = self._trans_t_x_alpha_beta(t=t, x=x, alpha=alpha, beta=beta)
        haz_mat = hazard_multi(alpha_beta, x_trans, t_trans, self.dist)
        if not_none(self.enc_t):  # Return to original scale
            haz_mat /= self.enc_t.max_abs_
        return haz_mat

    def survival(self, t:np.ndarray or None=None, x:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None) -> np.ndarray:
        """Calculate the survival probability (see hazard)"""
        t_trans, x_trans, alpha_beta = self._trans_t_x_alpha_beta(t=t, x=x, alpha=alpha, beta=beta)
        surv_mat = survival_multi(alpha_beta, x_trans, t_trans, self.dist)
        return surv_mat

    def pdf(self, t:np.ndarray or None=None, x:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None) -> np.ndarray:
        """Calculate the density (see hazard)"""
        t_trans, x_trans, alpha_beta = self._trans_t_x_alpha_beta(t=t, x=x, alpha=alpha, beta=beta)
        pdf_mat = pdf_multi(alpha_beta, x_trans, t_trans, self.dist)
        if not_none(self.enc_t):  # Return to original scale
            pdf_mat /= self.enc_t.max_abs_
        return pdf_mat


    def quantile(self, percentile:np.ndarray or float, x:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None, squeeze:bool=True) -> np.ndarray:
        """
        Calculate the quantiles for different percentiles.

        Inputs
        ------
        percentile:            A float or (q,) array specifying the percentiles we will calculate the quantile for: q=F^{-1}(p|x)
        squeeze:               Will return an (n,k) if percentile is of length one (default=True)
        See hazard() for other arguments
        
        Returns
        -------
        An (n,k,q) array of quantiles for the for each (n,k) point corresponding to a different distribution.
        """
        x_trans, alpha_beta = self._trans_x_alpha_beta(x=x, alpha=alpha, beta=beta)
        q_mat = quantile_multi(percentile, alpha_beta, x_trans, self.dist, squeeze)
        if not_none(self.enc_t):  # Return to original scale
            q_mat *= self.enc_t.max_abs_
        return q_mat


    def rvs(self, n_sim:int, censoring:float=0, seed:int or None=None, x:np.ndarray or None=None, alpha:np.ndarray or None=None, beta:np.ndarray or None=None, n_points:int=500, upper_constant:float=20, try_squeeze:bool=True) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate n_sim samples from the covariate conditional distribution.

        Inputs
        ------
        censoring:              Value of censoring (can be between [0,1))
        n_sim:                  Number of simulations draws

        Returns
        -------
        Two (n,k,n_sim) arrays of time measurements and censoring indicators. If x is (n,p) and beta is (p,k) then array size will be (n,k,n_sim) where [i,j,:] is an n_sim length array where i,k corresponds to the i'th covariate row and j'th distribution.
        """
        x_trans, alpha_beta = self._trans_x_alpha_beta(x=x, alpha=alpha, beta=beta)
        t, d = rvs_multi(censoring=censoring, n_sim=n_sim, alpha_beta=alpha_beta, x=x_trans, dist=self.dist, seed=seed, n_points=n_points, upper_constant=upper_constant, has_int=self.add_int)
        if not_none(self.enc_t):  # Return to original scale
            t *= self.enc_t.max_abs_
        if t.shape[2] == 1:
            if try_squeeze:
                t, d = t[:,:,0], d[:,:,0]
        return t, d