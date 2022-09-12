"""
Test the gradient solver for the non-regularized models
"""

# External modules
import numpy as np

# Internal modules
from paranet.utils import dist_valid
from paranet.models import parametric

def test_param_consistency(n:int=1000000, p:int=5, lst_dist:list=dist_valid, n_sim:int=10, tol_bhat:float=0.02, tol_moment:float=0.03):
    """
    For a fixed p, increasing n should achieve asymtotic convergence of the parameters
    """
    k = len(lst_dist)
    methods = ['hazard','survival','pdf']
    seed_range = np.arange(1, n_sim+1)
    for s in seed_range:
        # Generate ground-truth data
        np.random.seed(s)
        if s % 1 == 0:
            print(f'Seed {s}')
        alpha = np.random.rand(1,k)+0.5
        alpha[:,np.where(np.array(lst_dist) == 'exponential')[0][0]] = 1
        beta = np.random.uniform(-1,1,[p+1,k])
        alpha_beta = np.vstack([alpha, beta])
        x = 0.5*np.random.randn(n,p)
        # Draw data from covariates
        enc_dgp = parametric(lst_dist, x, alpha=alpha, beta=beta, add_int=True, scale_x=False, scale_t=False)
        t, d = enc_dgp.rvs(n_sim=1, seed=s)
        t, d = np.squeeze(t), np.squeeze(d)
        # Fit model
        enc_mdl = parametric(lst_dist, x, add_int=True, scale_x=False, scale_t=False)
        enc_mdl.fit(x, t, d)
        alpha_beta_hat = np.vstack([enc_mdl.alpha, enc_mdl.beta])
        # (i) Check coefficient convergence
        bhat_err1 = np.abs(alpha_beta_hat - alpha_beta).max()
        bhat_err2 = np.abs(alpha_beta_hat / alpha_beta - 1).max()
        bhat_err = min(bhat_err1, bhat_err2)
        assert bhat_err < tol_bhat, f'Largest discrepancy between actual and expected coefficient is greater than {tol_bhat}: {bhat_err:.6f} for simulation={s}'
        # (ii) Check that hazard/survival/pdf is close enough to ground-truth
        x_oos = np.random.randn(n_sim, p)
        t_oos = np.squeeze(enc_dgp.rvs(n_sim=1, x=x_oos)[0])
        for method in methods:
            moment_dgp = getattr(enc_dgp, method)(t_oos, x_oos)
            moment_hat = getattr(enc_mdl, method)(t_oos, x_oos)
            moment_err1 = np.abs(moment_hat - moment_dgp).max()
            moment_err2 = np.abs(moment_hat/moment_dgp-1).max()
            moment_err = min(moment_err1, moment_err2)
            assert moment_err < tol_moment, f'Largest discrepancy between actual and expected moment is greater than {tol_moment}: {moment_err:.6f} for simulation={s}, method={method}'


if __name__ == '__main__':
    # (ii) Check that l2 regularization can solve p > n problems...
    
    # (i) Check that with large enough sample size we can arbitraily close to expected solution
    n, p, n_sim = 1000000, 5, 10
    tol_bhat, tol_moment = 0.02, 0.03
    test_param_consistency(n, p, dist_valid, n_sim, tol_bhat, tol_moment)

    print('~~~ End of test_multivariate_solver.py ~~~')