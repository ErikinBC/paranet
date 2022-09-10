"""
Check that scaling the time and covariates does not affect inference. Although the scale of the parameters will be different, the normalized covariates should produce the same hazard, survival, and pdf functions. 
"""

# External modules
import numpy as np
from scipy import stats

# Internal modules
from paranet.utils import dist_valid, t_long
from paranet.models import parametric


def test_scale_x(n:int=250000, p:int=5, lst_dist:list=dist_valid, tol_pct:float=0.05, tol_method:float=0.085):
    # Set up the covariates
    mu_X = np.linspace(-10, 10, p)
    se_X = np.linspace(1, 10, p)
    x_raw = stats.norm(loc=mu_X, scale=se_X).rvs([n,p], random_state=n)
    alpha_raw = 0.5
    b_raw = stats.norm(loc=0,scale=1/se_X**2).rvs(p, random_state=n)
    b0_raw = -np.sum(b_raw * mu_X) - 1
    beta_raw = t_long(np.append([b0_raw],b_raw))
    abeta_raw = np.append([alpha_raw],beta_raw)

    # Quantile to evaluate
    pseq = [0.25, 0.5, 0.75]
    # Methods to evaluate
    methods = ['survival', 'hazard', 'pdf']

    for dist in lst_dist:
        print(f'-- Running for {dist} --')
        # (i) oracle model
        enc_oracle = parametric(dist, x_raw, alpha=alpha_raw, beta=beta_raw, scale_x=False, scale_t=False, add_int=True)
        t_raw, d_raw = enc_oracle.rvs(1, seed=n)
        t_raw, d_raw = np.squeeze(t_raw), np.squeeze(d_raw)

        # (ii) unscaled version
        enc_unscaled = parametric(dist, x_raw, t_raw, d_raw, scale_x=False, scale_t=False, add_int=True)
        enc_unscaled.fit()
        if dist == 'exponential':
            enc_unscaled.alpha = enc_oracle.alpha
        abeta_unscaled = np.append(enc_unscaled.alpha, enc_unscaled.beta)
        err_unscaled = np.abs(abeta_unscaled/abeta_raw-1).max() 
        assert err_unscaled < tol_pct, f'Unscaled coefficients should be within {tol_pct}: {err_unscaled:.4f}'

        # (iii) scaled_x version
        enc_scale_x = parametric(dist, x_raw, t_raw, d_raw, scale_x=True, scale_t=False, add_int=True)
        enc_scale_x.fit()
        if dist == 'exponential':
            enc_scale_x.alpha = enc_oracle.alpha
        # Recover original intercept by substracting off sum_j b_j * mu_j / se_j from the scaled intercept
        b0_trans_x = enc_scale_x.beta[0][0]-np.sum(enc_scale_x.beta[1:].flat*enc_scale_x.enc_x.mean_/enc_scale_x.enc_x.scale_)
        # "Recover" original coefficients by divided transformed model by scaling factor
        b_trans_x = enc_scale_x.beta[1:].flat / enc_scale_x.enc_x.scale_
        abeta_trans_x = np.append(np.append(enc_scale_x.alpha,[b0_trans_x]),b_trans_x)
        err_scale_x = np.abs(abeta_trans_x/abeta_raw-1).max()
        assert err_scale_x < tol_pct, f'Scaled coefficients divided by standard error should be within {tol_pct}: {err_scale_x:.4f}'

        # (iv) Scale t (and x) only
        for scale_x in [False, True]:
            enc_scale_t = parametric(dist, x_raw, t_raw, d_raw, scale_x=scale_x, scale_t=True, add_int=True)
            enc_scale_t.fit()
            if dist == 'exponential':
                enc_scale_t.alpha = enc_oracle.alpha
            # (a) Check that the three methods are the same
            for method in methods:
                method_oracle = getattr(enc_oracle, method)(t_raw, x_raw).flatten()
                method_scale_t = getattr(enc_scale_t, method)(t_raw, x_raw).flatten()
                # Using the 5th-95th percentile, compare results
                idx_sort = np.argsort(method_oracle)
                method_oracle, method_scale_t = method_oracle[idx_sort], method_scale_t[idx_sort]
                method_oracle = method_oracle[int(n*0.1):int(n*0.90)]
                method_scale_t = method_scale_t[int(n*0.1):int(n*0.90)]
                err_method_t = np.abs(method_scale_t / method_oracle-1).max()
                assert err_method_t < tol_method, f'Values for method {method} should be within {tol_method} of the oracle: {err_method_t:.4f}'
            # (b) Check that the quantile() works   
            err_quantile_scale_t = np.abs(enc_scale_t.quantile(pseq, x=x_raw) / enc_oracle.quantile(pseq, x=x_raw) - 1).max()
            assert err_quantile_scale_t < tol_pct, f'Quantile values should be within {tol_pct} of the orcale: {err_quantile_scale_t:.4f}'
            # (c) Check that rvs() works
            quant_rvs_oracle = np.quantile(enc_oracle.rvs(n,seed=1,x=x_raw[:4])[0],pseq,[1,2])
            quant_rvs_scale_t = np.quantile(enc_scale_t.rvs(n,seed=1,x=x_raw[:4])[0],pseq,[1,2])
            err_quant_rvs = np.abs(quant_rvs_scale_t / quant_rvs_oracle - 1).max()
            assert err_quant_rvs < tol_pct, f'Quantile values should be within {tol_pct} of the orcale: {err_quant_rvs:.4f}'



if __name__ == '__main__':
    # (i) Check covariate scaling
    n, p = 250000, 5
    tol_pct, tol_method = 0.05, 0.085
    test_scale_x(n, p, dist_valid, tol_pct, tol_method)


