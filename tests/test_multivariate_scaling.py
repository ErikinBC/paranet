"""
Check that scaling the time and covariates does not affect inference. Although the scale of the parameters will be different, the normalized covariates should produce the same hazard, survival, and pdf functions. 
"""

# External modules
import numpy as np
import pandas as pd
from scipy import stats

# Internal modules
from paranet.utils import dist_valid, t_long
from paranet.models import parametric

def test_scale_x():
    1

n, p = 250000, 5
tol_pct = 0.05
tol_dpct = 0.01
# Generate some exponential data
mu_X = np.linspace(-10, 10, p)
se_X = np.linspace(1, 10, p)
x_raw = stats.norm(loc=mu_X, scale=se_X).rvs([n,p], random_state=n)
b_raw = stats.norm(loc=0,scale=1/se_X**2).rvs(p) #np.atleast_2d().T
b0_raw = -np.sum(b_raw * mu_X) - 1
eta_raw = b0_raw + x_raw.dot(b_raw)
risk_raw = np.exp(eta_raw)
np.random.seed(n)
t_raw = np.random.exponential(scale=1/risk_raw, size=n)
d_raw = np.ones(n)


# (i) oracle model
beta_raw = t_long(np.append([b0_raw],b_raw))
enc_oracle = parametric('exponential', alpha=1, beta=beta_raw, scale_x=False, scale_t=False, add_int=True)

# (ii) unscaled version
enc_unscaled = parametric('exponential', x_raw, t_raw, d_raw, scale_x=False, scale_t=False, add_int=True)
enc_unscaled.fit()
err_unscaled = np.abs(enc_unscaled.beta/beta_raw-1).max() 
assert err_unscaled < tol_pct, f'Unscaled coefficients should be within {tol_pct}: {err_unscaled:.4f}'

# (iii) scaled_x version
enc_scale_x = parametric('exponential', x_raw, t_raw, d_raw, scale_x=True, scale_t=False, add_int=True)
enc_scale_x.fit()
# "Recover" original coefficients by divided transformed model by scaling factor
b_trans_x = enc_scale_x.beta[1:].flat / enc_scale_x.enc_x.scale_
err_scale_x = np.abs(b_trans_x/b_raw-1).max()
assert err_scale_x < tol_pct, f'Scaled coefficients divided by standard error should be within {tol_pct}: {err_scale_x:.4f}'
# Recover original intercept by substracting off sum_j b_j * mu_j / se_j from the scaled intercept
b0_trans_x = enc_scale_x.beta[0][0]-np.sum(enc_scale_x.beta[1:].flat*enc_scale_x.enc_x.mean_/enc_scale_x.enc_x.scale_)
err_int_x = np.abs(b0_trans_x/b0_raw-1)
assert err_int_x < tol_pct, f'Reconstructed intercept should be within {tol_pct}: {err_int_x:.4f}'


# (iv) Scale t only
enc_scale_t = parametric('exponential', x_raw, t_raw, d_raw, scale_x=False, scale_t=True, add_int=True)
enc_scale_t.fit()
# (i) Check that the survival values are the same
err_surv_t = np.abs(enc_scale_t.survival(t_raw, x_raw) - enc_oracle.survival(t_raw, x_raw)).max()
assert err_surv_t < tol_dpct, f'Expected survival functions to produce similar results for scale_t version {tol_dpct}: {err_surv_t:.4f}'
# (ii) Check that the ratio of pdf/hazards is a constant equal to maximum time scaler
ratio_pdf = enc_scale_t.pdf(t_raw, x_raw) / enc_oracle.pdf(t_raw, x_raw)
ratio_haz = enc_scale_t.hazard(t_raw, x_raw) / enc_oracle.hazard(t_raw, x_raw)
ratio_pdf_haz = np.append(ratio_pdf, ratio_haz)
err_ratio = np.abs(ratio_pdf_haz.mean() / enc_scale_t.enc_t.max_abs_ - 1).max()
assert err_ratio < tol_pct, f'Expected ratio of hazards/pdfs to be the same as the scaling constant {tol_pct}: {err_ratio:.4f}'
# (iii) Check that the quantile function works as expected
pseq = np.arange(0.1,1,0.1)
np.squeeze(enc_scale_t.quantile(pseq, x=x_raw[:2])) * enc_scale_t.enc_t.max_abs_
np.squeeze(enc_oracle.quantile(pseq, x=x_raw[:2]))


# (iv) Scale x and t
# --> check that the hazard/surv/pdf are exactly the same to the unscaled version
# --> check that the hazard/surv/pdf are close to the oracle
# --> Check the ppf


if __name__ == '__main__':
    # (i) Check covariate scaling
    test_scale_x()


