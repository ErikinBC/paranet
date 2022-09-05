"""
Check that the multivariate censoring approach works when the covariates come from a normal distribution
"""

# External modules
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from scipy.integrate import quad, dblquad, simpson
# Internal modules
from paranet.models import parametric
from paranet.multivariate.dists import integral_for_censoring_multi
from paranet.univariate.dists import integral_for_censoring, quantile, univariate_dist

# Limits of integration
a, b = 0, 10
# Distribution parameters
scale_C = 2
scale_T, shape_T = 1, 1
dist_T = 'weibull'
# Simulation parameters
n = 100000
k_sim = 1
n_sim = 250
enc_censor_uni = univariate_dist('exponential', scale_C, 1)
enc_T_uni = univariate_dist(dist_T, scale_T, shape_T)

# Generate censoring dist
t_cens = enc_censor_uni.rvs(n)[0].flatten()
t_uni_dist = enc_T_uni.rvs(n)[0].flatten()

# --- (i) Sanity check univariate --- #
# P(C < T) ~ 3/4
int_uni_theory = quad(func=integral_for_censoring, a=0, b=b, args=(scale_C, scale_T, shape_T, dist_T))[0]
int_uni_emp = np.mean(t_uni_dist > t_cens)
print(f'Integral for univariate theory {int_uni_theory:.3f}, empirical {int_uni_emp:.3f}')

# --- (ii) Calculate for covariates --- #
lst_dist = np.repeat('weibull', k_sim)
x = norm().rvs(n, random_state=k_sim)
alpha_mat = np.tile(shape_T, [1, k_sim])
beta_mat = np.tile([0,1],[k_sim,1]).T
l2_beta = np.sum(beta_mat[1]**2)
enc_dist = parametric(lst_dist, x, alpha=alpha_mat, beta=beta_mat, scale_x=False, scale_t=False)
# # Check that mean aligns with expectation
# mu_emp = enc_dist.rvs(0, n_sim)[0].mean(axis=(1,2))
# pd.DataFrame({'emp':mu_emp, 'theory':np.exp(-x)})
# Calculate empirical censoring rate
t_multi_dist = enc_dist.rvs(0, n_sim=1)[0].flatten()
int_multi_emp = np.mean(t_multi_dist > t_cens)

# --- (iii) Try simpson's method --- #
n_points = 500
b1 = quantile(1-1/n_points, scale_C, 1, 'exponential').max()
b2 = quantile(1-1/n_points, scale_T, shape_T, dist_T).max()
b_upper = 3*max(b1, b2)
t_seq = np.exp(np.linspace(np.log(1e-10), np.log(b_upper), n_points))
r_seq = t_seq.copy()#lognorm(s=1,loc=0,scale=l2_beta).ppf(percentile)

f_stack = np.hstack([integral_for_censoring_multi(t_seq, r, scale_C, shape_T, dist_T, l2_beta) for r in r_seq])
int_multi_theory = simpson([simpson(f_stack_t,r_seq) for f_stack_t in f_stack],r_seq)
print(f'Integral for multivariate theory {int_multi_theory:.3f}, empirical {int_multi_emp:.3f}')


