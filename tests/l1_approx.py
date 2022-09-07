"""
Compare different l1-approximators to the ground truth
"""

# External modules
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression


# (i) Load and normalize data
x, z = load_boston(return_X_y=True)
y = np.where(z > np.median(z), 1, 0)
x_trans = (x - np.atleast_2d(x.mean(0)))/np.atleast_2d(x.std(0,ddof=1))
print(f'Sum y={y.sum():.1f}, sum x_trans^2={(x_trans**2).sum():.1f}')
ix = np.c_[np.ones(len(x)), x_trans]

# (ii) Fit with approximate loss
def log_lik(beta, x, y):
    eta = x.dot(beta)
    py = 1 / (1 + np.exp(-eta))
    nll = -np.mean(y*np.log(py) + (1-y)*np.log(1-py))
    return nll

def grad_f(beta, x, y):
    n = x.shape[0]
    eta = x.dot(beta)
    py = 1 / (1 + np.exp(-eta))
    res = (y - py).reshape([n,1])
    grad = -x.T.dot(res).flatten() / n
    return grad

def grad_g1(beta, lam, alpha, eps:float=1e-5):
    """|x_i| ~ sqrt(x_i**2 + eps)"""
    grad = lam * (alpha * (2*beta/np.sqrt(beta**2 + eps)) + (1-alpha)*beta)
    return grad


# Compare to R-glmnet
r_coef = "-0.05856, 0, 0, 0, 0.00939, -0.05621, 0.35858, -0.17972, 0, 0, -0.06574, -0.33639, 0.02483, -0.78619"
r_coef = pd.Series(r_coef.split(', ')).astype(float)
#pd.DataFrame({'py':py_coef, 'r':r_coef})


# # This aligns with glm() in R
# x0_init = np.zeros(ix.shape[1])
# log_lik(x0_init, ix, y)
# grad_f(x0_init, ix, y)
# minimize(log_lik, x0_init, (ix, y), 'l-bfgs-b', grad_f).x


# # (ii) Fit using sklearn
# lam_r = 0.1
# alpha = 0.5
# n = len(x)
# c = (1/n)*(1/lam_r)
# mdl = LogisticRegression(penalty='elasticnet',fit_intercept=True, C=.1, l1_ratio=alpha, solver='saga', max_iter=100000)
# mdl.fit(x, y)
# py_coef = np.append(mdl.intercept_,mdl.coef_.flat)
# py_coef



