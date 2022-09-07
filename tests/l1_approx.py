"""
Compare different l1-approximators to the ground truth
"""

# External modules
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.datasets import load_boston

# (ii) Fit with approximate loss
def loss_g1(beta:np.ndarray, lam:np.ndarray or float, alpha:float, eps:float=1e-5) -> float:
    """|x_i| ~ sqrt(x_i**2 + eps)"""
    l1_approx = np.sqrt(beta**2 + eps)
    ll = np.sum(lam * (alpha*l1_approx + 0.5*(1-alpha)*beta**2))
    return ll

def grad_g1(beta:np.ndarray, lam:np.ndarray or float, alpha:float, eps:float=1e-5)  -> np.ndarray:
    """|x_i| ~ sqrt(x_i**2 + eps)"""
    l1_approx = beta/np.sqrt(beta**2 + eps)
    grad = lam * (alpha * l1_approx + (1-alpha)*beta)
    return grad

def loss_g2(beta:np.ndarray, lam:np.ndarray or float, alpha:float) -> float:
    """|x_i| ~ [log(1+exp(-eps*x)) + log(1+exp(eps*x))]/eps"""
    l1_approx = np.log(1 + np.exp(-beta)) + np.log(1 + np.exp(beta))
    ll = np.sum(lam * (alpha*l1_approx + 0.5*(1-alpha)*beta**2))
    return ll

def grad_g2(beta:np.ndarray, lam:np.ndarray or float, alpha:float, eps:float=1e-5) -> np.ndarray:
    l1_approx = 1/(1+np.exp(-beta)) - 1/(1+np.exp(beta))
    grad = lam * (alpha * l1_approx + (1-alpha)*beta)
    return grad


b = np.arange(0.1,1,0.1)
loss_g1(b, lam=1, alpha=1)
loss_g2(b, lam=1, alpha=1)

grad_g1(b, lam=1, alpha=1)
grad_g2(b, lam=1, alpha=1)


di_g = {1:{'loss':loss_g1, 'grad':grad_g1}, 2:{'loss':loss_g2, 'grad':grad_g2}}

def neg_loglik(beta:np.ndarray, x:np.ndarray, y:np.ndarray, lam:np.ndarray or float, alpha:float, di:dict, eps:float=1e-5):
    eta = x.dot(beta)
    py = 1 / (1 + np.exp(-eta))
    lik = -np.mean(y*np.log(py) + (1-y)*np.log(1-py))
    reg = di['loss'](beta, lam, alpha, eps)
    nll = lik + reg
    return nll

def grad_loglik(beta:np.ndarray, x:np.ndarray, y:np.ndarray, lam:np.ndarray or float, alpha:float, di:dict, eps:float=1e-5):
    n = x.shape[0]
    eta = x.dot(beta)
    py = 1 / (1 + np.exp(-eta))
    res = (y - py).reshape([n,1])
    grad_lik = -x.T.dot(res).flatten() / n
    grad_reg = di['grad'](beta, lam, alpha, eps)
    grad = grad_lik + grad_reg
    return grad


# (i) Load and normalize data
x, z = load_boston(return_X_y=True)
y = np.where(z > np.median(z), 1, 0)
x_trans = (x - np.atleast_2d(x.mean(0)))/np.atleast_2d(x.std(0,ddof=1))
print(f'Sum y={y.sum():.1f}, sum x_trans^2={(x_trans**2).sum():.1f}')
ix = np.c_[np.ones(len(x)), x_trans]

# (ii) Fit un-regularized model
x0_init = np.zeros(ix.shape[1])
neg_loglik(beta=x0_init, x=ix, y=y, lam=0, alpha=1, di=di_g[1])
grad_loglik(beta=x0_init, x=ix, y=y, lam=0, alpha=1, di=di_g[1])
lam, alpha = 0, 1
minimize(neg_loglik, x0_init, (ix, y, lam, alpha, di_g[1]), 'l-bfgs-b', grad_loglik).x.round(3)

# (iii) Fit l2-regularized model
lam, alpha = np.append([0],np.repeat(1,x.shape[1])), 0
minimize(neg_loglik, x0_init, (ix, y, lam, alpha, di_g[1]), 'l-bfgs-b', grad_loglik).x.round(3)

# (iv) Fit the l1-regularized model
lam, alpha = np.append([0],np.repeat(0.1,x.shape[1])), 1
minimize(neg_loglik, x0_init, (ix, y, lam, alpha, di_g[1], 1e-10), 'l-bfgs-b', grad_loglik).x.round(3)
minimize(neg_loglik, x0_init, (ix, y, lam, alpha, di_g[2], 1e-10), 'l-bfgs-b', grad_loglik).x.round(3)



# Compare to R-glmnet
r_coef = "-0.05856, 0, 0, 0, 0.00939, -0.05621, 0.35858, -0.17972, 0, 0, -0.06574, -0.33639, 0.02483, -0.78619"
r_coef = pd.Series(r_coef.split(', ')).astype(float)
#pd.DataFrame({'py':py_coef, 'r':r_coef})




# # (ii) Fit using sklearn
# from sklearn.linear_model import LogisticRegression
# lam_r = 0.1
# alpha = 0.5
# n = len(x)
# c = (1/n)*(1/lam_r)
# mdl = LogisticRegression(penalty='elasticnet',fit_intercept=True, C=.1, l1_ratio=alpha, solver='saga', max_iter=100000)
# mdl.fit(x, y)
# py_coef = np.append(mdl.intercept_,mdl.coef_.flat)
# py_coef



