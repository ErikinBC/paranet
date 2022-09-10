"""
Check that the l1-approximator gets a a similar result to R's glmnet for the boston dataset
"""

# External modules
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.datasets import load_boston

def loss_g(beta:np.ndarray, lam:np.ndarray or float, alpha:float, eps:float=1e-5) -> float:
    """|x_i| ~ sqrt(x_i**2 + eps)"""
    l1_approx = np.sqrt(beta**2 + eps)
    ll = np.sum(lam * (alpha*l1_approx + 0.5*(1-alpha)*beta**2))
    return ll

def grad_g(beta:np.ndarray, lam:np.ndarray or float, alpha:float, eps:float=1e-5)  -> np.ndarray:
    """|x_i| ~ sqrt(x_i**2 + eps)"""
    l1_approx = beta/np.sqrt(beta**2 + eps)
    grad = lam * (alpha * l1_approx + (1-alpha)*beta)
    return grad


def neg_loglik(beta:np.ndarray, x:np.ndarray, y:np.ndarray, lam:np.ndarray or float, alpha:float, eps:float=1e-10):
    eta = x.dot(beta)
    py = 1 / (1 + np.exp(-eta))
    lik = -np.mean(y*np.log(py) + (1-y)*np.log(1-py))
    reg = loss_g(beta, lam, alpha, eps)
    nll = lik + reg
    return nll

def grad_loglik(beta:np.ndarray, x:np.ndarray, y:np.ndarray, lam:np.ndarray or float, alpha:float, eps:float=1e-10):
    n = x.shape[0]
    eta = x.dot(beta)
    py = 1 / (1 + np.exp(-eta))
    res = (y - py).reshape([n,1])
    grad_lik = -x.T.dot(res).flatten() / n
    grad_reg = grad_g(beta, lam, alpha, eps)
    grad = grad_lik + grad_reg
    return grad

def test_py_r():
    # (i) Load and normalize data
    x, z = load_boston(return_X_y=True)
    y = np.where(z > np.median(z), 1, 0)
    x_trans = (x - np.atleast_2d(x.mean(0)))/np.atleast_2d(x.std(0,ddof=1))
    print(f'Sum y={y.sum():.1f}, sum x_trans^2={(x_trans**2).sum():.1f}')
    ix = np.c_[np.ones(len(x)), x_trans]

    # (ii) Fit un-regularized model
    x0_init = np.zeros(ix.shape[1])
    neg_loglik(beta=x0_init, x=ix, y=y, lam=0, alpha=1)
    grad_loglik(beta=x0_init, x=ix, y=y, lam=0, alpha=1)
    lam, alpha = 0, 1
    bhat_unreg = minimize(neg_loglik, x0_init, (ix, y, lam, alpha), 'l-bfgs-b', grad_loglik).x

    # (iii) Fit l2-regularized model
    lam, alpha = 1, 0
    lam_vec = np.append([0],np.repeat(lam,x.shape[1]))
    bhat_l2 = minimize(neg_loglik, x0_init, (ix, y, lam_vec, alpha), 'l-bfgs-b', grad_loglik).x


    # (iv) Fit the l1-regularized model
    lam, alpha = 0.1, 1
    lam_vec = np.append([0],np.repeat(lam,x.shape[1]))
    bhat_l1 = minimize(neg_loglik, x0_init, (ix, y, lam_vec, alpha), 'l-bfgs-b', grad_loglik).x

    # (v) Fit the elastic net model
    lam, alpha = 0.25, 0.5
    lam_vec = np.append([0],np.repeat(lam,x.shape[1]))
    bhat_elnet = minimize(neg_loglik, x0_init, (ix, y, lam_vec, alpha), 'l-bfgs-b', grad_loglik).x

    # (vi) Compare to glmnet
    df_py = pd.DataFrame({'unreg':bhat_unreg, 'l2':bhat_l2, 'l1':bhat_l1, 'elnet':bhat_elnet})
    df_r = """
                unreg          l2          l1        elnet
    s0      -0.02171431 -0.03496629 -0.09307086 -0.034838586
    crim    -0.52458294 -0.05707621  0.00000000  0.000000000
    zn       0.41492711  0.05953021  0.00000000  0.000000000
    indus    0.14763483 -0.09052322  0.00000000 -0.020222149
    chas     0.43672130  0.05225872  0.00000000  0.000000000
    nox     -0.76319181 -0.08767222  0.00000000 -0.008123666
    rm       1.07838719  0.14770144  0.06656291  0.140197931
    age     -0.70516629 -0.10793995  0.00000000 -0.073959914
    dis     -1.49748331  0.02799960  0.00000000  0.000000000
    rad      2.20168700 -0.05167393  0.00000000  0.000000000
    tax     -1.67680575 -0.08616742  0.00000000 -0.020354261
    ptratio -1.16724170 -0.12656863 -0.20162521 -0.127445195
    black    0.36748552  0.06765398  0.00000000  0.000000000
    lstat   -2.33241362 -0.18369119 -1.11138392 -0.456631660
    """
    df_r = df_r.split('\n')[1:-1]
    colnames_r = [z for z in df_r[0].split(' ') if len(z)>1]
    data = pd.Series(df_r[1:]).str.strip()
    data = data.str.split('\\s{1,}',1,True)
    data = data[1].str.split('\\s{1,}',len(colnames_r)-1,True)
    df_r = pd.DataFrame(data.astype(float))
    df_r.columns = colnames_r
    # Look at absolute difference
    assert np.abs(df_r - df_py).max().max() < 1e-2, 'Expected R and python l1-approx to be similar'


if __name__ == '__main__':
    # Check that we can get similar results to glmnet
    test_py_r()