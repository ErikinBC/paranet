"""
Demonstrating how to do hyperparameter tuning to maximize out-of-sample concordance prediction
"""

# Load modules
import numpy as np
import pandas as pd
from SurvSet.data import SurvLoader
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sksurv.metrics import concordance_index_censored as concordance
from paranet.models import parametric
from paranet.utils import dist_valid


# (i) Load the colon dataset
# https://stat.ethz.ch/R-manual/R-devel/library/survival/html/colon.html
loader = SurvLoader()
df, ref = loader.load_dataset('colon').values()
t_raw, d_raw = df['time'].values, df['event'].values
df.drop(columns=['pid','time','event'], inplace=True)

# (ii) Perpeare the encoding class
enc_fac = Pipeline(steps=[('ohe', OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore'))])
sel_fac = make_column_selector(pattern='^fac\\_')
enc_num = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])
sel_num = make_column_selector(pattern='^num\\_')
enc_df = ColumnTransformer(transformers=[('ohe', enc_fac, sel_fac),('num', enc_num, sel_num)])

# (iii) Split into a training and test set
frac_test, seed = 0.2, 1
t_train, t_test, d_train, d_test, x_train, x_test = train_test_split(t_raw, d_raw, df, test_size=frac_test, random_state=seed, stratify=d_raw)
rho_seq = np.arange(0.2, 1.01, 0.2).round(2)
n_gamma = 50
gamma_frac = 0

# (iv) Make "out of fold" predictions
n_folds = 10
skf_train = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
holder_paranet, holder_cox = [], []
for i, (train_idx, val_idx) in enumerate(skf_train.split(t_train, d_train)):
    print(f'--- Fitting models on fold {i+1} of {n_folds} ---')
    # Split Training data into "fit" and "val" splits
    t_fit, d_fit, x_fit = t_train[train_idx], d_train[train_idx], x_train.iloc[train_idx]
    t_val, d_val, x_val = t_train[val_idx], d_train[val_idx], x_train.iloc[val_idx]
    # Learn data encoding and one-encoded matrices
    enc_fold = enc_df.fit(x_fit)
    mat_fit, mat_val = enc_fold.transform(x_fit), enc_fold.transform(x_val)
    t_hazard_val = np.repeat(np.median(t_fit), mat_val.shape[0])    

    # Fit model
    mdl_para = parametric(dist_valid, mat_fit, t_fit, d_fit, scale_t=True, scale_x=True)
    # Fit along the solution path
    holder_fold = []
    for rho in rho_seq:
        print(rho)
        gamma_mat, thresh = mdl_para.find_lambda_max(mat_fit, t_fit, d_fit, rho=rho)
        p = gamma_mat.shape[0]
        gamma_max = gamma_mat.max(0)
        gamma_mat = np.exp(np.linspace(np.log(gamma_max), np.log(gamma_frac*gamma_max), n_gamma-1))
        gamma_mat = np.vstack([gamma_mat, np.zeros(gamma_mat.shape[1])])
        for j in range(n_gamma):
            gamma_j = np.tile(gamma_mat[[j]], [p,1])
            init_j = np.vstack([mdl_para.alpha, mdl_para.beta])
            mdl_para.fit(mat_fit, t_fit, d_fit, gamma_j, rho, thresh, alpha_beta_init=init_j, grad_tol=0.05)
            # Make out of sample predictions
            res_rho_j = pd.DataFrame(mdl_para.hazard(t_hazard_val,mat_val), columns=dist_valid)
            res_rho_j = res_rho_j.assign(d=d_val, t=t_val, j=j, rho=rho)
            holder_fold.append(res_rho_j)
    # Merge and calculate concordance
    res_rho = pd.concat(holder_fold).melt(['rho','j','t','d'],dist_valid,'dist','hazard')
    res_rho['d'] = res_rho['d'] == 1
    res_rho = res_rho.groupby(['dist','rho','j']).apply(lambda x: concordance(x['d'], x['t'], x['hazard'])[0]).reset_index()
    res_rho = res_rho.rename(columns={0:'conc'}).assign(fold=i)
    holder_paranet.append(res_rho)
# Find the best shrinkage combination
res_cv = pd.concat(holder_paranet).reset_index(drop=True)
res_cv = res_cv.groupby(['dist','rho','j'])['conc'].mean().reset_index()
param_cv = res_cv.query('conc == conc.max()').sort_values('rho',ascending=False).head(1)
dist_star, rho_star, j_star = param_cv[['dist','rho','j']].values.flat
j_star = int(j_star)
idx_dist_star = np.where([dist_star in dist for dist in dist_valid])[0][0]

# (v) Fit model with the "optimal" parameter
enc_train = enc_df.fit(x_train)    
mat_train, mat_test = enc_df.transform(x_train), enc_df.transform(x_test)
mdl_para = parametric(dist_valid, mat_train, t_train, d_train, scale_t=True, scale_x=True)
# Find the optimal gamma
gamma, thresh = mdl_para.find_lambda_max()
gamma_max = gamma.max(0)
gamma_star = np.exp(np.linspace(np.log(gamma_max), np.log(gamma_frac*gamma_max), n_gamma-1))
gamma_star = np.vstack([gamma_star, np.zeros(gamma_star.shape[1])])
gamma_star = gamma_star[j_star,idx_dist_star]
# Re-fit model
mdl_para.fit(gamma=gamma_star, rho=rho_star, beta_thresh=thresh)

# (vii) Make predictions on test set
t_haz_test = np.repeat(np.median(t_train), len(t_test))
hazhat_para = mdl_para.hazard(t_haz_test, mat_test)[:,idx_dist_star]
conc_para = concordance(d_test==1, t_test, hazhat_para)[0]
print(f'Out of smaple conconcordance: {conc_para*100:.1f}%')
