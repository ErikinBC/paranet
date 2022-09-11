"""
Check that we can solve the solution path for all datasets in SurvSet
"""

# External modules
import numpy as np
import pandas as pd
from time import time
from SurvSet.data import SurvLoader
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Internal modules
from paranet.models import parametric
from paranet.utils import dist_valid


def test_survset(rho:float=1, n_gamma:int=10, ratio_min:float=0.01) -> None:
    # (i) Perpeare the encoding class
    enc_fac = Pipeline(steps=[('ohe', OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore'))])
    sel_fac = make_column_selector(pattern='^fac\\_')
    enc_num = Pipeline(steps=[('impute', SimpleImputer(strategy='median'))])
    sel_num = make_column_selector(pattern='^num\\_')
    enc_df = ColumnTransformer(transformers=[('ohe', enc_fac, sel_fac),('num', enc_num, sel_num)])

    # (ii) Look over all datasets
    pct_gamma = np.linspace(1, ratio_min, n_gamma)
    loader = SurvLoader()
    # Use non-time dependent datasets
    lst_ds = loader.df_ds.query('~is_td')['ds'].to_list()
    holder = []
    for i, ds in enumerate(lst_ds):
        print(f'- Fitting model for dataset {ds} ({i+1} of {len(lst_ds)}) -')
        if i < 28:
            continue
        # Load data
        df, _ = loader.load_dataset(ds).values()
        df = df.query('time > 0')
        t, d = df['time'].values, df['event'].values.astype(float)
        df.drop(columns=['pid','event','time'], inplace=True)
        # Transform and create class
        enc_i = enc_df.fit(df)
        X_raw = enc_i.transform(df)
        n, p = X_raw.shape
        print(f'X ~ ({n},{p})')
        mdl = parametric(dist_valid, X_raw, t, d, scale_t=True, scale_x=True, add_int=True)
        # Fit model across a range of hyperparameters
        gamma_mat, thresh = mdl.find_lambda_max(rho=rho)
        gamma_max = gamma_mat.max(0)
        gamma_seq = np.linspace(gamma_max*ratio_min,gamma_max, n_gamma)[::-1]
        pct_sparse, runtime = np.zeros(n_gamma), np.zeros(n_gamma)
        for j, gamma_row in enumerate(gamma_seq):
            print(f'gamma {j+1} of {n_gamma}')
            gamma_j = np.tile(np.atleast_2d(gamma_row),[mdl.beta.shape[0],1])
            prev_alpha_beta = np.vstack([mdl.alpha, mdl.beta])
            stime = time()
            mdl.fit(gamma=gamma_j, rho=rho, beta_thresh=thresh, alpha_beta_init=prev_alpha_beta, maxiter=15000)
            pct_sparse[j] = np.mean(mdl.beta[1:] == 0)
            runtime[j] = time() - stime
        res_i = pd.DataFrame({'ds':ds, 'n':n, 'p':p, 'gamma':pct_gamma, 'sparse':pct_sparse, 'runtime':runtime})
        print(f'Took {runtime.sum():.0f} seconds to run\n')
        holder.append(res_i)

    # (iii) Combine datasets and plot
    res = pd.concat(holder).assign(sparse=lambda x: x['sparse']*100)
    res = res.melt(['ds','n','p','gamma'],None,'msr')
    di_msr = {'sparse':'Sparse (%)', 'runtime':'Runtime (seconds)'}
    import os
    import plotnine as pn
    from mizani.formatters import percent_format
    gg_survset = (pn.ggplot(res, pn.aes(x='gamma',y='value',group='ds')) + 
        pn.theme_bw() + 
        pn.labs(x='(%) of gamma-max',y='Measurement') + 
        pn.facet_wrap('~msr',scales='free_y', labeller=pn.labeller(msr=di_msr)) + 
        pn.scale_x_continuous(labels=percent_format()) + 
        pn.theme(subplots_adjust={'wspace': 0.25}) + 
        pn.geom_line(color='blue',alpha=0.5))
    gg_survset.save(os.path.join(os.getcwd(),'tests','figures','survset.png'),width=8,height=3)


if __name__ == '__main__':
    # Check that parametric can fit to every dataset in SurvSet
    rho = 1
    n_gamma = 10
    ratio_min = 0.01
    test_survset(rho, n_gamma, ratio_min)
