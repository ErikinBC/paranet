"""
Ensure that rvs(...,censoring=float) works and achieves expected censoring levels
"""

# External modules
import numpy as np

# Internal modules
from paranet.univariate.dists import surv_dist
from paranet.utils import dist_valid

# Experiment with different shape parameters
scale = [0.5, 1, 2]
shape = scale[::-1]
cens_seq = [0.25,0.5,0.75]

def test_cens_precision(tol_cens:float=5e-3, n_sim:int=10000000, seed:int=1):
    for dist in dist_valid:
        mdl_dist = surv_dist(dist, scale, shape)
        for cens in cens_seq:
            print(f'Running censoring sim for {dist} target of {cens}')
            _, d = mdl_dist.rvs(n_sim=n_sim, censoring=cens, seed=seed)
            hat_cens = 1 - d.mean(0)
            print(hat_cens.round(4))
            err_mx = np.abs(hat_cens - cens).max()
            assert err_mx < tol_cens, f'Absolute difference between expected and actual censoring is larger than {tol_cens}: {err_mx}'


def cens_plot(n_points:int=500, cens:float=0.5):
        import pandas as pd
        import plotnine as pn
        from paranet.utils import gg_save

        # Make an example plot of censored vs non-censored    
        dist_wei_1 = surv_dist('weibull', 1, 1)
        t_1_0, d_1_0 = dist_wei_1.rvs(n_points, 0, seed)
        t_1_1, d_1_1 = dist_wei_1.rvs(n_points, cens, seed)
        dist_wei_2 = surv_dist('weibull', 2, 1)
        t_2_0, d_2_0 = dist_wei_2.rvs(n_points, 0, seed)
        t_2_1, d_2_1 = dist_wei_2.rvs(n_points, cens, seed)
        
        df_1_0 = pd.DataFrame({'t':t_1_0.flat,'d':d_1_0.flat, 'lam':1, 'cens':False})
        df_1_1 = pd.DataFrame({'t':t_1_1.flat,'d':d_1_1.flat, 'lam':1, 'cens':True})
        df_2_0 = pd.DataFrame({'t':t_2_0.flat,'d':d_2_0.flat, 'lam':2, 'cens':False})
        df_2_1 = pd.DataFrame({'t':t_2_1.flat,'d':d_2_1.flat, 'lam':2, 'cens':True})
        df = pd.concat(objs=[df_1_0, df_1_1, df_2_0, df_2_1]).reset_index(drop=True)
        df['d'] = df['d'].astype(int).astype(str)
        df_mu = df.groupby(['d','lam','cens'])['t'].mean().reset_index().rename(columns={'t':'mu'}).assign(y=int(n_points*0.1))
        
        gg_dist = (pn.ggplot(df, pn.aes(x='t',fill='d')) + 
            pn.theme_bw() + pn.labs(y='Density',x='Time-to-event') + 
            pn.facet_grid('cens~lam',labeller=pn.label_both) + 
            pn.geom_histogram(alpha=0.5,position='identity',bins=30) + 
            pn.scale_x_log10() + 
            pn.geom_vline(pn.aes(xintercept='mu',color='d'),data=df_mu) + 
            pn.geom_text(pn.aes(x='mu',y='y',color='d',label='mu'),format_string='{:,.2f}',data=df_mu) + 
            pn.scale_fill_discrete(name='Censoring') + 
            pn.scale_color_discrete(name='Censoring'))
        gg_save('tdist_cens.png','tests',gg_dist,6,4)



if __name__ == "__main__":
    # (i) Test simulation of censoring precision
    tol_cens = 0.005
    n_sim = 10000000
    seed = 1
    test_cens_precision(tol_cens, n_sim, seed)

    # (ii) Make example plot
    cens_plot(n_points=500, cens=0.5)

    print('~~~ test_censoring completed without errors ~~~')    

