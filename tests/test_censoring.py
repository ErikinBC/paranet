"""
Ensure that rvs(...,censoring=float) works and achieves expected censoring levels
"""

import numpy as np
import pandas as pd
import plotnine as pn

from paranet.dists import surv_dist
from paranet.utils import dist_valid, gg_save

n_sim = 1000
n_censor_sim = 150
seed = 1
censoring = 0.25
lam = [1, 1, 1]
alph = [0.5, 1, 2]

t_1_0, d_1_0 = surv_dist('weibull', 1, 0.5).rvs(n_sim=n_sim, censoring=0, n_censor_sim=n_censor_sim, seed=1)
t_1_1, d_1_1 = surv_dist('weibull', 1, 0.5).rvs(n_sim=n_sim, censoring=0.5, n_censor_sim=n_censor_sim, seed=1)
t_2_0, d_2_0 = surv_dist('weibull', 2, 0.5).rvs(n_sim=n_sim, censoring=0, n_censor_sim=n_censor_sim, seed=1)
t_2_1, d_2_1 = surv_dist('weibull', 2, 0.5).rvs(n_sim=n_sim, censoring=0.5, n_censor_sim=n_censor_sim, seed=1)

t_1_0.mean();t_1_1.mean()
t_2_0.mean();t_2_1.mean()

t_1_0.max();t_1_1.max()
t_2_0.max();t_2_1.max()

df_1_0 = pd.DataFrame({'t':t_1_0.flat,'d':d_1_0.flat, 'lam':1, 'cens':False})
df_1_1 = pd.DataFrame({'t':t_1_1.flat,'d':d_1_1.flat, 'lam':1, 'cens':True})
df_2_0 = pd.DataFrame({'t':t_2_0.flat,'d':d_2_0.flat, 'lam':2, 'cens':False})
df_2_1 = pd.DataFrame({'t':t_2_1.flat,'d':d_2_1.flat, 'lam':2, 'cens':True})

df = pd.concat(objs=[df_1_0, df_1_1, df_2_0, df_2_1]).reset_index(drop=True)
df['d'] = df['d'].astype(int).astype(str)
df_mu = df.groupby(['d','lam','cens'])['t'].mean().reset_index().rename(columns={'t':'mu'}).assign(y=int(n_sim*0.1))

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

for dist in dist_valid[:1]:
    gen_dist = surv_dist(dist, scale=lam, shape=alph)
    t_dist, d_cens = gen_dist.rvs(n_sim=10000, censoring=0.25, seed=None, n_censor_sim=10000)
    d_cens.mean(0)
#     t_dist.mean(0)
#     (1-d_cens.mean(axis=0)) - censoring