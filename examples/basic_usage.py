"""
Show how to fit a regularized model on a toy dataset
"""

# Load modules
import numpy as np
import pandas as pd
import plotnine as pn
from scipy import stats
from paranet.models import parametric

# (i) Create a toy dataset
n, p, seed = 100, 5, 2
x = stats.norm().rvs([n,p],seed)
shape = 2
b0 = 0.25
beta = stats.norm(scale=0.5).rvs([p,1],seed)
eta = x.dot(beta).flatten() + b0
scale = np.exp(eta)
t = (-np.log(stats.uniform().rvs(n,seed))/scale)**(1/shape)
d = np.ones(n)

# (ii) Fit the (unregularized) model
mdl = parametric(dist=['exponential', 'weibull', 'gompertz'], x=x, t=t, d=d, scale_t=False, scale_t=False)
mdl.fit()

# (iii) Plot the individual survival, hazard, and density functions for five "new" observations
n_points = 100
n_new = 4
t_range = np.exp(np.linspace(np.log(0.25), np.log(t.max()), n_points))
x_new = stats.norm().rvs([n_new,p],seed)

# We can at look at the hazard for first out-of-sample individual
# Notice that for the exponential distribution (first column) the hazard is independent of time which is as expected
print(mdl.hazard(t_range, x_new[[0]]))

# We can then comprehensively calculate this for each method
methods = ['hazard', 'survival', 'pdf']
holder = []
for j in range(n_new):
    x_j = np.tile(x_new[[j]],[n_points,1])
    for method in methods:
        res_j = getattr(mdl, method)(t_range, x_j)
        res_j = pd.DataFrame(res_j, columns = mdl.dist).assign(time=t_range,method=method, sample=j+1)
        holder.append(res_j)

# Plot the results
res = pd.concat(holder).melt(['sample','time','method'],None,'dist')

gg_res = (pn.ggplot(res, pn.aes(x='time', y='value', color='dist')) + 
    pn.theme_bw() + pn.geom_line() + 
    pn.scale_color_discrete(name='Distribution') + 
    pn.facet_grid('method~sample',scales='free',labeller=pn.labeller(sample=pn.label_both)))
import os
gg_res.save(os.path.join(os.getcwd(),'examples','basic_usage.png'), width=7, height=5)
