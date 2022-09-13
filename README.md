# paranet: Parametric survival models with elastic net regularization

The `paranet` package allows for the fitting of elastic net regularized parametric survival models with right-censored time-to-event data in `python`. There are three parametric distributions which currently supported:

1. Exponential
2. Weibull
3. Gompertz

These distributions were chosen due to their common usage in practice and for their computational simplicity since sampling and quantile calculations can be done trivially with the inverse method. 

Parametric model support currently exists with the [`lifelines`](https://lifelines.readthedocs.io) and [`pysurvival`](https://square.github.io/pysurvival) packages, but these packages do not support regularization for these model classes. Elastic net survival models can be fit with the [`scikit-survival`](https://scikit-survival.readthedocs.io/en/stable/) package, but this is only for the Cox-PH model. While the Cox model is a very important tool for survival modelling, its key limitation for large-scale datasets is that i) it is not natively able to do inference on individual survival times, and ii) its loss function is non-parametric in that its run-time grows O(n^2) rather than O(n).

The `paranet` package allows users to fit a high-dimensional linear model on right-censored data and then provide individualized or group predictions on time-to-event outcomes. For example, fitting a parametric model on customer churn data can allow a data science to answer interesting questions such as: "out of these 100 customers, when do we first expect that 10% of them will have churned?", or "for this new customer, at what time point are they at highest risk of leaving us (i.e. maximum hazard)?", or "for an existing customer, what is the probability they will have churned in 10 months from now?".

<br>

## (0) Installation

The `paranet` package is available on PyPi and can be installed with `pip install paranet=0.1.4`. **NOTE** this package has been tested with python 3.9+. Using earlier versions of python may lead to errors.

<br>


## (1) Basic syntax

The `parametric` class is the workhouse model of this package. When initializing the model users will always need to the specify the `dist` argument. This can be a list or a string of valid distribution types. There are no limits on how long this list can be, but if it is of length `k`, then subsequent time measurements will either need to a vector or a matrix with `k` columns. 

Although each input argument is defined in the docstrings, several parameters will recur frequently throughout and are defined here for convenience.

1. `x`: An `(n,p)` array of covariates. Users can either manually add an intercept and scale the data, or set `add_int=True` and `scale_x=True`.
2. `t`: An `(n,k)` array of time measurements that should be non-zero. If $k\geq 0$ then the model assumes each column corresponds to a (potentially) different distribution.
3. `d`: An `(n,k)` array of censoring indicators whose values should be either zero or one. As is the usual convention, 0 corresponds to a censored observation and 1 to an un-censored one.
4. `gamma`: The strength of the regularization (see section (2) for a further description). If this variable is a vector or a matrix, it must correspond to the number of columns of `x` (including the intercept if it is given).
5. `rho`: The relative L1/L2 strength, where 0 corresponds to L2-only (i.e. Ridge regression) and 1 corresponds to L1-only (i.e. Lasso). 

As a rule  `paranet` will try to broadcast where possible. For example, if the time measurement array is `t.shape==(n,k)`, and `dist=='weibull'` then it will assume that each column of `t` is a Weibull distribution. In contrast, if `t.shape==(n,)` and `dist=['weibull','gompertz']`, it broadcast copies of `t` for each distribution.


The `parametric` class has X keys methods. If `x`, `t`, or `d` are initialized then arguments can be left empty.

1. `fit(x, t, d, gamma, rho)`: Will fit the elastic net model for a given `gamma`/`rho` penalty and enable methods like `hazard` to be executed.
2. `find_lambda_max(x, t, d, gamma)`: Uses the KKT conditions of the sub-gradient to the determine the largest `gamma` that is needed to zero-out all covariates except the scale and shape parameters.
3. `{hazard,survival,pdf}(t, x)`: Similar to `predict` is `sklearn`, these methods provide individualized estimates of the hazard, survival, and density functions. 
4. `quantile(percentile, x)`: Provides the quantile of the individualized survival distribution.
5. `rvs(n_sim, censoring)`: Generates a certain number of samples for a censoring target.


When initializing the `parametric` class, users can include the data matrices which will be saved for later for methods that require them. However, specifying these arguments in later methods will always override (but not replace) these inherited attributes.

1. `dist`: Required argument that is a string or a list whose elements must be one of: exponential, weibull, or gompertz.
2. `alpha`: The shape parameter can be manually defined in advance (needs to match the dimensionality of `dist`).
3. `beta`: The scale parameter can be defined in advance (needs to match the dimensionality of `dist` and `x`).
4. `scale_x`: Will standardize the covariates to have a mean of zero and a variance of one. This is highly recommended when using any regularization. If this argument is set to True, always provide the raw form of the covariates as they will be scaled during inference.
5. `scale_t`: Will normalize the time vector be a maximum of one for model fitting which can help with overflow issues. During inference, the output will always be returned to the original scale. However, the coefficients will change as a result of this.
<br>

## (2) Toy example

The code block below shows how to fit three parametric distributions to a single array of data generated by covariates. For other demonstrations of using `paranet` with a jupyter notebook see the [examples](examples) folder. 

```python
# Load modules
import numpy as np
import pandas as pd
import plotnine as pn
from scipy import stats
from paranet.models import parametric

# (i) Create a toy dataset
n, p, seed = 100, 5, 3
x = stats.norm().rvs([n,p],seed)
shape = 2
b0 = 0.25
beta = stats.norm(scale=0.5).rvs([p,1],seed)
eta = x.dot(beta).flatten() + b0
scale = np.exp(eta)
t = (-np.log(stats.uniform().rvs(n,seed))/scale)**(1/shape)
d = np.ones(n)

# (ii) Fit the (unregularized) model
mdl = parametric(dist=['exponential', 'weibull', 'gompertz'], x=x, t=t, d=d, scale_x=False, scale_t=False)
mdl.fit()

# (iii) Plot the individual survival, hazard, and density functions for five "new" observations
n_points = 100
n_new = 4
t_range = np.exp(np.linspace(np.log(0.25), np.log(t.max()), n_points))
x_new = stats.norm().rvs([n_new,p],seed)

# We can at look at the hazard for first out-of-sample individual
# Notice that for the exponential distribution (first column) the hazard is independent of time which is as expected
print(np.log(mdl.hazard(t_range, np.tile(x_new[[0]],[n_points,1]))).round(2))

# We can then comprehensively calculate this for each method
methods = ['hazard', 'survival', 'pdf']
holder = []
for j in range(n_new):
    x_j = np.tile(x_new[[j]],[n_points,1])
    for method in methods:
        res_j = getattr(mdl, method)(t_range, x_j)
        if method == 'hazard':
            res_j = np.log(res_j)
        res_j = pd.DataFrame(res_j, columns = mdl.dist).assign(time=t_range,method=method, sample=j+1)
        holder.append(res_j)

# Plot the results
res = pd.concat(holder).melt(['sample','time','method'],None,'dist')

gg_res = (pn.ggplot(res, pn.aes(x='time', y='value', color='dist')) + 
    pn.theme_bw() + pn.geom_line() + 
    pn.scale_color_discrete(name='Distribution') + 
    pn.facet_grid('method~sample',scales='free',labeller=pn.labeller(sample=pn.label_both)))
```

![toy_output](examples/basic_usage.png)

## (3) Probability distribution parameterization

Each parametric survival distribution is defined by a scale $\lambda$ and, except for the Exponential distribution, a shape $\alpha$ parameter. Each distribution has been parameterized so that a higher value of the scale parameter indicates a higher "risk". The density functions are shown below. The scale and shape parameters must also be positive, except for the case of the Gompertz distribution where the shape parameter can be positive or negative.

$$
\begin{align*}
    f(t;\lambda, \alpha) &= \begin{cases}
        \lambda \exp\{ -\lambda t \}  & \text{ if Exponential} \\
        \alpha \lambda  t^{\alpha-1} \exp\{ -\lambda t^{\alpha} \}  & \text{ if Weibull} \\
        \lambda \exp\{ \alpha t \} \exp\{ -\frac{\lambda}{\alpha}(e^{\alpha t} - 1) \}  & \text{ if Gompertz} \\
    \end{cases}
\end{align*}
$$

When moving from the univariate to the multivariate distribution, we assume that scale parameter takes is an exponential transform (to ensure positivity) of a linear combination of parameters: $\eta$. Optimization occurs by balancing the data likelihood with the magnitude of the coefficients, $R$: 

$$
\begin{align*}
    \lambda_i &= \exp\Big( \beta_0 + \sum_{j=1}^p x_{ij}\beta_j \Big) \\
    R(\beta;\gamma,\rho) &= \gamma\big(\rho \| \beta_{1:} \|_1 + 0.5(1-\rho)\|\beta_{1:}\|_2^2\big) \\
    \ell(\alpha,\beta,\gamma,\rho) &= \begin{cases}
        -n^{-1}\sum_{i=1}^n\delta_i\log\lambda_i - \lambda_i t_i + R(\beta;\gamma,\rho)  & \text{ if Exponential} \\
        -n^{-1}\sum_{i=1}^n\delta_i[\log(\alpha\lambda_i)+(\alpha-1)\log t_i] - \lambda t_i^\alpha + R(\beta;\gamma,\rho)  & \text{ if Weibull} \\
        -n^{-1}\sum_{i=1}^n\delta_i[\log\lambda + \alpha t] - \frac{\lambda}{\alpha}(\exp\{\alpha t_i \} -1) + R(\beta;\gamma,\rho)  & \text{ if Gompertz} \\
    \end{cases}
\end{align*}
$$


<br>

## (4) How is censoring calculated?

When calling the `parametric.rvs` method, the user can specify the censoring value. In `paranet`, censoring is generated by an exponential distribution taking on a value that is smaller than the actual value. Formally:

$$
\begin{align*}
	T^{\text{obs}} &= \begin{cases}
		T^{\text{act}} & \text{ if } T^{\text{act}} < C \\
		C & \text{ otherwise}
	\end{cases} \\
	C &\sim \text{Exp}(\lambda_C) \\
\end{align*}
$$


There are of course other processes that could generate censoring (such as type-I censoring where all observations are censored at a pre-specified point). The reason an exponential distribution is used in the censoring process is to allow for a (relatively) simple optimization problem of finding a single scale parameter, $\lambda_C$, which obtains an (asymptotic) censoring probability of $\phi$: 

$$
\begin{align*}
	\phi(\lambda_C) &= P(C \leq T_i) = \int_0^\infty f_T(u) F_C(u) du, \\
	\lambda_C^* &= \arg\min_\lambda \| \phi(\lambda) - \phi^* \|_2^2,
\end{align*}
$$


Where $F_C(u)$ is the CDF of an exponential distribution with $\lambda_C$ as the scale parameter, and $f_T(u)$ is the density of the target distribution (e.g. a Weibull-pdf). Finding the scale parameter amounts to a root-finding problem that can be carried out with `scipy`. Finding a single scale parameter is more complicated for the multivariate case because an assumption needs to be made about the distribution of $\lambda_i$ itself, which is random. While it is tempting to generate a censoring-specific distribution (i.e. $C_i$) this would break the non-informative censoring assumption since the censoring random variable is now a function of the realized risk scores. The `paranet` package assumes that the covariates come from a standard normal distribution: $x_{ij} \sim N(0,1)$ so that $\eta_i \sim N(0, \|\beta\|^2_2)$, and $\lambda_i \sim \text{Lognormal}(0, \|\beta\|^2_2)$. It is important that the data be at least normalized for this assumption to be credible.

$$
\begin{align*}
    P(C \leq T) &= \int_0^\infty \Bigg( \int_0^\infty P(C \leq T_i) di \Bigg) F_C(u) du \\ 
    &= \int_0^\infty\int_0^\infty F_C(u)f_{i}(u) f_\lambda(i) du di ,
\end{align*}
$$

Where $f_{i}(u)$ is the density of the target distribution evaluated at $u$, whereas $f_\lambda(i)$ is the pdf of a log-normal distribution evaluated at $i$. This is a much more complicated integral to solve and `paranet` currently uses a brute-force approach at integrating over a grid of values rather than using double quadrature as the latter approach was shown to be prohibitively expensive in terms of run-time.

<br>

## (5) How does optimization happen?

Unlike [`glmnet`](https://cran.r-project.org/web/packages/glmnet/index.html), the `paranet` packages does not use coordinate descent (CD). Instead, this packages uses a [smooth approximation](https://www.cs.ubc.ca/~schmidtm/Documents/2007_ECML_L1General.pdf) of the L1-norm to allow for direct optimization with `scipy` as shown below. Parametric survival models are not easily amenable to the iteratively-reweighted least squares (IRLS) approach used by `glmnet`, because of the presence of the shape parameter. In contract, an exponential model can be [easily fit](http://www.erikdrysdale.com/censored_reg_exponential/) leveraging existing CD-based elastic net solvers. Moving to proximal gradient descent would enable for direct optimization of the L1-norm loss and represents a possible future release.

$$
\begin{align*}
    R(\beta;\gamma,\rho) &= \gamma\big(\rho \| \beta_{1:} \|_1 + 0.5(1-\rho)\|\beta_{1:}\|_2^2\big) \\ 
    |\beta| &\approx \sqrt{\beta^2 + \epsilon} \\
    \frac{\partial R}{\partial \beta} &\approx \gamma\Bigg(\rho \frac{\beta}{\sqrt{\beta^2+\epsilon}} + (1-\rho)\beta\Bigg)
\end{align*}
$$

<br>

## (6) Making contributions

If you are interested in making contributions to this package feel free to email me or make a pull request. The main classes and functions of this package have received significant unit testing and to ensure that changes do not break the package, it is recommended running `source run_tests.sh` before making any pull requests. This package was built with a specific conda environment and developers can use `conda env create -f paranet.yml` and then `conda activate paranet`.
