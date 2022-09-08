# R version of scripts to check that l1 approximators obtain expected results

library(magrittr)
library(MASS)
library(glmnet)

# (i) Load and normalize data
MASS::Boston %>% head
z = MASS::Boston$medv
y = ifelse(z > median(z), 1, 0)
x = as.matrix(MASS::Boston[,-14])
x_trans = scale(x)
sprintf('Sum y=%.1f, sum x_trans^2=%.1f',sum(y),sum(x_trans**2))

# (ii) Fit unregularized model
mdl = glmnet(x=x_trans,y=y,family='binomial',alpha=1, lambda=0,standardize=FALSE)
bhat_unreg = c(mdl$a0,mdl$beta[,'s0'])

# (iii) Fit L2-only model
mdl = glmnet(x=x_trans,y=y,family='binomial',alpha=0, lambda=1,standardize=FALSE)
bhat_l2 = c(mdl$a0,mdl$beta[,'s0'])

# (iii) Fit L1-only model
mdl = glmnet(x=x_trans,y=y,family='binomial',alpha=1, lambda=0.1,standardize=FALSE)
bhat_l1 = c(mdl$a0,mdl$beta[,'s0'])

# (iv) Fit the elastic net model
mdl = glmnet(x=x_trans,y=y,family='binomial',alpha=0.5, lambda=0.25,standardize=FALSE)
bhat_elnet = c(mdl$a0,mdl$beta[,'s0'])

df = data.frame(unreg=bhat_unreg, l2=bhat_l2, l1=bhat_l1, elnet=bhat_elnet)
df
