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
round(c(mdl$a0,mdl$beta[,'s0']),3)

# (iii) Fit L2-only model
mdl = glmnet(x=x_trans,y=y,family='binomial',alpha=0, lambda=1,standardize=FALSE)
round(c(mdl$a0,mdl$beta[,'s0']),3)

# (iii) Fit L1-only model
mdl = glmnet(x=x_trans,y=y,family='binomial',alpha=1, lambda=0,standardize=FALSE)
round(c(mdl$a0,mdl$beta[,'s0']),3)

# Fit elastic net model for specific hyperparameter
mdl = glmnet(x=x_trans,y=y,family='binomial',alpha=0.5, lambda=c(0.1),standardize=FALSE)
abhat = c(as.numeric(mdl$a0),as.vector(mdl$beta))
cn = c('intercept',row.names(mdl$beta))
df = data.frame(cn=cn, val=abhat)
print(df)
# Coefficient values to be parsed in python
paste(as.character(round(df$val,5)),collapse=', ')

glm(y ~ x_trans, family=binomial)$coef
