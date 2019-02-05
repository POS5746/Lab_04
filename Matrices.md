Linear Algebra in R
-------------------

We will often find it useful to work with matrices in R, especially when
coding our own functions for analysis. Luckily, R makes it extremely
easy to work with matrices and we will estimate a simple OLS model today
using matrix algebra.

We first load in a dataset of interest. For this class we will be using
the birth weight dataset referenced in the Wooldridge textbook. There is
a package in R, `wooldridge` that allows access to all the datasets
referenced throughout the textbook.

    library(wooldridge)

    data("bwght")

When looking over our dataset, we notice that there are missing values
for some of the variables. Let's proceed to delete any such
observations. We create a new dataset called `df`.

    df <- na.omit(bwght)

We are interested in exploring the effects of different individual,
family, and county level variables on a child's weight at birth.
Specifically, we are interested in determining the effect that a
mother's cigarette smoking while pregnant has on her child. To estimate
this model, we also include relevant controls such as family income (in
logs), the education level of both parents, gender of the newborn, race
of newborn, and the tax rate on cigarette sales (if any). Our variable
of interest, `cigs` measures the number of cigarettes the mother smoked
per day while pregnant. Let's estimate the model, and generate fitted
values:

    fit0 <- lm(lbwght ~ lfaminc + fatheduc + motheduc + cigs + cigtax + male + white, data = df)
    yhat0 <- predict(fit0)
    res0 <- residuals(fit0)
    summary(fit0)

    ## 
    ## Call:
    ## lm(formula = lbwght ~ lfaminc + fatheduc + motheduc + cigs + 
    ##     cigtax + male + white, data = df)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -1.61350 -0.08261  0.01850  0.11708  0.83109 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  4.6793133  0.0373280 125.357  < 2e-16 ***
    ## lfaminc      0.0108444  0.0085664   1.266  0.20579    
    ## fatheduc     0.0033457  0.0026192   1.277  0.20171    
    ## motheduc    -0.0036716  0.0029724  -1.235  0.21699    
    ## cigs        -0.0051856  0.0010301  -5.034 5.55e-07 ***
    ## cigtax       0.0006816  0.0006876   0.991  0.32181    
    ## male         0.0330145  0.0107383   3.074  0.00216 ** 
    ## white        0.0419820  0.0152131   2.760  0.00588 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1844 on 1183 degrees of freedom
    ## Multiple R-squared:  0.04378,    Adjusted R-squared:  0.03812 
    ## F-statistic: 7.737 on 7 and 1183 DF,  p-value: 3.596e-09

Now that we have these results, we can estimate our model using linear
algebra. The least squares solution for our slope coefficients, in
matrix form, is *b* = (*X*′*X*)<sup>−1</sup>*X*′*Y*. The general form of
our regression model using matrix algebra is

$$\\hat{Y} = Xb = X(X'X)^{-1}X'Y$$

and our residuals are calculated as $e=Y-\\hat{Y}=(I-H)Y$ where
*H* = *X*(*X*′*X*)<sup>−1</sup>*X*′.

To estimate this equation in R, we first build our matrices.

    Y <- as.matrix(df$lbwght)
    X <- as.matrix(cbind(1, df$lfaminc, df$fatheduc, df$motheduc, df$cigs, df$cigtax, df$male, df$white))

Notice that we have to include a column of ones which will allow us to
estimate our intercept.

We can now solve our model:

    betas <- solve(t(X) %*% X) %*% t(X) %*% Y
    yhat1 <- X %*% betas
    res1 <- Y - yhat1

If you compare the coefficient vector *b* with the results from the OLS
regression, you will see that they are identical. Similarly, fitted
values and residual values are identical across both estimation
procedures.

    fitted <- as.matrix(cbind(yhat0, yhat1))

    coefficients <- as.matrix(cbind(fit0$coefficients, betas))

    resid <- as.matrix(cbind(res0, res1))

    head(fitted)

    ##      yhat0         
    ## 1 4.789870 4.789870
    ## 2 4.721439 4.721439
    ## 4 4.749386 4.749386
    ## 5 4.804277 4.804277
    ## 6 4.734170 4.734170
    ## 7 4.779939 4.779939

    head(coefficients)

    ##                      [,1]          [,2]
    ## (Intercept)  4.6793132906  4.6793132906
    ## lfaminc      0.0108444324  0.0108444324
    ## fatheduc     0.0033457133  0.0033457133
    ## motheduc    -0.0036715878 -0.0036715878
    ## cigs        -0.0051856056 -0.0051856056
    ## cigtax       0.0006815552  0.0006815552

    head(resid)

    ##          res0            
    ## 1 -0.09852166 -0.09852166
    ## 2  0.16890971  0.16890971
    ## 4  0.08689592  0.08689592
    ## 5  0.09356309  0.09356309
    ## 6  0.03651441  0.03651441
    ## 7  0.16170325  0.16170325
