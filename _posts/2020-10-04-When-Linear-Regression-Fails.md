---
layout: post
toc: true
title:  "When Should We NOT Run Linear Regression?"
categories: Statistics
tags: [python, statistics]
author: K.Asaba
description: This article describes 1. omitted variable bias, 2. bad control, 3. attenuation bias.
---
## TR;DR
Linear regression gives you incorrect results when there exists
1. Omitted variables bias
2. Bad control
3. Attenuation Bias
This article explains them with Python examples.

(There are many other cases that gives you wrong regression results. This article only gives you three of them)

## Omitted Variable Biases

Assume that you want to estimate **how much additional one year of education increases one's salary on average**.   
Your data has two variables: "years of education" and "salary". You decided to regress them as:

<div align="center">
$$\text{salary}_i = \alpha + \beta \times \text{educyear}_i + \varepsilon_i$$
</div>

Although you only have "educyear" in your variable, you realized that there exists an **omitted variable** "IQ" which affects obth "years of education" and "salary".  
While you don't have "IQ" data, it seems people with high IQ tend to receive more education and higher salary.

![OVB]({{ site.baseurl }}//assets/images/liner_reg/omitted_variable_bias.png)

In this case, running a regression with just "educyear" will over/under-estimate educyear's effect on salary.

Let's see a Python example with a simulated data.

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# generate data
n = 300
x_iq = 100 + np.random.normal(0, 15, size=n)  # IQ is centered at 100 and have sd=15
x_educyear = np.random.randint(6, 10, n) + np.round(x_iq*0.1)  # Assume higher IQ leads to higher educyear
y_salary = 150 + 0.7*x_iq + 1.1*x_educyear + np.random.normal(0, 10, size=n)

# plot data
fig = go.Figure(data=[go.Scatter3d(x=x_iq, y=x_educyear, z=y_salary,
                                   mode='markers')])
fig.update_traces(marker_size=1)
fig.update_layout(scene = dict(xaxis_title='IQ',
                               yaxis_title='EducYear',
                               zaxis_title='Salary'),
                  margin=dict(l=0, r=0, t=0, b=0),)
fig.show()
```

<iframe width="400" height="300" frameborder="0" scrolling="no" src="//plotly.com/~kasabaimo/21.embed"></iframe>

Now estimate the coefficients.

```python
X_modelA = np.column_stack((np.repeat(1, n), x_educyear))  # add y-intercept

ols = sm.OLS(y_salary, X_modelA)
ols_result = ols.fit()
print(ols_result.summary())
```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.462
Model:                            OLS   Adj. R-squared:                  0.460
Method:                 Least Squares   F-statistic:                     255.7
Date:                Tue, 05 Oct 2021   Prob (F-statistic):           5.52e-42
Time:                        12:18:32   Log-Likelihood:                -1180.1
No. Observations:                 300   AIC:                             2364.
Df Residuals:                     298   BIC:                             2372.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        131.7647      6.902     19.090      0.000     118.181     145.348
x1             6.2147      0.389     15.990      0.000       5.450       6.980
==============================================================================
Omnibus:                        1.788   Durbin-Watson:                   1.784
Prob(Omnibus):                  0.409   Jarque-Bera (JB):                1.559
Skew:                          -0.081   Prob(JB):                        0.459
Kurtosis:                       3.314   Cond. No.                         172.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

The OLS regression result shows coefficient of 6.2147, although the true coefficient is 1.1.

Just in case, let's test if we can estimate coefficients correctly when two variables are included.

```python
X_modelB = np.column_stack((np.repeat(1, n), x_educyear, x_iq))
ols = sm.OLS(y_salary, X_modelB)
ols_result = ols.fit()
print(ols_result.summary())
```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.678
Model:                            OLS   Adj. R-squared:                  0.676
Method:                 Least Squares   F-statistic:                     313.1
Date:                Tue, 05 Oct 2021   Prob (F-statistic):           7.17e-74
Time:                        13:12:51   Log-Likelihood:                -1102.9
No. Observations:                 300   AIC:                             2212.
Df Residuals:                     297   BIC:                             2223.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        144.9187      5.426     26.710      0.000     134.241     155.596
x1             0.9455      0.479      1.974      0.049       0.003       1.888
x2             0.7926      0.056     14.138      0.000       0.682       0.903
==============================================================================
Omnibus:                        0.131   Durbin-Watson:                   1.939
Prob(Omnibus):                  0.936   Jarque-Bera (JB):                0.027
Skew:                           0.007   Prob(JB):                        0.987
Kurtosis:                       3.044   Cond. No.                     1.02e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.02e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
```

For educyear, estimated=0.9455 vs true=1.1.  
For iq, estimated=0.7926 vs true=0.7.  
We can say that estimation is successful in this case.

To sum up, we need to always make sure we are not missing variables that affects both X and Y.

[Note]
When we realized that there are omitted variable biases but we can't collect missing variable, use *Instrument Variable* to obtain true coefficients.


## Bad Control

In **Omitted Variable Biases**, we got a wrong result by NOT having the important variable.  
In **Bad Control**, we get a wrong result by **including unnecessary variables**.

Assume again that you want to estimate the effect of "years of education" on "salary".  
You decided to regress "salary" with "years of education" and "number of pencils used in his/her life".  
Here, "number of pencils used in his/her life" is affected by "years of education" but has nothing to do with "salary", as shown in the below graph. 

![BadControl]({{ site.baseurl }}//assets/images/liner_reg/bad_control.png)

In this case, we should not include the irrelevant variable "number of pencils used" in the regression. What if we included in the regression model? Let's see what will happen with Python example.

```python
# generate data
n = 300
x_educyear = np.random.randint(6, 15, n)
y_salary = 150 + 8*x_educyear + np.random.normal(0, 10, size=n)  # x_book isn't there
x_book = 10*x_educyear + y_salary*0.6 + 10*np.random.randn(n)

# visualization
fig_2 = go.Figure(data=[go.Scatter3d(x=x_educyear, y=x_book, z=y_salary,
                                   mode='markers')])
fig_2.update_traces(marker_size=1)
fig_2.update_layout(scene = dict(xaxis_title='Educyear',
                                 yaxis_title='Books',
                                 zaxis_title='Salary'),
                    margin=dict(l=0, r=0, t=0, b=0),)
fig_2.show()
```

<iframe width="400" height="300" frameborder="0" scrolling="no" src="//plotly.com/~kasabaimo/23.embed"></iframe>

```python
# OLS with full data
X_badcontrol = np.column_stack((np.repeat(1, n), x_educyear, x_book))  # add y-intercept
ols_bc = sm.OLS(y_salary, X_badcontrol)
ols_bc_result = ols_bc.fit()
print(ols_bc_result.summary())
```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.864
Model:                            OLS   Adj. R-squared:                  0.863
Method:                 Least Squares   F-statistic:                     940.7
Date:                Tue, 05 Oct 2021   Prob (F-statistic):          3.08e-129
Time:                        16:59:08   Log-Likelihood:                -1070.8
No. Observations:                 300   AIC:                             2148.
Df Residuals:                     297   BIC:                             2159.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        101.1492      4.557     22.194      0.000      92.180     110.118
x1             0.4052      0.663      0.611      0.541      -0.899       1.709
x2             0.5220      0.044     11.982      0.000       0.436       0.608
==============================================================================
Omnibus:                        0.776   Durbin-Watson:                   1.932
Prob(Omnibus):                  0.678   Jarque-Bera (JB):                0.526
Skew:                           0.054   Prob(JB):                        0.769
Kurtosis:                       3.175   Cond. No.                     2.25e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.25e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
```

Although the true coefficient for x_educyear is 8, the OLS shows the coefficient of 0.4052.

Lesson: do not always include everything you have on your data.

You can check what are bad and good controls with [this paper](https://ftp.cs.ucla.edu/pub/stat_ser/r493.pdf)

## References
- [Wikipedia](https://en.wikipedia.org/wiki/Smoothing_spline#cite_note-EilersMarx1996-13)
- [Spline smoothing with model-based penalties](https://link.springer.com/article/10.3758/BF03200573)
- [Smoothing with penalized splines](https://csm.lshtm.ac.uk/wp-content/uploads/sites/6/2016/04/Antonio-Gasparrini-29-05-2015.pdf)


