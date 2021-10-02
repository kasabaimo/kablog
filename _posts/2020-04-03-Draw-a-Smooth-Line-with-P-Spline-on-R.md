---
layout: post
title:  "P-Spline Implementation on R"
categories: Data
tags: [R, Statistics]
author: K.Asaba
description: Use Penalized Smoothing Splines to draw a smooth line on two dimensional scatter plot on R
---
# Objective
By using P-Splines (Penalized Smoothing Splines) to draw a smooth line on two dimensional scatter plot on R.


# Code Examples

### Sample code from [Package ‘pspline’](https://cran.r-project.org/web/packages/pspline/pspline.pdf)


```r
data(cars)
attach(cars)
plot(speed, dist, main = "data(cars) & smoothing splines")  # scatter plot the original data
lines(sm.spline(speed, dist, df=10), lty=1, col = "red", )  # draw the P-Spline curve with degree of freedom 10
lines(sm.spline(speed, dist, df=100), lty=1, col = "green")  # draw the P-Spline curve with degree fo freedom 100
legend("topleft", legend = c('df=10', 'df=100'), col = c("red", "green"),  lty=c(1, 1))
```

 ![Xixia]({{ site.baseurl }}//assets/images/pspline/p1.png)

 We confirm that the larger `df` (=degree of freedom) leads to more zigzagged line (bias-variance tradeoff).


### Artificial Data

<div align="center">
$$y = \sin (x) - 1.5\cos (x/2 - 5) + 0.3 \sin (x * 10)$$
</div>

```r
x = (1:50)/3
y = sin(x) - cos(x/2 - 5)*1.5 + 0.3*sin(x*10)
plot(x, y)
lines(sm.spline(x, y, df=5), lty=1, col = "red")
lines(sm.spline(x, y, df=10), lty=1, col = "green")
legend("bottomleft", legend = c('df=5', 'df=10'), col = c("red", "green"),  lty=c(1, 1))
```

 ![Xixia]({{ site.baseurl }}//assets/images/pspline/p2.png)



# Application on Real Data
For example, we can apply P-Spline on stock transaction data to extract **intraday seasonality**

X-axis: time stamps transaction occurred.  \\
Y-axis: trade interval from the last trade
 ![Xixia]({{ site.baseurl }}//assets/images/pspline/p3.png)

We can confirm the intraday seasonality with P-Spline (shorter transaction interval right after market opens, and right before market closes.)

# Mathematical Background
Now, consider
<div align="center">
$$y _ j =x\left(t _ j \right)+\varepsilon _ j , \;  j=1, \ldots, n.$$
</div>
Here, $$x\left(t _ j \right)$$ is the spline's prediction, $$y _ j$$ are the actual observed points.

Now, how we decide $$x\left(t _ j \right)$$ ?

The first method comes to our mind is probably *least square method*.
<div align="center">
$$ S S E(x | y)=\sum_ j \left(y_ j -x\left(t_ j \right)\right) ^ 2$$
</div>

P-Spline utilizes this idea of least square method.

Now, we consider what kind of line we want to draw.


![Xixia]({{ site.baseurl }}//assets/images/pspline/p4.png)

If we have these ↑ points, the line we want to draw would look like this↓



![Xixia]({{ site.baseurl }}//assets/images/pspline/p5.png)

We can also draw a line like this:

![Xixia]({{ site.baseurl }}//assets/images/pspline/p6.png)
But this↑ is not what we wanted. We **penalize** this zigzag in the  **P**-Spline.

We define **penalty** as:
<div align="center">
$$ \sum _ {i=1} ^ N \left(y-\alpha-f(x ; \beta) +\sum _ {p=1} ^ P f\left(z _ {i p} \right)  \right) ^ 2 +\lambda \int \left(f ^ {\prime \prime} (x) \right) ^ 2 dx$$
</div>



## References
- [Wikipedia](https://en.wikipedia.org/wiki/Smoothing_spline#cite_note-EilersMarx1996-13)
- [Spline smoothing with model-based penalties](https://link.springer.com/article/10.3758/BF03200573)
- [Smoothing with penalized splines](https://csm.lshtm.ac.uk/wp-content/uploads/sites/6/2016/04/Antonio-Gasparrini-29-05-2015.pdf)


