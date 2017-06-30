# Probability Density Approximation using Graphics Processing Unit 

_gpda_ is an R package, conducting probability density approximation (PDA) 
(Turner & Sederberg, 2012; Holmes, 2015).  This package provides R functions 
and CUDA C API 
to harness the parallel computing power of graphics processing unit (GPU), 
making PDA computation efficient. Current release, version 0.18, mainly 
provides,

  * CUDA C API, which allow C programmers to construct their own 
  probability density approximation routines for biological or cognitive 
  models and,
  * R functions, which approximates two choice response time cognitive 
  models: Canonical linear ballistic accumulation and piecewise LBA 
  models (Holmes, Trueblood, & Heathcote, 2016).  

PDA calculates likelihood even when their analytic functions are 
unavailable (Turner & Sederberg, 2012; Holmes, 2015).  It allows 
researchers to model computationally complex biological processes, which in the
past could only be approached by overly simplified models. PDA is however 
computationally demanding.  It requires a large number of Monte Carlo 
simulations to attain satisfactory approximation precision. Monte Carlo 
simulations add a heavy computational burden on every step of PDA algorithm. 

We implement _gpda_, using Armadillo C++ and CUDA libraries in order to provide
a practical and efficient solution for PDA, which is ready for applying on 
Bayesian computation. _gpda_ enables parallel computations with millions 
threads using graphics processing unit (GPU) and avoids moving large chunk of 
memories back and forth system and GPU Memories. Hence, _gpda_ practically 
removes the computational burden that involves large numbers (>1e6) of model 
simulations without suffering the pitfall of moving memories. This solution 
allows one to rapidly approximate probability densities with ultra-high 
precision. 

This project is still under active development. We are glad if you find 
software here is useful.  If you've found any bugs or have any suggestions, 
please email the package maintainer at <yishin.lin@utas.edu.au>. 


## Getting Started

The main reason that _gpda_ compute fast is it easily simulates millions of 
random numbers without impeded by the bandwidth bottleneck. For example, it 
can simulates 2^20 random numbers from the LBA model quickly. 

```
require(gpda)
rm(list=ls())
n <- 2^20     ## This must be a power of two
dat1 <- gpda::rlba(n, nthread=64);  
dat3 <- rtdists::rLBA(n, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, 
   silent=TRUE)
names(dat3) <- c("RT","R")

## Trim extreme RTs ----
dat1 <- dat1[dat1$RT < 5, ]
dat3 <- dat3[dat3$RT < 5, ]

## Separate choice 1 (correct) RTs and choice 2 (error) RTs
dat1c <- dat1[dat1[,2]==1, 1]
dat1e <- dat1[dat1[,2]==2, 1]
dat3c <- dat3[dat3[,2]==1, 1]
dat3e <- dat3[dat3[,2]==2, 1]

den1c <- density(dat1c)
den3c <- density(dat3c)
den1e <- density(dat1e)
den3e <- density(dat3e)

## Identical PDFs
hist(dat1c, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="GPU-Choice 1", xlim=c(0, 3)) ## gpu float
lines(den3c, col="blue", lty="dashed",  lwd=3.0) ## rtdists

plot(den1c$x, den1c$y, type="l")
lines(den1e$x, den1e$y)

lines(den3c$x, den3c$y, col="blue", lwd=2, lty="dashed")
lines(den3e$x, den3e$y, col="blue", lwd=2, lty="dashed")


## Comparing the speed of using double-precision, single-precision, 
## and R's script. 'dp=F' stands for turning off double-precision. 
library(microbenchmark)
res <- microbenchmark(gpda::rlba(n, dp=F),
                      gpda::rlba(n, dp=T),
                      rtdists::rLBA(n, b=1, A=.5, mean_v=c(2.4, 1.6), 
                         sd_v=c(1, 1), t0=.5, silent=TRUE), times=10L)

## Unit: milliseconds
##                  expr            min           lq         mean       median    
## gpda::rlba(n, dp = F)       8.310137     8.465464     9.171308     8.529089     
## gpda::rlba(n, dp = T)      11.860449    11.955713    12.313426    12.061767    
## rtdists::rLBA(n,... )   13521.669045 13614.740048 13799.592982 13770.777719 

```

_gpda_ also provides functions to generate random numbers from truncated normal 
distributions. Again, _n_ must be power of 2.


```
n <- 2^20
dat1 <- gpda::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf)
dat2 <- tnorm::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf)
dat3 <- msm::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf)
den1 <- density(dat1)
den2 <- density(dat2)
den3 <- density(dat3)

summary(dat1)
summary(dat2)
summary(dat3)

hist(dat2, breaks="fd", freq=F)
lines(den1$x, den1$y,lwd=2) ## gpu
lines(den2$x, den2$y,lwd=2) ## tnorm
lines(den3$x, den3$y,lwd=2) ## msn


## Unit: milliseconds
##            expr       min         lq       mean     median         uq        max
## gpda::rtnorm(n)   1.173537   1.417016   1.978613   1.423757   1.580943   6.976541
## tnorm::rtnorm(n)  7.475374   8.317984   8.544317   8.345958   9.120224  10.220493
## msm::rtnorm(n)   54.597366 109.426265 103.025877 110.050924 119.054471 125.192521

```

The main R functions in _gpda_ are to estimate probability densities from 
complex cognitive models, for example, LBA and PLBA models.

```
## 1,000 RTs from 0 to 3 seconds 
RT <- seq(0, 3, length.out = 1e3);

## Default parameters are b = 1, A = 0.5, mean_v = c(2.4, 1.6),
## sd_v = c(1, 1), t0 = 0.5 with nthread = 32 and nsim = 1024 
den1 <- gpda::n1PDF(RT)

## Raising nsim to 2^20 improves approximation
den2 <- gpda::n1PDF(RT, nsim=2^20)
plot(RT,  den2, type="l")
lines(RT, den1, lwd=1.5)


## PLBA model 1
den3 <- gpda::n1PDF_plba1(RT, nsim=2^20, b=2.7, A=1.5, mean_v=c(3.3, 2.2), 
                          mean_w=c(1.5, 1.2), sd_v=c(1, 1), rD=.3, swt=.5,
                          t0=.08)
plot(x, den3, type="l")

```


## Installation 

```
## From github
devtools::install_github("TasCL/gpda")
## From source: 
install.packages("gpda_0.1.8.tar.gz", repos = NULL, type="source")
```

## Prerequisites
 - R (>= 3.0.2)
 - nvcc, gcc & g++ (>= 4.4)
 - Nvidia GPU card Compute Capability (>= 2.0)
 - Nvidia's CUDA toolkit (>= release 7.5)
 - [Armadillo](http://arma.sourceforge.net/download.html) (>= 5.100)
 - Ubuntu Linux >= 16.04 

## Known Workable Nvidia GPU Cards
 - GeForce GT 720M
 - GeForce GTX 980
 - Tesla K80
 
## Contributors

- Yi-Shin Lin <yishin.lin@utas.edu.au> 
- Andrew Heathcote 
- William Holmes 

## Acknowledgments
* gpda R packaging is based on gputools 1.1.

## References

* Holmes, W. (2015). A practical guide to the Probability Density
Approximation (PDA) with improved implementation and error characterization.
_Journal of Mathematical Psychology_, 68-69, 13--24,
doi: http://dx.doi.org/10.1016/j.jmp.2015.08.006.

* Holmes, W., Trueblood, J. S., & Heathcote, A. (2016). A new framework for 
modeling decisions about changing information: The Piecewise Linear Ballistic 
Accumulator model. _Cognitive Psychology_, 85, 1--29, 
doi: http://dx.doi.org/10.1016/j.cogpsych.2015.11.002.