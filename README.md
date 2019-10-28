# Probability Density Approximation using Graphics Processing Unit 

[![DOI](https://zenodo.org/badge/95934306.svg)](https://zenodo.org/badge/latestdoi/95934306)

Probability density approximation (PDA) is a one of the MCMC methods in approximate Bayesian Computation (ABC) (Turner & Sederberg, 2014, PBR; Holmes, 2015, JMP). _ppda_ implements this particular method is an R package, using a heterogeneous programming framework of GPU and CPU. This approach provides not only a user-friendly R interface, but also high-performance parallel computation, solving the computation bottleneck plagued often in ABC.  _ppda_ provides R functions and CUDA C API to harness the parallel computing power of graphics processing unit (GPU), making PDA efficient. Current release, version 0.1.8.6, 
provides,

  * CUDA C API, which allows C programmers to construct their own 
  probability density approximation routines for biological or cognitive 
  models and,
  * R functions, calculating approximated likelihoods of the canonical linear 
  ballistic accumulation (LBA) and piecewise LBA models 
  (Holmes, Trueblood, & Heathcote, 2016).  

PDA calculates likelihoods even when their analytic functions are 
unavailable.  It allows researchers to model computationally complex 
biological processes, which in the past could only be approached by simplified 
models. One criticl contrast is that the conventional methods of  
minimizing the cost function with least square requires one to decide what 
summary statistics to minimize. PDA instead can use all the observations. 
However, PDA is computationally demanding.  It conducts a large number 
of Monte Carlo simulations to attain satisfactory precision. Monte Carlo 
simulations when being conducted repeated, such as in Bayesian MCMC, demand 
huge computational resources. 

We implement _ppda_, in Armadillo C++ and CUDA C libraries to provide
a practical and efficient solution for PDA, which is ready to apply on 
Bayesian computation. _ppda_ enables parallel computation with millions of
threads using GPUs and avoids moving large chunk of memories back and forth 
the system and GPU memories. Hence, _ppda_ practically removes the computational
burden that involves large numbers (>1e6) of model simulations without 
suffering by the limitation of GPU memory bandwidth. This solution allows one to
rapidly approximate probability densities with ultra-high precision.

The paper associated with this package can be viewed / downloaded 
[here](http://link.springer.com/article/10.3758/s13428-018-1153-1) or [here](https://rdcu.be/bPYrT). We are glad if you find the software here is 
useful.  Please open an issue thread here or email the package maintainer at <yishinlin001@gmail.com>, if you find any bugs or have any suggestions.  

## Getting Started

The main reason that _ppda_ runs fast is it easily simulates millions of 
random numbers without being impeded by GPU bandwidth bottleneck. For example, 
it can simulates 2^20 random numbers from the LBA model quickly. 

```
require(ppda)
rm(list = ls())
n <- 2^20     ## This must be a power of two
dat1 <- ppda::rlba(n, nthread=64);  
dat3 <- rtdists::rLBA(n, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, 
   silent=TRUE)
names(dat3) <- c("RT","R")

## Trim RTs > 5 s ----
dat1 <- dat1[dat1$RT < 5, ]
dat3 <- dat3[dat3$RT < 5, ]

## Separate choice 1 (correct) RTs and choice 2 (error) RTs
dat1c <- dat1[dat1[,2]==1, 1]
dat1e <- dat1[dat1[,2]==2, 1]
dat3c <- dat3[dat3[,2]==1, 1]
dat3e <- dat3[dat3[,2]==2, 1]


## Identical PDFs
par(mfrow = c(1,2))
hist(dat1c, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="GPU-Choice 1", xlim=c(0, 3)) ## gpu float
lines(den3c, col="blue", lty="dashed",  lwd=3.0) ## rtdists


hist(dat1e, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="GPU-Choice 2", xlim=c(0, 3)) ## gpu float
lines(den3e, col="blue", lty="dashed",  lwd=3.0) ## rtdists
par(mfrow = c(1,1))

## Comparing the speed of using double-precision, single-precision, 
## and R's script. 'dp=F' stands for turning off double-precision. 
library(microbenchmark)
res <- microbenchmark(ppda::rlba(n, dp=F),
                      ppda::rlba(n, dp=T),
                      rtdists::rLBA(n, b=1, A=.5, mean_v=c(2.4, 1.6), 
                         sd_v=c(1, 1), t0=.5, silent=TRUE), times=10L)

## Unit: milliseconds
##                  expr            min           lq         mean       median    
## ppda::rlba(n, dp = F)       8.310137     8.465464     9.171308     8.529089     
## ppda::rlba(n, dp = T)      11.860449    11.955713    12.313426    12.061767    
## rtdists::rLBA(n,... )   13521.669045 13614.740048 13799.592982 13770.777719 

```

_ppda_ also provides functions to generate random numbers from truncated normal 
distributions. Again, _n_ must be power of 2.


```
n <- 2^20
dat1 <- ppda::rtnorm_gpu(n, mean=-1, sd=1.2, lower=0, upper=Inf)
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
lines(den2$x, den2$y,lwd=2, col = "blue") ## tnorm
lines(den3$x, den3$y,lwd=2, col = "red") ## msn

## tnorm is a small C++ package I developed when testing this package. It is
## defunct. Please see my ggdmc package for C++ tnorm functions.

## Unit: milliseconds
##            expr       min         lq       mean     median         uq        max
## ppda::rtnorm(n)   1.173537   1.417016   1.978613   1.423757   1.580943   6.976541
## tnorm::rtnorm(n)  7.475374   8.317984   8.544317   8.345958   9.120224  10.220493
## msm::rtnorm(n)   54.597366 109.426265 103.025877 110.050924 119.054471 125.192521

```

The main R functions in _ppda_ are to estimate probability densities from 
complex cognitive models: LBA and PLBA models.

```
## 1,000 response times (RTs) from 0 to 3 seconds 
RT <- seq(0, 3, length.out = 1e3);

## Default parameters are b = 1, A = 0.5, mean_v = c(2.4, 1.6),
## sd_v = c(1, 1), t0 = 0.5 with nthread = 32 and nsim = 1024 
den1 <- ppda::n1PDF(RT, A = .5, b = 1, mean_v = c(2.4, 1.6), sd_v = c(1, 1),
  t0 = .2, n = 1024)

## Raising nsim to 2^20 improves approximation
den2 <- ppda::n1PDF(RT, A = .5, b = 1, mean_v = c(2.4, 1.6), sd_v = c(1, 1),
  t0 = .2, n=2^20)

plot(RT,  den2, type="l")
lines(RT, den1, lwd=1.5)


## PLBA model 1
den3 <- ppda::n1PDF_plba1(RT, A=1.5, b=2.7, mean_v=c(3.3, 2.2), sd_v=c(1, 1),
  mean_w=c(1.5, 1.2), rD=.3, swt=.5, t0=.08, n=2^20)
plot(RT, den3, type="l")

```


## Installation 

```
## From github
devtools::install_github("yxlin/ppda")
## From source: 
install.packages("ppda_0.1.8.6.tar.gz", repos = NULL, type="source")
```

## Prerequisites
 - R (>= 3.0.2)
 - nvcc, gcc & g++ (>= 4.4)
 - Nvidia GPU card Compute Capability (>= 2.0)
 - Nvidia's CUDA toolkit (>= release 7.5)
 - [Armadillo](http://arma.sourceforge.net/download.html) (>= 5.100)
 - Ubuntu Linux >= 16.04 (may work on macOS and other Linux distributions)

## Known Workable Nvidia GPU Cards
 - GeForce GT 720M
 - GeForce GTX 980
 - GeForce GTX 1050
 - Tesla K80
 - Tesla K20
 
## Contributors
- Yi-Shin Lin <yishinlin001@gmail.com> 
- Andrew Heathcote 
- William Holmes 

## Acknowledgments
* ppda R packaging is based on gputools 1.1.

## Download the paper 
* Lin, Y.-S., Heathcote, A., & Holmes, W. (2019). [Parallel
Probability Density Approximation](https://rdcu.be/bPYrT). _Behavioral Research Methods_, 1--23. 

## References
* Holmes, W. (2015). A practical guide to the Probability Density
Approximation (PDA) with improved implementation and error characterization.
_Journal of Mathematical Psychology_, 68-69, 13--24,
doi: http://dx.doi.org/10.1016/j.jmp.2015.08.006.

* Holmes, W., Trueblood, J. S., & Heathcote, A. (2016). A new framework for 
modeling decisions about changing information: The Piecewise Linear Ballistic 
Accumulator model. _Cognitive Psychology_, 85, 1--29, 
doi: http://dx.doi.org/10.1016/j.cogpsych.2015.11.002.
