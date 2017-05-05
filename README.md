# Probability Density Approximation using Graphics Processing Unit 

The package uses general purpose graphics processing unit (GP-GPU) 
to conduct Monte Carlo simulations. Two cognitive models we conduct Monte
Carlo are the basic and the piece-wise linear ballistic accumulation models. 
The simulations are then used to approximate the model likelihood via a 
parallel implementation on the basis of a method described in Homles (2015). 

This project is still under active development. We are glad if you find 
software here is useful.  If you've found any bugs or have any suggestions, 
please email the package [maintainer] at <yishin.lin@utas.edu.au>. 


## Getting Started

Simulate more than 1,000,000 LBA model random numbers:

```
require(gpda)
m(list=ls())
n <- 2^20     ## This must be a power of two
dat1 <- gpda::rlba(n, nthread=64);  
dat3 <- rtdists::rLBA(n, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, 
   silent=TRUE)
names(dat3) <- c("RT","R")

## Trim extreme RTs ----
dat1 <- dat1[dat1$RT < 5, ]
dat3 <- dat3[dat3$RT < 5, ]

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

'gpda' generates random numbers from a truncated normal distribution, too. n 
must be power of 2.


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

## Installation 

```
## From github
devtools::install_github("TasCL/gpda")
## From source: 
install.packages("gpda_0.1.5.tar.gz", repos = NULL, type="source")
```

## Prerequisites
 - R (>= 3.0.2)
 - nvcc, gcc & g++ (>= 4.4)
 - Nvidia GPU card Compute Capability (>= 2.0)
 - Nvidia's CUDA toolkit (>= release 7.5)
 - [Armadillo](http://arma.sourceforge.net/download.html) (>= 5.100)
  
## Contributors

- Yi-Shin Lin <yishin.lin@utas.edu.au> 
- Andrew Heathcote 
- William Holmes 

## References
* Holmes, W. (2015). A practical guide to the Probability Density
Approximation (PDA) with improved implementation and error characterization.
Journal of Mathematical Psychology, 68-69, 13--24,
doi: http://dx.doi.org/10.1016/j.jmp.2015.08.006.

## Acknowledgments
* gpda R packaging is based on gputools 1.1.
