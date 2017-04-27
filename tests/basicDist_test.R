## runif ----
n <- 1e5
dat1 <- gpda::runif(n)
dat2 <- stats::runif(n)

den1 <- density(dat1)
den2 <- density(dat2)

## Identical 
par(mfrow=c(1,2))
hist(dat2, breaks="fd", freq=F, ylim=c(0,1.1))
lines(den1$x, den1$y,lwd=2)
hist(dat1, breaks="fd", freq=F, ylim=c(0,1.1))
lines(den2$x, den2$y,lwd=2)
par(mfrow=c(1,1))

require(microbenchmark)
res <- microbenchmark(gpda::runif(n), stats::runif(n), times=10L)
# Unit: milliseconds
# expr      min       lq     mean   median       uq      max neval cld
# gpda::runif(n) 1.274803 1.398778 2.051245 1.455317 1.521285 7.245396    10   a
# stats::runif(n) 2.457481 2.487725 2.758491 2.622419 2.639217 3.627238    10   a

## rnorm ----
rm(list=ls())
n <- 1e5
dat1 <- gpda::rnorm(n)
dat2 <- stats::rnorm(n)
den1 <- density(dat1)
den2 <- density(dat2)

## Identical 
par(mfrow=c(1,2))
hist(dat2, breaks="fd", freq=F)
lines(den1$x, den1$y,lwd=2)
hist(dat1, breaks="fd", freq=F)
lines(den2$x, den2$y,lwd=2)
par(mfrow=c(1,1))

require(microbenchmark)
res <- microbenchmark(gpda::rnorm(n), stats::rnorm(n), times=10L)
# Unit: milliseconds
#           expr      min       lq     mean   median       uq      max neval 
# gpda::rnorm(n) 1.337829 1.352705 1.858909 1.385913 1.570951 5.861834    10  
# stats::rnorm(n) 6.191541 6.197897 6.213164 6.213855 6.220036 6.251044   10 

## rtnorm ----
n <- 1e5
dat1 <- gpda::rtnorm(n, mean=2.4, sd=1, lower=0, upper=Inf)
dat2 <- tnorm::rtnorm(n, mean=2.4, sd=1, lower=0, upper=Inf)
den1 <- density(dat1)
den2 <- density(dat2)

## Identical 
par(mfrow=c(1,2))
hist(dat2, breaks="fd", freq=F, ylim=c(0, .5))
lines(den1$x, den1$y,lwd=2)
hist(dat1, breaks="fd", freq=F, ylim=c(0, .5))
lines(den2$x, den2$y,lwd=2)
par(mfrow=c(1,1))

require(microbenchmark)
res <- microbenchmark(gpda::rtnorm(n), tnorm::rtnorm(n), times=10L)

## Other parameter sets
n <- 1e5
dat1 <- gpda::rtnorm(n, mean=2.4, sd=1, lower=0, upper=Inf)
dat2 <- tnorm::rtnorm(n, mean=2.4, sd=1, lower=0, upper=Inf)
dat3 <- msm::rtnorm(n, mean=2.4, sd=1, lower=0, upper=Inf)
den1 <- density(dat1)
den2 <- density(dat2)
den3 <- density(dat3)

res <- microbenchmark(gpda::rtnorm(n), tnorm::rtnorm(n), msm::rtnorm(n), times=10L)
res


dat1 <- gpda::rtnorm(n, mean=100, sd=10,  lower=90, upper=110)
dat2 <- tnorm::rtnorm(n, mean=100, sd=10, lower=90, upper=110)
den1 <- density(dat1)
den2 <- density(dat2)

dat1 <- gpda::rtnorm(n, mean=100, sd=10,  lower=-Inf, upper=150)
dat2 <- tnorm::rtnorm(n, mean=100, sd=10, lower=-Inf, upper=150)
den1 <- density(dat1)
den2 <- density(dat2)

dat1 <- gpda::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf)
dat2 <- tnorm::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf)
dat3 <- msm::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf)
den1 <- density(dat1)
den2 <- density(dat2)
den3 <- density(dat3)

summary(dat1)
summary(dat2)
summary(dat3)

par(mfrow=c(1,3))
hist(dat2, breaks="fd", freq=F)
lines(den1$x, den1$y,lwd=2) ## gpu
lines(den2$x, den2$y,lwd=2) ## tnorm
lines(den3$x, den3$y,lwd=2) ## msn

hist(dat1, breaks="fd", freq=F)
lines(den1$x, den1$y,lwd=2) ## gpu
lines(den2$x, den2$y,lwd=2) ## tnorm
lines(den3$x, den3$y,lwd=2) ## msn

hist(dat3, breaks="fd", freq=F)
lines(den1$x, den1$y,lwd=2) ## gpu
lines(den2$x, den2$y,lwd=2) ## tnorm
lines(den3$x, den3$y,lwd=2) ## msn

par(mfrow=c(1,1))

res <- microbenchmark(gpda::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf), 
  tnorm::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf), 
  msm::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf),
  times=10L)

res




