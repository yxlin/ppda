## Reference: Saucier, R. (2000). Computer Generation of Statistical
## Distributions. ARMY RESEARCH LABORATORY.
rexp_gpu <- function(n, a, b) {
  if (b < 0) stop("b must be greater than 0.")
  z <- a-b*log(runif(n))
  return(z)
}

dat1 <- rexp(1e5)
dat2 <- rexp_gpu(1e5, 0, 1)

par(mfrow=c(1,2))
hist(dat1, breaks="fd")
hist(dat2, breaks="fd")
par(mfrow=c(1,1))
