#' Generate Uniform Random Numbers with GPU 
#'
#' This function generates uniform random numbers using GPU.
#'
#' @param n number of observations. This accepts only one integer.
#' @param min lower bound of the uniform distribution. Must be finite. 
#' This accepts only one double/numeric. 
#' @param max upper bound of the uniform distribution. Must be finite. 
#' This accepts only one double/numeric.
#' @param nThread number of threads launched per block. 
#' @return a double vector
#' @export
#' @examples
#' n <- 1e6
#' dat1 <- gpda::runif(n)
#' dat2 <- stats::runif(n)
#' den1 <- density(dat1)
#' den2 <- density(dat2)
#' ## Identical result
#' par(mfrow=c(1,2))
#' hist(dat2, breaks="fd", freq=F, ylim=c(0,1.1))
#' lines(den1$x, den1$y,lwd=2)
#' hist(dat1, breaks="fd", freq=F, ylim=c(0,1.1))
#' lines(den2$x, den2$y,lwd=2)
#' par(mfrow=c(1,1))
#' 
#' ## require(microbenchmark)
#' ## res <- microbenchmark(gpda::runif(n), stats::runif(n), times=10L)
#' ## Unit: milliseconds
#' ##           expr      min        lq     mean   median       uq      max neval   
#' ## gpda::runif(n)  1.274803 1.398778 2.051245 1.455317 1.521285 7.245396    10   
#' ## stats::runif(n) 2.457481 2.487725 2.758491 2.622419 2.639217 3.627238    10   
runif <- function(n, min=0, max=1, nThread=32) {
  if( length(min) != 1 | length(max) != 1 ) { 
    stop("min and max must be a scalar!") 
  }
  
  .C("runif_entry", as.integer(n),  as.double(min), as.double(max), 
    as.integer(nThread), numeric(n), PACKAGE='gpda')[[5]]
}

#' Generate Gaussian Random Numbers with GPU 
#'
#' This function generates random numbers using GPU from a normal distribution.
#'
#' @param n number of observations. This accepts only one integer
#' @param mean a mean. This accepts only one double/numeric.
#' @param sd a standard deviation. This accepts only one double/numeric.
#' @param nThread number of threads launched per block.
#' @return a double vector
#' @export
#' @examples
#' n <- 1e5
#' dat1 <- gpda::rnorm(n)
#' dat2 <- stats::rnorm(n)
#' den1 <- density(dat1)
#' den2 <- density(dat2)
#' 
#' ## Identical 
#' par(mfrow=c(1,2))
#' hist(dat2, breaks="fd", freq=F)
#' lines(den1$x, den1$y,lwd=2)
#' hist(dat1, breaks="fd", freq=F)
#' lines(den2$x, den2$y,lwd=2)
#' par(mfrow=c(1,1))
#' 
#' ## require(microbenchmark)
#' ## res <- microbenchmark(gpda::rnorm(n), stats::rnorm(n), times=10L)
#' ## Unit: milliseconds
#' ##           expr      min       lq     mean   median       uq      max neval 
#' ## gpda::rnorm(n) 1.337829 1.352705 1.858909 1.385913 1.570951 5.861834    10  
#' ## stats::rnorm(n) 6.191541 6.197897 6.213164 6.213855 6.220036 6.251044   10 
rnorm <- function(n, mean=0, sd=1, nThread=32) {
  if( length(mean) != 1 | length(sd) != 1 ) { 
    stop("mean and sd must be a scalar!") 
  }
  if(sd < 0) stop("sd must be greater than 0!") 
  .C("rnorm_entry", as.integer(n), as.double(mean), as.double(sd), 
    as.integer(nThread), numeric(n), PACKAGE='gpda')[[5]]
}

#' Generate Random numbers form a Truncated Normal Distribution with GPU 
#'
#' This function generates random numbers using GPU from a truncated normal 
#' distribution.
#' 
#' @param n number of observations. Only accept an integer scalar
#' @param mean mean. Only a scalar
#' @param sd standard deviations. Only accept a scalar
#' @param lower lower bound. Only accept a scalar
#' @param upper upper bound. Only accept a scalar
#' @param nThread number of threads launched per block.
#' @return a double vector
#' @export
#' @examples
#' dat1 <- gpda::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf)
#' dat2 <- tnorm::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf)
#' dat3 <- msm::rtnorm(n, mean=-1, sd=1.2, lower=0, upper=Inf)
#' den1 <- density(dat1)
#' den2 <- density(dat2)
#' den3 <- density(dat3)
#' 
#' summary(dat1)
#' summary(dat2)
#' summary(dat3)
#' 
#' par(mfrow=c(1,3))
#' hist(dat2, breaks="fd", freq=F)
#' lines(den1$x, den1$y,lwd=2) ## gpu
#' lines(den2$x, den2$y,lwd=2) ## tnorm
#' lines(den3$x, den3$y,lwd=2) ## msn
#' 
#' hist(dat1, breaks="fd", freq=F)
#' lines(den1$x, den1$y,lwd=2) ## gpu
#' lines(den2$x, den2$y,lwd=2) ## tnorm
#' lines(den3$x, den3$y,lwd=2) ## msn
#' 
#' hist(dat3, breaks="fd", freq=F)
#' lines(den1$x, den1$y,lwd=2) ## gpu
#' lines(den2$x, den2$y,lwd=2) ## tnorm
#' lines(den3$x, den3$y,lwd=2) ## msn
#' par(mfrow=c(1,1))
#' 
#' ## Unit: milliseconds
#' ##            expr       min         lq       mean     median         uq        max
#' ## gpda::rtnorm(n)   1.173537   1.417016   1.978613   1.423757   1.580943   6.976541
#' ## tnorm::rtnorm(n)  7.475374   8.317984   8.544317   8.345958   9.120224  10.220493
#' ## msm::rtnorm(n)   54.597366 109.426265 103.025877 110.050924 119.054471 125.192521
rtnorm <- function(n, mean=0, sd=1, lower=-Inf , upper=Inf, nThread=32) {
  if( length(mean) != 1 | length(sd) != 1 | length(lower) != 1 | 
      length(upper) != 1) { stop("mean, sd, lower, or upper must be scalar!") }
  if(upper <= lower) stop("upper must be greater than lower!")
  if(sd <= 0) stop("Standard deviation cannot be 0 or negative!") 

  out <- .C("rtnorm_entry", as.integer(n), as.double(mean),
      as.double(sd), as.double(lower), as.double(upper),
      as.integer(nThread), numeric(n), NAOK = TRUE, PACKAGE='gpda')[[7]]
  return(out)
}


#' The Random Number Generator of Cannoical Linear Ballistic Accumulator Model
#'
#' This function generates two-accumulator LBA random numbers using GPU.
#'
#' @param n number of observations. This accepts only integer
#' @param b upper bound. (b-A/2) is response caution.
#' @param A starting point interval or evidence in accumulator before
#' beginning of decision process. Start point varies from trial to trial in
#' the interval [0, A]. Average amount of evidence before evidence
#' accumulation across trials is A/2.
#' @param mean_v drift rate means
#' @param sd_v drift rate standard deviations
#' @param t0 non-decision times
#' @param nThreads number of threads launched per block.
#' @return a data frame
#' @export
#' @examples
#' 
rlba <- function(n, b=1, A=0.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=0.5, 
  nthread=32) {
  if (any(sd_v < 0)) stop("Standard deviation must be positive.\n")
  nmean_v <- length(mean_v)
  nsd_v   <- length(sd_v)
  
  if (nsd_v==1) { 
    sd_v  <- rep(sd_v, nmean_v)
    nsd_v <- length(sd_v)
  }
  if (nmean_v != nsd_v) stop("sd_v length must match that of mean_v!\n")

  result <- .C("rlba_entry", as.integer(n), as.double(b), as.double(A), 
    as.double(mean_v), as.integer(nmean_v), 
    as.double(sd_v), as.integer(length(sd_v)),
    as.double(t0), as.integer(nthread), 
    numeric(n), integer(n), PACKAGE='gpda')
  return(data.frame(RT=result[[10]], R=result[[11]]))
}

#' @export
rlba_test <- function(n, b=1, A=0.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=0.5, 
  nthread=32) {
  if (any(sd_v < 0)) stop("Standard deviation must be positive.\n")
  nmean_v <- length(mean_v)
  nsd_v   <- length(sd_v)
  
  if (nsd_v==1) { 
    sd_v  <- rep(sd_v, nmean_v)
    nsd_v <- length(sd_v)
  }
  if (nmean_v != nsd_v) stop("sd_v length must match that of mean_v!\n")
  
  result <- .C("rlba_test", as.integer(n), as.double(b), as.double(A), 
    as.double(mean_v), as.integer(nmean_v), 
    as.double(sd_v), as.integer(length(sd_v)),
    as.double(t0), as.integer(nthread), 
    numeric(n), integer(n), PACKAGE='gpda')
  return(data.frame(RT=result[[10]], R=result[[11]]))
}
