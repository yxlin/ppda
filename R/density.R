#' Approximate Node 1 Density of a Canonical LBA Model 
#'
#' This is the probability density function for the canonical 2-accumualtor LBA 
#' model, sampling drift rates from truncated normal distributions. The 
#' function approximates model likelihood, instead of calculating analytically.
#' 
#' @param x a numeric vector for estimating likelihood. 
#' @param nsim number of Monte Carlo simulations for building a simulated PDF. 
#' This must be a power of two.
#' @param b decision threshold. Default is 1.  
#' @param A start point variability. Because the LBA model draws  
#' realisation of start point from an uniform distribution, \code{A} is the 
#' upper bound of the uniform distribution. See the below reference for 
#' more details.  
#' @param mean_v mean drift rate. This must be a two-element numeric vector. 
#' @param sd_v standard deviation of the drift rate. This must be a two-element 
#' numeric vector. The LBA model draws realisation of drift rate for a trial 
#' from a normal distribution. \code{n1PDF} draws from a truncated normal 
#' distribution, eliminating realisation of negative drift rate.  Note in 
#' the canonical LBA model, mean drift rate could be negative, but not 
#' drift-rate realisation .        
#' @param t0 nondecision time.  
#' @param nthread number of GPU threads launched per block. Default is 32. 
#' Maximum thread per block is 1024 for K80 and 512 for most early GPU cards.
#' @param h kernal density bandwidth   
#' @param debug a debugging switch. This is for advance users who wish to use 
#' CUDA C API.
#' @return a numeric likelihood vector. 
#' @references
#' Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice 
#' response time: Linear ballistic accumulation. Cognitive psychology, 57(3), 
#' 153-178. \url{https://doi.org/10.1016/j.cogpsych.2007.12.002}
#' @export
#' @examples
#' ##### n1PDF examples 
#' data <- seq(0, 3, length.out = 1e3);
#' 
#' ## Default parameters are b = 1, A = 0.5, mean_v = c(2.4, 1.6),
#' ## sd_v = c(1, 1), t0 = 0.5 with nthread = 32 and nsim = 1024 
#' den1 <- gpda::n1PDF(data)
#' 
#' ## Raising nsim to 2^20 can improve approximation
#' den2 <- gpda::n1PDF(data, nsim=2^20)
#' plot(data,  den2, type="l")
#' lines(data, den1, lwd=1.5)
#' 
#' \dontrun{
#' den3 <- rtdists::n1PDF(data, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), 
#'                        t0=.5, silent=T)
#' lines(data, den3, lwd=2)
#' all.equal(den1, den3)
#' all.equal(den2, den3)
#' ## "Mean relative difference: 0.1675376"
#' ## "Mean relative difference: 0.007108101"
#' }
#'
#' ##### An extreme case that rlba does not match dlba 
#' ## When approximated PDF is almost perfect with 2^20 simulations,
#' ## one can still observe noise in Bayesian computation. One possible reason
#' ## is the following:   
#' den4 <- gpda::n1PDF(data, b=.09, A=.07, mean_v=c(-7.37, -4.36), 
#' sd_v=c(1, 1), t0=.94, nsim=2^20)
#' 
#' \dontrun{
#' den5 <- rtdists::n1PDF(data, b=.09, A=.07, mean_v=c(-7.37, -4.36), 
#' sd_v=c(1, 1), t0=.94, silent=T)
#' par(mfrow=c(1,2))
#' plot(data,  den4, type="l")
#' lines(data, den5, lwd=1.5)
#' 
#' plot(data, den5,  type="l")
#' lines(data, den4, lwd=1.5)
#' all.equal(den4, den5)
#' ## "Mean relative difference: 0.9991495"
#' }
#' 
#' ## Note both shapes are similar, but dlba method estimates smaller values, 
#' ## relative to rlba method. This happens in behaviourally less plausible  
#' ## parameter sets. It takes longer iterations to smooth out the 
#' ## noise.  That is when a sampler no longer propose these less plausible
#' ## parameter sets. 
#' @export
n1PDF <- function(x, nsim = 1024, b = 1, A = 0.5, mean_v = c(2.4, 1.6),
  sd_v = c(1, 1), t0 = 0.5, nthread = 64, h = NA, debug = FALSE) {

  if (debug) {  
    if (any(sd_v < 0))   {stop("Standard deviation must be positive.\n")}
    if (any(sd_v == 0))  {stop("0 sd causes rtnorm to stall.\n")}
    if (length(b)  != 1) {stop("b must be a scalar.\n")}
    if (length(A)  != 1) {stop("A must be a scalar.\n")}
    if (length(t0) != 1) {stop("t0 must be a scalar.\n")}
    if (nsim %% 2 != 0 || nsim < 512) { 
      stop("nsim must be power of 2 and at least 2^9.\n")
    }
  }
  
  out <- .C("n1PDF", 
    as.double(x),       as.integer(length(x)), 
    as.integer(nsim),   as.double(b),  as.double(A),
    as.double(mean_v),  as.integer(length(mean_v)), 
    as.double(sd_v),    as.double(t0),     
    as.integer(nthread),as.double(h),
    as.logical(debug),  numeric(length(x)),
    NAOK = TRUE,        PACKAGE='gpda')
  return(out[[13]])
}

#' Approximate Node 1 Likelihood of pLBA Model 
#'
#' This is the approximated density function for 2-accumualtor \bold{piecewise}
#' LBA model, sampling drift rates from truncated normal distributions. The 
#' function uses probability density approximation to estimate likelihood. 
#' 
#' @param x a data vector for estimating likelihood. 
#' @param nsim number of simulations. This must be a power of two.
#' @param b threshold. Must be a scalar for plba1. Must be a two-element vector
#' for plba2.
#' @param B travelling distance stage 1. The distance between starting point (
#' drawn randomly from an uniform distribution) to the threshold.  This applies
#' for plba3 only. Please note B differs from b. Must be a two-element vector.
#' @param A starting point upper bound. Must be a scalar for plba1. Must be a 
#' two-element vector for plba2 and plba3.
#' @param C travelling distance stage 2. The distance between updated threshold 
#' and original threshold This applies for plba3 only. Must be a two-element 
#' vector. Note this is uppercase.
#' @param mean_v mean drift rate stage 1. This must be a two-element vector. 
#' @param mean_w mean drift rate stage 2. This must be a two-element vector. 
#' @param sd_v standard deviation of drift rate stage 1. This must be a 
#' two-element vector.
#' @param sd_w standard deviation of drift rate stage 2. This must be a 
#' two-element vector.
#' @param rD an internal psychological delay time for drift rate.   
#' @param tD an internal psychological delay time for threshold. This applies
#' for plba3 only.   
#' @param swt an external switch time when task information changes.   
#' @param t0 non-decision time.  
#' @param nthread numbers of launched GPU threads. Default is a wrap.
#' @param debug a debugging switch. 
#' @return a likelihood vector. 
#' @references Holmes, W., Trueblood, J. S., & Heathcote, A. (2016). A new 
#' framework for modeling decisions about changing information: The Piecewise 
#' Linear Ballistic Accumulator model \emph{Cognitive Psychology}, \bold{85},
#' 1--29, \cr doi: \url{http://dx.doi.org/10.1016/j.cogpsych.2015.11.002}.
#' @export
#' @examples
#' n <- 2^20
#' x <- seq(0, 3, length.out = 1e3);
#' 
#' ## plba1
#' den1 <- gpda::n1PDF_plba1(x, nsim=n, b=2.7, A=1.5, mean_v=c(3.3, 2.2), 
#'                           mean_w=c(1.5, 1.2), sd_v=c(1, 1), rD=.3, swt=.5,
#'                           t0=.08)
#' plot(x, den1, type="l")
#' 
#' ## plba2
#' den2 <- gpda::n1PDF_plba2(x, nsim=n, b=c(2.7, 2.7), A=c(1.5,1.5),
#' mean_v=c(3.3, 2.2), mean_w=c(1.5, 1.2),
#' sd_v=c(1, 1), rD=.3, swt=.5, t0=.08)
#' plot(x, den2, type="l")
#' 
#' ## plba3
#' pvec1 <- c(A1 = 1.51, A2 = 1.51, B1 = 1.2, B2 = 1.2,   C1 = .3, C2 = .3,
#'   v1 = 3.32, v2 = 2.24, w1 = 1.51, w2 = 3.69, sv1 = 1, sv2 = 1,
#'   sw1 = 1, sw2 = 1, rD = 0.1, tD = .1, swt = 0.5, t0 = 0.08)
#' 
#' pvec2 <- c(A1 = 1.51, A2 = 1.51, B1 = 1.2, B2 = 1.2, C1 = .3, C2 = .3,
#'   v1 = 3.32, v2 = 2.24, w1 = 1.51, w2 = 3.69, sv1 = 1, sv2 = 1,
#'   sw1 = 1, sw2 = 1, rD = 0.1, tD = .15, swt = 0.5, t0 = 0.08)
#' pvec3 <- c(A1 = 1.51, A2 = 1.51, B1 = 1.2, B2 = 1.2, C1 = .3, C2 = .3,
#'   v1 = 3.32, v2 = 2.24, w1 = 1.51, w2 = 3.69, sv1 = 1, sv2 = 1,
#'   sw1 = 1, sw2 = 1, rD = 0.15, tD = .1, swt = 0.5, t0 = 0.08)
#' 
#' den3 <- gpda::n1PDF_plba3(x, nsim = n, B = pvec1[3:4], A =pvec1[1:2],
#'   C = pvec1[5:6], mean_v = pvec1[7:8],
#'   mean_w = pvec1[9:10], sd_v = pvec1[11:12],
#'   sd_w = pvec1[13:14], rD = pvec1[15], tD = pvec1[16],
#'   swt = pvec1[17], t0 = pvec1[18])
#' 
#' plot(x, den3, type="l")
#' 
n1PDF_plba1 <- function(x, nsim = 1024, b=2.7, A=1.5, mean_v=c(3.3, 2.2), 
  mean_w=c(1.5, 1.2), sd_v=c(1, 1), rD=.3, swt=.5, t0=.08, nthread=32, 
  debug=FALSE) {
  result <- .C("n1PDF_plba1", 
    as.double(x), as.integer(length(x)), as.integer(nsim),  
    as.double(b),
    as.double(A), 
    as.double(mean_v),
    as.integer(length(mean_v)),
    as.double(sd_v),
    as.double(t0),
    as.double(mean_w),
    as.double(rD),
    as.double(swt),
    as.integer(nthread), 
    as.logical(debug), 
    #numeric(nsim), integer(nsim),
    numeric(length(x)), PACKAGE = "gpda")
  return(result[[15]])
  ##return(data.frame(RT=result[[15]], R=result[[16]], Den=result[[17]]))
  ##return(list(RT=result[[15]], R=result[[16]], Den=result[[17]]))
}

#' @rdname n1PDF_plba1
#' @export
n1PDF_plba2 <- function(x, nsim = 1024, b=c(2.7, 2.7), A=c(1.5, 1.5), 
  mean_v=c(3.3, 2.2), mean_w=c(1.5, 3.7), sd_v=c(1, 1), sd_w=c(1.5, 1.2), rD=.3,
  swt=.5, t0=.08, nthread=32, debug=FALSE) {
  result <- .C("n1PDF_plba2", 
    as.double(x), as.integer(length(x)), as.integer(nsim),  
    as.double(b),
    as.double(A), 
    as.double(mean_v),
    as.integer(length(mean_v)),
    as.double(sd_v),
    as.double(sd_w),
    as.double(t0),
    as.double(mean_w),
    as.double(rD),
    as.double(swt),
    as.integer(nthread), 
    as.logical(debug), 
    ## numeric(nsim), integer(nsim),
    numeric(length(x)), PACKAGE = "gpda")
  return(result[[16]])
  ## return(list(RT=result[[16]], R=result[[17]], Den=result[[18]]))
}


#' @rdname n1PDF_plba1
#' @export
n1PDF_plba3 <- function(x, nsim = 1024, B=c(1.2, 1.2), A=c(1.5, 1.5), C=c(.3, .3),
                        mean_v=c(3.3, 2.2), mean_w=c(1.5, 3.7), sd_v=c(1, 1),
                        sd_w=c(1, 1), rD=.1, tD=.1, swt=.5, t0=.08,
                        nthread=32, debug=FALSE) {
  result <- .C("n1PDF_plba3", 
    as.double(x), as.integer(length(x)), as.integer(nsim),  
    as.double(B),
    as.double(A),
    as.double(C), 
    as.double(mean_v),
    as.integer(length(mean_v)),
    as.double(sd_v),
    as.double(sd_w),
    as.double(t0),
    as.double(mean_w),
    as.double(rD),
    as.double(tD),
    as.double(swt),
    as.integer(nthread), 
    as.logical(debug), 
    ## numeric(nsim), integer(nsim),
    numeric(length(x)), PACKAGE = "gpda")
    return(result[[18]])
  ## return(list(RT=result[[18]], R=result[[19]], Den=result[[20]]))
  
}

#' @export
n1PDF_ngpu <- function(x, nsim = 1024, b = 1, A = 0.5, mean_v = c(2.4, 1.6),
  sd_v = c(1, 1), t0 = 0.5, nthread = 64, h = NA, debug = FALSE) {
  
  if (debug) {
    if (any(sd_v < 0))   {stop("Standard deviation must be positive.\n")}
    if (any(sd_v == 0))  {stop("0 sd causes rtnorm to stall.\n")}
    if (length(b)  != 1) {stop("b must be a scalar.\n")}
    if (length(A)  != 1) {stop("A must be a scalar.\n")}
    if (length(t0) != 1) {stop("t0 must be a scalar.\n")}
    if (nsim %% 2 != 0 || nsim < 512) { 
      stop("nsim must be power of 2 and at least 2^9.\n")
    }
  }
  
  out <- .C("n1PDF_ngpu", as.double(x), as.integer(length(x)), 
    as.integer(nsim),  as.double(b),  as.double(A),
    as.double(mean_v), as.integer(length(mean_v)), 
    as.double(sd_v),    
    as.double(t0),     as.integer(nthread),
    as.double(h),
    as.logical(debug), numeric(length(x)),
    NAOK = TRUE, PACKAGE='gpda')
  return(out[[13]])
  
}

