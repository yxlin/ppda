#' Approximated Probability Density of the Canonical LBA Model at Node One 
#'
#' Calculate approximated densities of the LBA model, drawing drift 
#' rates from truncated normal distributions. 
#' 
#' This implementation presumes no variability for non-decision times. When 
#' negative standard deviations are entered, \code{n1PDF} returns \code{1e-10}.
#' Note that lowercase, \code{b}, and uppercase \code{B} are treated 
#' differently. The former refers to the decision threshold and the latter, 
#' traveling distance of an accumulator. That is, \eqn{b = A + B}.  
#' 
#' Brown and Heathcote's (2008) LBA model draws drift rates from a normal 
#' distribution. \code{n1PDF} draws from a truncated normal distribution, with 
#' a lower bound at 0 [See Robert (1995) for further details]. This makes 
#' negative _mean_ drift rates possible, but not a trial will not have a 
#' negative drift rate
#' . 
#' @param x vector of quantiles.
#' @param A start point variability. The LBA model draws a start point
#' from a uniform distribution. \code{A} is the upper bound of the distribution.
#' @param b decision threshold. This must be a scalar 
#' @param mean_v mean drift rates. This must be a numeric vector with two 
#' elements.
#' @param sd_v standard deviations of the drift rate. This must be a numeric 
#' vector with two elements.
#' @param t0 non-decision time.
#' @param n number of model simulations. This must be a power of two.
#' @param nthread number of GPU threads launched per block. Default is 32. 
#' Maximum thread per block is 1024 or 512 depending on GPU. 
#' @param gpuid a nominal ID to identify which GPU to use on a multiple GPU PC.
#' Typically, the ID number starts from 0. 
#' @param h KDE kernel bandwidth
#' @param debug a debugging switch. This is for advance users who wish to use
#' CUDA C API. When setting it to TRUE, debugging information will be printed. 
#' @return a likelihood vector.
#' @references
#' Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice
#' response time: Linear ballistic accumulation. Cognitive psychology, 57(3),
#' 153-178. \url{https://doi.org/10.1016/j.cogpsych.2007.12.002}. \cr
#' Robert, C. P. (1995). Simulation of truncated normal variables. Statistics
#' and Computing, 5(2), 121-125. \url{https://doi.org/10.1007/BF00143942}.
#' @export
#' @examples
#' ##### n1PDF examples
#' x <- seq(0, 3, length.out = 1e3);
#'
#' ## Default n (number of simulations) = 1024
#' den1 <- ppda::n1PDF(x, A = .5, b = 1, mean_v = c(2.4, 1.6), sd_v = c(1, 1),
#' t0 = .5)
#'
#' ## Raising number of simulations to 2^20 can improve approximation
#' den2 <- ppda::n1PDF(x,  A = .5, b = 1, mean_v = c(2.4, 1.6), sd_v = c(1, 1),
#' t0 = .5, n = 2^20, nthread = 32)
#' plot(x,  den2, type = "l", xlab = "Quantiles", ylab = "Density")
#' lines(x, den1, lwd = 1.5)
#'
#' ## If you have multiple GPUs on one machine, you can select different a GPU.
#' ## Raise simulations to 2^27 further improves precision, but not too much
#' den3 <- ppda::n1PDF(x, A = .5, b = 1, mean_v = c(2.4, 1.6), sd_v = c(1, 1),
#' t0 = .5, n = 2^27, nthread = 32, gpuid = 1)
#'
#' \dontrun{
#' ## Checking against an existing method for calculating n1PDF.
#' ## The analytic solution of n1PDF is implemented in rtdists
#' den4 <- rtdists::n1PDF(x, b = 1, A = .5, mean_v = c(2.4, 1.6), 
#'                        sd_v = c(1, 1), t0 = .5, silent = T)
#' lines(x, den3, lwd = 2, col = "red")
#' all.equal(den1, den4)
#' all.equal(den2, den4)
#' all.equal(den3, den4)
#' ## "Mean relative difference: 0.1675376"
#' ## "Mean relative difference: 0.007108101"
#' ## "Mean relative difference: 0.005110652"
#' }
#'
#' ##### An extreme case that rlba does not match dlba
#' ## This happens even when we approximate PDF with 2^20 simulations.
#' ## In trace plots, one can still observe overly noise in Bayesian 
#' ## computation. 
#' den5 <- ppda::n1PDF(x, A = .07, b = .09,  mean_v = c(-7.37, -4.36),
#'                     sd_v = c(1, 1), t0 = .94, n = 2^20)
#'
#' \dontrun{
#' den6 <- rtdists::n1PDF(x, A = .07, b = .09, mean_v = c(-7.37, -4.36),
#'                        sd_v = c(1, 1), t0 = .94, silent = T)
#' par(mfrow = c(1,2))
#' plot(x,  den5, type = "l")
#' lines(x, den6, lwd = 1.5)
#'
#' plot(x, den5,  type = "l")
#' lines(x, den4, lwd = 1.5)
#' all.equal(den4, den5)
#' ## "Mean relative difference: 0.9991495"
#' }
#'
#' ## Note both shapes are similar, but dlba method estimates smaller values,
#' ## relative to rlba method. This happens in behaviourally less plausible
#' ## parameters. It takes some burn-in iterations to smooth out the
#' ## noise. 
#' @export
n1PDF <- function(x, A, b, mean_v, sd_v, t0, n = 1024, nthread = 32,
  gpuid = 0, h = NA, debug = FALSE) {

  if (debug) {
    if (any(sd_v <= 0))  {stop("less than 0 sd causes rtnorm to stall.\n")}
    if (length(b)  != 1) {stop("b must be a scalar.\n")}
    if (length(A)  != 1) {stop("A must be a scalar.\n")}
    if (length(t0) != 1) {stop("t0 must be a scalar.\n")}
    if (n %% 2 != 0 || n < 512) {
      stop("n must be power of 2 and at least 2^9.\n")
    }
  }

  ##if(any(sd_v <= 0) | b <0 | A<0 | t0 <0) {
  if(any(sd_v < 0)) {
    out <- rep(1e-10, length(x))
  } else {
    res <- .C("n1PDF",
      as.double(x),        as.integer(length(x)),
      as.integer(n),       as.double(b),  as.double(A),
      as.double(mean_v),   as.integer(length(mean_v)),
      as.double(sd_v),     as.double(t0),
      as.integer(nthread), as.integer(gpuid),
      as.double(h),
      as.logical(debug),  numeric(length(x)),
      NAOK = TRUE,        PACKAGE='ppda')
    out <- res[[14]]
  }
  return(out)
}

#' Approximate Node 1 Likelihood of the pLBA Model
#'
#' This is the approximated density function for 2-accumualtor \emph{piecewise}
#' LBA model, drawing drift rates from truncated normal distributions. 
#' 
#' \code{n1PDF_plba0} is the node-one probability density function for the naive
#' piecewise LBA model. It draws stage-one drift rates from the truncated normal 
#' distributions with means, \code{mean_v} and standard deviations, \code{sd_v}. 
#' After switch, it redraws drift rates from the same truncated normal 
#' distributions. See the last paragraph in Section 3.1, page 13 (Holmes, 
#' Trueblood, & Heathcote, 2016).
#' 
#' \code{n1PDF_plba1} is the node-one probability density function for two-stage
#' piecewise LBA model. It draws stage-one drift rate from truncated normal
#' distributions with means, \code{mean_v} and standard deviations, \code{sd_v}.
#' After switch, it redraws drift rate from different truncated normal 
#' distributions  with means, \code{mean_w} and standard deviations, 
#' \code{sd_v}.
#'
#' @param x vector of quantiles. 
#' @param A starting point variability. This must be a scalar in plba1. It 
#' must be a two-element vector in plba2 and plba3.
#' @param n number of simulations. This must be a power of two.
#' @param b threshold. Must be a scalar for plba1. Must be a two-element vector
#' for plba2.
#' @param mean_v mean drift rate stage 1. This must be a two-element vector.
#' @param mean_w mean drift rate stage 2. This must be a two-element vector.
#' @param sd_v standard deviation of drift rate stage 1. This must be a
#' two-element vector.
#' @param rD an internal psychological delay time for drift rate.
#' @param swt an external switch time when task information changes.
#' @param t0 non-decision time.
#' @param gpuid select which GPU to conduct model simulation, if running on 
#' multiple GPU machines.
#' @param h kernel bandwidth
#' @param nthread numbers of launched GPU threads. Default is a wrap.
#' @param debug a debugging switch.
#' @return a likelihood vector.
#' @references Holmes, W., Trueblood, J. S., & Heathcote, A. (2016). A new
#' framework for modeling decisions about changing information: The Piecewise
#' Linear Ballistic Accumulator model \emph{Cognitive Psychology}, 85,
#' 1--29.  \url{http://dx.doi.org/10.1016/j.cogpsych.2015.11.002}.
#' @export
#' @examples
#' rm(list=ls())
#' n <- 2^20
#' x <- seq(0, 3, length.out = 1e3);
#' #########################30
#' ## plba0 vs plba1 -------30
#' #########################30
#' den0 <- ppda::n1PDF_plba0(x, A = 1.5, b = 2.7, mean_v = c(3.3, 2.2), 
#'   sd_v = c(1, 1), t0 = .08, mean_w = c(1.5, 1.2), rD = .3, swt = .5, n = n, 
#'   h = .01, debug = FALSE)
#' 
#' den1 <- ppda::n1PDF_plba1(x, A = 1.5, b = 2.7, mean_v = c(3.3, 2.2), 
#'   sd_v = c(1, 1), t0 = .08, mean_w = c(1.5, 1.2), rD = .3, swt = .5, n = n, 
#'   h = .01, debug = FALSE)
#' 
#' ## Use the second GPU card, if there is any
#' ## den2 <- ppda::n1PDF_plba1(x, A = 1.5, b = 2.7, mean_v = c(3.3, 2.2), 
#' ##   sd_v = c(1, 1), t0 = .08, mean_w = c(1.5, 1.2), rD = .3, swt = .5, n = n, 
#' ##   h = .01, gpuid = 1, debug = FALSE)
#' 
#' palette1 <- palette()
#' plot(x, den0, type="l", col = palette1[1], xlab = "Quantiles", 
#'         ylab = "Density")
#' lines(x, den1, lwd = 1, col = palette1[2])
#' ## lines(data, den2, lwd = 1, col = palette1[3])
#' 
#' #########################30
#' ## plba2          -------30
#' #########################30
#' \dontrun{
#' den2 <- ppda::n1PDF_plba2(x, nsim=n, b=c(2.7, 2.7), A=c(1.5,1.5),
#' mean_v=c(3.3, 2.2), mean_w=c(1.5, 1.2),
#' sd_v=c(1, 1), rD=.3, swt=.5, t0=.08)
#' plot(x, den2, type="l")
#' 
#'
#' #########################30
#' ## plba3          -------30
#' #########################30
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
#' den3 <- ppda::n1PDF_plba3(x, nsim = n, B = pvec1[3:4], A =pvec1[1:2],
#'   C = pvec1[5:6], mean_v = pvec1[7:8],
#'   mean_w = pvec1[9:10], sd_v = pvec1[11:12],
#'   sd_w = pvec1[13:14], rD = pvec1[15], tD = pvec1[16],
#'   swt = pvec1[17], t0 = pvec1[18])
#'
#' plot(x, den3, type="l")
#' }
n1PDF_plba0 <- function(x, A, b, mean_v, sd_v, t0, mean_w, rD, swt, n = 1024,
  nthread = 32, gpuid = 0, h = NA, debug = FALSE) {

  if(any(sd_v < 0))  {
    out <- rep(1e-10, length(x))
  } else {
    res <- .C("n1PDF_plba0",
      as.double(x), as.integer(length(x)), as.integer(n),
      as.double(b),
      as.double(A),
      as.double(mean_v),
      as.integer(length(mean_v)),
      as.double(sd_v),
      as.double(t0),
      as.double(mean_w),
      as.double(rD),
      as.double(swt),
      as.integer(nthread), as.integer(gpuid),
      as.double(h), as.logical(debug),
      numeric(length(x)), NAOK = TRUE, PACKAGE = "ppda")
    out <- res[[17]]
  }
  return(out)
}

#' @rdname n1PDF_plba0
#' @export
n1PDF_plba1 <- function(x, A, b, mean_v, sd_v, t0, mean_w, rD, swt, n = 1024,
  nthread = 32, gpuid = 0, h = NA, debug = FALSE) {
  if(any(sd_v < 0))  {
    out <- rep(1e-10, length(x))
  } else {
    res <- .C("n1PDF_plba1",
      as.double(x), as.integer(length(x)), as.integer(n),
      as.double(b),
      as.double(A),
      as.double(mean_v),
      as.integer(length(mean_v)),
      as.double(sd_v),
      as.double(t0),
      as.double(mean_w),
      as.double(rD),
      as.double(swt),
      as.integer(nthread), as.integer(gpuid),
      as.double(h), as.logical(debug),
      numeric(length(x)), NAOK = TRUE, PACKAGE = "ppda")
  out <- res[[17]]
  }
  return(out)
}

n1PDF_plba2 <- function(x, nsim = 1024, b=c(2.7, 2.7), A=c(1.5, 1.5),
  mean_v=c(3.3, 2.2), mean_w=c(1.5, 3.7), sd_v=c(1, 1), sd_w=c(1.5, 1.2), rD=.3,
  swt=.5, t0=.08, nthread=32, gpuid=0, h = NA, debug=FALSE) {
  if(any(sd_v < 0) | any(sd_w < 0)) {
    out <- rep(1e-10, length(x))
  } else {
    res <- .C("n1PDF_plba2",
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
      as.integer(nthread), as.integer(gpuid),
      as.double(h), as.logical(debug),
      numeric(length(x)), NAOK = TRUE, PACKAGE = "ppda")
    out <- res[[18]]
  }
  return(out)
}

n1PDF_plba3 <- function(x, nsim = 1024, B=c(1.2, 1.2), A=c(1.5, 1.5), C=c(.3, .3),
                        mean_v=c(3.3, 2.2), mean_w=c(1.5, 3.7), sd_v=c(1, 1),
                        sd_w=c(1, 1), rD=.1, tD=.1, swt=.5, t0=.08,
                        nthread=64, gpuid=0, h = NA, debug=FALSE) {

  if(any(sd_v < 0) | any(sd_w < 0) | any(A < 0) | any(B < 0) | any(C < 0)
    | any(t0 < 0)) {
    out <- rep(1e-10, length(x))
  } else {
    res <- .C("n1PDF_plba3",
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
      as.integer(nthread), as.integer(gpuid), as.double(h),
      as.logical(debug),
      numeric(length(x)), NAOK = TRUE, PACKAGE = "ppda")
    out <- res[[20]]
  }
  return(out)
}

# n1PDF_ngpu <- function(x, nsim = 1024, b = 1, A = 0.5, mean_v = c(2.4, 1.6),
#   sd_v = c(1, 1), t0 = 0.5, nthread = 64, h = NA, deviceid = 0, debug = FALSE) {
# 
#   if (debug) {
#     if (any(sd_v < 0))   {stop("Standard deviation must be positive.\n")}
#     if (any(sd_v == 0))  {stop("0 sd causes rtnorm to stall.\n")}
#     if (length(b)  != 1) {stop("b must be a scalar.\n")}
#     if (length(A)  != 1) {stop("A must be a scalar.\n")}
#     if (length(t0) != 1) {stop("t0 must be a scalar.\n")}
#     if (nsim %% 2 != 0 || nsim < 512) {
#       stop("nsim must be power of 2 and at least 2^9.\n")
#     }
#   }
# 
#   out <- .C("n1PDF_ngpu", as.double(x), as.integer(length(x)),
#     as.integer(nsim),  as.double(b),  as.double(A),
#     as.double(mean_v), as.integer(length(mean_v)),
#     as.double(sd_v),
#     as.double(t0),     as.integer(nthread),
#     as.double(h),
#     as.logical(debug), numeric(length(x)), as.integer(deviceid),
#     NAOK = TRUE, PACKAGE='ppda')
#   return(out[[13]])
# 
# }

