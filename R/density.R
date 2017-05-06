#' Approximate Node 1 Likelihood in a Two-accumulator LBA Model 
#'
#' This is the approximated density function for the basic 2-accumualtor LBA 
#' model, sampling drift rates from truncated normal distributions. The 
#' function uses probability density approximation to estimate likelihood. 
#' 
#' @param data a data vector. 
#' @param nsim number of simulations, passed to internal rlba_gpu. The default 
#' is 1e5.  
#' @param b threshold. Default is 1.  
#' @param A starting point upper bound  
#' @param mean_v mean drift rate. This must be a two-element vector. 
#' @param sd_v standard deviation of drift rate. This must be a two-element 
#' vector.
#' @param t0 non-decision time.  
#' @param nthread number of threads launched in GPU. Default is 32. Maximum is
#' 1024.
#' @return a likelihood vector. 
#' @references Holmes, W. (2015). A practical guide to the Probability Density
#' Approximation (PDA) with improved implementation and error characterization.
#' \emph{Journal of Mathematical Psychology}, \bold{68-69}, 13--24,
#' doi: http://dx.doi.org/10.1016/j.jmp.2015.08.006.
#' @export
#' @examples
#' #################
#' ## n1PDF examples 
#' #################
#' data <- seq(0, 3, length.out = 1e3);
#' 
#' ## Default parameter set is b = 1, A = 0.5, mean_v = c(2.4, 1.6),
#' ## sd_v = c(1, 1), t0 = 0.5 with nthread = 64 and conduct only 1024 
#' ## simulations
#' den1 <- gpda::n1PDF(data)
#' den2 <- rtdists::n1PDF(data, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), 
#'                        t0=.5, silent=T)
#' 
#' ## Verify that we are not checking near 0 densities                          #' plot(data, den1, type="l")
#' lines(data, den2, lwd=2)
#' all.equal(den1, den2)
#' ## "Mean relative difference: 0.2002878"
#' ## The approximation is not good enough, so we raise the simulation to 2^20 
#' den1 <- gpda::n1PDF(data, nsim=2^20)
#' plot(data, den1, type="l")
#' lines(data, den2, lwd=2)
#' all.equal(den1, den2)
#' ## "Mean relative difference: 0.007108101"
#' ## Now the difference goes down to 2 decimal places below 0.
#'
#' ##########################################
#' ## The cases that rlba does not match dlba 
#' ##########################################
#' ## When approximated PDF is almost perfect with 2^20 simulations,
#' ## one can still observe noise in Bayesian computation. One possible reason
#' ## is the following:   
#' den1 <- gpda::n1PDF(data, b=.09, A=.07, mean_v=c(-7.37, -4.36), 
#' sd_v=c(1, 1), t0=.94, nsim=2^20, debug=T)
#' den2 <- rtdists::n1PDF(data, b=.09, A=.07, mean_v=c(-7.37, -4.36), 
#' sd_v=c(1, 1), t0=.94, silent=T)
#' par(mfrow=c(1,2))
#' plot(data, den1, type="l")
#' lines(data, den2, lwd=2)
#' 
#' plot(data, den2, type="l")
#' lines(data, den1, lwd=2)
#' all.equal(den1, den2)
#' ## [1] "Mean relative difference: 0.9991495"
#' 
#' ## Note both shapes are similar, but dlba estimates smaller values, 
#' ## relative to rlba. These happen in behaviourally implausible parameter 
#' ## sets. It takes relatively longer iterations to smooth out the noise.   
#' @export
n1PDF <- function(x, nsim = 1024, b = 1, A = 0.5, mean_v = c(2.4, 1.6),
  sd_v = c(1, 1), t0 = 0.5, nthread = 64, debug = FALSE) {
  if (any(sd_v < 0))   {stop("Standard deviation must be positive.\n")}
  if (any(sd_v == 0))  {stop("0 sd causes rtnorm to stall.\n")}
  if (length(b)  != 1) {stop("b must be a scalar.\n")}
  if (length(A)  != 1) {stop("A must be a scalar.\n")}
  if (length(t0) != 1) {stop("t0 must be a scalar.\n")}
  if (nsim %% 2 != 0 || nsim < 512) {stop("nsim must be power of 2 and at least 2^9.\n")}
  out <- .C("n1PDF", as.double(x), as.integer(length(x)), 
    as.integer(nsim),  as.double(b),  as.double(A),
    as.double(mean_v), as.integer(length(mean_v)), 
    as.double(sd_v),    
    as.double(t0),     as.integer(nthread), 
    as.logical(debug), numeric(length(x)),
    numeric(length(x)),
    PACKAGE='gpda')
  return(out[[12]])
}


#' @export
n1PDF_plba1 <- function(x, nsim = 1024, b=2.7, A=1.5, mean_v=c(3.3, 2.2), 
  mean_w=c(1.5, 3.7), sd_v=c(1, 1), rD=.3, swt=.5, t0=0.08, nthread=32, 
  debug=TRUE) {
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


#' @export
n1PDF_plba2 <- function(x, nsim = 1024, b=c(2.7, 2.7), A=c(1.5, 1.51), mean_v=c(3.3, 2.2),
  mean_w=c(1.5, 3.7), sd_v=c(1, 1), sd_w=c(1.5, 1.2), rD=.3, swt=.5, t0=0.08, nthread=32,
  debug=TRUE) {
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
    numeric(nsim), integer(nsim),
    numeric(length(x)), PACKAGE = "gpda")
  ##return(result[[16]])
  
  return(list(RT=result[[16]], R=result[[17]], Den=result[[18]]))
  
}


#' @export
n1PDF_plba3 <- function(x, nsim = 1024, B=c(1.2, 1.2), A=c(1.5, 1.5), C=c(.3, .3),
                        mean_v=c(3.3, 2.2), mean_w=c(1.5, 3.7), sd_v=c(1, 1),
                        sd_w=c(1, 1), rD=.1, tD=.1, swt=.5, t0=.08,
                        nthread=32, debug=TRUE) {
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
    numeric(nsim), integer(nsim),
    numeric(length(x)), PACKAGE = "gpda")
  ##return(result[[16]])
  return(list(RT=result[[18]], R=result[[19]], Den=result[[20]]))
  
}
