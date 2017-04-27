#' The Random Number Generator of Uniform Distribution 
#'
#' This function generates random numbers in [0, 1), using GPU. 
#'
#' @param x vector of quantiles. 
#' @param mu vector of Gaussian mean.
#' @param sigma vector of scale parameter.
#' @param nu vector of nu parameter.
#' @param nThreads number of threads launched per block.
#' @return a double vector
#' @export
#' @examples
#' gpda::dexG(10)
dexGaussian <- function(x, mu=5, sigma=1, nu=1, nThreads=2) {
  if (any(sigma <= 0) ) stop("sigma must be greater than 0 \n ") 
  if (any(nu <= 0) ) stop("nu must be greater than 0 \n") 
  n <- length(x)
  .C("dexG", as.integer(n), as.double(x),  as.double(mu), as.double(sigma), 
     as.double(nu), as.integer(nThreads), numeric(n), PACKAGE='gpda')[[5]]
}

