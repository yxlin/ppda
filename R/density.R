#' Compute Log-likelihood via KDE-FFT 
#'
#' This function uses GPU to calculate log-likelihood via KDE-FFT 
#' algorithm. 
#'
#''\code{logLik_fft1} returns summed log-likelihood.  \code{logLik_fft2} returns
#' a matrix storing individual data point (column 1) and log-likelihood for 
#' these data points (column 2).
#' 
#' @param y a vector storing empirical data (e.g., RTs)
#' @param yhat a vector storing simulated data.
#' @param h KDE bandwidth
#' @param m a multiplier to adjust \code{h} proportationally. Default is 0.8. 
#' Set \code{m=1}, if one wish not adjust bandwidth. 
#' @param p a precision parameter defines the number of grid as power of 2.
#' Default value is 10 (i.e., \code{ngrid=2^10}). This is for \code{logLik_fft}.
#' @param nThreads number of threads per block. Default is 2
#' @return Log-likelihood. 
#' @references Holmes, W. (2015). A practical guide to the Probability Density
#' Approximation (PDA) with improved implementation and error characterization.
#' \emph{Journal of Mathematical Psychology}, \bold{68-69}, 13--24,
#' doi: http://dx.doi.org/10.1016/j.jmp.2015.08.006.
#' @export
#' @examples
#' ## ----- Use piecewise LBA data  ----- ## 
#' ## Use piecewise LBA data as an example ----
#' data(lba); bandwidth <- .02;
#' tmp1 <- cpda::logLik_fft(plba$DT1, plba$eDT1, bandwidth)
#' tmp2 <- gpda::logLik_fft(plba$DT1, plba$eDT1, bandwidth)
#' tmp1
#' tmp2
#' 
#' tmp3 <- cpda::logLik_pw(plba$DT2, plba$eDT2, bandwidth)
#' tmp4 <- gpda::logLik_pw(plba$DT2, plba$eDT2, bandwidth)
#' sum(tmp3) 
#' sum(tmp4)
#' 
#' ## Set ngrid to 2^20
#' tmp5 <- gpda::logLik_fft(plba$DT1, plba$eDT1, bandwidth, p=20)
#' 
#' ## ----- Use MATLAB data ----- ## 
#' tmp1 <- read.table("tests/Datavec.txt") ## From matlab
#' y <- as.double(sort(tmp1$V1))
#' tmp2 <- read.table("tests/sampvec.txt") ## From matlab
#' yhat <- as.double(sort(tmp2$V1))
#' ## or use R to simulate; -1435.704
#' yhat <- as.double(sort(rnorm(ns, 5, 1)))
#' ns <- as.integer(length(yhat))
#' ny <- as.integer(length(y))
#' 
#' ## Calculate pointwise likelihood  
#' LL3 <- gpda::logLik_fft(y, yhat, pointwise=TRUE); 
#' LL4 <- gpda::logLik_fft2(plba$DT1, plba$eDT1, bandwidth)
#' sum(LL4[,2])  ## Column 1 store data; column 2 store log-likelihoods
#' 
logLik_fft <- function(y, yhat, h=NA, m=.8, p=10, nThreads=2) {
  y    <- as.double(sort(y))
  yhat <- as.double(sort(yhat))
  ny   <- as.integer(length(y))
  ns   <- as.integer(length(yhat))
  h_   <- ifelse(is.na(h), 0, as.double(m*h))
  m_   <- as.double(m)
  nt   <- as.integer(nThreads)
  p_   <- as.integer(p)
  .C("logLik_fft", y, yhat, ny, ns, h_, m_, p_, nt, as.double(1), 
    PACKAGE='gpda')[[9]]
}

#' @rdname logLik_fft
#' @export
logLik_fft2 <- function(y, yhat, p=10, nThreads=2) {
  y    <- as.double(sort(y))
  yhat <- as.double(sort(yhat))
  ny   <- as.integer(length(y))
  ns   <- as.integer(length(yhat))
  nt   <- as.integer(nThreads)
  p_   <- as.integer(p)
  out  <- .C("logLik_fft2", y, yhat, ny, ns, p_, nt, 
    numeric(ny), PACKAGE='gpda')[[7]]
  return(cbind(y, out))
}

#' Calculating Two-accumulator LBA Densities Using PDA Method 
#'
#' This is the density function of the cannoical 2-accumualtor LBA model, 
#' sampling drift rates from the truncated normal distribution. The function
#' uses probability density approximation to estimate likelihood. 
#' 
#' @param data a data frame. Its 1st column must be RT and the 2nd
#' column must be R with level 1 and 2. The column names must be c('RT', 'R').
#' data for n1PDF is a vector. 
#' @param nsim number of simulations, passed to internal rlba_gpu. The default 
#' is 1e5.  
#' @param b threshold. Default is 1.  
#' @param A starting point upper bound  
#' @param mean_v mean drift rate. This must be a two-element vector. 
#' @param sd_v standard deviation of drift rate. This must be a two-element 
#' vector.
#' @param t0 non-decision time 
#' @param nThread number of threads launched in GPU. Default is 32. Max is 1024.
#' @return Likelihood. 
#' @references Holmes, W. (2015). A practical guide to the Probability Density
#' Approximation (PDA) with improved implementation and error characterization.
#' \emph{Journal of Mathematical Psychology}, \bold{68-69}, 13--24,
#' doi: http://dx.doi.org/10.1016/j.jmp.2015.08.006.
#' @export
#' @examples
#' ## Use either rlba_gpu in 'gpda' or rLBA in 'rtdists' to generate LBA random
#' ## observations. 'rtdists' uses c(rt, response) as column names, so I change
#' ## it to our standard c('RT', 'R') 
#' ## dat1 <- rlba_gpu(n, nThreads=1024); dplyr::tbl_df(dat1)
#' n <- 1e2
#' dat1 <- rtdists::rLBA(n, A=.5, b=1, mean_v=c(2.4, 1.6), sd_v=1, t0=.5, 
#' silent=T); 
#' names(dat1) <- c("RT","R")
#' 
#' ## Get the densities with dlba_gpu
#' den3 <- dlba_gpu(dat1, nsim=2e5, nThread=32)
#' 
#' ## Check den3 against n1PDF in 'lba' package and dLBA in 'rtdists' package
#' den.c <- as.vector(lba::n1PDF(dat1[dat1$R==1, "RT"], A=0.5, b=1, 
#'                    mean_v=c(2.4,1.6), sd_v=1, t0=0.5))
#' den.e <- as.vector(lba::n1PDF(dat1[dat1$R==2, "RT"], A=0.5, b=1, 
#'                    mean_v=c(1.6,2.4), sd_v=1, t0=0.5))
#' ## Bind densities with RT, for plotting                  
#' den2.c <- data.frame(RT=dat1[dat1$R==1, "RT"], D=den.c)
#' den2.e <- data.frame(RT=dat1[dat1$R==2, "RT"], D=den.e)
#' den2 <- rbind(den2.c,den2.e)  ## den2 from 'lba' package
#' 
#' ## den1 from 'rtdists' package
#' den1 <- rtdists::dLBA(dat1$RT, dat1$R, A=.5, b=1, mean_v=c(2.4,1.6), sd_v=1, 
#'      t0=.5, silent=T)
#'      
#' ## Again bind data with densities       
#' den1 <- data.frame(RT=dat1$RT, R=dat1$R, D=den1); 
#' den1.c <- den1[den1$R==1,"D"]
#' den1.e <- den1[den1$R==2,"D"]
#' 
#' ## Verify the 'lba' and 'rtdists' produce identical densities
#' all.equal(den.c, den1.c)
#' all.equal(den.e, den1.e)
#' 
#' ## Calculate densities, using dlba_gpu.  Call it den3
#' den3 <- dlba_gpu(dat1, nsim=2e5, nThread=32)
#' 
#' ## dlba_gpu return a 2-element list. First element is choice 1 (correct
#' ## responses) and 2nd element is choice 2 (error responses). 
#' den3.c <- data.frame(RT=dat1[dat1$R==1, "RT"], D=den3$C1[,2])
#' den3.e <- data.frame(RT=dat1[dat1$R==2, "RT"], D=den3$C2[,2])
#' den3 <- rbind(den3.c,den3.e)
#' 
#' ## Plot the densities calculated from three different packages, separately,
#' par(mfrow=c(1,3))
#' plot(den1$RT, den1$D)
#' plot(den2$RT, den2$D, pch=2, col="red")
#' plot(den3$RT, den3$D, pch=3, col="darkgreen")
#' 
#' ## and together
#' par(mfrow=c(1,1))
#' plot(den1$RT, den1$D)
#' points(den2$RT, den2$D, pch=2, col="red")
#' points(den3$RT, den3$D, pch=3, col="darkgreen")
#' 
#' ## n1PDF examples
#' 
#' data <- seq(0, 1, length.out = 10)
#' gpda::n1PDF(data)
#' 
#' ## Test an unreasonable parameter vector ----
#' gpda::n1PDF(10, b=3.04, A=1.76, mean_v = c(2.79, -7.04), sd_v=c(1, 1), 
#'  t0=0.35)
#'  
#' ## Test a safe parameter vector  
#' n <- 100
#' A <- .5; b <- 1; t0 <- .5
#' mean_v <- c(2.4, 1.6)
#' sd_v <- c(1,1)
#' 
#' ## You can test an unreasonable parameter vector, too.
#' ## But note that this set won't produce error RTs.  
#' # n <- 100
#' # A <- 1.76; b <- 3.04; t0 <- .35
#' # mean_v <- c(2.79, -7.04)
#' # sd_v <- c(1,1)
#' 
#' dat1 <- rtdists::rLBA(n, A=A, b=b, mean_v=mean_v, sd_v=sd_v, t0=t0, silent=T); 
#'              
#' names(dat1) <- c("RT","R")
#' datc1 <- dat1[dat1$R==1, "RT"]
#' date1 <- dat1[dat1$R==2, "RT"]
#' str(dat1); str(datc1); str(date1)
#'
#' ## Use lba, cpda and gpda package
#' denc1 <- as.vector(lba::n1PDF(datc1, A=A, b=b, 
#'          mean_v=mean_v, sd_v=sd_v, t0=t0))
#' dene1 <- as.vector(lba::n1PDF(date1, A=A, b=b, 
#'          mean_v=c(mean_v[2], mean_v[1]), sd_v=sd_v, t0=t0))
#' denc2 <- gpda::n1PDF(datc1, nsim=1e5, A=A, b=b, 
#'          mean_v=mean_v, sd_v=sd_v, t0=t0)
#' dene2 <- gpda::n1PDF(date1, nsim=1e5, A=A, b=b, 
#'          mean_v=c(mean_v[2], mean_v[1]), sd_v=sd_v, t0=t0)
#' denc3 <- cpda::n1PDF(datc1, nsim=1e5, A=A, b=b, 
#'          mean_v=mean_v, sd_v=sd_v, t0=t0)
#' dene3 <- cpda::n1PDF(date1, nsim=1e5, A=A, b=b, 
#'          mean_v=c(mean_v[2], mean_v[1]), sd_v=sd_v, t0=t0)
#'          
#' ## First, plot correct RTs
#' plot(datc1, denc1)
#' points(datc1, denc2, col="grey")
#' points(datc1, denc3, col="lightblue")
#' ## Then error RTs
#' points(date1, dene1, pch=2)
#' points(date1, dene2, pch=2, col="grey")
#' points(date1, dene3, pch=2, col="lightblue")
#' @export
dlba_gpu <- function(data, nsim=1e5, b=1, A=0.5, mean_v=c(2.4, 1.6),
                     sd_v=c1, t0=0.5, nThread=32) {
    if (sd_v < 0) {stop("Standard deviation must be positive.\n")}
    RT0  <- as.double(data[data[,2]==1, 1])
    RT1  <- as.double(data[data[,2]==2, 1])
    nRT0 <- as.integer(length(RT0))
    nRT1 <- as.integer(length(RT1))
    nsim  <- as.integer(nsim)
    b  <- as.double(b)
    A  <- as.double(A)
    v  <- as.double(mean_v)
    nv <- as.integer(length(mean_v))
    sv <- as.double(sd_v)
    t0 <- as.double(t0)
    nThread <- as.integer(nThread)

    out <- .C("dlba_gpu", RT0,   RT1, nRT0, nRT1,  
                          nsim,   b,  A, 
                          v,     nv,  sv, t0, 
                          nThread, numeric(nRT0), numeric(nRT1),PACKAGE='gpda')
    return(list(C1=cbind(RT0, out[[14]]), 
                C2=cbind(RT1, out[[15]])))
}


#' @rdname dlba_gpu
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
    ##as.logical(debug), numeric(nsim), integer(nsim),
    numeric(length(x)),
    PACKAGE='gpda')
  #cat("Receive call from lba_B-gpda.R\n")

  return(out[[12]])
  ##result <- data.frame(RT = out[[12]], R = out[[13]])
  ##names(result) <-  c("RT", "R")
  ##return(result)
}


