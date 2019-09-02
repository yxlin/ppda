#' Parallel Probability Density Approximation
#'
#' Parallel probability density approximation (pPDA) uses graphics processing 
#' units (GPU) to conduct Monte Carlo simulations of the canonical and piecewise 
#' linear ballistic accumulation models. It enables fast likelihood 
#' approximation of cognitive/biological models.  
#'
#' @keywords ppda
#' @name ppda
#' @docType package
#' @author  Yi-Shin Lin <yishin.lin@utas.edu.au> \cr
#' Andrew Heathcote <andrew.heathcote@utas.edu.au> \cr
#' William Holmes <william.holmes@vanderbilts.edu>
#' @references 
#' Holmes, W. (2015). A practical guide to the Probability Density
#' Approximation (PDA) with improved implementation and error characterization.
#' \emph{Journal of Mathematical Psychology}, 68-69, 13--24.
#' \url{http://dx.doi.org/10.1016/j.jmp.2015.08.006} \cr
#' 
#' Holmes, W., Trueblood, J. S., & Heathcote, A. (2016). A new framework for
#' modeling decisions about changing information: The Piecewise Linear Ballistic
#' Accumulator model. \emph{Cognitive Psychology}, 85, 1--29. 
#' \url{http://dx.doi.org/10.1016/j.cogpsych.2015.11.002} \cr
#' 
#' Robert, C. P. (1995). Simulation of truncated normal variables. 
#' \emph{Statistics and computing}, 5(2), 121-125. 
#' \url{http://dx.doi.org/doi:10.1007/BF00143942} \cr
#' 
#' Silverman, B. W. (1982). Algorithm as 176: Kernel density estimation using 
#' the fast Fourier transform. \emph{Journal of the Royal Statistical Society. 
#' Series C (Applied Statistics)}, 31(1), 93-99. 
#' \url{http://www.jstor.org/stable/2347084} \cr
#'  
#' Silverman, B. W. (1986). \emph{Density estimation for statistics and data 
#' analysis}. Vol. 26. CRC press.  
#'  
#' @useDynLib ppda
NULL
