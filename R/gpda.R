#' Probability Density Approximation Using Graphics Processing Unit 
#'
#' The package uses general purpose graphics processing unit (GP-GPU) 
#' to conduct Monte Carlo simulations of basic and piece-wise linear ballistic
#' accumulator models. The simulations are then used to approximate the model
#' likelihood via a slihgtly modified method described in Homles (2015). 
#'
#' @keywords gpda
#' @name gpda
#' @docType package
#' @author  Yi-Shin Lin <yishin.lin@utas.edu.au> \cr
#' Andrew Heathcote <andrew.heathcote@utas.edu.au> \cr
#' William Holmes <william.holmes@vanderbilts.edu>
#' @references Holmes, W. (2015). A practical guide to the Probability Density
#' Approximation (PDA) with improved implementation and error characterization.
#' \emph{Journal of Mathematical Psychology}, \bold{68-69}, 13--24,
#' doi: http://dx.doi.org/10.1016/j.jmp.2015.08.006.
#' @useDynLib gpda
NULL
