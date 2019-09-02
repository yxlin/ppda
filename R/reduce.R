#' GPU Math Functions 
#'
#' \code{sum}, \code{min}, \code{max}, \code{minmax} and \code{sd} conduct basic
#' math operations on a vector, to find summation, minimal, maximal, minimal 
#' and maximal at one go, and standard deviation. These functions are meant to 
#' operate in GPU memory. These R entry points are mainly for testing. My tests
#' shown in the examples sugges that the R functions in base and stats packages
#' are faster.   
#' 
#' \code{count} is to count the numbers of responses in each response type in 
#' a 2AFC data set. Again, it meants to operate in GPU memory.
#' 
#' \code{n1min} is to find minimal value for node 1 (i.e., the first accumulator
#' in a 2 or more accumulator model). 
#' 
#' @param x a numeric vector. count function takes a vector with either 1 or 2,
#' standing for choice 1 or choice 2.
#' @param debug whether to print debugging information
#' @return a scalar or a pair of scalar values. 
#' @export
#' @examples
#' ## The smallest vector must be greater than 512 (= 2^9) elements
#' dat0   <- rlba(2^20, nthread = 64); str(dat0)
#' result <- min_gpu(dat0$RT); result
#' result <- max_gpu(dat0$RT); result
#' result <- minmax(dat0$RT); result
#' result <- sum_gpu(dat0$RT); result
#' result <- sd_gpu(dat0$RT); result
#' 
#' stats::sd(dat0$RT);
#' base::min(dat0$RT); 
#' base::max(dat0$RT)
#" base::sum(dat0$RT);
#' @export
sum_gpu <- function(x, debug=FALSE) {
    .C("sum_entry", as.double(x), as.integer(length(x)),
      as.logical(debug), numeric(1), PACKAGE='ppda')[[4]]
}

#' @rdname sum_gpu
#' @export
min_gpu <- function(x, debug=FALSE) {
  .C("min_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "ppda")[[4]]
}

#' @rdname sum_gpu
#' @export
max_gpu <- function(x, debug=FALSE) {
  .C("max_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "ppda")[[4]]
}

#' @rdname sum_gpu
#' @export
minmax <- function(x, debug=FALSE) {
  .C("minmax_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(2), PACKAGE = "ppda")[[4]]
}

#' @rdname sum_gpu
#' @export
sqsum <- function(x, debug=FALSE) {
  .C("sqsum_entry", as.double(x), as.integer(length(x)), 
    as.logical(debug), numeric(1),
    PACKAGE="ppda")[[4]]
}

#' @rdname sum_gpu
#' @export
sd_gpu <- function(x, debug=FALSE) {
  .C("sd_entry", as.double(x), as.integer(length(x)), 
    as.logical(debug), numeric(1), PACKAGE="ppda")[[4]]
}

#' @rdname sum_gpu
#' @export
count <- function(x, debug=FALSE) {
  .C("count_entry", as.integer(length(x)), as.integer(x),
    as.logical(debug), numeric(2), PACKAGE="ppda")[[4]]
}

#' @rdname sum_gpu
#' @export
n1min <- function(x, debug=FALSE) {
  .C("n1min_entry", as.double(x), as.integer(length(x)), as.logical(debug), 
    numeric(1), PACKAGE="ppda")[[4]]
}

#' Test if an integer is a power of 2
#' 
#' This is a convenient function to test if an integer number is a power of 2.
#' Because  the number of model simulation must be a power of 2 in 
#' \code{rlba} and \code{rplba}, this function helps to check this limitation. 
#' 
#' @param n an interger. 
#' @return a boolean TRUE or FALSE
#' @export
isp2 <- function(n) {
  ## test if n is a power of 2.
  if (n == 0) stop("n cannot be 0.")
  if (!is.integer(n)) stop("n must be an integer.")
  .C("isp2", as.integer(n), as.logical(1), PACKAGE = "ppda")[[2]]
}


## Beta testing code ----------------------
## The following function is to test different reduction algorithms to calculate
## sum and squared sum. They have not been thoroughly tested and it seems that
## these reduction algorithms do not run faster.  
sumur <- function(x, debug=TRUE) {
  .C("sumur_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "ppda")[[4]]
}

sqsumur <- function(x, debug=TRUE) {
  .C("sqsumur_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "ppda")[[4]]
}

sqsumurd <- function(x, debug=TRUE) {
  .C("sqsumurd_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "ppda")[[4]]
}
