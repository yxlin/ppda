#' GPU Math Functions 
#'
#' sum, min, max, minmax and sd conduct basic math operations on a vector, to
#' find summation, minimal, maximal, minimal and maximal at one go, and
#' standard deviation. These functions are meant to operate in GPU memory. These
#' R entry points are for testing. 
#' 
#' count is to count the numbers of responses in each response type in a 2AFC
#' data set. Again, it meants to operate in GPU memory.
#' 
#' n1min designs to find minimal value for node 1. 
#' 
#' @param x a numeric vector. count function takes a vector with either 1 or 2,
#' standing for choice 1 or choice 2.
#' @param debug whether to print debugging information
#' @return a scalar or a pair of scalar values. 
#' @export
#' @examples
#' 
#' ## The smallest vector must be no less than 512 (= 2^9) elements
#' dat0   <- rlba(2^20, nthread=64); str(dat0)
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
      as.logical(debug), numeric(1), PACKAGE='gpda')[[4]]
}



#' @rdname sum_gpu
#' @export
min_gpu <- function(x, debug=FALSE) {
  .C("min_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "gpda")[[4]]
}


#' @rdname sum_gpu
#' @export
max_gpu <- function(x, debug=FALSE) {
  .C("max_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "gpda")[[4]]
}

#' @rdname sum_gpu
#' @export
minmax <- function(x, debug=FALSE) {
  .C("minmax_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(2), PACKAGE = "gpda")[[4]]
}

#' @rdname sum_gpu
#' @export
sqsum <- function(x, debug=FALSE) {
  .C("sqsum_entry", as.double(x), as.integer(length(x)), 
    as.logical(debug), numeric(1),
    PACKAGE="gpda")[[4]]
}


#' @rdname sum_gpu
#' @export
sd_gpu <- function(x, debug=FALSE) {
  .C("sd_entry", as.double(x), as.integer(length(x)), 
    as.logical(debug), numeric(1), PACKAGE="gpda")[[4]]
}

#' @rdname sum_gpu
#' @export
count <- function(x, debug=FALSE) {
  .C("count_entry", as.integer(length(x)), as.integer(x),
    as.logical(debug), numeric(2), PACKAGE="gpda")[[4]]
}


#' @rdname sum_gpu
#' @export
n1min <- function(x, debug=FALSE) {
  .C("n1min_entry", as.double(x), as.integer(length(x)), as.logical(debug), 
    numeric(1), PACKAGE="gpda")[[4]]
}


sumur <- function(x, debug=TRUE) {
  .C("sumur_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "gpda")[[4]]
}

sqsumur <- function(x, debug=TRUE) {
  .C("sqsumur_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "gpda")[[4]]
}

sqsumurd <- function(x, debug=TRUE) {
  .C("sqsumurd_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "gpda")[[4]]
}
