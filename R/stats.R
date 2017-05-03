#' GPU Math Functions 
#'
#' sum, min, max, minmax and sd are 5 basic mathematical functions to sum a
#' vector, to find minimal or maximal value, to find minimal and maximal
#' in one go, and to calculate standard deviation.
#' 
#' @param x a data vector.
#' @param debug whether to print debug information
#' @return a scalar or a pair of scalar values. 
#' @export
#' @examples
#' rm(list=ls())
#' ## min 512 = 2^9 elements
#' 
#' dat0   <- gpda::rlba(2^21, nthread=64); str(dat0)
#' result <- gpda::min(dat0$RT); result
#' result <- gpda::max(dat0$RT); result
#' result <- gpda::minmax(dat0$RT); result
#' result <- gpda::sum(dat0$RT); result
#' result <- gpda::sd(dat0$RT); result
#' 
#' stats::sd(dat0$RT);
#' base::min(dat0$RT); base::max(dat0$RT)
#" base::sum(dat0$RT);
#' @export
sum_gpu <- function(x, debug=TRUE) {
    .C("sum_entry", as.double(x), as.integer(length(x)),
      as.logical(debug), numeric(1), PACKAGE='gpda')[[4]]
}

#' @export
sumur <- function(x, debug=TRUE) {
    .C("sumur_entry", as.double(x), as.integer(length(x)),
      as.logical(debug), numeric(1), PACKAGE = "gpda")[[4]]
}

#' @export
sqsumur <- function(x, debug=TRUE) {
    .C("sqsumur_entry", as.double(x), as.integer(length(x)),
      as.logical(debug), numeric(1), PACKAGE = "gpda")[[4]]
}

#' @export
sqsumurd <- function(x, debug=TRUE) {
    .C("sqsumurd_entry", as.double(x), as.integer(length(x)),
      as.logical(debug), numeric(1), PACKAGE = "gpda")[[4]]
}


#' @rdname sum
#' @export
min_gpu <- function(x, debug=FALSE) {
  .C("min_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "gpda")[[4]]
}


#' @rdname sum
#' @export
max_gpu <- function(x, debug=FALSE) {
  .C("max_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(1), PACKAGE = "gpda")[[4]]
}

#' @rdname sum
#' @export
minmax <- function(x, debug=FALSE) {
  .C("minmax_entry", as.double(x), as.integer(length(x)),
    as.logical(debug), numeric(2), PACKAGE = "gpda")[[4]]
}

#' @rdname sum
#' @export
sqsum <- function(x, debug=FALSE) {
  .C("sqsum_entry", as.double(x), as.integer(length(x)), 
    as.logical(debug), numeric(1),
    PACKAGE="gpda")[[4]]
}


#' @rdname sum
#' @export
sd_gpu <- function(x, debug=FALSE) {
  .C("sd_entry", as.double(x), as.integer(length(x)), 
    as.logical(debug), numeric(1), PACKAGE="gpda")[[4]]
}

#' @rdname sum
#' @export
count <- function(R, debug=FALSE) {
  .C("count_entry", as.integer(length(R)), as.integer(R),
    as.logical(debug), numeric(2), PACKAGE="gpda")[[4]]
}


#' @rdname sum
#' @export
n1min <- function(x, debug=TRUE) {
  .C("n1min_entry", as.double(x), as.integer(length(x)), as.logical(debug), 
    numeric(1), PACKAGE="gpda")[[4]]
}
