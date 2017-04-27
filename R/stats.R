#' GPU Math Functions 
#'
#' sum, min, max, minmax and sd are 5 basic mathematical functions to sum a
#' vector, to find minimal or maximal value, to find minimal and maximal
#' in one go, and to calculate standard deviation.
#' 
#' @param x a data vector.
#' @param nthread numbers of thread to launch the kernel
#' @param debug whether to print debug information
#' @return a scalar or a pair of scalar values. 
#' @export
#' @examples
#' rm(list=ls())
#' ## min 512 = 2^9 elements
#' 2^21
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
sum <- function(x, nthread=32, debug=FALSE) {
    .C("sum_entry", as.double(x), as.integer(length(x)),
      as.integer(nthread), as.logical(debug), numeric(1),
      PACKAGE='gpda')[[5]]
}


#' @rdname sum
#' @export
min <- function(x, nthread = 32, debug=FALSE) {
  .C("min_entry", as.double(x), as.integer(length(x)),
    as.integer(nthread), as.logical(debug), numeric(1),
    PACKAGE = "gpda")[[5]]
}

#' @rdname sum
#' @export
max <- function(x, nthread = 32, debug=FALSE) {
  .C("max_entry", as.double(x), as.integer(length(x)),
    as.integer(nthread), as.logical(debug), numeric(1),
    PACKAGE = "gpda")[[5]]
}


#' @rdname sum
#' @export
minmax <- function(x, nthread = 32, debug=FALSE) {
  .C("minmax_entry", as.double(x), as.integer(length(x)),
    as.integer(nthread), as.logical(debug), numeric(2),
    PACKAGE = "gpda")[[5]]
}

#' @rdname sum
#' @export
sqsum <- function(x, nthread=32, debug=FALSE) {
  .C("sqsum_entry", as.double(x), as.integer(length(x)), 
    as.integer(nthread), as.logical(debug), numeric(1),
    PACKAGE="gpda")[[5]]
}

#' @rdname sum
#' @export
sd <- function(x, nthread=32, debug=FALSE) {
  .C("sd_entry", as.double(x), as.integer(length(x)), 
    as.integer(nthread), as.logical(debug), numeric(1),
    PACKAGE="gpda")[[5]]
}

#' @rdname sum
#' @export
count <- function(R, nthread=32, debug=FALSE) {
  .C("count_entry", as.integer(length(R)), as.integer(R),
    as.integer(nthread), as.logical(debug), numeric(2),
    PACKAGE="gpda")[[5]]
}

