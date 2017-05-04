rm(list=ls())
## sum test
dat0   <- gpda::rlba(2^20, nthread=256); str(dat0)
result <- gpda::sum_gpu(dat0$RT, debug=T); result
base::sum(dat0$RT)
result <- gpda::sumur(dat0$RT, debug=T); result


res <- microbenchmark::microbenchmark(
                           gpda::sum_gpu(dat0$RT, debug=FALSE),
                           gpda::sumur(dat0$RT, debug=FALSE),
                           base::sum(dat0$RT),
  times=100L)

res





result <- gpda::min(dat0$RT, debug=T); result
result <- gpda::minunroll(dat0$RT,debug=T); result


result <- gpda::min(dat0$RT, debug=T); result
result <- gpda::max(dat0$RT); result
result <- gpda::minmax(dat0$RT); result
result <- gpda::sd(dat0$RT); result

base::sum(dat0$RT); 
base::min(dat0$RT); base::max(dat0$RT)
minmax <- function(dat) {c(base::min(dat), base::max(dat))}
minmax(dat0)
stats::sd(dat0$RT);

res <- microbenchmark::microbenchmark(
  gpda::sd(dat0$RT, debug=FALSE),
  sd(dat0$RT),
  times=100L)

res


result <- gpda::n1min(dat0)












res <- microbenchmark::microbenchmark(
  gpda::minmax(dat0$RT, debug=FALSE),
  findminmax(dat0$RT),
  times=100L)

res


res <- microbenchmark::microbenchmark(
  gpda::max(dat0$RT, debug=FALSE),
  gpda::maxtest(dat0$RT, debug=FALSE),
  base::max(dat0$RT),
  times=100L)

res



res <- microbenchmark::microbenchmark(
  gpda::min(dat0$RT, debug=FALSE),
  gpda::min(dat0$RT, debug=TRUE),
  base::min(dat0$RT),
  times=100L)

res


res <- microbenchmark::microbenchmark(gpda::sum(dat0$RT, debug=FALSE),
                                      base::sum(dat0$RT),
                               times=100L)
res

res <- microbenchmark::microbenchmark(gpda::sqsum(dat0$RT, debug=FALSE),
                                      base::sum(dat0$RT^2),
                               times=100L)

res
