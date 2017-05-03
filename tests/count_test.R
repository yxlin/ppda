rm(list=ls())
dat1 <- gpda::rlba(2^15); str(dat1)
table(dat1$R)

tmp1 <- gpda::count(dat1$R)
tmp1

head(dat1)


res <- microbenchmark::microbenchmark(
  gpda::count(dat1$R),
  table(dat1$R),
  times=10L)

res
