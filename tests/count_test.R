rm(list=ls())
dat1 <- gpda::rlba(2^10)
str(dat1)
table(dat1$R)


tmp1 <- gpda::count(dat1$R)
tmp1

head(dat1)
