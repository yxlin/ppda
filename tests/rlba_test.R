rm(list=ls())
n <- 2^20; n
dat1 <- gpda::rlba(n, nthread=64); str(dat1)
dat2 <- gpda::n1PDF(n, nsim=n, nthread=64); str(dat2)
dat3 <- rtdists::rLBA(n, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, silent=TRUE)
names(dat3) <- c("RT","R")

dat14 <- gpda::rlba_n1(n, nthread=64, dp=FALSE); str(dat14)


dat1c <- dat1[dat1[,2]==1, 1]
dat1e <- dat1[dat1[,2]==2, 1]
dat2c <- dat2[dat2[,2]==1, 1]
dat2e <- dat2[dat2[,2]==2, 1]
dat3c <- dat3[dat3[,2]==1, 1]
dat3e <- dat3[dat3[,2]==2, 1]

den1c <- density(dat1c)
den2c <- density(dat2c)
den3c <- density(dat3c)
den1e <- density(dat1e)
den2e <- density(dat2e)
den3e <- density(dat3e)

str(dat1); str(dat2); str(dat3)
summary(dat1c); summary(dat2c); summary(dat3c)
summary(dat1e); summary(dat2e); summary(dat3e)
round(rbind(table(dat1$R)/n, table(dat2$R)/n, table(dat3$R)/n), 3)

table(dat1$R)
table(dat2$R)
table(dat3$R)

head(dat2)

## Trim ----

sum(dat1$RT>5)
sum(dat2$RT>5)
sum(dat3$RT>5)

dat1 <- dat1[dat1$RT < 5, ]
dat2 <- dat2[dat2$RT < 5, ]
dat3 <- dat3[dat3$RT < 5, ]

dat1c <- dat1[dat1[,2]==1, 1]
dat1e <- dat1[dat1[,2]==2, 1]
dat2c <- dat2[dat2[,2]==1, 1]
dat2e <- dat2[dat2[,2]==2, 1]
dat3c <- dat3[dat3[,2]==1, 1]
dat3e <- dat3[dat3[,2]==2, 1]

den1c <- density(dat1c)
den2c <- density(dat2c)
den3c <- density(dat3c)
den1e <- density(dat1e)
den2e <- density(dat2e)
den3e <- density(dat3e)


par(mfrow=c(1,3))
hist(dat1c, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="GPU-Choice 1", xlim=c(0, 3)) ## gpu float
lines(den2c, col="red",  lty="dashed",  lwd=1.5) ## cpu
lines(den3c, col="blue", lty="dashed",  lwd=3.0) ## rtdists

hist(dat2c, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="CPU-Choice 1", xlim=c(0, 3)) ## cpu 
lines(den1c, col="red",  lty="dashed",  lwd=1.5) ## gpu float
lines(den3c, col="blue", lty="dashed",  lwd=3.0) ## rtdists


hist(dat3c, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="R-Choice 1", xlim=c(0, 3)) ## rtdists
lines(den1c, col="red",  lty="dashed",  lwd=1.5) ## gpu float
lines(den2c, col="blue", lty="dashed",  lwd=3.0) ## cpu
par(mfrow=c(1,1))

par(mfrow=c(1,3))
hist(dat1e, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="GPU-Choice 2", xlim=c(0, 3)) ## gpu
lines(den2e, col="red",  lty="dashed",  lwd=1.5) ## cpu
lines(den3e, col="blue", lty="dashed",  lwd=3.0) ## rtdists

hist(dat2e, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="CPU-Choice 2", xlim=c(0, 3)) ## cpu
lines(den1e, col="red",  lty="dashed",  lwd=1.5) ## gpu
lines(den3e, col="blue", lty="dashed",  lwd=3.0) ## rtdists

hist(dat3e, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="R-Choice 2", xlim=c(0, 3)) ## rtdists
lines(den1e, col="red",  lty="dashed",  lwd=1.5) ## gpu
lines(den2e, col="blue", lty="dashed",  lwd=3.0) ## cpu
par(mfrow=c(1,1))


plot(den1c$x, den1c$y, type="l")
lines(den1e$x, den1e$y)

lines(den2c$x, den2c$y, col="red", lwd=2, lty="dotted")
lines(den2e$x, den2e$y, col="red", lwd=2, lty="dotted")

lines(den3c$x, den3c$y, col="blue", lwd=2, lty="dashed")
lines(den3e$x, den3e$y, col="blue", lwd=2, lty="dashed")

plot(den1c$x, den1c$y, type="l")
lines(den1e$x, den1e$y)

lines(den2c$x, den2c$y, col="red", lwd=2, lty="dotted")
lines(den2e$x, den2e$y, col="red", lwd=2, lty="dotted")

lines(den3c$x, den3c$y, col="blue", lwd=2, lty="dashed")
lines(den3e$x, den3e$y, col="blue", lwd=2, lty="dashed")


library(microbenchmark)
res <- microbenchmark(gpda::rlba(n, dp=F),
                      gpda::rlba(n, dp=T),
                      cpda::rlba_test(n),
                      rtdists::rLBA(n, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, silent=TRUE),
  times=10L)

res



## Unit: milliseconds
##                  expr            min           lq         mean       median           uq         max neval cld
## gpda::rlba(n, dp = F)       8.310137     8.465464     9.171308     8.529089     9.213674    11.58632    10   a
## gpda::rlba(n, dp = T)      11.860449    11.955713    12.313426    12.061767    12.169148    14.85524    10   a
## cpda::rlba_test(n)        201.833239   202.047791   209.318144   202.836404   225.028719   225.71750    10   b
## rtdists::rLBA(n,  . )   13521.669045 13614.740048 13799.592982 13770.777719 13919.770909 14177.51434    10   c


n <- 2^20; n
library(microbenchmark)
res <- microbenchmark(gpda::rlba_n1(n, dp=F),
                      gpda::rlba_n1(n, dp=T),
  times=100L)

res


head(dat13)
tmp <- dat13[dat13[,2] == 2,]
unique(tmp$RT)

## dat2 <- gpda::rlba_test(n, nthread=64); str(dat2)
str(dat2)

tmp1 <- gpda::n1min(dat2$RT0)
tmp1

dat2no0 <- dat2[dat2$RT0!=0,]
base::min(dat2no0$RT0)

head(dat2)
nrow(dat2)

n1min(dat2)

tail(dat2)

min(dat1$RT); max(dat1$RT); sum(dat1$RT); sum(dat1$RT^2); sd(dat1$RT)

dat2 <- gpda::rlba_test(n, nthread=64); 
min(dat2$RT); max(dat2$RT); sum(dat2$RT); sum(dat2$RT^2); sd(dat2$RT)

str(dat2)
