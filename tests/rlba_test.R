rm(list=ls())
n <- 2^13; n
dat1 <- gpda::rlba(n, nthread=64); str(dat1)

min(dat1$RT); max(dat1$RT); sum(dat1$RT); sum(dat1$RT^2); sd(dat1$RT)

dat2 <- gpda::rlba_test(n, nthread=64); 
min(dat2$RT); max(dat2$RT); sum(dat2$RT); sum(dat2$RT^2); sd(dat2$RT)

str(dat2)




dat2 <- cpda::rlba_test(n)
dat3 <- rtdists::rLBA(n, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, silent=TRUE)
names(dat3) <- c("RT","R")

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
  main="GPU-Choice 1", xlim=c(0, 3)) ## gpu
lines(den2c, col="red",  lty="dashed",  lwd=1.5) ## cpu
lines(den3c, col="blue", lty="dashed",  lwd=3.0) ## rtdists

hist(dat2c, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="CPU-Choice 1", xlim=c(0, 3)) ## cpu
lines(den1c, col="red",  lty="dashed",  lwd=1.5) ## gpu
lines(den3c, col="blue", lty="dashed",  lwd=3.0) ## rtdists


hist(dat3c, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="R-Choice 1", xlim=c(0, 3)) ## rtdists
lines(den1c, col="red",  lty="dashed",  lwd=1.5) ## gpu
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


hist(dat3c, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="R-Choice 2", xlim=c(0, 3)) ## rtdists
lines(den1c, col="red",  lty="dashed",  lwd=1.5) ## gpu
lines(den2c, col="blue", lty="dashed",  lwd=3.0) ## cpu
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
res <- microbenchmark(gpda::rlba(n),
  cpda::rlba_test(n),
  rtdists::rLBA(n, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, silent=TRUE),
  times=10L)

res
