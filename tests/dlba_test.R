rm(list=ls())
require(gpda)

n    <- 10
dat1 <- gpda::rlba(n, b=3.04, A=1.76, mean_v = c(2.79, -7.04), sd_v=1, t0=0.35)
dat2 <- cpda::rlba()
dat1 <- rtdists::rLBA(n, A=1.76, b=3.04, mean_v=c(2.79, -7.04), 
  sd_v=1, t0=0.35, silent=F); 

0-2.79
0+7.04
0.5 * (sqrt(2.79*2.79+4) - 2.79)

rnorm(10, -7.04, 1)
tmp1 <- 1.76*runif(1e5)
tmp2 <- tnorm::rtnorm(1e5, -7.04, 1, lower=0, upper=Inf)
dt1 <- (3.04-tmp1)/tmp2
hist(tmp2, breaks="fd")
hist(dt1, breaks="fd")

head(dat1)
hist(dat1$rt, breaks="fd")
summary(dat1$rt)
table(dat1$response)

sum(is.infinite(dat1$RT))
sum(is.infinite(dat1$R))


# den.c <- as.vector(lba::n1PDF(dat1[dat1$R==1, "RT"], A=0.5, b=1, 
#   mean_v=c(2.4,1.6), sd_v=1, t0=0.5))
# den.e <- as.vector(lba::n1PDF(dat1[dat1$R==2, "RT"], A=0.5, b=1, 
#   mean_v=c(1.6,2.4), sd_v=1, t0=0.5))

den2.c <- data.frame(RT=dat1[dat1$R==1, "RT"], D=den.c)
den2.e <- data.frame(RT=dat1[dat1$R==2, "RT"], D=den.e)
den2 <- rbind(den2.c,den2.e)

den1 <- rtdists::dLBA(dat1$RT, dat1$R, A=.5, b=1, mean_v=c(2.4,1.6), sd_v=1, t0=.5, silent=T)
den1 <- data.frame(RT=dat1$RT, R=dat1$R, D=den1); dplyr::tbl_df(den1)
den1.c <- den1[den1$R==1,"D"]
den1.e <- den1[den1$R==2,"D"]
all.equal(den.c, den1.c)
all.equal(den.e, den1.e)


den3 <- dlba_gpu(dat1, nsim=2e5, nThread=32)
head(cbind(den3$C1, den.c))
head(cbind(den3$C2, den.e))


round(den3$C1[,2]  - den.c, 2)
round(den3$C2[,2]  - den.e, 2)

den3.c <- data.frame(RT=dat1[dat1$R==1, "RT"], D=den3$C1[,2])
den3.e <- data.frame(RT=dat1[dat1$R==2, "RT"], D=den3$C2[,2])
den3 <- rbind(den3.c,den3.e)


par(mfrow=c(1,3))
plot(den1$RT, den1$D)
plot(den2$RT, den2$D, pch=2, col="red")
plot(den3$RT, den3$D, pch=3, col="darkgreen")
par(mfrow=c(1,1))

plot(den1$RT, den1$D)
points(den2$RT, den2$D, pch=2, col="red")
points(den3$RT, den3$D, pch=3, col="darkgreen")


## Unit: milliseconds
##                                                                                                       expr
##                                                                dlba_gpu(dat1, nsim = 1e+05, nThread = 512)
##  rtdists::dLBA(dat1$RT, dat1$R, A = 0.5, b = 1, mean_v = c(2.4,      1.6), sd_v = 1, t0 = 0.5, silent = T)
##       min       lq     mean   median       uq       max neval cld
##  6.371201 6.507254 6.799131 6.622912 6.886566 12.354989    50   b
## 1.100641 1.127811 1.162635 1.159483 1.189271  1.297736    50  a
pVec <- c(b1=1, b2=1, A1=.5, A2=.5, mu1=2.4, mu2=1.6, sigma1=1, sigma2=1,
          t01=.5, t02=.5)

library(microbenchmark)
res <- microbenchmark(dlba_gpu(dat1, nsim=1e5, nThread=32),
                      rtdists::dLBA(dat1$RT, dat1$R, A=.5, b=1, mean_v=c(2.4,1.6), sd_v=1, t0=.5, silent=T),
                      rtdists::rLBA(1e5, A=.5, b=1, mean_v=c(2.4,1.6), sd_v=1, t0=.5, silent=T),
                      cpda::rlba(1e5, pVec),
                      times=10L)
res

## Unit: milliseconds
##                                                                                                       expr
##                                                                 dlba_gpu(dat1, nsim = 1e+05, nThread = 32)
##  rtdists::dLBA(dat1$RT, dat1$R, A = 0.5, b = 1, mean_v = c(2.4,      1.6), sd_v = 1, t0 = 0.5, silent = T)
##            rtdists::rLBA(1e+05, A = 0.5, b = 1, mean_v = c(2.4, 1.6), sd_v = 1,      t0 = 0.5, silent = T)
##                                                                                    cpda::rlba(1e+05, pVec)
##          min          lq        mean      median          uq         max neval
##     6.128742    6.373957    7.325938    6.611038    6.796784   14.207622    10
##     1.071453    1.131518    1.143489    1.139061    1.157429    1.191932    10
##  1383.860027 1411.418002 1461.351988 1445.932039 1474.183621 1597.549716    10
##    18.236829   18.303737   18.999417   18.327205   20.560766   20.736978    10
##  cld
##   a 
##   a 
##   b
##   a 









