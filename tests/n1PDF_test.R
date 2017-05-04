rm(list=ls())

##   A    b  mean_v sd_v,    t0
##0.07 0.09 -7.37   -4.36  0.94  0.049 

data <- seq(0, 3, length.out = 1e3);
den1 <- gpda::n1PDF(data)
den2 <- rtdists::n1PDF(data, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), 
                        t0=.5, silent=T)

plot(data, den1, type="l")
lines(data, den2, lwd=2)
all.equal(den1, den2)

den1 <- gpda::n1PDF(data, nsim=2^20)

plot(data, den1, type="l")
lines(data, den2, lwd=2)
all.equal(den1, den2)


den1 <- gpda::n1PDF(data, b=.09, A=.07, mean_v=c(-7.37, -4.36), sd_v=c(1, 1), 
  t0=.94, nsim=2^20, debug=T)
den2 <- rtdists::n1PDF(data, b=.09, A=.07, mean_v=c(-7.37, -4.36), sd_v=c(1, 1),
  t0=.94, silent=T)

par(mfrow=c(1,2))
plot(data, den1, type="l")
lines(data, den2, lwd=2)

plot(data, den2, type="l")
lines(data, den1, lwd=2)
all.equal(den1, den2)



tmp5 <- cpda::rlba_test(2^20, b=pvec[2], A=pvec[1], mean_v=mean_v, sd_v=sd_v, 
  t0=pvec[3])
tmp6 <- rtdists::rLBA(2^20, b=pvec[2], A=pvec[1], mean_v=mean_v, sd_v=sd_v, 
  t0=pvec[3], silent=T)
names(tmp6) <- c("RT","R")




all.equal(as.vector(tmp1), as.vector(tmp2))
all.equal(as.vector(tmp1), as.vector(tmp3))
all.equal(as.vector(tmp1), as.vector(tmp4))
all.equal(as.vector(tmp2), as.vector(tmp3))
all.equal(as.vector(tmp2), as.vector(tmp4))

round(tmp1, 2)
round(tmp4, 3)

dat1 <- gpda::rlba(2^20, b=pvec[2], A=pvec[1], mean_v=mean_v, sd_v=sd_v, 
  t0=pvec[3], nthread=64); str(dat1)
dat2 <- cpda::rlba_test(2^20, b=pvec[2], A=pvec[1], mean_v=mean_v, sd_v=sd_v, 
  t0=pvec[3]); str(dat2)
dat3 <- rtdists::rLBA(2^20, b=pvec[2], A=pvec[1], mean_v=mean_v, sd_v=sd_v,
  t0=pvec[3], silent=T)
names(dat3) <- c("RT","R"); str(dat3)

table(dat1$R)
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


lines(data,tmp1)
lines(data,tmp4, lwd=3, col="red")

plot(data,tmp4, lwd=3, col="red", type="l")

par(mfrow=c(1,3))
hist(dat1c, breaks="fd",  col="grey", freq=FALSE, xlab="RT (s)", 
  main="GPU-Choice 1", xlim=c(0, 3)) ## gpu float
lines(den2c, col="red",  lty="dashed",  lwd=1.5) ## cpu
lines(den3c, col="blue", lty="dashed",  lwd=3.0) ## rtdists
lines(den12c, col="green", lty="dashed",  lwd=4.0) ## gpu double

plot(den1c$x, den1c$y, type="l", ylim=c(0, 5))
lines(den1e$x, den1e$y)

lines(den2c$x, den2c$y, col="red", lwd=2, lty="dotted")
lines(den2e$x, den2e$y, col="red", lwd=2, lty="dotted")

lines(den3c$x, den3c$y, col="blue", lwd=2, lty="dashed")
lines(den3e$x, den3e$y, col="blue", lwd=2, lty="dashed")

plot(den1c$x, den1c$y, type="l", ylim=c(0,5))
lines(den1e$x, den1e$y)

lines(den2c$x, den2c$y, col="red", lwd=2, lty="dotted")
lines(den2e$x, den2e$y, col="red", lwd=2, lty="dotted")

lines(den3c$x, den3c$y, col="blue", lwd=2, lty="dashed")
lines(den3e$x, den3e$y, col="blue", lwd=2, lty="dashed")



all.equal(as.vector(tmp3), as.vector(tmp4))

base::min(tmp1$RT[tmp1$RT!=0])
base::max(tmp1$RT[tmp1$RT!=0])
base::max(tmp1$RT)
table(tmp1$R)

base::sum(tmp1$RT==0)
base::sum(tmp1$RT)
base::sum(tmp1$RT^2)

stats::sd(tmp1$RT[tmp1$RT!=0])

tmp2 <- cpda::n1PDF(data, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, nsim=2^20, debug=F)
tmp3 <- clba::n1PDFfixedt0(data, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5)
tmp4 <- rtdists::n1PDF(data, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, silent=T)







## NB1: nsim = 1e6
## NB2: lba:n1PDF and rtdists::n1PDF used analytic probability density function
## NB3: gpda::n1PDF use double precision
## Unit: microseconds
## expr                                 min         lq        mean      median         uq        max neval cld
## lba::n1PDF(data, ...)             35.970     65.513     82.3868     87.5135     94.148    115.031    10   a
## rtdists::n1PDF(data, ...)        355.428    482.959    533.1549    523.8525    598.199    777.553    10   a
## gpda::n1PDF_float(data, ...)  115802.444 116215.700 122231.5587 116987.1755 124510.151 145378.491    10   b
## gpda::n1PDF(data, ...)        120642.014 121562.045 133174.1721 124230.2235 149902.235 159919.483    10   b
## cpda::n1PDF(data, ...)        310518.881 328194.527 355060.7977 351149.1915 370555.805 451738.621    10   c

n <- 2^20
2^20
res <- microbenchmark::microbenchmark(
                           gpda::n1PDF(data, nsim=1048576, debug=F),
                           lba::n1PDF(data, A=0.5, b=1, mean_v=c(2.4,1.6), sd_v=c(1,1), t0=0.5),
                           cpda::n1PDF(data, nsim=n, A=0.5, b=1, mean_v=c(2.4,1.6), sd_v=c(1,1),
                                       t0=0.5, debug=F),
                           rtdists::n1PDF(data, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, silent=T),
                           times=10L)

res


## Test an unreasonable parameter vector ----
tmp1 <- gpda::n1PDF(10, b=3.04, A=1.76, mean_v = c(2.79, -7.04), sd_v=c(1, 1),
  t0=0.35)
tmp2 <- cpda::n1PDF(10, b=3.04, A=1.76, mean_v = c(2.79, -7.04), sd_v=c(1, 1),
  t0=0.35)
round(tmp1,5)
round(tmp2,5)

## range ----
rm(list=ls())
n <- 100
# A <- 1.76; b <- 3.04; t0 <- .35
# mean_v <- c(2.79, -7.04)
# sd_v <- c(1,1)
A <- .5; b <- 1; t0 <- .5
mean_v <- c(2.4, 1.6)
sd_v <- c(1,1)


## dat1 <- rtdists::rLBA(n, A=.5, b=1, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, silent=T);
dat1 <- rtdists::rLBA(n, A=A, b=b, mean_v=mean_v, sd_v=sd_v, t0=t0, silent=T);
names(dat1) <- c("RT","R")
datc1 <- dat1[dat1$R==1, "RT"]
date1 <- dat1[dat1$R==2, "RT"]
str(dat1); str(datc1); str(date1)

denc1 <- as.vector(lba::n1PDF(datc1, A=A, b=b, mean_v=mean_v, sd_v=sd_v, t0=t0))
dene1 <- as.vector(lba::n1PDF(date1, A=A, b=b, mean_v=c(mean_v[2], mean_v[1]), sd_v=sd_v, t0=t0))

denc2 <- gpda::n1PDF(datc1, nsim=1e5, A=A, b=b, mean_v=mean_v, sd_v=sd_v,
  t0=t0)
dene2 <- gpda::n1PDF(date1, nsim=1e5, A=A, b=b, mean_v=c(mean_v[2], mean_v[1]),
  sd_v=sd_v,
  t0=t0)

denc3 <- cpda::n1PDF(datc1, nsim=1e5, A=A, b=b, mean_v=mean_v, sd_v=sd_v,
  t0=t0)
dene3 <- cpda::n1PDF(date1, nsim=1e5, A=A, b=b, mean_v=c(mean_v[2], mean_v[1]),
  sd_v=sd_v,
  t0=t0)

str(datc1); str(denc1); str(denc2)
str(date1); str(dene1); str(dene2)


sort(round(denc1,2))
sort(round(denc2,2))
sort(round(denc3,2))

plot(datc1, denc1)
points(datc1, denc2, pch=2, col="grey")
points(datc1, denc3, pch=2, col="lightblue")

points(date1, dene1, col="lightblue")
points(date1, dene2, pch=2, col="blue")



res <- microbenchmark::microbenchmark(
  lba::n1PDF(datc1, A=0.5, b=1, mean_v=c(2.4,1.6), sd_v=c(1,1), t0=0.5),
  gpda::n1PDF(datc1, nsim=1e6, A=0.5, b=1, mean_v=c(2.4,1.6), sd_v=c(1,1),
    t0=0.5),
  cpda::n1PDF(datc1, nsim=1e6, A=0.5, b=1, mean_v=c(2.4,1.6), sd_v=c(1,1),
    t0=0.5),
  times=10L)
res


## Test out-of-range ----
gpda::n1PDF(10, b=1.04, A=.44, t0=.18, mean_v=c(2.2, 1.6), sd_v=c(1,1))

# [b, A, t0]: [1.040 0.436 0.179 ]
# mean_v and sd_v[0]: [2.160 1.000]
# mean_v and sd_v[1]: [1.611 1.000]
