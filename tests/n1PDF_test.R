rm(list=ls())

data <- seq(0, 3, length.out = 1e3)
tmp1 <- gpda::n1PDF(data, nsim=2^22)
tmp2 <- cpda::n1PDF(data, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, nsim=1e5, debug=F)
tmp3 <- lba::n1PDFfixedt0(data, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5)
tmp4 <- rtdists::n1PDF(data, b=1, A=.5, mean_v=c(2.4, 1.6), sd_v=c(1, 1), t0=.5, silent=T)
tmp5 <- gpda::n1PDF_test(data, nsim=2^20)


all.equal(as.vector(tmp1), as.vector(tmp3))
all.equal(as.vector(tmp1), as.vector(tmp4))
all.equal(as.vector(tmp2), as.vector(tmp3))
all.equal(as.vector(tmp2), as.vector(tmp4))
all.equal(as.vector(tmp3), as.vector(tmp4))

all.equal(as.vector(tmp5), as.vector(tmp3))
all.equal(as.vector(tmp5), as.vector(tmp4))


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

res <- microbenchmark::microbenchmark(
                           gpda::n1PDF(data, nsim=1e5),
                           gpda::n1PDF(data, nsim=2^20),
                           gpda::n1PDF_test(data, nsim=2^20),
                           gpda::n1PDF_test(data, nsim=2^21),
                           lba::n1PDF(data, A=0.5, b=1, mean_v=c(2.4,1.6), sd_v=c(1,1), t0=0.5),
                           cpda::n1PDF(data, nsim=2^20, A=0.5, b=1, mean_v=c(2.4,1.6), sd_v=c(1,1),
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
