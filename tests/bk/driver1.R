## Test logLik_pw, logLik_fft and logLik_fft2 ------------------------
gpda::runif_gpu(10, 1, 2)
gpda::rnorm_gpu(10)

answer <- .C("runif_gpu", as.integer(10),  as.double(0), as.double(1), 
  as.integer(2), numeric(10), PACKAGE='gpda')
str(answer)



rbenchmark::benchmark(replications=rep(3, 3),
          gpda::rnorm_gpu(3e5),
          stats::rnorm(3e5), 
          columns=c('test', 'elapsed', 'replications'))


require(rbenchmark)
set.seed(123)
n    <- 1e5
obs  <- rnorm(n)
h    <- bw.nrd0(obs);h             ## KDE Bandwidth  = 0.08997601
x    <- seq(-3, 3, length.out=100) ## Support

## logLik_pw returns point-wise log-likelihood. That is, it
## uses Gaussian kernel to calculate log-likelihood for each observation with 
## bandwidth = h * 0.8.
benchmark(replications=rep(3, 1),
          pw_g <- gpda::logLik_pw(x, obs),
          pw_c <- cpda::logLik_pw(x, obs),
          columns=c('test', 'elapsed', 'replications'))
##                              test elapsed replications
## 2 pw_c <- cpda::logLik_pw(x, obs)   8.921            3
## 1 pw_g <- gpda::logLik_pw(x, obs)   8.853            3

system.time(pw_g <- gpda::logLik_pw(x, obs))
system.time(pw_c <- cpda::logLik_pw(x, obs))
system.time(pw_tmp1 <- cpda::logLik_pw(x, obs, h=h, m=0.6))
system.time(pw_tmp2 <- gpda::logLik_pw(x, obs, h=h, m=0.6))
system.time(fft1_g <- gpda::logLik_fft(x, obs))
system.time(fft1_c <- cpda::logLik_fft(x, obs))
system.time(fft2_g <- gpda::logLik_fft2(x, obs))
system.time(fft2_c <- cpda::logLik_fft2(x, obs)[["PDF"]])

fft1_c
fft1_g
sum(pw_g)
sum(pw_c)
sum(fft2_g[,2])
sum(fft2_c[,2])

head(fft2_g)
head(fft2_c)

benchmark(replications=rep(3, 1),
          fft1_g <- gpda::logLik_fft(x, obs),
          fft1_c <- cpda::logLik_fft(x,obs),
          columns=c('test', 'elapsed', 'replications'))
##                                 test elapsed replications
## 2 fft1_c <- cpda::logLik_fft(x, obs)   2.456            3
## 1 fft1_g <- gpda::logLik_fft(x, obs)   2.211            3

benchmark(replications=rep(3, 1),
          fft2_g <- gpda::logLik_fft2(x, obs),
          fft2_c <- cpda::logLik_fft2(x, obs)[["PDF"]],
          columns=c('test', 'elapsed', 'replications'))
##                                           test elapsed replications
## 2 fft1_c <- cpda::logLik_fft2(x, obs)[["PDF"]]   2.619            3
## 1          fft1_g <- gpda::logLik_fft2(x, obs)   2.170            3

## See R help page.
?cpda::logLik_pw
?gpda::logLik_pw


system.time(fft1_g <- gpda::logLik_fft2(x,obs))
system.time(fft1_c <- cpda::logLik_fft2(x,obs)[["PDF"]])


png(filename="tests/gaussian.png", width=1024, height=768)
plot(x,  exp(pw_g),  type="l",         lty="dotted")
lines(x, exp(fft2_g[,2]), col="green", lty="dotdash")
lines(x, dnorm(x), col="red",  lty="longdash")
dev.off()


## Test genRng.R  ------------------------
unirng_c <- runif(1e3)
unirng_g <- gpda::runif_gpu(1e3)

plot(sort(unirng_g),  dunif(sort(unirng_g)),  type="l", lty="dotted")
lines(sort(unirng_c), dunif(sort(unirng_c)), col="red",  lty="longdash", lwd=3)

unirng_c <- rnorm(1e3)
unirng_g <- gpda::rnorm_gpu(1e3)

plot(sort(unirng_g),  dnorm(sort(unirng_g)), type="l", lty="dotted")
lines(sort(unirng_c), dnorm(sort(unirng_c)), col="red",  lty="longdash")

## Test dexG  ------------------------
x    <- seq(-3, 3, length.out=100) ## Support
dexGaussian(x)

## Use piecewise LBA data as an example ----
data(lba); bandwidth <- .02;
tmp1 <- cpda::logLik_fft(plba$DT1, plba$eDT1, bandwidth)
tmp2 <- gpda::logLik_fft(plba$DT1, plba$eDT1, bandwidth)
tmp1;
tmp2

tmp3 <- cpda::logLik_pw(plba$DT2, plba$eDT2, bandwidth)
tmp4 <- gpda::logLik_pw(plba$DT2, plba$eDT2, bandwidth)
sum(tmp3); 
sum(tmp4)

tmp5 <- gpda::logLik_fft(plba$DT1, plba$eDT1, bandwidth, p=20)
str(tmp5)


## Compare arma::accu and arma::sum
## Result: accu is slightly faster on average
rbenchmark::benchmark(replications=rep(300, 1),
          tmp1 <- gpda::logLik_fft(plba$DT1, plba$eDT1, bandwidth),
          tmp2 <- gpda::logLik_fft2(plba$DT1, plba$eDT1, bandwidth),
          columns=c('test', 'elapsed', 'replications'))

tmp1 <- cpda::logLik_fft(plba$DT1, plba$eDT1, bandwidth)
tmp2 <- cpda::logLik_fft2(plba$DT1, plba$eDT1, bandwidth)
str(tmp2)
dat <- tmp2[[2]]
sum((dat[,2]))


tmp2 <- gpda::logLik_fft2(plba$DT1, plba$eDT1, bandwidth)
sum(tmp2[,2])

dat1 <- head(dat)
dat2 <- head(cbind(y, tmp2))

h <- NA
m <- .8
nThreads <- 2
p <- 10
y    <- as.double(sort(plba$DT1))
yhat <- as.double(sort(plba$eDT1))
ny   <- as.integer(length(y))
ns   <- as.integer(length(yhat))
h_   <- ifelse(is.na(h), as.double(m*stats::bw.nrd0(yhat)), as.double(m*h))
nt   <- as.integer(nThreads)
p_   <- as.integer(p)
numeric(ny)
out  <- .C("logLik_fft2", y, yhat, ny, ns, h_, p_, nt, numeric(ny), PACKAGE='gpda')[[8]]
str(out)

## Use Holmes's MATLAB data ----
tmp1 <- read.table("tests/Datavec.txt") ## From matlab
tmp2 <- read.table("tests/sampvec.txt") ## From matlab
## or use R to simulate; -1435.704
## yhat <- as.double(sort(rnorm(ns, 5, 1)))

## Calculate pointwise likelihood  
LL1 <- gpda::logLik_fft(tmp1, tmp2); ## Use GPU
LL2 <- gpda::logLik_pw(y, yhat);  ## Use CPU  
LL1; sum(LL2)


