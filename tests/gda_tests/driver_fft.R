## Use LBA data
library(gda)

tmp1 <- read.table("tests/Datavec.txt") ## From matlab
tmp2 <- read.table("tests/sampvec.txt") ## From matlab
data       <- as.double(sort(tmp1$V1))
simulation <- as.double(sort(tmp2$V1))

LL3    <- gda::logLik_fft(data, simulation, pointwise=TRUE); 
LL4    <- gda::logLik_fft(data, simulation, pointwise=FALSE); 

data(lba)
bandwidth <- .02;
nsample <- length(plba$eDT1)

tmp1 <- logLik_fft(y=plba$DT1, yhat=plba$eDT1)
str(tmp1)

## Use Gaussian data ---------
## -1.4362e+03
## or use R to simulate; -1435.704
## yhat <- as.double(sort(rnorm(ns, 5, 1)))

ns <- as.integer(length(yhat))
ny <- as.integer(length(y))
require(gda)
## as.double(1) is to leave a memory space for output


LL2    <- gda::logLik_fft(y, yhat, h=NA, pointwise=FALSE, nThreads=32)
LL2    <- gda::logLik_fft(y, yhat); LL2


answer <- .C("logLik_fft", y, yhat, ny, ns, 0, as.double(1), PACKAGE='gda')
LL3    <- answer[[6]]; LL3


for(i in 1:10) {
  tmp0 <- gda::logLik_fft(plba$DT2[i], plba$eDT2)
  cat("LL: ", round(tmp0, 3),"\t")
}


## Use pda package
m <- min(y)
M <- max(y)
pda::logLik_fft(y, yhat, m, M, h, ns)


require(rbenchmark)
within(benchmark(GPU=answer1 <- .C("logLik_fft", y, yhat, ny, ns, h, as.double(1), 
                                   PACKAGE='gda'),
                 CPU=answer2 <- pda::logLik_fft(y, yhat, m, M, h, ns),
                 replications=rep(1e5, 3),
                 columns=c('test', 'replications', 'elapsed'),
                 order=c('test', 'replications')),
       { average = elapsed/replications })

