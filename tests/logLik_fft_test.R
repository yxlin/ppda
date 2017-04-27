## Use LBA data
rm(list=ls())
library(gpda)
tmp1 <- read.table("tests/Datavec.txt") ## From matlab
data <- as.double(sort(tmp1$V1))

# logLik_lba <- function(y, h=NA, m=.8, nThreads=2) 

dat1 <- logLik_lba(data, nThreads=32)  
h <- 0.8*bw.nrd0(data)
min(data) - 3*h
max(data) + 3*h

LL1 <- gpda::logLik_fft(data, simulation); 
LL2 <- cpda::logLik_fft(data, simulation)
cat("LL1 and LL2: ", LL1, "\t", LL2, "\n")

u <- 8.770941
l <- 1.369797
tmp1 <- seq(l, u, length.out=1024)
head(tmp1)
tail(tmp1)


dt <- (u-l)/1023
term1 <- tmp1 - dt/2
term2 <- tmp1[1024] + dt/2
out <- c(term1, term2)  
head(out)
tail(out)

head(cbind(tmp1, term1))


c(l+0*step, l+step, l+2*step)

l+1023*step
l+1024*step

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

