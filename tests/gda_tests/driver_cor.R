numAvars <- 5
numBvars <- 10
numSamples <- 1e2
A <- matrix(runif(numAvars*numSamples), numSamples, numAvars)
B <- matrix(runif(numBvars*numSamples), numSamples, numBvars)

gda::gpuCor(A, B, method="pearson")
cor(A,B)

rm(list=ls())
goodPairs <- rpois(10, lambda=5)
coeffs <- runif(10)
gda::gpuTtest(goodPairs, coeffs)


require(rbenchmark)
within(benchmark(gpu=gda::gpuCor(A, B, method="pearson"),
                 cpu=cor(A,B),
                 replications=rep(1e2, 3),
                 columns=c('test', 'replications', 'elapsed'),
                 order=c('test', 'replications')),
                 { average = elapsed/replications })

#   test replications elapsed average
# 2  cpu          100   1.173 0.01173
# 4  cpu          100   1.342 0.01342
# 6  cpu          100   1.241 0.01241
# 1  gpu          100   6.630 0.06630
# 3  gpu          100   6.683 0.06683
# 5  gpu          100   6.798 0.06798
