size <- 1e6
matA <- matrix(runif(size*2), size, 2)
matB <- matrix(runif(size*4), size, 4)
gpuCrossprod(matA, matB)
crossprod(matA, matB)

require(rbenchmark)
within(benchmark(gpu=gda::gpuCrossprod(matA, matB),
                 cpu=crossprod(matA, matB),
                 replications=rep(10, 3),
                 columns=c('test', 'replications', 'elapsed'),
                 order=c('test', 'replications')),
       { average = elapsed/replications })
